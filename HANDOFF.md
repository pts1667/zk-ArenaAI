# HANDOFF: zk-ArenaAI, the learned policy

Updated 2026-11-10.  Branch `learned-policy-plan`.  Repo: `C:/Users/Thoma/Documents/sources/zk-ArenaAI`.

The current state only.  How things got here is in git history.  How everything works, with its numbers, is in
`docs/learned-policy/` (start at `00-overview.md`) and `docs/model.md`.

## Goal and status

The goal is a Zero-K AI that beats CircuitAI, pursued as a Pluto-style learned policy.  It's a network trained in a simplified
simulator (`sim/`): first by cloning scripted players, then DAgger, then PPO.

* **Nothing plays in the engine yet.**  `bots/zkbot` hosts the policy: it chooses a commander and sends no commands.  The rules
  bot that was there is at the tag `rules-bot-final` (local only).
* **In the sim**, both sides play a scripted opening.  The C++ economy script runs their constructors (or the policy does, in a
  share of PPO games).  The C++ army script, or the policy, runs their fighters, factories, missiles and the Trinity.
* **The actor** (`tools/policy_latent.py`, 10.5M parameters) reads a frozen frame encoder's 40 latent tokens a frame, over a
  window of 4 frames in each of 4 layers.  It decides every 3 s, as actions one after another: a unit, then its order.
* **One command runs everything** (`tools/policy_pipeline.py`): frames, encoder, 10,000 steps of cloning as `clone-8` with z
  from the variant, two DAgger rounds, an eval, PPO.  While it runs it serves a dashboard at `http://localhost:8765`: the log as
  it's written, each stage's results, and replays of test games (4 after each test, 4 every hour of PPO).
* **`runs/full-1` is its first whole run** (2026-11-10), against the army script, told its variant:
  * the clone picks a unit the scripts told 91% of the time held out, and wins 5% in play;
  * DAgger took it to 13% in one round, the second adding nothing.  The stall in expansion stays: at 6 minutes it has about
    half the script's extractors;
  * PPO reached 22-31% test wins by iteration 88 (360,552 decisions), with explained variance 0.94, when the user stopped it.
    It's not marked done: delete `full-1/ppo/stop` and run the pipeline's command again to resume.
* **PPO plays as CUDA graphs** (`policy_graphs.py`), optionally with collector processes (`policy_collectors.py`): 158 and 178
  decisions a second against 92 eager.  The GPU is now the limit.  A round's action steps stop at the first cut where every
  decision is done: 6% more with 4 concurrent collectors (186 against 175 from `full-1`'s PPO actor), and no more memory.
* **The KL to the clone and the cloning loss fade by iteration** (`--kl-fade`, `--clone-fade`, the user, 2026-11-10): from
  their weights at iteration 0 to 20% at 100 and nothing at 500.  The KL had fallen over the run's decisions instead, and was
  still 0.48 at iteration 88 of `full-1`, about 0.14 of the loss against the policy gradient's 0.003; the cloning loss, kept
  at 0.5, was about 1.2.  Resumed, `full-1` goes on at 0.15 for each.  Once the cloning loss is gone the army script stops
  advising, and it never advises in tests.
* **The policy loss is SPO's** (Simple Policy Optimization, the user, 2026-11-10) in place of PPO's clip, and dual-clip is
  gone (`--dual-clip` no longer exists).  The stage and its directory are still called PPO.  Checked by a unit test of the
  objective, a short CPU run, and `spo-1` below.
* **`runs/spo-1`** (2026-11-10, the user's): SPO and the fades from `full-1`'s DAgger actor, as `full-1`'s PPO began, with
  `--collectors 4 --concurrent` (156 decisions a second with tests).  The user stopped it at iteration 140 (573,440 of 4.3M
  decisions, `ppo/ckpt/it-000140.pt`), with test wins of 12-28% and no trend (03-training.md, "What the earlier runs showed").
  Its clone, DAgger rounds and evals are links into `full-1` (don't `--redo` them there).  It played the scripts as they were
  before the economy change below; resumed, it would play the new ones.
* **The economy script's factories** (the user, 2026-11-10).  It used to spend floating metal on copies of its first factory, 3
  by minute 10 and up to 6.  Now it builds one plate beside the factory, then caretakers in reach of both.  The second factory
  comes at 40 metal a second (the user's rule), at a median of 7.6 minutes in the sim, and there's no third.  Two more
  constructors are made while metal floats.  On 48 games, a side's value at 19 minutes went from 27,800 to 31,700.
  * The sim now models plates (`unit_factory_plate.lua`: they build only near their factory) and idle caretakers (they help what's
    being built in their reach, as Zero-K's default widget has them patrol).  01-simulator.md, "Economy" and "The economy script".
  * Plates are factories in the policy's type table (masks only; FEATURE_VERSION stays 7, since no data held one).
  * `clone/army-6` was recorded by the old scripts, so a clone of it learns factory copies: new recordings are needed first.

## Invalid since 2026-11-10 (FEATURE_VERSION 7)

The id tables were resized for free-for-all (`featurize.MAX_PLAYERS` 64, `MAX_ALLY_TEAMS` 8).  So:

* `runs/encoder-big` and every clone (`clone-7`, `clone-8`, `clone-7s`, and older) no longer load;
* `frames/pool4` and `pool5` were stored featurized, so they were rejected too; they're deleted.

The recorded games (`clone/army-6`, its `z.npz`) are raw and still good.  **`runs/full-1/encoder/vae-last.pt` is the new
encoder** (recon R² 0.95).  A pipeline run gives it with `--encoder` instead of `--train-encoder`, which would collect 51 GB of
frames and train another.  `policy_latent.ENCODER` still names `encoder-big`.

## Environment

* **Zero-K**: portable install at `C:/Users/Thoma/Downloads/zero-k-portable`; engine 2026.07.04; Zero-K v1.14.10.3;
  ControllerAI 0.3.
* **Read-only sources**: `../RecoilEngine` (CircuitAI under `AI/Skirmish/CircuitAI`), `../Zero-K`, `../ControllerAI`.
* **Windows Python 3.13** runs the bot (`pip install requests websockets orjson numpy numba`).  Without numba, featurize runs in
  plain Python, about 50x slower, and says so on import.
* **WSL (Ubuntu)** runs the sim and training: `~/zkml/bin/python` (PyTorch, RTX 4070), with pytest in `~/zksim/pt`.  So
  `PYTHONPATH=$HOME/zksim/pt:bots`.
* **The sim's libraries**: `cmake -S sim -B ~/zksim/build && cmake --build ~/zksim/build -j8`.  That builds `libzksim.so`,
  `libzkscripts.so` and `libzkrecord.so` (`ZKSIM_LIB` overrides).  The recorder needs `libzstd-dev`.  Rebuild after any change
  under `sim/`.
* **`sim/data/maps`** is gitignored: after a pull, `python tools/sim_data.py maps`.
* **Data on D:**, outside git, under `/mnt/d/zk-learned` (174 GB free on 2026-11-10):

  | Path | What |
  |---|---|
  | `maps/` | the map pool: 98 real and 200 made-up maps, 21 held out.  About 7.5 GB in memory when loaded |
  | `clone/army-6` | what's cloned: 83,820 script games with variants, 300.6M decisions, 105 GB, data version 6, with `z.npz` and `sides.npz` |
  | `clone/army-6-test` | about 1,000 games of the same, for `check` and smoke runs |
  | `evals/` | the six closed-loop evals of `clone-7`, `clone-8` and `clone-7s` |
  | `frames/` | encoder frames: only `smoke-run` now (`pool4` and `pool5` deleted) |
  | `runs/full-1` | the first whole pipeline run: its frames (51 GB), encoder, clone, DAgger games (530 MB) and rounds, evals, PPO (1.4 GB), replays |
  | `runs/` | every other training run; of those, since feature version 7 none of the encoders or actors load.  `ppo-rate`, `ppo-profile`, `ppo-aligned` and `ppo-collectors-*` are scratch timing runs |
  | `dashboard-demo/` | a made-up run directory: earlier runs' logs and 10 replays, for working on the dashboard |
  | `dashboard-cache/` | the dashboard's map pictures and icons, made once |

## Layout

* **`bots/zkbot/`**: the engine side.
  * `main.py` connects to ControllerAI and drives `host.py`, which keeps what a raw frame is read from and runs
    `policy/runtime.py` once a decision.
  * `policy/`: the raw frame schema (`rawframe.py`, which also generates `sim/include/zksim/rawframe.hpp` via
    `tools/policy_schema.py`), the map bundle, type tables, featurizer (`featurize.py`, `featurize_kernels.py`), actions and
    group orders.
  * Unit data: `units.py`, `catalog.py`, `data/*.json`, rebuilt by `tools/extract_weapons.py`, `extract_abilities.py` and
    `dump_defs.py`.  Terrain: `terrain.py`, `hazards.py`, `mapdata.py`.
* **`sim/`**: the C++ simulator (`src/`), the economy and army scripts (`scripts/`), the recorder (`record/`), their ctypes
  bindings (`py/`, and `tools/policy_scripts_cpp.py` for the scripts), data (`data/`) and tests (`tests/`).
* **`tools/`**:
  * the sim's data and checks against the engine: `sim_data.py`, `sim_engine.py`, `sim_battery.py`, `sim_batch_bench.py`,
    `map_gen.py`;
  * the sim as a game: `policy_sim_env.py`, `policy_opening.py`, `policy_variants.py`, and the scripts' tables (`policy_eco.py`,
    `policy_mix.py`, `policy_script.py`);
  * frames and the encoder: `frame_world.py`, `frame_data.py`, `frame_model.py`, `frame_vae.py`;
  * the policy: `policy_model.py`, `policy_latent.py`, `policy_clone_data.py`, `policy_clone_train.py`, `policy_clone_eval.py`,
    `policy_train.py`, `policy_pipeline.py`, `policy_golden.py`;
  * PPO's play: `policy_graphs.py` (each round as a CUDA graph), `policy_collectors.py` (collector processes);
  * the run's dashboard: `policy_dashboard.py` (server), `dashboard/` (page, replay viewer), `policy_replay.py` (test games
    kept for it).
* **Tests**: `tests/policy/`, `tests/test_*.py`, `sim/tests/`.  None needs the engine.
* **Engine games**: `zk-startup.ps1`, `zk-maps.ps1`, `maps.md`, `README.md`.

## Running

In WSL, from `/mnt/c/Users/Thoma/Documents/sources/zk-ArenaAI`, with `PYTHONPATH=$HOME/zksim/pt:bots`:

    ~/zkml/bin/python -m pytest -q sim/tests tests                     # about 3 minutes; all pass
    ~/zkml/bin/python tools/policy_pipeline.py --root /mnt/d/zk-learned/runs/NAME --train-encoder [--dry-run]
        # everything, stage by stage; the same command again carries on.  A `stop` file in ROOT halts it after the running stage.
        # While it runs: the dashboard at http://localhost:8765 (log, results, replays)
    ~/zkml/bin/python tools/policy_dashboard.py --root /mnt/d/zk-learned/runs/NAME     # the dashboard alone, during a run or after
    ~/zkml/bin/python tools/policy_replay.py demo --out DIR     # replays of the army scripts' games, for the viewer
    ~/zkml/bin/python tools/policy_replay.py repair DIR...      # replays written before format 2, with the right units' names
    ~/zkml/bin/python tools/policy_clone_train.py --data /mnt/d/zk-learned/clone/army-6 --maps /mnt/d/zk-learned/maps \
        --run /mnt/d/zk-learned/runs/NAME --decide-every 6 --samples 8 --accumulate 8 --lr 6e-4 --muon-lr 0.008 --workers 15 \
        --steps 10000 --z --z-dropout 0.2 --economy-weight 2.5 --finish-winners 0.1 --encoder ENCODER/vae-last.pt
    ~/zkml/bin/python tools/policy_clone_eval.py --actor RUN/actor.pt --maps /mnt/d/zk-learned/maps --games 64 --workers 12 \
        --z variant --encoder ENCODER/vae-last.pt --out RESULT.json   # against the army script; --record DIR keeps its games

Recording, `check` and `zindex` are in 03-training.md, "Script data"; PPO's flags in "PPO", and how it plays (aligned
decisions, graphs, `--collectors N`, `--concurrent`) in "Playing".  A `stop` file in a PPO run's directory ends it at its next
report, and the same command resumes it.

`--encoder` is needed by every tool that loads an actor until `policy_latent.ENCODER` names a new encoder: its default is
`encoder-big`, which no longer loads.

In the engine (Windows), recording raw frames from a game: `$env:ZKBOT_FLAGS='rawframe'`, then
`.\zk-startup.ps1 ControllerAI CircuitAI:Normal -Headless -Map 'TitanDuel 2.2' -NoDownload -Client 'python bots\zkbot\main.py {team}'`.
README.md has the rules for engine games: headless at max speed, windowed only when the user asks.

## Next

1. **PPO's speed, next**: with concurrent collectors, training (about 22 s an iteration) is now longer than collecting (12 s).
   In a round's replay the decision itself (encoder, time transformer, critic: 41 ms at 64 decisions) is the bigger part; a
   step is about 1 ms of small kernels, which fusing (`torch.compile` of a step, now the shapes are fixed) could shrink.
2. **More variety before the next whole run** (the user, 2026-11-10; in progress, none of it in the tree yet): team games,
   several alliances, free-for-all with up to 8 players and chicken defense, each on the maps that support it (`maps.json`);
   a generator for Violet Rampart, whose terrain is made at game start; a small reward for winning sooner; and the lead rule
   (2.5x the value) drawn per game, from 90% of games at PPO's iteration 0 to none at 100.
3. **Then run the pipeline again whole**: record the scripts' games with the new economy (plates, caretakers, the second
   factory at 40 metal a second) and the new kinds of game, train an encoder on frames of them, clone, DAgger, PPO.
   `clone/army-6` teaches factory copies.
4. **The stall in expansion** survived DAgger.  Whether PPO's economy share (the policy running its own economy in more games)
   moves the extractor count is the thing to watch in its tests.
5. **The engine hand-off**:
   * a numpy forward of the actor and encoder for the bot;
   * decoded orders sent as commands;
   * the opening and the economy script in the engine;
   * the checks listed in 02-policy.md, "Not yet checked live".
6. **The user's open questions** (`docs/model.md`, "Open questions").

## Traps

* **Leftover build directories**: `~/zksim/build` is the one in use (libzksim ABI 18, libzkscripts ABI 8, rebuilt
  2026-11-10).  `~/zksim/build-next`, `build-head` and the other `build-*` beside it are leftovers of comparisons, safe to delete.
* **Memory**: check `wsl free -g` before a launch.  Never run two jobs that load the map pool at once.  Every GPU trainer caps
  itself at 85% of the card (`--gpu-memory`): past the card, the Windows driver spills into system RAM instead of failing.
  With collectors, the learner's cap leaves each `--collector-gb` plus about 0.4 GB of CUDA's own.
* **Forking with torch**: a child forked after the parent ran a parallel CPU op hangs at its first one (OpenMP), so collectors
  set one thread first.  A parent that has started CUDA can't fork children that use it: the trainer forks its collectors
  before the learner touches the GPU, and a test of them on CUDA runs the trainer in a fresh process.
* **Pinned memory in WSL**: a collector's 20 MB `cuMemHostAlloc` failed as out of memory with 7 GB of the card free.  The
  graphs' inputs aren't pinned.
* **`pkill -f` / `pgrep -f` from `wsl bash -c '...'`** match that shell too, whose command line holds the pattern.  Write the
  pattern as `[p]olicy_train`.  A `kill -STOP` that way stopped the shell itself.
* **A background process started by `wsl bash -lc 'nohup X &'`** dies when that call returns.  `setsid nohup X > LOG 2>&1 <
  /dev/null &` outlives it.
* **The dashboard's port**: a standalone `policy_dashboard.py` left on 8765 makes the pipeline run without its own (it says so
  and carries on).  Stop it before a launch, or give the pipeline another `--port`.
* **Engine games**: never start one, or kill engine processes, while the user is playing.  Check
  `tasklist | grep -i "spring\|zero-k"` first.
* **`set_settings`** at game start can make the engine exit, so the host sends it after frame 2.  A sync-mode client that stops
  calling `finish_frame` stalls the match.
* **numba's cache** knows only the kernel file, so it holds rawframe's numbers as literals.  They're checked on import.
* **A cloning run appends** to its `log.jsonl`: clear it before rerunning a run by hand.  The pipeline sets the old one aside.
* **Line endings**: several files are CRLF.  Edit them without changing their line endings.
* **Never committed**: `results/` and `.vscode/`.
