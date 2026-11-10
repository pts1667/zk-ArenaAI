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
* **Cloning works open-loop and stalls in play.**  Held out, `clone-8` picks a unit the scripts told 86% of the time.  Against
  the army script in the sim, every clone wins 2-5%: each keeps up for 3 minutes, then stops expanding.  DAgger is the next
  step, and is built.
* **One command runs everything** (`tools/policy_pipeline.py`): encoder, 10,000 steps of cloning as `clone-8` with z from the
  variant, two DAgger rounds, an eval, PPO.  Not run yet; the user starts it.  While it runs it serves a dashboard at
  `http://localhost:8765`: the log as it's written, each stage's results, and replays of test games (4 after each test, 4 every
  hour of PPO).

## Invalid since 2026-11-10 (FEATURE_VERSION 7)

The id tables were resized for free-for-all (`featurize.MAX_PLAYERS` 64, `MAX_ALLY_TEAMS` 8).  So:

* `runs/encoder-big` and every clone (`clone-7`, `clone-8`, `clone-7s`, and older) no longer load;
* `frames/pool4` and `pool5` were stored featurized, so they were rejected too; they're deleted.

The recorded games (`clone/army-6`, its `z.npz`) are raw and still good.  **The pipeline must run with `--train-encoder`** until a
new encoder exists.  It collects fresh frames into `ROOT/frames` (about 51 GB, as `pool5` took; D: has 228 GB free).

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
* **Data on D:**, outside git, under `/mnt/d/zk-learned` (228 GB free on 2026-11-10):

  | Path | What |
  |---|---|
  | `maps/` | the map pool: 98 real and 200 made-up maps, 21 held out.  About 7.5 GB in memory when loaded |
  | `clone/army-6` | what's cloned: 83,820 script games with variants, 300.6M decisions, 105 GB, data version 6, with `z.npz` and `sides.npz` |
  | `clone/army-6-test` | about 1,000 games of the same, for `check` and smoke runs |
  | `evals/` | the six closed-loop evals of `clone-7`, `clone-8` and `clone-7s` |
  | `frames/` | encoder frames: only `smoke-run` now (`pool4` and `pool5` deleted) |
  | `runs/` | every training run (5.3 GB); since feature version 7 none of the encoders or actors load |
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
    ~/zkml/bin/python tools/policy_clone_train.py --data /mnt/d/zk-learned/clone/army-6 --maps /mnt/d/zk-learned/maps \
        --run /mnt/d/zk-learned/runs/NAME --decide-every 6 --samples 8 --accumulate 8 --lr 6e-4 --muon-lr 0.008 --workers 15 \
        --steps 10000 --z --z-dropout 0.2 --economy-weight 2.5 --finish-winners 0.1 --encoder ENCODER/vae-last.pt
    ~/zkml/bin/python tools/policy_clone_eval.py --actor RUN/actor.pt --maps /mnt/d/zk-learned/maps --games 64 --workers 12 \
        --z variant --encoder ENCODER/vae-last.pt --out RESULT.json   # against the army script; --record DIR keeps its games

Recording, `check` and `zindex` are in 03-training.md, "Script data"; PPO's flags in "PPO".  A `stop` file in a PPO run's
directory ends it at its next report, and the same command resumes it.

`--encoder` is needed by every tool that loads an actor until `policy_latent.ENCODER` names a new encoder: its default is
`encoder-big`, which no longer loads.

In the engine (Windows), recording raw frames from a game: `$env:ZKBOT_FLAGS='rawframe'`, then
`.\zk-startup.ps1 ControllerAI CircuitAI:Normal -Headless -Map 'TitanDuel 2.2' -NoDownload -Client 'python bots\zkbot\main.py {team}'`.
README.md has the rules for engine games: headless at max speed, windowed only when the user asks.

## Next

1. **The whole run** (the user starts it): `policy_pipeline.py --train-encoder`, watched on its dashboard.  Before it, the user
   may want to delete the runs that no longer load and that they don't want to keep (`runs/clone-*`, `encoder-*`), and
   `runs/replay-check` (a plumbing test of the replays: an untrained encoder, a 2-step clone and a tiny PPO).
2. **After it**: whether DAgger fixes the stall in expansion (`evals/record-N.json` and `final.json` against the clone's
   2-5%).  Then PPO's rate in the fine sim, and whether 2M decisions is the right length (`--ppo-hours` caps it).
3. **Not yet measured**: cloning's rate with the numba featurize.
4. **The engine hand-off**:
   * a numpy forward of the actor and encoder for the bot;
   * decoded orders sent as commands;
   * the opening and the economy script in the engine;
   * the checks listed in 02-policy.md, "Not yet checked live".
5. **The user's open questions** (`docs/model.md`, "Open questions").

## Traps

* **Memory**: check `wsl free -g` before a launch.  Never run two jobs that load the map pool at once.  Every GPU trainer caps
  itself at 85% of the card (`--gpu-memory`): past the card, the Windows driver spills into system RAM instead of failing.
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
