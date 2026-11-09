# HANDOFF: zk-ArenaAI, the learned policy

Updated 2026-11-09.  Branch `learned-policy-plan`.  Repo: `C:/Users/Thoma/Documents/sources/zk-ArenaAI`.

This file is the current state only.  How things got here is in git history and in `docs/learned-policy/`, whose numbered
sections (`03-training.md` 11.x for the latest work) hold every result with its numbers.

## 1. Goal and status

A Zero-K AI that beats CircuitAI, now pursued as a Pluto-style learned policy: a network trained in a simplified simulator
(`sim/`), first by cloning a scripted player, then by self-play.  The plan starts at `docs/learned-policy/00-overview.md`.

* **The rules bot is gone** (2026-11-08, at the user's request).  It was the hand-written bot in `bots/zkbot` that beat
  CircuitAI Hard at high game speed and about 1 game in 16 at 1x, plus ML stages 1 to 4 around it (fight evaluator, value net,
  observer, parameter tuning).  It's kept at the git tag `rules-bot-final` (local; push it with `git push origin rules-bot-final`
  if wanted).  `bots/zkbot` is now the policy's host in the engine (section 3).
* **Nothing plays in the engine.**  The host chooses a commander and sends no commands, until the policy's hand-off is built:
  its decoded orders sent as commands, and the opening and the economy script as the sim plays them.
* **In the sim**, both sides play a scripted opening (`tools/policy_opening.py`), the economy script runs their constructors,
  and the army script or the policy runs their fighters, factories, silo missiles and the Trinity.  The economy and army
  scripts are C++ (`sim/scripts/`, `libzkscripts.so`, through `tools/policy_scripts_cpp.py`), threaded over games.  The Python
  versions they were ported from are gone (in git at `8e9bca7`); `tools/policy_eco.py` and `tools/policy_mix.py` describe what
  the scripts do and build the tables they're given.  The encoder's frame games (`frame_world.FrameGames`,
  `frame_data.ScenarioEnv`) play their own Python scripts instead (`SimEnv.own_scripts`).
* **The actor** (`tools/policy_latent.py`) reads a frozen frame encoder's 40 latent tokens, frame after frame, with what it did
  at each frame as 4 action tokens after that frame's: `[frame 1 latents] [actions at 1] [frame 2 latents] ...`, causal,
  a 4-frame window in each of 2 layers.  It decides every 3 s (`decide_every` 6, when the army script looks; 2026-11-09) and
  reads only those frames.  It gives its orders as Pluto does, as actions one after another (up to 16, then done):
  each picks a unit, the anchor, and an order for it, which may go as a group order.  A unit no action picks keeps.
* **Cloning** (`tools/policy_clone_train.py`) learns the army script's choices from recorded games
  (`tools/policy_clone_data.py`), its orders as actions in a random order, each anchor scored as any unit still to be told.
  The army and economy scripts both, on `army-6`: `clone-7` is the baseline, `clone-8` adds a strategy z and weights the
  economy (`03-training.md` 11.21, 11.22).  `tools/policy_clone_eval.py` plays a clone against the script in the sim.
* **PPO's reward** has Pluto's diversity bonus as well (11.19): zero-sum, for finished units of types the policy rarely makes.
* **Recording** is C++ (`sim/record/`, `libzkrecord.so`, through `sim/py/zkrecord.py`): straight from the batch's arrays, a
  `.zkr` file a game, both sides, narrowed and compressed (192 bytes a decision against the npz's 516).  Data version 5 (group
  orders as labels); the trainer and `check` read the old npz files too.
* **Script variants** (`tools/policy_variants.py`): with `--variants`, each side's army script draws its own role mix and
  aggression at game start, recorded in the game's file.  Without it, play is as before.
* **How the model works**, with diagrams: `docs/model.md`.
* **Air** (2026-11-08): the sim flies every aircraft (planes turn in circles, bombers rearm at pads, gunships strafe, what a
  plane plant makes lands when idle), and the scripts build a second factory, planes, gunships or a strider hub, once it's both
  18 minutes in and 50 metal a second without reclaim.  Air plants have their own mix.
* **Group orders** (2026-11-08): one unit's move, fight or attack can go as well to every unit of its side of some roles within
  one of ten radii of it (`bots/zkbot/policy/groups.py`).  The C++ army script sends its fighters' identical orders that way
  wherever that's exact, they're recorded that way as labels, and the policy chooses them: alone or a radius, then the roles.

## 2. Environment

* Zero-K portable `C:/Users/Thoma/Downloads/zero-k-portable`; engine 2026.07.04; Zero-K v1.14.10.3; ControllerAI 0.3.
* Read-only sources: `../RecoilEngine` (CircuitAI under `AI/Skirmish/CircuitAI`), `../Zero-K`, `../ControllerAI`.
* Windows Python 3.13 runs the bot (`pip install requests websockets orjson numpy numba`).  Without numba, featurize's kernels
  run in plain Python, right but ~50x slower (18 ms a frame of 45 units, more with more), and say so on import.
* WSL (Ubuntu) runs the sim and training: `~/zkml/bin/python` (PyTorch, RTX 4070), pytest in `~/zksim/pt`
  (`~/zkml/bin/python -m pip install -q --target ~/zksim/pt pytest`), so `PYTHONPATH=$HOME/zksim/pt:bots`.
* The sim library: `cmake -S sim -B ~/zksim/build && cmake --build ~/zksim/build -j8` gives `~/zksim/build/libzksim.so`,
  and the scripts' `libzkscripts.so` and the recorder's `libzkrecord.so` beside it (`ZKSIM_LIB` overrides).  The recorder needs
  `libzstd-dev`.  Rebuild after any change under `sim/`.
* Data on D:, not in git (`/mnt/d/zk-learned`):

  | Path | What |
  |---|---|
  | `maps/` | the map pool: 98 real and 200 made-up maps (`tools/map_gen.py`).  About 7.5 GB in memory when loaded |
  | `frames/pool4` | 1M script frames and 1M scenario frames (`tools/frame_data.py`), with air and second factories |
  | `frames/pool5` | 1M frames each of script games, scenarios (60% themed: air battles and raids, fleets, heavies, swarms) and the C++ scripts' games |
  | `runs/encoder-big/final/vae-last.pt` | the frozen encoder the actor reads (`policy_latent.ENCODER`): 40 tokens, 1.65M parameters, trained on `pool5` |
  | `runs/encoder-air/final/vae-last.pt` | the one before, trained on `pool4` |
  | `runs/encoder-questions/final/vae-last.pt` | the one before that, which never saw aircraft |
  | `clone/army-5` | 114,223 games, 401.5M decisions (138 GB) of the C++ scripts with variants, .zkr, data version 5: what's cloned |
  | `runs/clone-6/actor.pt` | the cloned actor at a decision every 3 s (saved every 500 steps); `clone-5` the one at every slot |
  | `runs/` | every training run, one directory each |

  On 2026-11-09 the user was given a command to delete what's no longer used: `clone/army-1` to `army-4` (older data
  versions), `frames/pool2` and `pool3` (frames for the encoders before air) and the runs from before the latent actor.  If
  they're still there, the command wasn't run; `frames/pool4` (25 GB, the held-out set of 11.16-11.17) was left to the user.

* `sim/data/maps` is gitignored too: after a pull, `python tools/sim_data.py maps`.

## 3. Layout

* `bots/zkbot/`: the engine side.
  * `main.py` connects to ControllerAI and drives `host.py`, which keeps what a raw frame is read from and runs
    `policy/runtime.py` once a decision.
  * `policy/`: the raw frame schema (`rawframe.py`, also generating `sim/include/zksim/rawframe.hpp` via
    `tools/policy_schema.py`), the map bundle, type tables, featurizer (`featurize.py`, its loops in `featurize_kernels.py`),
    action encode and decode.
  * Unit data: `units.py`, `catalog.py`, `data/*.json` (rebuilt by `tools/extract_weapons.py`, `tools/extract_abilities.py`,
    `tools/dump_defs.py`).  Terrain: `terrain.py`, `hazards.py`, `mapdata.py`.  `params.py`: `ZKBOT_FLAGS`.
* `sim/`: the C++ simulator (`src/`), the economy and army scripts (`scripts/`), the recorder (`record/`), their ctypes
  bindings (`py/zksim.py`, `py/zknav.py`, `py/zkrecord.py`, and `tools/policy_scripts_cpp.py` for the scripts), data (`data/`)
  and tests (`tests/`).  A batch steps its games on a pool of threads of its own.
* `tools/`:
  * Sim data and checks against the engine: `sim_data.py`, `sim_engine.py` (with `simgame/` gadgets), `sim_battery.py`,
    `sim_profile.py`, `sim_batch_bench.py`, `sim_segment_check.py`, `terrain_map.py`, `map_gen.py`.
  * The sim as a game: `policy_sim_env.py`, `policy_opening.py`, `policy_scripts_cpp.py` (the C++ scripts' binding), and the
    tables it's given: `policy_eco.py`, `policy_mix.py`, `policy_script.py`.  `sim_step_bench.py` times the sim's step alone.
  * Frames and encoders: `frame_world.py` (games for frames, with the first fight script), `frame_data.py`, `frame_model.py`
    (the encoder, batches, shards, probes), `frame_vae.py` (its training).
  * The policy: `policy_model.py` (batches, heads' sampling, the critic), `policy_latent.py` (the actor), `policy_train.py` (PPO),
    `policy_clone_data.py` (recording, `check`, `zindex`), `policy_clone_train.py`, `policy_clone_eval.py` (a clone against
    the script in the sim; DAgger's recording), `policy_golden.py`.
* `tests/policy/`, `tests/test_*.py`, `sim/tests/`: offline tests; nothing needs the engine.
* `zk-startup.ps1`, `zk-maps.ps1`, `maps.md`, `README.md`: the workspace template for running engine games.

## 4. Running

In WSL, from the repo (`cd /mnt/c/Users/Thoma/Documents/sources/zk-ArenaAI`), with `PYTHONPATH=$HOME/zksim/pt:bots`:

    ~/zkml/bin/python -m pytest -q sim/tests tests                     # about 3 minutes; all should pass
    ~/zkml/bin/python tools/policy_clone_data.py record --out /mnt/d/zk-learned/clone/NAME --games 32 --decisions 10000000 \
        --workers 12 --threads 12 --seed 1 --maps /mnt/d/zk-learned/maps          # --level: zstd's, 3 by default
    ~/zkml/bin/python tools/policy_clone_data.py check --data /mnt/d/zk-learned/clone/NAME --maps /mnt/d/zk-learned/maps
    ~/zkml/bin/python tools/policy_clone_train.py --data /mnt/d/zk-learned/clone/army-5 --maps /mnt/d/zk-learned/maps \
        --run /mnt/d/zk-learned/runs/NAME --steps 20000 --workers 8    # --decide-every 6 by default; --blind: no action tokens
    ~/zkml/bin/python tools/policy_clone_data.py zindex --data /mnt/d/zk-learned/clone/army-6 --workers 15   # z.npz, for --z
    ~/zkml/bin/python tools/policy_clone_eval.py --actor /mnt/d/zk-learned/runs/clone-7/actor.pt --maps /mnt/d/zk-learned/maps \
        --games 64 --workers 12 --out RESULT.json     # against the script; --record DIR keeps its games for DAgger
    ~/zkml/bin/python tools/policy_train.py --run /mnt/d/zk-learned/runs/NAME        # the encoder is policy_latent.ENCODER
        # from a clone: --init RUN/actor.pt --fine-sim --decide-every 6, and 11.22's flags (KL, cloning loss, critic)

To stop a PPO run, create a file named `stop` in its directory; the same command resumes from its last checkpoint.

In the engine (Windows), recording raw frames from a game: `$env:ZKBOT_FLAGS='rawframe'`, then
`.\zk-startup.ps1 ControllerAI CircuitAI:Normal -Headless -Map 'TitanDuel 2.2' -NoDownload -Client 'python bots\zkbot\main.py {team}'`.
README.md has the template's rules for engine games (headless, max speed; windowed only when the user asks).

## 5. Where the learned policy is

Latest first; each item's numbers are in the section named.

* **P3-P8, C1 and C4-C8 against `clone-7`** (`03-training.md` 11.22, 2026-11-09): `clone-8` (z, economy weight 2.5, the last
  10% of steps on winners) is even with `clone-7` held out: builds and reclaims better, fights and done worse; z is worth 0.03
  nats.  The small actor (`clone-7s`) is worse on every measure.  Blind (no action tokens) is worse everywhere, so no action
  dropout.  In closed loop (six evals of 64 games against the army script) every clone wins 2-5% and loses 77-80%, at
  temperature 1.0 or 0.8, with z or without: it keeps up for 3 minutes, then stops expanding (5-6 extractors at minute 10 to
  the script's 15-18), so the next step is DAgger, not tuning.  The sim must step every slot with the actor deciding every 6th, as the data was recorded (`--fine-sim` for PPO).
  DAgger and PPO-from-a-clone are built; PPO is smoke-tested only.  C1 doesn't apply (no queue in the sim).
* **Cloning 18x faster** (`03-training.md` 11.20): 52 to about 950 of the script's decisions a second, by deciding every 3 s
  (the user's choice), decoding only what a sample needs, indexing the age bias by frame (its backward was 77% of the GPU's
  time) and one encoder call a step.  `clone-6` ran at about 1,050 a second once started; stopped at step 2,000 (for the
  economy), the likeliest anchor was one the script told 67.5% of the time and the kind right 98.9%, as `clone-5` at 11,500.
* **`army-5` and `clone-5`** (11.20): 401.5M decisions recorded in 2 hours once the workers shared each map's data.  `clone-5`, at
  a decision every slot, reached anchor right 0.68 and kind right 0.99 at 11,500 steps, still improving.
* **A diversity bonus** (`03-training.md` 11.19): at the default 0.05, about 2% of the shaped reward a game on script games, 14%
  at most.  Not yet trained with.
* **Actions one after another** (11.18): the army script's grouping now takes in units already told that order, so a look's
  orders went from a 99th percentile of 28 actions (max 111) to 11 (max 41); 16 cuts 1.1% of them.  The actor's decoder picks
  anchors in turn; 600 CPU cloning steps took the likeliest anchor being a unit the script told from 5% to 39%.
* **The bigger encoder** (11.17): on `pool4`'s held-out maps, better on everything than `encoder-air`: where any within 0.05
  0.743 to 0.865, aircraft type 0.579 to 0.759, own ground counts R2 0.41 to 0.73, reconstruction 0.924 to 0.945.  The actor
  reads it now.  What part is size, tokens or data isn't separated.
* **The encoder retrained with air** (`03-training.md` 11.16): on `pool4`'s held-out maps, better on the questions (where any
  within 0.05: 0.706 to 0.743; aircraft found 0.744 to 0.761, type 0.559 to 0.579), worse on the count grid (own ground R2 0.50
  to 0.41) and aircraft health (0.47 to 0.43).  The actor reads it now.
* **Group orders, the policy's too** (`01-simulator.md` 8.19, `03-training.md` 11.15): on 16 games x 2,000 decisions, the same
  frames with the script's and without; 29% of its army orders went in group orders of 3.4 units, so the sim got 79% as many.
  What's recorded tells every unit what it was told; the policy's group orders reach exactly the units they select.  The actor
  has two more heads (group, roles) and isn't trained with them yet.
* **Air, and a second factory** (`01-simulator.md` 8.18, `03-training.md` 11.14): plane turns within 10% of the game's, bomber
  rearm times within 3 frames, strafing circles within 8 elmos.  The second plant needs both 18 minutes and 50 metal a
  second.  In 16 script games to 25 minutes, 17 of 32 sides built one (11 plane, 5 gunship, 1 hub), between 18.0 and 24.5
  minutes; 8 sides' games ended before 18 minutes.  AA gets built once air is seen.
* **Script variants** (`03-training.md` 11.13): 4 presets (standard, rush, turtle, skirmish) with jittered knobs.  On 64 games
  their production mixes differ as meant and each wins and loses.  The script's orders are group orders: where units act, a median
  of 3 share 1 distinct order.
* **Recording in C++** (`03-training.md` 11.12): a game's file is 2.7x smaller than the npz pair (192 bytes a decision on the
  same 6 `army-3` games), and 64 games record at 62,600 decisions a second in one process, the recorder taking 32% of it.
* **A faster sim, and the scripts in C++** (`03-training.md` 11.11): with both sides scripted, 128 games step at 92,000 seat
  decisions a second on 8 threads, 17x the Python scripts.  The sim alone is 1.7 to 3.3x faster and plays the same games
  bit for bit; the C++ scripts give the same orders as the Python ones on the same frames.
* **The trim** (11.10): the rules bot, the first actor and its numpy forward, the toy, the JEPA encoder and
  the raider-only script in the sim are gone.
* **Cloning, with and without action tokens** (11.9): 3,000 steps on `army-3` each.  No measurable difference: the NLL of the
  script's acts is 3.51 with tokens and 3.49 without, an act's kind right 24% and 23%; tokens cost 19% in speed.  Both are
  undertrained (NLL still falling at the end), so it doesn't say the tokens are useless to an actor that has learned the script.
* **The economy script and missiles** (11.9): constructors expand, build energy and spend; metal lost at the storage cap is
  2 to 12% of income by game phase, from 62%.  Cerberus, Lucifer and the silo from 16 minutes and 55 metal a second while map
  control is stalled; the Trinity by chance from 18 minutes and 60.  Missiles and the Trinity are the policy's to aim.
* **Factories are the policy's, and script games are recorded as labels** (11.8).
* **The army script** (11.7): factories build a role mix by time and enemy, and each role fights its own way.
* **The actor on latents** (11.6), and the encoder it reads (11.5).
* **The sim** (`01-simulator.md` 8.x): combat, movement, economy, overdrive, abilities, shields, unit AI, checked against the
  engine by `tools/sim_engine.py` and `tools/sim_battery.py`.  Gate 1 (replayed fights) failed and was waived by the user.

## 6. Next

* **The policy runs the economy too** (the user's decision, 2026-11-09): the plan is `docs/learned-policy/04-economy.md`, the
  user's five decisions in its section 5.  `clone-6` was stopped for it at step 2,000 (`runs/clone-6/actor.pt`: held-out
  anchor right 0.675, kind right 0.989, NLL of actions 3.89, at about 1,050 decisions a second), since cloning starts again on a
  dataset with the economy script's orders as labels.  Steps 0-2 are done (section 4 there): constructors are controllable,
  BUILD (type, a place among our units and the sites, an offset) and ASSIST are in the featurizer, `actions`, the actor and the
  sim, feature version 6 (the version-5 encoder still loads: `ENCODER_FEATURE_VERSIONS`), data version 6.  `clone-6`'s checkpoint
  doesn't load into the new actor (new layers); `clone-7` starts fresh.
* **Reclaim** (the user, 2026-11-09, before recording): the economy script reclaims (2 dedicated constructors at most, idle ones
  near; avoids enemies, goes home when hurt; libzkscripts ABI 6), and the policy has RECLAIM with a wreck pointer.  4% of a side's
  metal income in the scripts' games; the round trip gives the same wreck every time (04-economy.md, "Reclaim").
* **`army-6` is recorded** (`army-5` is deleted): 83,820 games, 300.6M decisions, 105 GB, stopped at the user's word at 75% of the
  400M asked (`policy_clone_data.py record --out /mnt/d/zk-learned/clone/army-6 --games 32 --workers 12 --threads 12 --seed 5
  --variants --maps /mnt/d/zk-learned/maps`; log `/mnt/d/zk-learned/clone/army-6.log`).  Every game reads back whole.  The
  ~1,000-game `army-6-test` passed `check` (missed 0.35%; build, assist and reclaim all there).
* **S1-S5, P1, P2 are in** (`05-literature.md`, "Taken up so far").  Every trainer caps its GPU memory (`--gpu-memory` 0.85): past
  the card, the Windows driver spills into system RAM instead of failing (30 GB on 2026-11-09).
* **The actor is 10.5M parameters** (`policy_latent.SIZE`: d 256, 8 heads, 4 time layers, 4 plan layers; `SMALL` is the old 1.9M)
  and the critic 3.2M.  At 8 stretches of 48 decisions a forward pass it peaks at 4.5 GB; 16 doesn't fit in 85% of the card.
* **The time transformer's attention is flex attention in cloning** (`--attention flex`, the default on CUDA): a block mask skips
  the keys outside the window, ~93% of them, and the age bias rides in 18 extra query/key dimensions (`TimeBlock.bias_dims`).  A
  score_mod bias was 2.4x slower than dense: its gradient came a pair of tokens at a time as atomic adds.  The main process alone:
  230 ms a step dense, 157 ms flex.  `--compile reduce-overhead` (CUDA graphs) on top: 147 ms; `default`: 155 ms; left off.
* **Cloning is limited by making samples**, not the GPU (`05-literature.md` S1b): 791 decisions a second at first, 1,462 now with
  15 workers, against the GPU's 2,450.  The trainer no longer takes a whole step's samples before its first forward; a worker cuts
  4 stretches one after another from each game it decodes, each frame featurized once; `featurize._masks` lost its loop over units.
  What's left is the main process sharing the 8 cores (Ryzen 7 5800X3D, 16 threads) with the workers, `/mnt/d` (~15% slower than
  WSL's own disk), and decoding (0.67 ms a decision).
* **featurize's per-unit loops are numba kernels** (`featurize_kernels.py`, 2026-11-09): selection, the unit rows (instance,
  order memory, rays, DPS around), the regions' live columns and the masks; what a table or bundle gives is packed once and kept
  on it.  The same tensors to the bit as the numpy before (47,000 army-6 frames, 2,500 of them tiled past the caps, all fields,
  dtypes and shapes; golden tests pass): float32 stays float32 through float32 constants, means are numpy's pairwise sum, and
  bincount's float64 sums keep their order.  1.6 ms a frame to 0.4; a cloning worker's whole decision 2.85 ms to 1.54.
  Compiling takes 13 s once; numba's cache (`__pycache__`) loads in 0.4 s.  That cache knows only the kernel file, so it holds
  rawframe's numbers as literals, checked on import.  Cloning's decisions a second with it: not yet measured.
* **The `clone-7` baseline is done** (2026-11-09, 44 minutes; `03-training.md` 11.21 has its curve): at step 1,000 held out,
  anchor right 0.856, kind right 0.938, done right 0.958, NLL of actions 2.84, ordered BCE 0.148 against 0.331 guessing.
  Builds are the weak part (NLL 7.8, kind right 0.81) and reclaim's kind (0.57).  Still improving at the end.  P3-P8, C1 and
  C4-C8 were each measured against it with the same command and a run name of its own (section 5).  It was:
  `policy_clone_train.py --data /mnt/d/zk-learned/clone/army-6 --maps /mnt/d/zk-learned/maps --run /mnt/d/zk-learned/runs/clone-7
  --decide-every 6 --samples 8 --accumulate 8 --lr 6e-4 --muon-lr 0.008 --workers 15 --steps 1000 --check-every 250` (the user:
  1,000 steps is enough for the baseline; a whole cosine schedule of them, not a longer run cut short), about 40 minutes; output
  in `runs/clone-7/train.out`, rows in `log.jsonl` (appended to: clear it before rerunning a run).  Its start read 83,820
  files' headers over /mnt/d, about 3 minutes; now 7 s, from the dataset's `sides.npz`.
* **DAgger, a round** (P3): the better of `clone-7` and `clone-8` plays 256 games recorded with the scripts' advice
  (`policy_clone_eval.py --record /mnt/d/zk-learned/clone/dagger-1`), then 500 steps of fine-tuning on half those and half
  `army-6` (`--mix --mix-share 0.5 --init`), then the same eval.  On the current encoder.
* **The encoder retrained from scratch** (the user, later; ~1 hour last time), for the unit allyteam and team fields, its id
  embeddings sized for FFA: 64 players and 8 allyteams (`featurize.MAX_PLAYERS` 32 and `MAX_ALLY_TEAMS` 16 now).  The actor's
  embeddings are sized from the same constants, so every clone is redone on it.
* **PPO from the clone** waits for that: `policy_train.py --init` takes the clone stage's last output (after DAgger), not the
  base clone.  The smoke run's settings are in 11.22.
* Cloning's decisions a second with the numba featurize (not measured yet), and where the closed-loop actor's economy stalls.
* The user's open questions on the actor:
  * one token per action between frames too, in place of the 4 summary tokens (within a decision the actions are such tokens now);
  * full attention over all past frames, through summary tokens, in place of the 9 s window.
* PPO with the diversity bonus.
* The engine hand-off: a numpy forward of the latent actor and its encoder for the bot, decoded orders sent as commands, and the
  opening and economy script in the engine.

## 7. Traps

* **Memory**: check `wsl free -g` before a launch.  Never run two jobs that load the map pool at once (it thrashed on 2026-11-08).
* **Listing `army-5`** takes 4 to 12 minutes (a header read for each of 114,224 files over `/mnt/d`); a profile or smoke run
  should use a few hundred files linked into a directory of their own (`~/army5-small`).
* **`pkill -f` / `pgrep -f` from `wsl bash -c '...'`** match that shell too, whose command line holds the pattern: write the
  pattern as `[p]olicy_...`.  A `kill -STOP` that way stopped the shell itself.
  The cloning trainer loads only map bundles.
* **Engine games**: never start one, or kill engine processes, while the user is playing.  Check
  `tasklist | grep -i "spring\|zero-k"` first.
* `pkill -f policy_train` in WSL also matches the shell running it: anchor the pattern, as in
  `pkill -f "^/home/pts1667/zkml/bin/python tools/polic[y]_train"`.
* `set_settings` at game start can make the engine exit; the host sends it after frame 2.
* A sync-mode client that stops calling `finish_frame` stalls the match.
* Several files are CRLF; edit them without changing their line endings.
* `results/` and `.vscode/` are never committed.
