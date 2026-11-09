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
  windowed 8 s in each of 2 layers.  It gives its orders as Pluto does, as actions one after another (up to 16, then done):
  each picks a unit, the anchor, and an order for it, which may go as a group order.  A unit no action picks keeps.
* **Cloning** (`tools/policy_clone_train.py`) learns the army script's choices from recorded games
  (`tools/policy_clone_data.py`), its orders as actions in a random order, each anchor scored as any unit still to be told.
  Only smoke runs on the action form so far (`03-training.md` 11.18).
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
* Windows Python 3.13 runs the bot (`pip install requests websockets orjson numpy`).
* WSL (Ubuntu) runs the sim and training: `~/zkml/bin/python` (PyTorch, RTX 4070), pytest in `~/zksim/pt`
  (`~/zkml/bin/python -m pip install -q --target ~/zksim/pt pytest`), so `PYTHONPATH=$HOME/zksim/pt:bots`.
* The sim library: `cmake -S sim -B ~/zksim/build && cmake --build ~/zksim/build -j8` gives `~/zksim/build/libzksim.so`,
  and the scripts' `libzkscripts.so` and the recorder's `libzkrecord.so` beside it (`ZKSIM_LIB` overrides).  The recorder needs
  `libzstd-dev`.  Rebuild after any change under `sim/`.
* Data on D:, not in git (`/mnt/d/zk-learned`):

  | Path | What |
  |---|---|
  | `maps/` | the map pool: 98 real and 200 made-up maps (`tools/map_gen.py`).  About 7.5 GB in memory when loaded |
  | `frames/pool2`, `frames/pool3` | 1M script frames and 1M scenario frames each (`tools/frame_data.py`), for encoders |
  | `frames/pool4` | the same, collected with air and second factories |
  | `frames/pool5` | 1M frames each of script games, scenarios (60% themed: air battles and raids, fleets, heavies, swarms) and the C++ scripts' games |
  | `runs/encoder-big/final/vae-last.pt` | the frozen encoder the actor reads (`policy_latent.ENCODER`): 40 tokens, 1.65M parameters, trained on `pool5` |
  | `runs/encoder-air/final/vae-last.pt` | the one before, trained on `pool4` |
  | `runs/encoder-questions/final/vae-last.pt` | the one before that, which never saw aircraft |
  | `clone/army-1`, `army-2`, `army-3` | recorded army-script games, data versions 1 to 3 (npz, a file a side) |
  | `clone/army-4` | 3,243 games (3.8 GB) of the standard script, .zkr; a 100 GB run stopped early to add variants first |
  | `runs/` | every training run, one directory each |

* `sim/data/maps` is gitignored too: after a pull, `python tools/sim_data.py maps`.

## 3. Layout

* `bots/zkbot/`: the engine side.
  * `main.py` connects to ControllerAI and drives `host.py`, which keeps what a raw frame is read from and runs
    `policy/runtime.py` once a decision.
  * `policy/`: the raw frame schema (`rawframe.py`, also generating `sim/include/zksim/rawframe.hpp` via
    `tools/policy_schema.py`), the map bundle, type tables, featurizer, action encode and decode.
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
    `policy_clone_data.py`, `policy_clone_train.py`, `policy_golden.py`.
* `tests/policy/`, `tests/test_*.py`, `sim/tests/`: offline tests; nothing needs the engine.
* `zk-startup.ps1`, `zk-maps.ps1`, `maps.md`, `README.md`: the workspace template for running engine games.

## 4. Running

In WSL, from the repo (`cd /mnt/c/Users/Thoma/Documents/sources/zk-ArenaAI`), with `PYTHONPATH=$HOME/zksim/pt:bots`:

    ~/zkml/bin/python -m pytest -q sim/tests tests                     # about 3 minutes; all should pass
    ~/zkml/bin/python tools/policy_clone_data.py record --out /mnt/d/zk-learned/clone/NAME --games 32 --decisions 10000000 \
        --workers 12 --threads 12 --seed 1 --maps /mnt/d/zk-learned/maps          # --level: zstd's, 3 by default
    ~/zkml/bin/python tools/policy_clone_data.py check --data /mnt/d/zk-learned/clone/NAME --maps /mnt/d/zk-learned/maps
    ~/zkml/bin/python tools/policy_clone_train.py --data /mnt/d/zk-learned/clone/army-3 --maps /mnt/d/zk-learned/maps \
        --run /mnt/d/zk-learned/runs/NAME --steps 3000 --workers 12          # --blind: no action tokens, for comparison
    ~/zkml/bin/python tools/policy_train.py --run /mnt/d/zk-learned/runs/NAME        # the encoder is policy_latent.ENCODER

To stop a PPO run, create a file named `stop` in its directory; the same command resumes from its last checkpoint.

In the engine (Windows), recording raw frames from a game: `$env:ZKBOT_FLAGS='rawframe'`, then
`.\zk-startup.ps1 ControllerAI CircuitAI:Normal -Headless -Map 'TitanDuel 2.2' -NoDownload -Client 'python bots\zkbot\main.py {team}'`.
README.md has the template's rules for engine games (headless, max speed; windowed only when the user asks).

## 5. Where the learned policy is

Latest first; each item's numbers are in the section named.

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

* `army-5` (`/mnt/d/zk-learned/clone/army-5`, 400M decisions, about 95 GB, `--variants`, seed 5, the map pool) was being
  recorded on 2026-11-09; its log is `army-5.log` beside it, which ends with `check`'s output and the size once it's done.
  The first try was stopped: each worker made its own copy of each map's masks and labels, Python and C++, as it came to the
  map (about 6 MB a map, 1.6 GB a worker over the pool).  `policy_sim_env.share_maps` now makes them before the workers fork;
  12 workers level off at about 5.4 GB together.  Then clone on it with the action form and the bigger encoder: older
  checkpoints don't load into this actor.
* Whether the policy should run the economy too (constructors and buildings, as Pluto does), which the economy script does now:
  asked by the user on 2026-11-09, not yet decided.  It would need the economy script's orders recorded as labels.
* The user's open questions on the actor:
  * one token per action between frames too, in place of the 4 summary tokens (within a decision the actions are such tokens now);
  * full attention over all past frames, through summary tokens, in place of the 16 s window.
* Then the cloned actor playing the army script in the sim, and PPO with the diversity bonus.
* The engine hand-off: a numpy forward of the latent actor and its encoder for the bot, decoded orders sent as commands, and the
  opening and economy script in the engine.

## 7. Traps

* **Memory**: check `wsl free -g` before a launch.  Never run two jobs that load the map pool at once (it thrashed on 2026-11-08).
  The cloning trainer loads only map bundles.
* **Engine games**: never start one, or kill engine processes, while the user is playing.  Check
  `tasklist | grep -i "spring\|zero-k"` first.
* `pkill -f policy_train` in WSL also matches the shell running it: anchor the pattern, as in
  `pkill -f "^/home/pts1667/zkml/bin/python tools/polic[y]_train"`.
* `set_settings` at game start can make the engine exit; the host sends it after frame 2.
* A sync-mode client that stops calling `finish_frame` stalls the match.
* Several files are CRLF; edit them without changing their line endings.
* `results/` and `.vscode/` are never committed.
