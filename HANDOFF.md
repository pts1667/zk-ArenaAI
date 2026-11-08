# HANDOFF: zk-ArenaAI, the learned policy

Updated 2026-11-08.  Branch `learned-policy-plan`.  Repo: `C:/Users/Thoma/Documents/sources/zk-ArenaAI`.

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
* **In the sim**, both sides play a scripted opening (`tools/policy_opening.py`), the economy script runs their constructors
  (`tools/policy_eco.py`), and the army script (`tools/policy_army.py`) or the policy runs their fighters, factories, silo
  missiles and the Trinity.
* **The actor** (`tools/policy_latent.py`) reads a frozen frame encoder's latent tokens, frame after frame, with what it did at
  each frame as 4 action tokens after that frame's: `[frame 1 latents] [actions at 1] [frame 2 latents] ...`, causal, windowed
  8 s in each of 2 layers.
* **Cloning** (`tools/policy_clone_train.py`) learns the army script's choices from recorded games
  (`tools/policy_clone_data.py`), with keep weighted down.  Latest dataset `army-3`; latest result in `03-training.md` 11.9.

## 2. Environment

* Zero-K portable `C:/Users/Thoma/Downloads/zero-k-portable`; engine 2026.07.04; Zero-K v1.14.10.3; ControllerAI 0.3.
* Read-only sources: `../RecoilEngine` (CircuitAI under `AI/Skirmish/CircuitAI`), `../Zero-K`, `../ControllerAI`.
* Windows Python 3.13 runs the bot (`pip install requests websockets orjson numpy`).
* WSL (Ubuntu) runs the sim and training: `~/zkml/bin/python` (PyTorch, RTX 4070), pytest in `~/zksim/pt`
  (`~/zkml/bin/python -m pip install -q --target ~/zksim/pt pytest`), so `PYTHONPATH=$HOME/zksim/pt:bots`.
* The sim library: `cmake -S sim -B ~/zksim/build && cmake --build ~/zksim/build` gives `~/zksim/build/libzksim.so`
  (`ZKSIM_LIB` overrides).  Rebuild after any change under `sim/`.
* Data on D:, not in git (`/mnt/d/zk-learned`):

  | Path | What |
  |---|---|
  | `maps/` | the map pool: 98 real and 200 made-up maps (`tools/map_gen.py`).  About 7.5 GB in memory when loaded |
  | `frames/pool2`, `frames/pool3` | 1M script frames and 1M scenario frames each (`tools/frame_data.py`), for encoders |
  | `runs/encoder-questions/final/vae-last.pt` | the frozen encoder the actor reads (`policy_latent.ENCODER`) |
  | `clone/army-1`, `army-2`, `army-3` | recorded army-script games, data versions 1 to 3; only `army-3` matches the current scripts |
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
* `sim/`: the C++ simulator (`src/`), its ctypes binding (`py/zksim.py`, `py/zknav.py`), data (`data/`) and tests (`tests/`).
* `tools/`:
  * Sim data and checks against the engine: `sim_data.py`, `sim_engine.py` (with `simgame/` gadgets), `sim_battery.py`,
    `sim_profile.py`, `sim_batch_bench.py`, `sim_segment_check.py`, `terrain_map.py`, `map_gen.py`.
  * The sim as a game: `policy_sim_env.py`, `policy_opening.py`, `policy_eco.py`, `policy_army.py`, `policy_mix.py`,
    `policy_script.py` (the scripts' shared data).
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
        --workers 12 --threads 12 --seed 1 --maps /mnt/d/zk-learned/maps
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

* **The trim** (`03-training.md` 11.10): the rules bot, the first actor and its numpy forward, the toy, the JEPA encoder and
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

* The user's open question: full attention over all past frames, through summary tokens, in place of the 16 s window.
* Cloning learns slowly: an act's kind is right a quarter of the time after 3,000 steps.  Whether to run it longer, with or
  without action tokens (the A/B didn't separate them), or to change what it learns from first, is the user's to decide.
  Then the cloned actor playing the army script in the sim.
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
