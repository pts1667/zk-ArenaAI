# zk-ArenaAI, the learned policy: where it stands

Updated 2026-11-13.  Branch `learned-policy-plan`.  Repo: `C:/Users/Thoma/Documents/sources/zk-ArenaAI`.  This file was
`HANDOFF.md`.

The current state only.  How things got here is in git history.  How everything works, with its numbers, is in
`docs/learned-policy/` (start at `00-overview.md`) and `docs/model.md`.

## Goal and status

The goal is a Zero-K AI that beats CircuitAI, pursued as a Pluto-style learned policy.  It's a network trained in a simplified
simulator (`sim/`): first by cloning scripted players, then DAgger, then PPO.

* **Nothing plays in the engine yet.**  `bots/zkbot` hosts the policy: it chooses a commander and sends no commands.  The rules
  bot that was there is at the tag `rules-bot-final` (local only).
* **In the sim**, every team plays a scripted opening.  The C++ economy script runs their constructors (or the policy does, in a
  share of PPO games).  The C++ army script, or the policy, runs their fighters, factories, missiles, the Trinity
  and air transports.
* **The sim plays four kinds of game** (the user, 2026-11-10): 1v1s, team games of 2 to 4 a side, free-for-alls of 3 to 8,
  and chicken defense, each on the pool maps that hold it, for up to 80 minutes.  See "Variety" below: it's built and
  tested, and **nothing has been trained on it yet**.
* **Shots fly in the sim** (2026-11-12, the user): a shot that takes time is a projectile, aimed ahead of its target as the
  engine aims it, that hits whoever is where it comes down, and raiders on a fight order jink up to slow shooters as Zero-K's
  tactical AI has them do.  Before, 13 Ronins beat 18 Glaives every time; now the Glaives win, as in a game.  The sim's
  movement is also 1.1 to 1.7 times faster, with no game changed by that.  01-simulator.md, "Weapons and damage", "Unit AI"
  and "Validation".  Everything recorded or trained before it played the old sim.
* **The rest of Zero-K's tactical AI is in the sim** (2026-11-12, the user), data-driven from the gadget's tables: ward fire
  at personal shields (38 types), the tables for the sea and for fight-only targets, the parameters single units have, the
  AI's turn every 20 frames, and the AI being off for nine types until an order turns it on.  A game's Ronin, Glaive and
  Mace wear a Thug's shield down as the sim's do.  01-simulator.md, "Unit AI", "Validation", and "Known gaps" for what
  isn't modelled.
* **No side goes first in the sim** (2026-11-12, the user).  The policy plays team 0, whose units are made first, and in
  mirror fights that decided who won: of 4 Maces a side, side 1 every time, of 8 Ronins side 0 every time.  The tactical
  AI's turn went by the handle, a unit moved before the ones made after it had said what they'd do, and a stun stopped a
  unit made later from firing back that tick.  Now the turn and a unit's first side are drawn by the seed, and a tick's
  moves, shots and deaths are settled before any takes effect: of 1,600 mirror fights with the starts shaken, the side made
  first won 642 and the other 689, side 0 672 and side 1 659.  A game's own 96 went 51 to 40 for the side made first.
  01-simulator.md, "Unit AI" ("No one goes first") and "Validation" ("Mirror fights"); "Known gaps" for Glaives against
  Glaives, which take twice a game's time.
* **Zero-K's end of a game, whole** (2026-11-13, the user): an alliance of AIs with no finished builder or factory left is
  destroyed (`game_over.lua`'s `vitalAlive`; libzksim ABI 25, `rules.json`'s `vital`), so a game of two ends.  Every team
  in the sim stands for an AI, so it's always on but for the chickens.  The 25x reveal was already in.  The lead rule and
  the time limit stay, on top, to hurry games.  01-simulator.md, "Teams and alliances" ("No builder left").
* **Two damage stats a team** (2026-11-12, the user; libzksim ABI 24, `ZK_STATS` 6): `[4]` the damage its units have done
  to enemies' units, in metal, and `[5]` what enemies' have done to its own, for a combat-efficiency head.  01-simulator.md,
  "Teams and alliances".
* **The actor** (`tools/policy_latent.py`, 10.5M parameters) reads a frozen frame encoder's 40 latent tokens a frame, over a
  window of 4 frames in each of 4 layers.  It decides every 3 s, as actions one after another: a unit, then its order.
* **One command runs everything** (`tools/policy_pipeline.py`): recording the scripts' games, frames, encoder, 10,000 steps
  of cloning as `clone-8` with z from the variant, two DAgger rounds, an eval, PPO.  While it runs it serves a dashboard at `http://localhost:8765`: the log as
  it's written, each stage's results, and replays of test games (4 after each test, 4 every hour of PPO).
* **`runs/full-1` is its first whole run** (2026-11-10), against the army script, told its variant:
  * the clone picks a unit the scripts told 91% of the time held out, and wins 5% in play;
  * DAgger took it to 13% in one round, the second adding nothing.  The stall in expansion stays: at 6 minutes it has about
    half the script's extractors;
  * PPO reached 22-31% test wins by iteration 88 (360,552 decisions), with explained variance 0.94, when the user stopped it.
    It can't be resumed: its encoder and actor don't load since feature version 7.
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
  before the economy change below, and like `full-1` it can't be resumed.
* **The economy script's factories** (the user, 2026-11-10).  It used to spend floating metal on copies of its first factory, 3
  by minute 10 and up to 6.  Now it builds one plate beside the factory, then caretakers in reach of both.  The second factory
  comes at 40 metal a second (the user's rule), at a median of 7.6 minutes in the sim, and there's no third.  Two more
  constructors are made while metal floats.  On 48 games, a side's value at 19 minutes went from 27,800 to 31,700.
  * The sim now models plates (`unit_factory_plate.lua`: they build only near their factory) and idle caretakers (they help what's
    being built in their reach, as Zero-K's default widget has them patrol).  01-simulator.md, "Economy" and "The economy script".
  * Plates are factories in the policy's type table (masks only, and no new feature version for it, since no data held one).
  * `clone/army-6` was recorded by the old scripts, so a clone of it learns factory copies: new recordings are needed first.

## `runs/full-2` (started 2026-11-11 04:15)

The first whole run on the variety work, started by me while the user slept:

    policy_pipeline.py --root /mnt/d/zk-learned/runs/full-2 --train-encoder --data /mnt/d/zk-learned/clone/army-7 \
        --clone-steps 5000 --ppo-decisions 4000000 --ppo-args "--collectors 4 --concurrent --economy-start 1 --economy-end 1"

* The policy runs its own economy in every PPO game (economy share 1 throughout, the user), for 4M decisions.
* Its first attempt filled WSL's 40 GB in 5 minutes of recording: the recorder's buffers, fixed in `sim/record/src/record.cpp`
  (03-training.md, "Script data").
* The critic sees up to 256 hidden enemies, not 64 (the user, 2026-11-11; 02-policy.md, "Kinds").
* Record 87 minutes (11,938 games, 110M decisions, 156 GB), frames 27, encoder 163 (held-out region R² 0.966), `zindex` 28.
* The clone ran out of the card in 5 minutes, at 4 samples a pass.  Resumed with passes split when they run out, it made 323
  decisions a second (27 hours for 10,000 steps), and went NaN between steps 500 and 1000; I stopped it.  The user's PC
  restarted.  Then, the user asked, cloning was made faster, to about 750 a second (03-training.md, "Speed"), and its
  fixes kept: passes by size, splitting, a non-finite pass logged in `clone/nonfinite.jsonl`.
* The clone is 5,000 steps, not 10,000 (the user, 2026-11-11: about 5-6 hours): `--clone-steps 5000`.
* Stopped at 15:48, the user asked, to make it faster still: no waiting on the GPU each pass, reads taken as they're done, and
  `--compile dynamic` (the pipeline's now), to about 1,070 a second.  Restarted at 17:08: about 4 hours for the clone.
* The NaN's cause, found from `nonfinite.jsonl` at 17:40: dying units' health below zero made a summary token's `log1p`
  NaN (`featurize._summaries`, fixed).  The running clone's workers featurize as before the fix, so such a stretch still
  skips its step there (1 in its first 500); every later stage featurizes with the fix.
* The clone finished at 21:12 (245 minutes, 5 steps of 5,000 skipped): held out, the unit the scripts told 84%, the kind
  96%.  In 64 games against the army script (by kind: 38 1v1s, 9 team games, 9 free-for-alls, 8 chicken games), told its
  variant and running its own economy:

  | Actor | Wins / draws / losses | 1v1 wins | Extractors at 6 min (script's) |
  |---|---|---|---|
  | clone | 9% / 5% / 86% | 5% | 5.2 (9.2) |
  | after DAgger round 1 | 17% / 2% / 81% | 11% | 5.2 (8.6) |

  The stall in expansion stays.  `record-1` and `record-2` took 97 and 128 minutes; recording is faster since (03-training.md,
  "DAgger").  After round 2, the final eval (64 1v1s): 8% / 5% / 88%, extractors at 6 minutes 4.6 (10.5).
* PPO started at 02:52 on 2026-11-12, after the critic was made to read the frame with learnt queries and Muon trains it (the
  user; 03-training.md, "PPO").  Neither had been tried at scale before it.  Explained variance reached 0.41 by iteration 7.
* A collector ran out of its 0.75 GB at iteration 8 (`ppo-oom-0.75`); at 1.0 one reached 0.96 by iteration 16, holding its
  previous graph shape while capturing the next.  Training had fallen from 252 decisions a second to 70: as frames grew to
  about 730 tokens, minibatches went from 141 decisions to 34, under a token weighting fitted when attention paired every
  token, while the learner used little of its memory (the user saw 8 of the card's 12 GB in use).  I stopped it at
  iteration 18 (set aside as `ppo-pairs-18`).  Now a graph shape narrower than the widest goes before the next capture, and
  the budget counts tokens linearly, with 200 more a decision; a minibatch that doesn't fit is split, and the budget shrinks
  by a tenth once in an iteration and grows back when there's room (03-training.md, "PPO" and "Playing").  Restarted three
  times for it: shrinking at every split (`ppo-shrink-4`), then 800 a decision, which planned big frames' minibatches too
  big (`ppo-k800-7`, 145-222 decisions a second overall all the same), then at 04:07 as it is now.  Each minibatch's
  memory is logged in `ppo/minibatches.jsonl`, to fit the 200 to.
* The A/B (the user, 2026-11-12): 50 iterations of the pipeline's PPO with Adam on the actor (04:07-04:49), then 50 with
  Muon on its hidden matrices (`--actor-muon-lr 1.3e-4`, `runs/full-2-actor-muon`).  Adam won: wins level (tests at 20
  and 40: 16 of 128 against 14), Muon's KL an iteration twice Adam's and rising, its cloning loss higher, 10% slower
  (03-training.md, "PPO").  Adam's arm went on as the real PPO stage from its iteration 51, with the budget starting at
  100,000 (fitted on its 2,042 minibatches) and growing back on what the tensors held.
* At about 100 decisions a second.  As the fade took the KL to the clone's weight to 0.094, PPO's KL an iteration rose to
  0.052 by iteration 126, with 36% of ratios past 1 ± eps; tests at 100 and 120 won 8% and 9%, against 11-16% before.  I
  stopped it at 127 (kept as `ppo-kl-127`).  The user chose: the actor's rate slows 1.5x after an iteration past twice
  `--kl-target` 0.01 and comes back after one under half of it, and a rollback to iteration 80 (best test, 15.6%).
  Resumed from 80 at 15:11 (`results/rollback_ppo.sh` cut the logs there).  The test at 100 won 11% (KL 0.012, the rate
  unslowed).  From iteration 101 (15:30) it tests every 100 iterations, not 20 (the user): a test of 64 games took 220-460 s
  with the learner waiting, 30-60% on top of 20 iterations.  `results/resume-full2.sh` passes `--eval-every 100`.
* The sim changed under it twice (the user): projectiles and lead from iteration 296 (17:58), the rest of the tactical AI
  from 562 (20:52).  Tests (64 1v1s): 11% at 100, 8% at 200, 6% at 300, 11% at 400, 3% at 500 and 600.  183 decisions a
  second overall at 600.  Explained variance was 0.99-1.0 from about iteration 550: beta at
  0.85 made the reward mostly the result, and the policy lost about 95% of its games.
* So the critic now has the reward's terms apart, a head each (the user, 2026-11-12; 03-training.md, "Reward"):
  `--terms result:1 score:0.5 economy:0.3 combat:0.3` in the pipeline, with `--critic-span 2`.  Built and tested, never
  trained with: a resume onto other terms keeps the critic but its heads, which start afresh while the actor waits 10
  iterations.
* PPO kept getting worse: training games' 1v1 wins fell from 16% (iterations 100-199) to 5% (600-699), team games' from
  39% to 20%; tests won 0% at 700 and 5% at 800 and 900.  It steepened as the KL to the clone and the cloning loss faded to
  nothing by 500.  It finished its 4M decisions at iteration 971 (2026-11-13 00:06) on the old critic.  The user
  judged that PPO isn't working, and asked first for Zero-K's whole end of a game in the sim (above).

## Variety (2026-11-10 and 11): built, tested, not yet trained on

The user asked for it before the next whole run.  All of it is in the tree and described in `docs/learned-policy/`:

* **Teams**: the sim, the scripts and the recorder take up to 8 teams in alliances (libzksim ABI 25, libzkscripts ABI 12 then,
  libzkrecord ABI 2, clone data version 7).  A 1v1 plays bit for bit as it did.  Allies share income, overdrive and overflow.
* **Kinds of game**: `SimEnv`, the recorder, the clone eval, the trainer and the pipeline all take `--kinds` (default
  `1v1:0.5 teams:0.2 ffa:0.2 chicken:0.1`), `--slots`, `--test-kinds` and `--chicken-difficulties`.  The policy plays team 0
  and scripts play its allies and enemies.  The headline test stays a 1v1.
* **Maps**: each pool map's `modes` say what it plays (`tools/map_modes.py`); 18 Violet Rampart maps come from a port of
  the map's own generator (`tools/map_rampart.py`).  The pool is 404 maps since 2026-11-11, when 89 of the install's were
  added: 381 play 1v1s, 332 teams, 64 free-for-alls, 5 chickens only.  THISspace and ZeroWars play nothing, and a map
  made for chickens plays only them (the user).  01-simulator.md, "Data", lists what's not in the pool and why.
* **Start points**: where a box's start point is ground the commander can't stand on (Esker Creek, Pools of Ilys, Acidic
  Quarry: a ridge only spiders climb), the start is the nearest place in the box with room, as Zero-K does.  A commander
  started on one never moved.
* **Chicken defense** runs inside the sim (`sim/src/chickens.cpp`, a port of `unit_spawner.lua` at Zero-K's timings), with
  Zero-K's reveal rule and its rule for what keeps a team in the game (`sim/data/rules.json`).  The army script has a
  posture for it.  Four scripted players won at Very Easy and Easy, 1 game in 6 at Normal, none at Hard.
* **Longer games**: the limit is 80 minutes (it was 20).  The reward has a small zero-sum term for winning sooner
  (`--speed-weight` 0.1), and the lead rule (2.5x the value) is drawn per game: 90% of PPO's games at iteration 0, none
  from 100 (`--lead-fade`); tests and recordings keep it, frame collection has it off.
* **Caps**: a frame gives the policy 1,280 unit tokens, 512 of them controllable, and 512 wrecks (`FEATURE_VERSION` 8).
* **Air transports** (the user, 2026-11-11; `FEATURE_VERSION` 9, raw frame schema 5):
  * the sim has the Charon and the Hercules (`sim/src/transport.cpp`): one unit each, by its weight class; LOAD takes one
    of its own, or anyone's that stands still in sight; UNLOAD sets it down; DROP lets it fall, hurt by its fall (a Jack
    about a tenth of its health, an Ogre a fifth).  01-simulator.md, "Transports";
  * the policy has the three kinds (24 in all, 20 in the sim's games), two flags on a unit (129 numbers), and aircraft
    among its controllable units, which they weren't.  02-policy.md, "Kinds";
  * the army script keeps a wing (`Seat::lifts`): a plant that can make transports makes one for every two assaults or
    riots it could carry, up to 4 of a kind; they take the dearest to the enemy's dearest building the wing's metal
    covers, drop what loses 15% or less to the fall and set the rest down, and each landed unit attacks that building,
    then the next near it.  None in chicken games.  The scripts have no third factory, so a strider hub and a Hercules
    never meet in their games.
* **Frames**: army games of every kind, and hand-built scenarios with allies, several enemies and chickens
  (`frame_data.ScenarioEnv`).
* **Building placement** follows the engine: the footprint's rectangle on ground level enough for the def, in the opening,
  the scripts (Python and C++), scenarios and the policy's builds (01-simulator.md, "Where a building can go").
* **Replays** are format 3 (teams, alliances, pages), and the viewer shows any number of teams.

**My choices the user hasn't confirmed** (each a constant, easy to change): the scenario kinds' weights (40% 1v1s, 20%
each of the rest) and their chicken numbers; a scenario team's 480 units at most; chicken difficulty drawn evenly from
Very Easy, Easy and Normal, not set from the measurement above; chicken games drawn on team maps too, as Zero-K lists only
two maps for chickens; the reward zero-sum over alliances; the speed term unfaded by beta.  From the transports and the
maps (2026-11-11): aircraft controllable by the policy (one line in `featurize_kernels.select`); the wing's numbers in
`sim/scripts/src/army.cpp` (a transport for every 2 loads, 4 of a kind at most, a drop at 15% loss or less, 45 s of
waiting for the rest of the wing); and Zero-K's "special" mark counting as no kind of game, which makes Requiem Outpost
a chicken map.

## Invalid since 2026-11-10 (FEATURE_VERSION 7, now 10)

The id tables were resized for free-for-all (version 7: `featurize.MAX_PLAYERS` 64, `MAX_ALLY_TEAMS` 8), then the caps were
raised (version 8), then air transports came (version 9, 2026-11-11), then the DPS sectors were made even and the frame put
over 80 minutes (version 10, the user, 2026-11-13).  Nothing was trained at 7, 8 or 10.  So:

* **no encoder or actor loads**: `runs/full-2`'s (trained at 9: its encoder, clone, DAgger rounds and PPO) and
  `runs/grpo-trial`'s, `runs/full-1`'s, `runs/encoder-big`, every clone, `spo-1`.  The GRPO trial's PPO arm runs from a copy
  of the code at 9 (`grpo-trial/code`);
* frame pools were stored featurized, so none is left (`frames/pool4` and `pool5` are deleted; `full-1/frames` is rejected).

The recorded games (`clone/army-6`, its `z.npz`) are raw and still read (data version 6).  But they're 1v1s by the scripts as
they were before the economy change, so the next run records anew.  `policy_latent.ENCODER` still names `encoder-big`: every
tool that loads an actor needs `--encoder`.

## Environment

* **Zero-K**: portable install at `C:/Users/Thoma/Downloads/zero-k-portable`; engine 2026.07.04; Zero-K v1.14.10.3;
  ControllerAI 0.3.
* **Read-only sources**: `../RecoilEngine` (CircuitAI under `AI/Skirmish/CircuitAI`), `../Zero-K`, `../ControllerAI`.
* **Windows Python 3.13** runs the bot (`pip install requests websockets orjson numpy numba`).  Without numba, featurize runs in
  plain Python, about 50x slower, and says so on import.
* **WSL (Ubuntu)** runs the sim and training: `~/zkml/bin/python` (PyTorch, RTX 4070, pytest-xdist), with pytest in
  `~/zksim/pt`.  So `PYTHONPATH=$HOME/zksim/pt:bots`.
* **The sim's libraries**: `cmake -S sim -B ~/zksim/build && cmake --build ~/zksim/build -j8`.  That builds `libzksim.so`,
  `libzkscripts.so`, `libzkrecord.so` and `libzkpolicy.so` (`ZKSIM_LIB` and `ZKPOLICY_LIB` override).  The recorder needs `libzstd-dev`.  Rebuild after any change
  under `sim/`.
* **`sim/data/rules.json`** is made by `tools/sim_rules.py`, which needs `lupa`.  It's only in `~/zksim/agent-maps-lupa`
  (add it to `PYTHONPATH`), not in `~/zkml`.
* **`sim/data/maps`** is gitignored: after a pull, `python tools/sim_data.py maps`.
* **Data on D:**, outside git, under `/mnt/d/zk-learned` (174 GB free on 2026-11-10):

  | Path | What |
  |---|---|
  | `maps/` | the map pool: 204 real (18 of them Violet Rampart) and 200 made-up maps, 31 held out, each with its `modes`.  It loads in 61 s to 10.8 GB (14.3 GB after two envs had played on every map) |
  | `maps-old/` | the pool's old Violet Rampart map, set aside |
  | `clone/army-6` | what was cloned: 83,820 script 1v1s with variants, 300.6M decisions, 105 GB, data version 6, with `z.npz` and `sides.npz`.  By the scripts before the economy change |
  | `clone/army-6-test` | about 1,000 games of the same, for `check` and smoke runs |
  | `evals/` | the six closed-loop evals of `clone-7`, `clone-8` and `clone-7s` |
  | `frames/` | encoder frames: only `smoke-run` now (`pool4` and `pool5` deleted) |
  | `runs/full-1` | the first whole pipeline run: its frames (51 GB), encoder, clone, DAgger games (530 MB) and rounds, evals, PPO (1.4 GB), replays.  None of it loads since feature version 7 |
  | `runs/` | every other training run; none of their encoders or actors load.  `ppo-rate`, `ppo-profile`, `ppo-aligned` and `ppo-collectors-*` are scratch timing runs |
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
  * the sim's data and checks against the engine: `sim_data.py`, `sim_rules.py`, `sim_engine.py`, `sim_battery.py`,
    `sim_batch_bench.py`, `map_gen.py`, `map_modes.py`, `map_rampart.py`;
  * the sim as a game: `policy_sim_env.py`, `policy_opening.py`, `policy_variants.py`, and the scripts' tables (`policy_eco.py`,
    `policy_mix.py`, `policy_script.py`);
  * frames and the encoder: `frame_world.py`, `frame_data.py`, `frame_model.py`, `frame_vae.py`;
  * the policy: `policy_model.py`, `policy_latent.py`, `policy_clone_data.py`, `policy_clone_train.py`, `policy_clone_eval.py`,
    `policy_train.py`, `policy_pipeline.py`, `policy_golden.py`;
  * PPO's play: `policy_graphs.py` (each round as a CUDA graph), `policy_collectors.py` (collector processes);
  * the run's dashboard: `policy_dashboard.py` (server), `dashboard/` (page, replay viewer), `policy_replay.py` (test games
    kept for it).
* **Tests**: `tests/policy/`, `tests/test_*.py`, `sim/tests/`, run as `pytest.ini` says.  None needs the engine.
* **Engine games**: `zk-startup.ps1`, `zk-maps.ps1`, `maps.md`, `README.md`.

## Running

In WSL, from `/mnt/c/Users/Thoma/Documents/sources/zk-ArenaAI`, with `PYTHONPATH=$HOME/zksim/pt:bots`:

    ~/zkml/bin/python -m pytest -q                     # about a minute, on 8 processes (pytest.ini); all pass
    ~/zkml/bin/python -m pytest -q -n 0 tests/policy/test_train.py     # one process: for a debugger, or to time a test
    ~/zkml/bin/python tools/policy_pipeline.py --root /mnt/d/zk-learned/runs/NAME --train-encoder \
        --data /mnt/d/zk-learned/clone/NEW [--dry-run]
        # everything, stage by stage; the same command again carries on.  A `stop` file in ROOT halts it after the running stage.
        # --data with no recording in it is recorded first (the `record` stage); without --data it clones army-6, the old scripts' 1v1s.
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

**The tests** (819 pass, 3 of them need CUDA and are skipped without it): `pytest.ini` runs them on 8 processes, a file's tests in
one process so that what they share is made once, and `tests/policy/conftest.py` keeps each process's maths to one thread.
That took the suite from 211 s to about 60 (2026-11-11), with the slowest learning runs cut down and those of paths the
pipeline doesn't use removed: the variational encoder's fits, checkpoints from before feature version 7, the trainer's
in-process concurrent collection.  `--durations=20` shows what's slow; most of what's left is each process importing torch
and the tools (about 15 s).

In the engine (Windows), recording raw frames from a game: `$env:ZKBOT_FLAGS='rawframe'`, then
`.\zk-startup.ps1 ControllerAI CircuitAI:Normal -Headless -Map 'TitanDuel 2.2' -NoDownload -Client 'python bots\zkbot\main.py {team}'`.
README.md has the rules for engine games: headless at max speed, windowed only when the user asks.

## Next

1. **GRPO from branched games** (the user, 2026-11-13, after `full-2`'s PPO got worse), in place of PPO's critic: a game
   copies in the sim and the scripts (libzksim ABI 26, libzkscripts ABI 13) and in `SimEnv` (01-simulator.md, "The batch"),
   and `policy_train.py --group 8` trains by GRPO (03-training.md, "GRPO from branched games").  **The trial**
   (`/mnt/d/zk-learned/runs/grpo-trial`, at FEATURE_VERSION 9), from `full-2`'s DAgger actor, which won 9.4% of 64 1v1s on
   the sim as it is now: GRPO won 14.1%, 15.6% and 14.1% at iterations 25, 50 and 75, PPO 6.2%, 7.8% and 20.3%.  The user
   stopped it at PPO's 75th, its results not needed.  At 64 games a test it says no more than that neither fell; GRPO
   played 1.5-2.5x as fast (no critic).  The next run (item 3) records and clones anew at FEATURE_VERSION 10 anyway.
2. **The bot's inference in C++** (02-policy.md, "Deployment"; the nets in ONNX not for now, the user, 2026-11-13).
   Built and checked, not yet used: `libzkpolicy` (`sim/policy/`, `zkpolicy.h`, binding `sim/py/zkpolicy.py`,
   table file `bots/zkbot/data/policy_table.bin`, written by `python sim/py/zkpolicy.py table`) has the type table, the
   featurizer and the actions.
   * Parity: bit for bit with the Python over 49,451 frames of the four kinds (`tools/policy_parity.py`, results in
     `/mnt/d/zk-learned/parity/`).  Speed: 1.13 ms a frame against numba's 2.13, and 0.19 ms on 16 threads.
   * Next: training's featurizing (`_worker_chunk`, `_worker_batch`, cloning's samples, frame data) and the orders'
     encoding and decoding through it, after the GRPO trial; then the map bundle.
   * What it found in the Python is fixed on both sides, the user asked (2026-11-13): the DPS sectors even and the frame
     over 80 minutes (`FEATURE_VERSION` 10), a Lobster's throw by its real name (it threw at its own feet when a training
     game dropped its name), decode's fine point in float64 as encode's, and encode's last order whole.  Parity held after
     (`/mnt/d/zk-learned/parity/check-fv10.json`).  Left: the bits match on glibc's libm only, so the bot on Windows will
     need the same.
3. **Run the pipeline again whole** (the user starts it): record the scripts' games with the new economy and the new kinds
   of game into a new `--data`, collect frames, train an encoder, clone, DAgger, PPO.  Before it: a small smoke run of the
   whole pipeline hasn't been done since the variety work (only `--dry-run`, the suite, and each stage's own tests), and
   the unconfirmed choices under "Variety" are the user's to settle.  Check `wsl free -g` first: the pool is 10.8 GB now
   (it was 7.5), and a sample from late in an 80-minute game takes 1.3 to 2 GB of the card.
4. **PPO's speed**: with concurrent collectors, training was longer than collecting, and in `full-2` fell to 70 decisions a
   second as frames grew.  A step takes about 0.5 s whatever its size, so with minibatches as big as memory allows
   (2026-11-12) collecting (about 150 a second in `full-2`) should be the limit; check it in `metrics.jsonl`'s `train_rate`
   and `collect_rate`.  In a round's replay the decision itself (encoder, time transformer, critic: 41 ms at 64 decisions) is the bigger part; a
   step is about 1 ms of small kernels, which fusing (`torch.compile` of a step, now the shapes are fixed) could shrink.
5. **The stall in expansion** survived DAgger.  Whether PPO's economy share (the policy running its own economy in more games)
   moves the extractor count is the thing to watch in its tests.
6. **The engine hand-off**:
   * the rest of the bot's inference in C++ (item 2), with the nets in onnxruntime, nothing to install (the user, 2026-11-12);
   * decoded orders sent as commands;
   * the opening and the economy script in the engine;
   * the checks listed in 02-policy.md, "Not yet checked live".
7. **The user's open questions** (`docs/model.md`, "Open questions").

## Traps

* **Leftover build directories**: `~/zksim/build` is the one in use (libzksim ABI 26 and libzkscripts ABI 13 since a game
  could be copied, 2026-11-13; libzkrecord ABI 2).  One built before that needs rebuilding.  The `build-*` and `agent*` directories beside it are leftovers of comparisons and of agents' tree
  copies, safe to delete, except `agent-maps-lupa` (above).
* **Memory**: check `wsl free -g` before a launch.  Never run two jobs that load the map pool at once.  Every GPU trainer caps
  itself at 85% of the card (`--gpu-memory`): past the card, the Windows driver spills into system RAM instead of failing.
  With collectors, the learner's cap leaves each `--collector-gb` plus about 0.4 GB of CUDA's own.
* **Forking with torch**: a child forked after the parent ran a parallel CPU op hangs at its first one (OpenMP), so collectors
  set one thread first.  A parent that has started CUDA can't fork children that use it: the trainer forks its collectors
  before the learner touches the GPU, and the tests of them run the trainer in a fresh process (in the test process, with
  every earlier test's threads behind it, a collector sometimes hung).
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
* **`git status` inside WSL** shows files as modified that aren't (line endings): use Windows git.
* **Adding maps to the pool** needs the engine (`sim_engine.py facts` starts it six times a map): 01-simulator.md, "Data".
* **Never committed**: `results/` and `.vscode/`.
