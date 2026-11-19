# zk-ArenaAI, the learned policy: where it stands

Updated 2026-11-19.  Branch `gru-memory`; `main` and `learned-policy-plan` are at `1d80deb`.  The user says when to
commit.  Repo: `C:/Users/Thoma/Documents/sources/zk-ArenaAI`.

The current state only, kept small (the user, 2026-11-19).  Elsewhere:

* **`docs/training_logs.md`**: every run, its tests and what was seen in it, each hourly check of the run going, the
  dated changes to the sim and the features, and my choices the user hasn't confirmed.  Anything dated goes there.
* **`docs/learned-policy/`** (start at `00-overview.md`) and **`docs/model.md`**: how everything works, with its numbers.
  `06-select-then-command.md` and `05-order-autoencoder.md` there are what's been built since `rdnn-5`.

## Goal and status

The goal is a Zero-K AI that beats CircuitAI, pursued as a Pluto-style learned policy: a network trained in a simplified
simulator (`sim/`).  First by cloning scripted players, then DAgger, then PPO or GRPO from the clone; since 2026-11-17 by
RL from nothing, against itself, with a recurrent memory.

* **Nothing plays in the engine yet.**  `bots/zkbot` hosts the policy: it chooses a commander and sends no commands.  The
  rules bot that was there is at the tag `rules-bot-final` (local only).
* **No policy beats the scripts.**  From nothing: `gru-1` won 3 of 128 1v1s against the script at iteration 1250 (the
  score 0.184 from 0.148), `rdnn-3` 1 of 128 at 1000 (0.166 from 0.151) and `rdnn-5` 2 of 128 at 750 (0.171 from
  0.144), each with 2.5 to 3 extractors at 6 minutes for the script's 10 or 11.  From a clone: 8-17% of test games won
  after DAgger (`full-2` to `full-4`) with about half the script's extractors, and no RL stage moved that beyond noise.
* **In the sim** (01-simulator.md) every team plays a scripted opening; the C++ economy script or the policy runs its
  constructors, and the C++ army script or the policy the rest.  Four kinds of game (1v1s, team games of 2 to 4 a side,
  free-for-alls of 3 to 8, chicken defense), each on the pool maps that hold it (404 maps, 31 held out), for up to 80
  minutes.  The libraries: libzksim ABI 32, libzkscripts 17, libzkrecord 2, libzkpolicy 4.
* **A test compares only with tests on the same sim and script.**  Against the script as it is since 2026-11-18:
  `rdnn-1`'s test at 500 and all of `rdnn-2`'s to `rdnn-5`'s.  Every earlier one was against a richer script.
* **Since 2026-11-18** (the user; the log, "What changed under the runs"): one extractor a spot in the scripts and the
  opening, and frames that say who holds each spot (`SimEnv.site_status`; 01-simulator.md, "Known gaps"); nothing a
  seat only remembers among what the nets read, the actor's own memory keeping what it saw (02-policy.md, "Nothing
  remembered"; `FEATURE_VERSION` stays 10); an area mex, the policy's kind 15 (02-policy.md, "Kinds").
* **Since 2026-11-19** (the user; from `rdnn-4`): an attack that's over leaves no order in a unit's record
  (`World::end_attack`; the unit's idle flag was right already); `production_lead` is capped at half a team's metal
  income (`PRODUCTION_SHARE`); and `--lead-draws` keeps the lead rule out of training games while they mostly draw
  (03-training.md, "Reward"), `--lead-fade` giving the share of games that have it after.
* **Since 2026-11-19, after `rdnn-5`** (the user; no run has had it): a build sticks.  A unit on a build (its record's
  order a BUILD, and not idle) can be picked for one thing, the policy's kind 12, `CANCEL_BUILD`, which ends its
  order and its area mex; no group order of the policy's reaches it; and an untrained actor picks such a unit 3 nats
  less (`CANCEL_START`, mine).  Only where the cancel is among the kinds: `policy_sim_env.KINDS` has it, a saved
  run's own list hasn't.  02-policy.md, "A build sticks".
* **Since 2026-11-19, later still** (the user; `code-1` is the first run with any of it; the log, "After `rdnn-5`"):
  * *In the sim* (libzksim ABI 32; 01-simulator.md, "Economy"): several told to BUILD one type at one point build one
    thing between them (`World::frame_to_join`, `SimEnv._placed`), and a RECLAIM is an area's: 32 elmos round a
    feature named, or a radius round a place (`zksim.reclaim_area`).
  * *Orders as commands* (`bots/zkbot/policy/selections.py`; 06-select-then-command.md): a decision is up to 16
    commands, each selecting units by roles, a circle and a portion of it and telling them one of 21 kinds, with
    places as fractions of the map.  Nothing in one points into a frame.
  * *The order autoencoder* (`tools/order_vae.py`; 05-order-autoencoder.md): a decision's commands to 128 numbers and
    back.  It reads no frame, is trained on the scripts' commands with plausible made-up ones mixed in, and is then
    frozen.  What it says is plausible too: roles that can be told the kind, a type they make (`selections.able`).
  * *The actor that chooses its code* (`tools/policy_code.py`, `policy_train.py --size code`; docs/model.md, "An
    actor that chooses a code"): the frame's latents, the RDNN and a Gaussian over the code.  It reads no unit, and
    its update featurizes no frame.
* **The actor** (`tools/policy_latent.py`, `policy_train.py --size rdnn`; docs/model.md, "An RDNN in the GRU's place"):
  `full-4`'s frozen frame encoder, then a recurrent divisive normalization network of 1,024 units (arXiv 2608.01947)
  whose state is all a seat keeps, told to the rest as one vector.  It decides every 3 s, as actions one after another:
  a unit, then its order.  `--size gru` is a GRU in its place; the actors that cloning trains read a window of 4 frames.
* **Training** (`tools/policy_train.py`; 03-training.md, "GRPO from branched games", "Reward", "Playing"): bootstrapped
  GRPO.  Groups of 8 copies of a game play stretches apart, each branch judged against its group on 15 reward terms
  with a small critic a term.  Against itself only, the policy running its own economy, each seat told its own draw of
  the terms' weights (z), in collector processes (two till `code-2`, which has six).  The stage and its directory are still called PPO.
* **The cloning pipeline** (`tools/policy_pipeline.py`: record, frames, encoder, clone, DAgger, eval, PPO, with a
  dashboard at `http://localhost:8765`) isn't the current path.  It last ran as `full-4`, whose encoder every run since
  reads.

## A run is going: `code-2`, a 6-hour trial of the code actor with 6 collectors, to 2026-11-20 00:17

**`code-2`** (the user, 2026-11-19: "stop the current training run and start a new one, from iteration 0", then "try
6-8 collectors", then "skip 8 collectors"; started 18:17 by `results/start-code.sh 6`): `code-1` again, the actor that
chooses the order autoencoder's code on `vae-3`, from nothing with `rdnn-5`'s flags (`results/launch-code.sh`), with 6
collectors of 16 games and the collector as it is since ("A faster collector", the log).  Under
`/mnt/d/zk-learned/runs/code-2` (`ppo/` with a checkpoint every 50 iterations, `logs/ppo.out`, `replays/`, `end`,
`begin`, `spread-floor`), its dashboard at `http://localhost:8765`.

* **Its rate** was 2,650 decisions a second in its first 9 iterations (collecting 3,350, training 3,780).  In checks
  from `code-1`'s last checkpoint at the same iterations: 1,480 with the collector as it was and 2 collectors, 1,920
  as it is now, 2,640 with 4 and 3,060 with 6.  A run's rate falls as its games grow old (`code-1` to five sixths).
* **Its card and memory**: `--collector-gb 0.8 --gpu-share 0.76` (mine: a collector's process costs the card 0.4 GB
  before its cap); a collector holds 0.23 GB and the learner 1.35, the card 4.9 GB in all, WSL 26 GB to spare.
* **Its spread's least** falls from 0.1 to 0.03 over its six hours, by the clock (`--spread-floor 0:0.1 6:0.03
  --spread-floor-began`).
* **Its test at iteration 0**: no win of 128, the score 0.155, 2.2 extractors at 6 minutes for the script's 10.3.
  Tests every 250 iterations; the log has them, and each hourly check (`code-2`).
* **Its hourly check** is a cron job in the session that started it (`results/code-status.sh`,
  `results/code_terms.py 5`, and D:'s free space: a checkpoint is 0.39 GB and D: had 44 GB); its watchdog and memory
  guard are up.  `results/stop-code.sh` stops it and waits, `results/resume-code.sh` carries it on to its `end`.

### Before it: `code-1`, stopped at iteration 252

The same actor and flags with 2 collectors of 48 games, 2026-11-19 16:38 to 17:46, stopped by the user for the
collector's work and `code-2`.  Under `/mnt/d/zk-learned/runs/code-1` (`ppo/ckpt/it-000252.pt`, one every 20).

* **Nothing learnt by 250**: the score against the script 0.163 at iteration 0 and 0.164 at 250, 1 win of 128 each,
  2.6 and 2.2 extractors at 6 minutes for the script's 10.3 (`rdnn-5`: 0.144 and 0.160).  Its spread had barely moved
  from the unit Gaussian's (0.995; the mean's size 0.021) and its floor was at 0.087.
* **Its rate** was 1,200 to 1,300 decisions a second from iteration 20 on, 1.7 to 2 times `rdnn-5`'s at the same
  iterations, not the four times the smokes were read as; collecting limited it, on a machine three quarters idle.
* **A faster collector** (the user, 2026-11-19: "Proceed with 2 and 3"; the log has what, the checks and the tools,
  `results/faster/`): every seat's reward terms at once after a step (`SimEnv.seats_terms`, the same numbers to the
  bit), no sort of a team's units when none is new, no frame kept in a code's step, a code's pass taking to the card
  only what the frame encoder reads (`collate(frame_only=True)`: a fifth of a collector's time was the rest), and
  wider passes for it.  A collector second makes 21% more decisions; the wider passes alone made no difference.

### Before those: `rdnn-5`, stopped at iteration 769

`rdnn-3` again from iteration 0 with four changes (the user, 2026-11-19, stopping `rdnn-3` at iteration 1149: "I don't
think it's learning"): an attack order that's over, its target dead or off the radar, leaves no order in the unit's
record; `production_lead` counts no more of a team's spending on units than half its metal income; no lead ends a
training game (`--lead-draws 0.8`) until fewer than 80% of the last 800 ended seats drew at the time limit; and from
then on the lead rule is in 80% of the games that start (`--lead-fade 0:0.8`), the rest played out.  The user stopped
it 7 of its 24 hours in, for checks.  All of it is under `/mnt/d/zk-learned/runs/rdnn-5` (`ppo/` with
`ckpt/it-000769.pt` and one every 20 iterations, `logs/ppo.out`, `replays/`), its flags `results/launch-rdnn.sh`'s
defaults.  The log has its tests, every check and what was seen.

* **Where it stood**: the score against the script 0.144, 0.160, 0.159 and 0.171 at iterations 0, 250, 500 and 750,
  with 1, 2, 1 and 2 wins of 128: learning by the rule (at 750 the score +4.3 SE, own value, income and extractors past
  2 SE), and level with `rdnn-3` (0.166 at 750), with 2.85 extractors at 6 minutes for the script's 10.
* **What the checks after it found** (the log, "The checks after it"; `results/checks/`; one run of 32 to 128 games
  each, at the checkpoints 100, 140 and 769):
  * *It stopped building because building paid nothing.*  BUILD and area-mex actions fell from 2,150 an iteration to
    450 in a burst of KL at 114 to 116 and to none from 700 (`rdnn-3` the same by its 1000).  At 100 a team that could
    build won 31 and lost 35 against the same policy that couldn't, 61 build orders a game leaving it 1.7 extractors
    ahead: a BUILD's type was drawn evenly from every structure, its place a median 1,760 elmos off, and the policy's
    own next order to the unit ended 53% of builds, a median 21 s in.
  * *Left to finish, an area mex pays in the economy terms and wins nothing.*  With area mexes alone and no order
    reaching a unit on a build for 120 s: 9.8 extractors for 2.5 and 2.5 times the income, +0.58 of weighted return
    (+3.9 SE), and 34 won, 32 lost, with no more value: at 100 the metal isn't turned into army.
  * *Its fighters wait for their turn.*  Against the script 71% of fighter-time is idle at 769 for 53% at 140, half of
    it at home.  It gives under one action a decision, to a unit chosen without regard to whether it's idle, its
    group orders have gone (24% of actions to 1%), and its army has doubled: an idle fighter waits 76 s.
  * *The frozen encoder is behind neither.*  Against the script it keeps the policy's frames as well as the scripts'
    own; in self-play it keeps less of where the policy's scattered army stands (R^2 0.41 for 0.80).
* **The user's answer for the builds** (2026-11-19): a build that sticks, and a cancel ("Goal and status", "Since
  2026-11-19, after `rdnn-5`").  With it and the same weights at 100, a team that builds had 8.4 extractors for 3.0,
  +0.58 of weighted return (+3.6 SE) and more value (+2.8 SE), and won 45 and lost 31 (+1.6 SE: not shown); 17% of
  its builds ended by a cancel.  No run has trained by it.
* **The user's answer for the rest** (2026-11-19): orders as a player gives them, select then command; an autoencoder
  of a decision's commands, frozen; and an actor that chooses its code and reads no unit ("Goal and status"; 06 and
  05).  Built and checked by play, not run (the log, "After `rdnn-5`"):
  * *The scripts play as well in commands*: 62 won, 64 lost of 128 1v1s against themselves, for 67 and 57 with their
    orders left as they are.
  * *The first autoencoder's 64 numbers held four commands*: through its code the scripts won 32 and lost 89, its
    made-up commands (any roles, any type) came back in part, and codes drawn at random said nothing 72% of the time.
  * *The user's choices after it and a second* (2026-11-19: "1. vae-2 2. bigger code should be fine 3. either is fine
    4. commands should be plausible/valid"): the second's stronger pull to the Gaussian and few emptied decisions, a
    bigger code, the actor's start either way (the unit Gaussian, as built), and made-up commands some unit could obey.
  * *`vae-3`, the one in use* (128 numbers, mine; `policy_latent.ORDER_CODER`): the scripts' decisions of up to 8
    commands (90% of them) come back right in everything 96 to 99.6% of the time, those of 9 or more 5%; a made-up
    decision's roles 82% and its type 95%.  **Through its code the scripts won 60 and lost 62.**  A fourth with 256
    numbers, trained as long, gave back less.
  * *An untrained actor on it* says 1.7 to 2.1 commands a decision, all plausible, and 0.5 to 0.6 units a decision
    obey one (the trainer's smoke).
  * *A place is only as near as the actor's spread lets it be*: a code off by 0.3 keeps its kinds, roles and types
    (97%, 94%, 90%) and puts its places 650 elmos off; by 0.1, 220; by 0.03, 70.  So the spread's least falls over a
    run (the user, 2026-11-19: "Start with 0.1, decreasing to 0.03 as the training run continues"):
    `--spread-floor AT:SPREAD ...`, its points iterations or, with `--spread-floor-began`, hours since the run began,
    which is how `start-code.sh HOURS` gives it (`0:0.1 HOURS:0.03`; by the clock is mine).
* **To carry it on, or start another, on the user's word only**: `results/resume-rdnn.sh END` (it reads the run's
  `end`, 2026-11-20 01:55, starts every game anew and no watcher; carried on, it would play by the new rule, its
  kinds being the code's), or a new directory's name in `results/*rdnn*` and
  `results/start-rdnn.sh HOURS`.  For the code actor: `results/start-code.sh HOURS` starts
  `/mnt/d/zk-learned/runs/code-2` (`results/launch-code.sh` has the trainer's flags, `rdnn-5`'s with `--size code`
  and 6 collectors;
  `CODER` gives another autoencoder), with `code-status.sh`, `code_terms.py N`, `stop-code.sh`, `resume-code.sh`,
  `kill-code.sh` and its watchdog and memory log, each the `rdnn` run's under a `code` name; `code-1` and `code-2` are the
  runs on them.  A run has an hourly check by a cron job in the session that starts it (`results/rdnn-status.sh` or
  `code-status.sh`, `results/rdnn_terms.py N` or `code_terms.py N`, no torch; a report to the user and a line in the
  log), a watchdog (25 minutes without a report, 3 restarts at most), and no change in the middle of it: not the
  memory, the segment, the rates, the terms or the code.
* **From the user**: a small memory state early is no fault ("The fading memory is to be expected early on; most games
  end early and there's not much to remember"), and no KL guard on the recurrent weights.

## What loads

* **`FEATURE_VERSION` is 10** since 2026-11-13.  Trained at 10: `full-3`'s and `full-4`'s encoders and actors, `gru-1`'s
  and the `rdnn` runs' actors (on `full-4`'s encoder).  All but `rdnn-3`'s to `rdnn-5`'s were trained with the
  remembered buildings as tokens, which the featurizer no longer makes.  Nothing trained before 10 loads.
* **`policy_latent.ENCODER` still names `encoder-big`**, which doesn't load: every tool that loads an actor needs
  `--encoder` (`policy_train.py`: `--latent-encoder`), `/mnt/d/zk-learned/runs/full-4/encoder/vae-last.pt`.
* **Recorded games** are raw and still read: `clone/army-8` (data version 7, every kind of game).  But they're by the
  scripts before one extractor a spot, and their frames read every spot free.
* A run carried on with other reward terms keeps the critics of those it had, by name; the new ones start from nothing
  while the actor waits 10 iterations.
* **The order autoencoder** (`/mnt/d/zk-learned/orders/vae-3/order-vae.pt`, `policy_latent.ORDER_CODER`) is of the
  commands as they are: `order_vae.load` refuses a file made for other kinds, roles or radii (`selections.KINDS`,
  `groups.roles`, `selections.RADII`).  A `--size code` checkpoint holds its weights and names its path
  (`Size.coder`; `--order-coder` gives another).

## Environment

* **Zero-K**: portable install at `C:/Users/Thoma/Downloads/zero-k-portable`; engine 2026.07.04; Zero-K v1.14.10.3;
  ControllerAI 0.3.  Read-only sources beside the repo: `../RecoilEngine` (CircuitAI under `AI/Skirmish/CircuitAI`),
  `../Zero-K`, `../ControllerAI`.
* **Windows Python 3.13** runs the bot (`pip install requests websockets orjson numpy numba`).  Without numba, featurize
  runs in plain Python, about 50x slower, and says so on import.
* **WSL (Ubuntu)** runs the sim and training: `~/zkml/bin/python` (PyTorch, RTX 4070 of 12.3 GB, pytest-xdist), with
  pytest in `~/zksim/pt`.  So `PYTHONPATH=$HOME/zksim/pt:bots`.  WSL has 40 GB, Windows 48.
* **The sim's libraries**: `cmake -S sim -B ~/zksim/build && cmake --build ~/zksim/build -j8` builds `libzksim.so`,
  `libzkscripts.so`, `libzkrecord.so` and `libzkpolicy.so` (`ZKSIM_LIB` and `ZKPOLICY_LIB` override; the recorder needs
  `libzstd-dev`).  Rebuild after any change under `sim/`.  `bots/zkbot/data/policy_table.bin` is written by
  `python sim/py/zkpolicy.py table`.
* **`sim/data/rules.json`** is made by `tools/sim_rules.py`, which needs `lupa`: it's only in `~/zksim/agent-maps-lupa`
  (add it to `PYTHONPATH`), not in `~/zkml`.  **`sim/data/maps`** is gitignored: after a pull,
  `python tools/sim_data.py maps`.
* **Data on D:**, outside git, under `/mnt/d/zk-learned` (44 GB free on 2026-11-19, before `code-2`'s checkpoints):
  * `maps/`: the pool of 204 real (18 of them Violet Rampart) and 200 made-up maps, 31 held out, each with its `modes`.
    It loads through the map cache in 27 s, to 1.4 GB of a process's own memory (01-simulator.md, "The map cache");
  * `clone/army-8`: 175 GB, about 8,800 script games of every kind, 100M decisions, recorded 2026-11-13;
  * `orders/`: `commands-1` (68 MB: 1.5M decisions of the scripts' commands in games of every kind, recorded
    2026-11-19) and the order autoencoders trained on it: `vae-3`, the one in use, and `vae-1`, `vae-2` and `vae-4`,
    kept for the comparison in 05;
  * `runs/`: `code-2`, `code-1`, `rdnn-5`, `rdnn-4`, `rdnn-3`, `rdnn-2`, `rdnn-1`, `gru-1`, `full-4`, `full-3`, `full-2`, `grpo-trial`; the rest is older runs
    that don't load, and scratch;
  * `parity/` (`tools/policy_parity.py`'s results), `dashboard-cache/` and `dashboard-demo/` (the dashboard's map
    pictures and icons; a made-up run directory for working on it).

## Layout

* **`bots/zkbot/`**, the engine side: `main.py` drives `host.py`, which runs `policy/runtime.py` once a decision.
  `policy/` has the raw frame schema (`rawframe.py`, which also generates `sim/include/zksim/rawframe.hpp` via
  `tools/policy_schema.py`), the map bundle, type tables, featurizer, actions, group orders and orders as commands
  (`selections.py`).
* **`sim/`**: the C++ simulator (`src/`), the economy and army scripts (`scripts/`), the recorder (`record/`), the type
  table, featurizer and actions in C++ (`policy/`: libzkpolicy), their ctypes bindings (`py/`, and
  `tools/policy_scripts_cpp.py` for the scripts), data (`data/`) and tests (`tests/`).
* **`tools/`**: the sim's data and checks against the engine (`sim_*.py`, `map_*.py`); the sim as a game
  (`policy_sim_env.py`, `policy_opening.py`); frames and the encoder (`frame_*.py`); the policy (`policy_model.py`,
  `policy_latent.py`, `policy_code.py`, `policy_clone_*.py`, `policy_train.py`, `policy_pipeline.py`,
  `policy_parity.py`); the scripts' commands recorded and their autoencoder (`order_data.py`, `order_vae.py`); the
  trainer's play (`policy_graphs.py`, `policy_collectors.py`); the dashboard (`policy_dashboard.py`, `dashboard/`,
  `policy_replay.py`).
* **`results/`** (never committed; on this machine only): the runs' launch files and helpers.  For the `rdnn` runs:
  `start-rdnn.sh HOURS` (from nothing, with its guard, memory log, watchdog and dashboard), `launch-rdnn.sh` (the
  trainer's flags), `rdnn-status.sh` and `rdnn_terms.py N` (no torch: safe beside it), `stop-rdnn.sh`,
  `resume-rdnn.sh` (reads the run's `end`; it starts no watcher), `kill-rdnn.sh`, `check-rdnn.sh NAME N` (a copy of
  the last checkpoint carried on N iterations), `paired_tests.py`, `compare_tests.py`, `replay_summary.py`,
  `inspect_rdnn.py`, and `run-tests.sh` (the suite; it refuses beside a trainer).  The GRU run's are the same under
  `gru` names.  `checks/` has the checks of a checkpoint by play made after `rdnn-5` (`checks/run.sh` runs one and
  lists them; they load torch and the map pool), and `checks/language_check.py`, the scripts played through commands
  or the autoencoder.  For the code actor: `order-data.sh`, `order-vae.sh` and `order-vae-eval.sh` (record the
  scripts' commands, train the autoencoder, measure it), `launch-code.sh` (`rdnn-5`'s flags with `--size code`),
  `smoke-code.sh N` (N iterations of it in a scratch directory), and the run's helpers under `code` names
  (`start-code.sh HOURS`, `code-status.sh`, `stop-code.sh`, `resume-code.sh`, `kill-code.sh`, the watchdog), which
  name `code-2` now.  `faster/` has the collector's checks: `check-code.sh TREE NAME N` carries `code-1`'s last
  checkpoint on N iterations by a tree's trainer (set two side by side with `check_compare.py`), `env_ab.sh` plays the
  env's bookkeeping in two trees and says whether the numbers are the same, `spy-one.sh` (as root) and
  `profile_report.py` say where a collector's time goes.
* **Tests**: `tests/policy/`, `tests/test_*.py`, `sim/tests/`, run as `pytest.ini` says.  None needs the engine.
* **Engine games**: `zk-startup.ps1`, `zk-maps.ps1`, `maps.md`, `README.md`.

## Running

In WSL, from `/mnt/c/Users/Thoma/Documents/sources/zk-ArenaAI`, with `PYTHONPATH=$HOME/zksim/pt:bots`:

    ~/zkml/bin/python -m pytest -q                     # about a minute, on 8 processes (pytest.ini); all pass
    ~/zkml/bin/python -m pytest -q -n 0 tests/policy/test_train.py     # one process: for a debugger, or to time a test
    bash results/start-rdnn.sh 24                      # a run from nothing as rdnn-5 is; first give results/*rdnn* a new directory's name
    bash results/rdnn-status.sh                        # where the run going, or the last, stands; no torch
    bash results/checks/run.sh fighter_check.py RUN CKPT:fight     # a checkpoint checked by play (torch)
    bash results/stop-rdnn.sh; bash results/resume-rdnn.sh     # stop it and wait; carry it on to its end (every game anew)
    bash results/order-data.sh OUT DECISIONS           # the scripts' games played in commands, each look's kept (the map pool)
    bash results/order-vae.sh DATA OUT                 # the order autoencoder trained on them (the card; half an hour)
    bash results/checks/run.sh language_check.py RUN 128 coded 7 CODER    # the scripts through the autoencoder, against the scripts
    bash results/smoke-code.sh 3                       # the code actor's trainer for 3 iterations (the pool and the card)
    bash results/start-code.sh 24                      # a run of the code actor from nothing, as code-2 (on the user's word)
    bash results/faster/check-code.sh TREE NAME 8      # code-1's last checkpoint carried on 8 iterations by a tree's trainer (the card)
    bash results/code-status.sh                        # where it stands; no torch
    ~/zkml/bin/python tools/policy_pipeline.py --root /mnt/d/zk-learned/runs/NAME --train-encoder \
        --data /mnt/d/zk-learned/clone/NEW [--dry-run]
        # the cloning pipeline, stage by stage; the same command again carries on.  A `stop` file in ROOT halts it after
        # the running stage.  --data with no recording in it is recorded first
    ~/zkml/bin/python tools/policy_dashboard.py --root /mnt/d/zk-learned/runs/NAME     # the dashboard alone, during a run or after
    ~/zkml/bin/python tools/policy_replay.py demo --out DIR     # replays of the army scripts' games, for the viewer
    ~/zkml/bin/python tools/policy_parity.py check              # the C++ featurizer and actions against the Python

Recording, cloning and the clone's eval are in 03-training.md ("Script data", "Cloning"); the trainer's flags in "PPO",
and how it plays in "Playing".  A `stop` file in a trainer's run directory ends it at its next report, and the same
command resumes it.

**The tests** (859 pass, 4 of them need CUDA and are skipped without it) run on 8 processes, a file's tests in one
process, in about 60 s; `--durations=20` shows what's slow.  The user wants the suite lean: no redundant, sure-to-pass
or slow tests.

In the engine (Windows), recording raw frames from a game: `$env:ZKBOT_FLAGS='rawframe'`, then
`.\zk-startup.ps1 ControllerAI CircuitAI:Normal -Headless -Map 'TitanDuel 2.2' -NoDownload -Client 'python bots\zkbot\main.py {team}'`.
README.md has the rules for engine games: headless at max speed, windowed only when the user asks.

## Next

0. **`code-2`, the 6-hour trial going**: its hourly checks, and at its end (00:17) its tests beside `code-1`'s and
   `rdnn-5`'s at the same iterations, what its spread and its commands did, its rate, and whether it learned by the
   log's rule ("Reading a test").  Then, on the user's word, a longer run (`results/*code*` given a new directory's
   name, `start-code.sh HOURS`).  If collecting is to be faster still: `--pass-seats` past 2 for the team games, 8
   collectors on 128 games, and what a pass copies that could be collated in place.
   Open, put to the user: a site's status still says "enemy" of a spot from an extractor only remembered.
1. **If the `rdnn` runs learn no better than `gru-1`**: the GRU with the memory told as a vector and the same rates,
   which separates the vector and the rates from the RDNN; and the missing arms of the paper's task (the log,
   `rdnn-1`).
2. **To set a run beside the older ones**, test their checkpoints anew against the script as it is (`gru-1` and
   `rdnn-1` keep one every 20 iterations, none of iteration 0); it needs the card, so no run going.
3. **Why the clone stops expanding** (mine, put to the user and not answered).  Held out, `full-4`'s clone gets the
   kind right for 83% of the script's builds (97-100% for the other orders), and a whole build costs 7.5 nats against
   0.3-2.4 for the rest (`clone/log.jsonl`): split that by type, place and offset, and by what's built.
4. **The bot's inference in C++** (02-policy.md, "Deployment"; the nets in onnxruntime, nothing to install, the user).
   Built: libzkpolicy has the type table, the featurizer and the actions, bit for bit with the Python, and the RL
   trainer featurizes through it.  Next: cloning's samples and frame data and the orders' encoding and decoding through
   it, then the map bundle.  The bits match on glibc's libm only, so the bot on Windows will need the same.
5. **The engine hand-off**: the rest of item 4; decoded orders sent as commands; the opening and the economy script in
   the engine; the checks listed in 02-policy.md, "Not yet checked live".
6. **The user's open questions** (`docs/model.md`, "Open questions"), and my unconfirmed choices (the log).

## Traps

* **Nothing with torch beside a running stage**: no pytest, smoke run or measurement, on the CPU or the card.  The memory
  guard killed `full-3` and `full-4` that way.  A change is made and tested in a tree copy (its sim tests run without
  torch), then applied with the run stopped.  A check run must end inside the watchdog's 25 minutes, or it takes that
  trainer for the run's.
* **A script that's running**: bash reads a script as it goes, so an edit to `results/launch-*.sh` during its run is run
  from wherever bash had got to.  Stop the run first (`results/stop-rdnn.sh` waits for the launch file too).  The tools'
  Python is safe to edit beside a run only in that its modules are loaded already.
* **A restart starts every game anew**: a run's games are young again for an hour, and the KL can burst 10 to 14
  iterations after.  If libzksim changed, the map cache is made again first (about 2 minutes).  A copy carried on needs
  `--decisions` past the checkpoint's, or it ends at once.  The trainer appends to `metrics.jsonl`, `minibatches.jsonl`
  and `logs/ppo.out`: after a stop that lost iterations, cut them at the checkpoint before carrying on.
* **Memory**: check `wsl free -g` before a launch, and never run two jobs that load the map pool at once.  Every GPU
  trainer caps itself at a share of the card (`--gpu-memory`, `--gpu-share`): past the card the Windows driver spills
  into system RAM instead of failing, and at about 11.3 GB of the 12.3 a process stalls.  With collectors, the learner's
  cap leaves each `--collector-gb` plus about 0.4 GB of CUDA's own.  The windowed actor's first training step on the CPU
  grows to 23 GB.
* **Build directories**: `~/zksim/build` is the one in use.  The `build-*` and `agent*` beside it are leftovers, safe to
  delete, except `agent-maps-lupa`.  Another agent changes the sim and the scripts at times (the user): a long run plays
  the build it starts with.
* **A site's status** is free in the sim's own records (`batch.frame`, a recording): `SimEnv.frame` fills it.  And the sim
  builds an extractor wherever it's told and pays each one on a spot: whoever places one keeps off a held spot itself.
* **The map cache** (`~/zksim/map-cache`, 6.1 GB; `ZKSIM_MAP_CACHE` moves it, empty turns it off) is made again when a
  map file or libzksim changes.  A change to what's kept, or to how `zksim.Map` makes what it gives the sim, needs
  `zksim.MAP_CACHE_VERSION` raised, or the old files are read.
* **Forking with torch**: a child forked after the parent ran a parallel CPU op hangs at its first one, so collectors set
  one thread first.  A parent that has started CUDA can't fork children that use it: the trainer forks its collectors
  before the learner touches the GPU, and their tests run the trainer in a fresh process.  The graphs' inputs aren't
  pinned (a 20 MB `cuMemHostAlloc` failed in WSL with 7 GB of the card free).
* **`wsl bash -c '...'`**: put anything more than a plain command in a script file.  A `$` is taken by the shell wsl
  starts first, where the variable isn't set.  `pkill -f` and `pgrep -f` match that shell too: write the pattern as
  `[p]olicy_train`.  A background process dies when the call returns unless started as
  `setsid nohup X > LOG 2>&1 < /dev/null &`.
* **A trainer started through `sudo -u`** dies at its first training step ("CUDA driver error: unknown error": no
  `/usr/lib/wsl/lib` on that path).  Start it as the user, and attach py-spy as root from another call.
* **`NAME` in WSL** is the PC's name, set for every shell: the launch files read `RUN_NAME`.
* **The dashboard's port**: a standalone `policy_dashboard.py` left on 8765 makes the pipeline run without its own.  Stop
  it before a launch, or give the pipeline another `--port`.
* **The cloning pipeline** plans a `frames` stage whenever a run's `frames/manifest.json` is missing: keep it when
  deleting frames.  A cloning run appends to its `log.jsonl`: clear it before rerunning one by hand.
* **Engine games**: never start one, or kill engine processes, while the user is playing (check
  `tasklist | grep -i "spring\|zero-k"` first).  `set_settings` at game start can make the engine exit, so the host
  sends it after frame 2; a sync-mode client that stops calling `finish_frame` stalls the match.
* **numba's cache** knows only the kernel file, so it holds rawframe's numbers as literals.  They're checked on import.
* **`share_kind{k}`** in a run's metrics and its log lines is by rawframe's numbers: 10 is BUILD, 11 ASSIST, 12 the
  cancel of a build, 13 RECLAIM, 15 the area mex.  The checks before 2026-11-19 08:55, and the log's entries from
  them, called 11 BUILD.
* **A unit's `ord_kind`** is the last order it was given, and stays when that's over (but an attack's, and after a
  stop).  Whether it's doing anything is its IDLE flag: a count of units "with an order" by the record is wrong, as
  two of mine were on 2026-11-19.
* **Line endings**: several files are CRLF; edit them without changing that.  `git status` inside WSL shows files as
  modified that aren't: use Windows git.
* **Adding maps to the pool** needs the engine (`sim_engine.py facts` starts it six times a map): 01-simulator.md, "Data".
* **A smoke's rate isn't a run's.**  A run's decisions a second fall by half over its first 40 iterations, as its games
  grow old (`rdnn-5`: 1,100 at iterations 2 and 3, 890 at 8 to 20, 520 over seven hours).  Set two actors side by side
  at the same iterations, and plan nothing by a run's iterations an hour before it has run: on 2026-11-19 a smoke's
  first three were read as four times `rdnn-5`'s rate, and a schedule laid by them was two to three times out.
* **Two checks from one checkpoint aren't the same games**: they draw differently from the first pass on, and a few
  percent between their rates is nothing.  Set them side by side over the same iterations, and read a profile
  (`results/faster/spy-one.sh`) for what a change did.
* **An edit script goes to the tree it's meant for.**  On 2026-11-19 one meant for a copy ran in the repo, beside a
  run; `git checkout` then gave the file CRLF (`core.autocrlf` is true and most files are LF in the tree).  The ones in
  `results/faster/` refuse outside a scratch copy unless told `--repo`.
* **A new tool on the card** caps itself before anything else and sends nothing through a net in one great batch
  (`order_vae.cap_card`, `CHUNK`): on 2026-11-19 `order_vae.py eval` put 100,000 decisions through at once, the card
  spilled and the machine's memory filled.  The same for a timing or a scratch script.
* **An order autoencoder's file and the commands go together**: a change to `selections.KINDS`, the roles or `RADII`
  makes every saved one unreadable (`order_vae.load` says so), and every `--size code` checkpoint with it.
* **Never committed**: `results/` and `.vscode/`.
