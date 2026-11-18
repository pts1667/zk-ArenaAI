# zk-ArenaAI, the learned policy: where it stands

Updated 2026-11-18.  Branch `gru-memory`, committed and pushed 2026-11-18 at about 07:30 with the run `rdnn-3` still going
(the user says when to commit).  `main` and `learned-policy-plan` are both at `1d80deb`.  Repo:
`C:/Users/Thoma/Documents/sources/zk-ArenaAI`.

The current state only.  How things got here is in git history: this file as it was at `f8a156c` has the long record of
every run to 2026-11-18 (each change made in the middle of one, each hourly check, the timings).  How everything works,
with its numbers, is in `docs/learned-policy/` (start at `00-overview.md`) and `docs/model.md`.

## Goal and status

The goal is a Zero-K AI that beats CircuitAI, pursued as a Pluto-style learned policy: a network trained in a simplified
simulator (`sim/`).  First by cloning scripted players, then DAgger, then PPO or GRPO from the clone; since 2026-11-17 by
RL from nothing, against itself, with a recurrent memory.

* **Nothing plays in the engine yet.**  `bots/zkbot` hosts the policy: it chooses a commander and sends no commands.  The
  rules bot that was there is at the tag `rules-bot-final` (local only).
* **No policy beats the scripts.**  From nothing, the best is `gru-1`'s 3 wins of 128 1v1s against the script at iteration
  1250 (the score 0.184 from 0.148), with 3 extractors at 6 minutes for the script's 11.  From a clone, 8-17% of test
  games won after DAgger (`full-2` to `full-4`) with about half the script's extractors at 6 minutes, and no RL stage
  since `full-1`'s moved that beyond noise.  "Earlier runs" below.
* **In the sim** every team plays a scripted opening.  The C++ economy script runs its constructors, or the policy does;
  the C++ army script, or the policy, runs its fighters, factories, missiles, the Trinity and air transports.  Four
  kinds of game: 1v1s, team games of 2 to 4 a side, free-for-alls of 3 to 8, and chicken defense, each on the pool maps
  that hold it (404 maps, 31 held out), for up to 80 minutes.
* **What the sim models** is in 01-simulator.md, and since the first runs: shots as projectiles aimed ahead and the whole
  of Zero-K's tactical AI ("Weapons and damage", "Unit AI"); no side going first ("Unit AI", "Validation"); Zero-K's end
  of a game, an alliance with no builder or factory left ("Teams and alliances"); plates and idle caretakers, and an
  economy script that builds them, a second factory at 40 metal a second and defences ("Economy", "The economy
  script"); water weapons, floats and sonar ("Water", "Sensing"); air transports ("Transports"); chicken defense; 12
  stats a team ("Teams and alliances").  The libraries: libzksim ABI 31, libzkscripts 17, libzkrecord 2, libzkpolicy 4.
* **A test compares only with tests on the same sim and script.**  The sim changed on 2026-11-12 (projectiles, the
  tactical AI, no side first), 2026-11-13 (the end of a game, the script's defences) and 2026-11-16 (water, sonar).  And
  on 2026-11-18 the script and the opening stopped standing several extractors on one spot, each of which the sim pays:
  the scripts' own games went from 9.6 extractors on 8.7 spots at 6 minutes and 23.8 on 16.0 at 12, for 30.6 and 75.6
  metal a second, to 8.5 on 8.5 and 12.6 on 12.6, for 27.0 and 43.8.  So every test, eval and recording before
  `rdnn-1`'s iteration 364 was against a richer script.  Against the script as it is: `rdnn-1`'s test at 500, and all of
  `rdnn-2`'s and `rdnn-3`'s.
* **One extractor a spot, and frames that say who holds it** (the user, 2026-11-18): `Pass::going` in
  `sim/scripts/src/eco.cpp` and `policy_opening.Seat._taken` (a spot a constructor is walking to is held);
  `SimEnv.site_status` (each seat's sites its own copy, with who holds each spot from the extractors its frame shows,
  made once a step); `SimEnv._going` (the policy's extractor keeps off a spot another of the team's units is on its way
  to).  Not fixed: a recording keeps a game's spots once, so its frames read every spot free; two allies' constructors
  can still go to one spot; the sim itself builds wherever it's told.  01-simulator.md, "Known gaps" ("Extractors on
  one spot").
* **Nothing remembered among the tokens** (the user, 2026-11-18; libzkpolicy ABI 4): a building a seat saw once and no
  longer sees ("unit ghosts": `KNOW_MEMORY` records, standing or dead, till their place is seen empty) was a unit token,
  a count in its region and firepower in its neighbours' sectors.  The featurizer, in Python and in C++, now gives the
  nets none of it; the actor's own memory keeps what it saw.  The raw frame still has the records, a site's status
  still goes by them, and a privileged critic has them among its hidden units.  `FEATURE_VERSION` stays 10 (no column
  moved), so `full-4`'s frozen encoder, trained with those tokens, still loads and now reads frames without them.
  02-policy.md, "Nothing remembered".
* **An area mex** (the user, 2026-11-18; the policy's kind 15): extractors on the free spots within a radius of a place,
  the nearest to the unit first, carried out by `SimEnv` a BUILD at a time; any other order to the unit ends it.
  02-policy.md, "Kinds".
* **The actor** (`tools/policy_latent.py`, `policy_train.py --size rdnn`; docs/model.md, "An RDNN in the GRU's place"):
  `full-4`'s frozen frame encoder (192 wide, 64 latent tokens a frame), then a recurrent divisive normalization network
  of 1,024 units (arXiv 2608.01947) over each decision's latents and a summary of what the last decision did.  Its state
  is all a seat keeps, told to the rest as one vector added to each controllable unit and each of the plan's slots.  It
  decides every 3 s, as actions one after another: a unit, then its order.  `--size gru` is a GRU in its place, told as
  4 memory tokens.  The actors that cloning trains read a window of 4 frames instead (10.5M parameters; `WIDE` 27.0M).
* **Training** (`tools/policy_train.py`; 03-training.md, "GRPO from branched games", "Reward", "Playing"): bootstrapped
  GRPO.  Groups of 8 copies of a game play stretches apart, and each branch is judged against its group on 15 reward
  terms, a small critic a term (`policy_model.MemoryCritic`) valuing where a branch ended.  Against itself only, the
  policy running its own economy, each seat told its own draw of the terms' weights (z), two collector processes playing
  as CUDA graphs and featurizing through libzkpolicy.  The stage and its directory are still called PPO, and the policy
  loss is SPO's.
* **The cloning pipeline** (`tools/policy_pipeline.py`: record, frames, encoder, clone, DAgger, eval, PPO, with a
  dashboard at `http://localhost:8765`) isn't the current path.  It last ran as `full-4`, whose encoder every run since
  reads.

## RL from nothing with an RDNN and nothing remembered among the tokens (2026-11-18): `rdnn-3`, a run of 24 hours, going

The user, 2026-11-18 at about 04:35, two hours into `rdnn-2`: "Currently, the encoder is given units that have been
seen in the past but aren't actually there ("unit ghosts" to help memorize where units were last seen); in addition, or
consequently, the policy is making "attack unit" commands against unit ghosts. Since there is now a recurrent memory
block, the unit ghost feature where units are still "seen" when out of line-of-sight can be removed. Afterwards, restart
the run from iteration 0."

* **The run**: `results/start-rdnn.sh 24`, started 2026-11-18 05:23:30 from nothing, so `rdnn-3/ppo`,
  `rdnn-3/logs/ppo.out`, `rdnn-3/end` (2026-11-19 05:23), the dashboard at `http://localhost:8765` on it,
  `rdnn-3/memguard.log` (3 GB), `rdnn-3/pss.log`, `rdnn-3/watchdog.log`.  Every `results/*rdnn*` helper names `rdnn-3`.
  24 hours from its start is mine.
* **What it is** (`results/launch-rdnn.sh`'s defaults), all of it from iteration 0: the RDNN of 1,024 told as a vector;
  a whole stretch as one segment (`--segment 128`) and the compiled loop (`--compile-memory`); the schedule from 2e-3
  (`--lr-start`), the actor at what the KL guard leaves of it, from 0.15 (`--lr-scale-start`), the recurrent weights at
  all of it in an Adam of their own (`--recurrent-lr 2e-3`), the critics at 3e-4; 96 games in two collectors of 48
  (`--gpu-share 0.65`, `--featurize-threads 4`); against itself only, the lead rule in every game, the policy's own
  economy; 15 terms, the last `production_lead`; a seat's own objective (`--z objective --objective-sigma 0.3
  --explore-share 0.25 --explore-weight 1`); the area mex; 1v1s on the pool's maps till iteration 500, then the other
  kinds ramped in to half by 2,500; a test of 128 1v1s against the script every 250 iterations and 4 replays an hour.
  It is `rdnn-2` with one thing changed in the code: the remembered buildings out of what the nets read.
* **What was removed**: the frame's `KNOW_MEMORY` records as unit tokens, as a count in each region (column 32, now 0),
  and so as firepower in the eight sectors of the tokens near them.  `featurize_kernels.select` and `region_block`,
  `featurize._hidden`, and the same three places of `sim/policy/src/featurize.cpp`.  Checked on 600 frames of the
  scripts' games (1,247 such tokens before, none after, everything else the same but column 32, in Python and C++
  alike), the five golden frames (156 tokens gone, 21 rows moved in sector firepower alone) and
  `tools/policy_parity.py check` (49,451 frames, no mismatch: `/mnt/d/zk-learned/parity/check-noghost.json`).
* **The policy made no attack on a ghost, as far as the masks go**: in those 600 frames no legal pointer of any kind was
  at a remembered building.  Every head is masked, an ATTACK_UNIT is at a foe in sight (a missile's in sight or on
  radar), and the trainer stops on a choice outside its mask (`masked_out`).  No term penalises an order.  What a
  replay shows that looks like it: an attack on a building stands when the building goes out of sight, as the engine's
  does (`World::target_lost`); a unit's record keeps the last order it was told till the next; and the viewer draws
  that order to wherever its target really is.  None of that changed.  Told to the user.
* **My choices**:
  * the featurizer, not the frame: the sim's and the host's records stay (the scripts read the enemy's own frame;
    recordings keep `knowledge`);
  * **a site's status still remembers**: a spot the enemy was seen to hold reads "enemy" from a remembered extractor
    till it's seen free.  The policy's extractors are placed by it, and the sim would stand one on a spot held out of
    sight and pay it.  It's the one input left that says what was seen and isn't now.  Put to the user, not answered;
  * a privileged critic has them among its hidden units, with how long ago each was seen.  This run's critics read
    the actor's latents and memory, so nothing of it reaches them;
  * `full-4`'s encoder is not trained again: it now reads frames without those tokens, as it read every frame before
    a first sighting;
  * `rdnn-2` ran on to its test at 250 so that this run has one to stand beside.
* **Reading its tests**: each paired with its own iteration 0 (`paired_tests.py`) and beside `rdnn-2`'s at the same
  iteration (`compare_tests.py`; both in `rdnn-status.sh`).  `rdnn-2` is like for like and has only 0 and 250.  Past
  250 the only outside test against this script is `rdnn-1`'s at 500 (0.165, 2 wins, extractors 2.5), a run with five
  restarts behind it.  Learning, as put to the user: wins, or the score up by more than 2 SE on iteration 0's, or two
  of own value, income, damage dealt and extractors at 6 minutes up by more than 2 SE.
* **Its tests**:
  * iteration 0: 2 wins of 128, the score 0.151 +- 0.008, extractors at 6 minutes 2.1 (the script's 10.4); within 1.4
    SE of `rdnn-2`'s iteration 0 on every measure;
  * iteration 250 (about 07:20), against its iteration 0: 1 win, the score 0.160 (+0.009 +- 0.007, 1.4 SE), own value at the
    end 4,374 to 5,681 (+4.8), income over the game +3.0, damage dealt -0.1, extractors at 3 minutes 2.4 to 3.1 (+5.2)
    and at 6 2.1 to 2.6 (+3.0, the script's 11.0), games 28 s longer.  Learning by the rule (own value, income,
    extractors).  Against `rdnn-2`'s 250 over the same games: the score -0.003 +- 0.004 (-0.9 SE), income -1.8 SE,
    extractors at 6 minutes the same, nothing past 2 SE.  So at 250 the remembered buildings make no difference that
    128 games show.
* **The memory** (the user, 03:15, of `rdnn-1`: "The fading memory is to be expected early on; most games end early and
  there's not much to remember"): a small state early isn't a fault.  The checks give its numbers as facts and act on
  neither.  And (the user, 2026-11-18 00:45): "No need to put recurrent weights under a KL guard".
* **The checks** (a line or two each):
  * to iteration 16: the KL an iteration 0.013 to 0.046 with the guard taking the scale from 0.15 to 0.03 (`rdnn-2`'s
    was 0.026 falling to 0.007-0.011 at a scale of 0.10-0.15); the state's mean size 0.33; 570-700 decisions a second.
  * 05:49, iteration 58 (1.00M decisions): the KL 0.005-0.009 an iteration since 20, the scale back up to 0.51; 706
    decisions a second over the last 50 iterations; the recurrent weights' step 0.008% of their mean, the state's mean
    size 0.20; BUILD 2.5-3.5% of actions and the area mex the same; the critics at idle 0.59-0.72, constructors
    0.41-0.60, economy 0.28-0.38, extractors 0.26-0.33, the rest 0; the card 7.7 GB, a collector 0.88 GB, WSL 28 GB
    available; no error.
  * 06:49, iteration 198 (3.39M decisions): 643 decisions a second over the last 50 iterations; the KL 0.006 with the
    scale whole (0.020 over 130-139); the recurrent weights' step 0.007%, the state's mean size 0.051; BUILD 3-5% of
    actions to iteration 159, 9% at 160-169 and 15% at 170-199, the area mex 3% and then 5%: the rise at the iteration
    of `rdnn-2`'s (161: BUILD 8-21% there, the area mex 4-13%), not looked into; the critics at result 0.52-0.61,
    economy 0.78, the three older leads 0.74-0.86, `production_lead` 0.47-0.54, score 0.56, map 0.2, combat 0.1,
    experience 0, much as `rdnn-2`'s at 170; the replays at 145 (4 games) with 1.2 extractors at 6 minutes (the
    script's 11.5) and no constructor made; the card 8.9 GB, a collector 1.22 GB, WSL 27 GB available; no error, the
    watchdog only watching.
* **No change in the middle of it** (mine, from the user's question about `rdnn-1`; this run exists to have none).  The
  hourly check (cron job `bb15e96e` in the session that started it, at about 49 minutes past) reports, and carries
  the run on with `results/resume-rdnn.sh` only if it has died and the cause is understood; anything else is put to
  the user.  A watchdog (`results/rdnn-watchdog.sh`) kills a trainer 25 minutes without a report and carries it on
  from its last checkpoint, 3 times at most, each said in `rdnn-3/watchdog.log`.  At its end: no resume, the cron job
  deleted, `touch /mnt/d/zk-learned/runs/rdnn-3/stop-pss`, a summary.
* **To watch**: the tests at 500 and on; the shares of BUILD and of the area mex; the replays' extractors and
  constructors; the KL an iteration and the lr scale; the state's size and `memory_step`; the share of other kinds
  from 500; the card (a collector's share grows with the games' age: `rdnn-1` reached 9.6 GB of 12.3 with games an
  hour old, and `gru-1` stalled at 11.3); `collect_rate`, `wait_s`.

## Earlier runs

Each is under `/mnt/d/zk-learned/runs/` unless said.  Since `full-4` a test is 128 1v1s against the script, the same games
each time, read as each game's change from iteration 0's (`results/paired_tests.py`); the score is `eval_fight_score`.
Read them with "A test compares only with tests on the same sim and script" above.

* **`rdnn-2`** (2026-11-18, 03:18 to 05:20; 4.7 GB): `rdnn-3` with the remembered buildings as tokens.  The first run with
  everything from iteration 0 and nothing changed in the middle (the user, of `rdnn-1`: "Could the bad/lackluster
  results be the results of applying our recent changes in the middle of the run? Could a fresh run at iteration 0
  give better results?").  Stopped at iteration 251 (4.29M decisions, `ppo/ckpt/it-000251.pt`) for `rdnn-3`.
  * Its test at 250 against its iteration 0: 1 win each, the score 0.145 to 0.163 (+0.019 +- 0.004, 4.9 SE), own value
    at the end 4,285 to 5,942 (+6.3), income +3.8, damage dealt +1.7, extractors at 3 minutes 2.3 to 3.0 (+4.9) and at
    6 2.1 to 2.6 (+3.4, the script's 10.8), games 80 s longer.
  * At 170: the KL 0.011 an iteration at 0.75 of the rate, the state's mean size 0.059, 723 decisions a second, the
    card 8.3 GB.  At the end: the KL 0.002-0.004, 430-500 decisions a second, the state 0.03-0.04.
  * To carry it on: the helpers naming `rdnn-2` again, no other trainer running, and the featurizer as it was before
    `rdnn-3`'s change (the three places above; never committed apart).
* **`rdnn-1`** (2026-11-17 21:43 to 2026-11-18 03:18; 13 GB).  The user: "Instead of a GRU, consider implementing the
  Recurrent Divisive Normalization Network (RDNN)", "a new 24-hour run with a higher learning rate (around 2e-3) and
  RDNN enabled", "pass in memory input as a vector".  `gru-1` as that ended, with the memory, the vector and the rates
  changed at once, then changed five times in the middle, each a restart with every game anew: at 146 whole stretches,
  the compiled loop and the recurrent weights at 5e-3; at 160 faster collection; at 219 featurizing through
  libzkpolicy; at 324 `production_lead` and the area mex; at 364 one extractor a spot and the sites' status.  Stopped
  at 526 (9.01M decisions).
  * Its tests: the score 0.149, 0.159, 0.165 at 0, 250, 500, with 1, 1 and 2 wins.  At 500 against 0: +0.016 +- 0.006
    (2.4 SE), extractors at 6 minutes 2.0 to 2.5 (the script's 11.1).  About where `gru-1` was by 750.
  * **The rates**: at the whole of 2e-3 the KL was 1.9 to 10.6 an iteration and the anchor head's entropy fell from
    0.85 to 0.03 in two iterations; from 0.15 of it under the guard, 0.08 falling to 0.006.  In the actor's own Adam
    (eps 1e-5) the recurrent weights stopped moving, their gradient being the weight's times about 1 / units: hence
    their own Adam with eps 1e-8.
  * **Bursts of KL** of 0.03-0.07 an iteration came 10 to 14 iterations after each restart and settled in a few.  In
    one the recurrent weights at 5e-3 moved it, which the guard doesn't slow; the user wants no guard on them (above).
  * **The memory's state shrank**: its mean size 0.08-0.13 with games of every age, under 0.01 twice (iterations
    308-346, 418-461), a restart or a new term bringing it back for a while.  The weights
    (`results/inspect_rdnn.py CHECKPOINTS`, with the run stopped): a rate's gain on itself (J over eta) from 1 to 0.91
    at 219 and 0.81 at 324, so what a unit holds fades over about 55 decisions unless it's driven.
  * **The area mex**, added to a policy 324 iterations old, was chosen less and less (0.1% of actions from 380).  From
    iteration 0 it's as common as BUILD.
  * **On the paper's own task** (`results/rdnn_check.py`: angle integration as I rebuilt it, 128 units, 1,000 steps,
    one seed): the RDNN at the paper's settings (5e-3, the gradient whole) reached -3.8 dB and was still improving
    (the paper's: -15.6 dB after 100 epochs); at 2e-3 with the gradient cut at 16 it learnt nothing (-0.4 dB) where a
    GRU reached -10.3 dB.  Not separated: the cut or the rate (arms at 5e-3 cut at 16, 2e-3 whole and a cut at 64
    would say).  The paper has no RL and nothing of this size.
  * **A segment** ends with its stretch, 20 to 60 decisions: further back needs the memory's inputs kept from the
    game's start (about a day's work, put to the user, not built).  Compiled, 5 segments of 60 take 24 ms, for 72
    uncompiled in one pass and 109 a step at a time.
  * **A collector's time** before the changes at 160 and 219 (py-spy): 57% in its forward passes, 12% featurizing, 6.5%
    counting the leads, 4% the sim.  Counting the leads at once and sharing a collection by each collector's speed took
    collecting from 555-713 decisions a second to 760-980 at the same age after a restart, and featurizing through
    libzkpolicy from 680-870 to 880-1,070: training is now as much the limit.  Not done: a third collector (the card),
    fusing the forward pass's kernels.
* **`gru-1`** (2026-11-17 07:53 to 20:54; 28 GB).  The user: an RL-only run with a GRU for memory as Pluto has, the parts
  the GRU makes needless out, and a small critic for each of the reward's terms.  From nothing, bootstrapped GRPO.
  Stopped at iteration 1269 (22.32M decisions) for the RDNN.  docs/model.md, "A GRU for memory".
  * Its tests, all against the richer script: the score 0.148, 0.150, 0.158, 0.167, 0.169 and 0.184 at iterations 0,
    250, 500, 750, 1000 and 1250, with 0, 0, 0, 1, 1 and 3 wins.  At 1250 against 0 over the 96 games they share:
    +0.036 +- 0.011 (3.4 SE), own value 4,596 to 6,417 (+6.7), extractors at 6 minutes 2.2 to 2.9 (+3.3, the script's
    11.6).  It learnt slowly to the end and never built more than 3 extractors by 6 minutes.
  * Within noise of its start at 250 (the KL an iteration down to 0.0005-0.001).  From 386 a seat's own objective and
    the five exploratory `made_*` terms (the user: "it's clear this training run isn't improving. Apply the changes
    sooner then restart the run"); the first test above iteration 0 was at 500.  From 1149 the three leads in
    `extractors`' place (the user), too late to read.  Every term but the `made_*` is zero-sum over alliances
    (checked to the bit on 48 script games).
  * **It stalled just past iteration 1100 with the card full**: 11.3 GB of 12.3, where Windows stops giving a process the
    card's own memory, a collector holding about 2.5 GB late in a run.  Hence two collectors of 48 games,
    `--gpu-share 0.65`, and the watchdog.
  * 1,250 decisions a second on the five training maps, 610-750 on the pool, 470-640 late in the run.  WSL's memory in
    use rose 1 GB an hour for a few hours, then held at about 18 GB.
  * **If the policy stops moving** (the user, 10:05): small random variation of the terms' weights and rarely-on
    exploratory terms (both built, as a seat's own objective), or MARA (arXiv 2510.20817; not built: it needs a KL
    penalty to a reference policy, which these runs haven't).  My rule then: two tests in a row within noise of the one
    before with the KL under 0.001, and the policy is made to move more (`--group-step 1024` first).
  * To carry it on: `results/resume-gru.sh "DATE TIME"`, with no other trainer running.
* **Self-play from `full-4`'s DAgger actor** (2026-11-16 14:17 to 2026-11-17 06:21; `full-4/ppo`).  The user, after
  Pluto's README: train against itself, a tenth of the games against the scripts, 1v1s first.  Unbootstrapped GRPO on
  seven terms, `--self-play 0.9`.  Stopped at iteration 275 (4.80M decisions) as not learning: test wins 10.9%, 10.2%,
  11.7%, 17.2%, 13.3%, 13.3% at 0 to 250; the last four less iteration 0, the score +0.027 +- 0.017 and extractors at 6
  minutes +0.55 +- 0.20 (4.8 to 5.6, the script's 9.2-9.5).  119 decisions a second.  Seen and not changed:
  `adv_abs_max` in the thousands in iterations where no group's copies differ in who won (the clip at 3 bounds it).
* **`full-4`** (2026-11-14 to 16): a bigger model on `army-8`, the encoder 192 wide with 64 latent tokens and the actor
  `WIDE` (27.0M).  The clone (held out: kind 0.971, anchor 0.910) and one DAgger round played as `full-3`'s: 9.4% of 64
  1v1s, extractors at 6 minutes 4.4 for the script's 9.9.  Bootstrapped GRPO with resignation for 120 iterations
  (`ppo-boot`): the critic learnt (explained variance 0.7-0.8), the policy barely moved, tests 15.6%, 10.9%, 14.8%.
  Then unbootstrapped on seven terms for 141 (`ppo-script`): it moved and the tests didn't rise.  The gradient checks
  found noise but for the seven terms mixed (03-training.md, "Signal or noise").  Its frames are deleted, the
  `manifest.json` kept so that the pipeline, run again, plans no `frames` stage (`full-3`'s too).
* **`full-3`** (2026-11-13 to 14): recorded `army-8`, the encoder stopped by the user at step 20,600 of 75,000 (levelled
  off), a clone of 10,000 steps and one DAgger round (10.9% of 64 1v1s), then GRPO with the policy's own economy: tests
  17.2%, 7.8%, 10.9% at iterations 50, 100, 150 (its last checkpoint is 141).  The memory guard killed it twice.
* **`grpo-trial`** (2026-11-13, at feature version 9): from `full-2`'s DAgger actor (9.4% of 64 1v1s), GRPO won 14.1%,
  15.6% and 14.1% at iterations 25, 50 and 75, and PPO 6.2%, 7.8% and 20.3%.  GRPO's orders swung from FIGHT to MOVE at
  72-79, since fixed (03-training.md, "GRPO from branched games").
* **`full-2`** (2026-11-11 to 13, at feature version 9, so nothing of it loads): the first whole run on every kind of
  game.  The clone won 9% of 64 mixed games and 17% after a DAgger round; PPO then got worse over 971 iterations
  (tests from 11% to 0-5%) as the KL to the clone and the cloning loss faded.  The user judged that PPO isn't working.
* **`full-1`, `spo-1` and before** (no longer on disk): 03-training.md, "What the earlier runs showed".

## What loads

* **`FEATURE_VERSION` is 10** since 2026-11-13.  Trained at 10: `full-3`'s and `full-4`'s encoders and actors, `gru-1`'s
  and the `rdnn` runs' actors (on `full-4`'s encoder).  All but `rdnn-3`'s were trained with the remembered buildings as
  tokens, which the featurizer no longer makes.  Nothing trained before 10 loads.
* **`policy_latent.ENCODER` still names `encoder-big`**, which doesn't load: every tool that loads an actor needs
  `--encoder` (`policy_train.py`: `--latent-encoder`), `/mnt/d/zk-learned/runs/full-4/encoder/vae-last.pt`.
* **Recorded games** are raw and still read: `clone/army-8` (data version 7, every kind of game).  But they're by the
  scripts before one extractor a spot, and their frames read every spot free.
* A run carried on with other reward terms keeps the critics of those it had, by name; the new ones start from nothing
  while the actor waits 10 iterations.

## My choices the user hasn't confirmed

Each is a constant or a flag, easy to change.

* **Nothing remembered** (2026-11-18): the ones under `rdnn-3` above, the site's status first.
* **`production_lead`**: its name; the rate over one step of the env (3 s) and the cap of 30 on each team's rate, not on
  the lead or an average; a player's rate in a team game; the 10 minutes' average and the weight 0.3, as the other
  leads'.
* **The area mex**: the reserved kind 15 and not a 25th; its middle one of a BUILD's places and not a free point; the
  radii the group head's 10 (60 to 2,250 elmos); a unit an action, so no "centre of selected units"; two units of an
  alliance don't go to one spot; a unit 300 elmos or more from its spot when someone else takes it goes to the next;
  a cloak's order doesn't end an area.  Between two extractors the unit stands for up to a step (3 s).
* **The RDNN** (docs/model.md has each): relu and not tanh; no noise on the state; eta 1 and the rates' spread at first;
  the recurrent weights' own Adam; the vector not LayerNormed and added in three places, the memory tokens gone; the
  actor's start at 0.15 of the schedule; the critics' rate unchanged; 1,024 units.
* **The recurrent runs**: the GRU's sizes (1,024, 4 memory tokens, 1 block, 2 plan layers); the critics two layers of
  256, detached from the memory; no script games in training and the lead rule on throughout; 96 games; a seat's
  objective at sigma 0.3, a quarter of games with an exploratory term at weight 1, the five classes and their caps (10,
  30, 10, 4, 8), the critics not told the draws; the leads as terms of their own, averaged over 10 minutes, their
  scales (20 extractors or idle fighters a player, 5 constructors) and weights 0.3.
* **The kinds of game** (2026-11-10 and 11): the scenario kinds' weights (40% 1v1s, 20% each of the rest) and their
  chicken numbers; a scenario team's 480 units at most; chicken difficulty drawn evenly from Very Easy, Easy and Normal;
  chicken games drawn on team maps too; the reward zero-sum over alliances; the speed term unfaded by beta; aircraft
  controllable by the policy; the wing's numbers in `sim/scripts/src/army.cpp`; Zero-K's "special" mark counting as no
  kind of game, which makes Requiem Outpost a chicken map; the share of other kinds split by `--kinds`' weights.

## Environment

* **Zero-K**: portable install at `C:/Users/Thoma/Downloads/zero-k-portable`; engine 2026.07.04; Zero-K v1.14.10.3;
  ControllerAI 0.3.
* **Read-only sources**: `../RecoilEngine` (CircuitAI under `AI/Skirmish/CircuitAI`), `../Zero-K`, `../ControllerAI`.
* **Windows Python 3.13** runs the bot (`pip install requests websockets orjson numpy numba`).  Without numba, featurize runs in
  plain Python, about 50x slower, and says so on import.
* **WSL (Ubuntu)** runs the sim and training: `~/zkml/bin/python` (PyTorch, RTX 4070 of 12.3 GB, pytest-xdist), with
  pytest in `~/zksim/pt`.  So `PYTHONPATH=$HOME/zksim/pt:bots`.  WSL has 40 GB, Windows 48.
* **The sim's libraries**: `cmake -S sim -B ~/zksim/build && cmake --build ~/zksim/build -j8`.  That builds `libzksim.so`,
  `libzkscripts.so`, `libzkrecord.so` and `libzkpolicy.so` (`ZKSIM_LIB` and `ZKPOLICY_LIB` override).  The recorder needs
  `libzstd-dev`.  Rebuild after any change under `sim/`.  `bots/zkbot/data/policy_table.bin` is written by
  `python sim/py/zkpolicy.py table`.
* **`sim/data/rules.json`** is made by `tools/sim_rules.py`, which needs `lupa`.  It's only in `~/zksim/agent-maps-lupa`
  (add it to `PYTHONPATH`), not in `~/zkml`.
* **`sim/data/maps`** is gitignored: after a pull, `python tools/sim_data.py maps`.
* **Data on D:**, outside git, under `/mnt/d/zk-learned` (81 GB free on 2026-11-18):

  | Path | What |
  |---|---|
  | `maps/` | the map pool: 204 real (18 of them Violet Rampart) and 200 made-up maps, 31 held out, each with its `modes`.  It loads through the map cache in 27 s, to 1.4 GB of a process's own memory (01-simulator.md, "The map cache") |
  | `clone/army-8` | 175 GB: about 8,800 script games of every kind, 100M decisions, recorded 2026-11-13 for `full-3` |
  | `runs/` | the runs above (`rdnn-3`, `rdnn-2`, `rdnn-1`, `gru-1`, `full-4`, `full-3`, `full-2`, `grpo-trial`); the rest is older runs that don't load, and scratch |
  | `parity/` | `tools/policy_parity.py`'s results |
  | `dashboard-cache/`, `dashboard-demo/` | the dashboard's map pictures and icons, made once; a made-up run directory for working on the dashboard |

## Layout

* **`bots/zkbot/`**: the engine side.
  * `main.py` connects to ControllerAI and drives `host.py`, which keeps what a raw frame is read from and runs
    `policy/runtime.py` once a decision.
  * `policy/`: the raw frame schema (`rawframe.py`, which also generates `sim/include/zksim/rawframe.hpp` via
    `tools/policy_schema.py`), the map bundle, type tables, featurizer (`featurize.py`, `featurize_kernels.py`), actions and
    group orders.
  * Unit data: `units.py`, `catalog.py`, `data/*.json`, rebuilt by `tools/extract_weapons.py`, `extract_abilities.py` and
    `dump_defs.py`.  Terrain: `terrain.py`, `hazards.py`, `mapdata.py`.
* **`sim/`**: the C++ simulator (`src/`), the economy and army scripts (`scripts/`), the recorder (`record/`), the type
  table, featurizer and actions in C++ (`policy/`: libzkpolicy), their ctypes bindings (`py/`, and
  `tools/policy_scripts_cpp.py` for the scripts), data (`data/`) and tests (`tests/`).
* **`tools/`**:
  * the sim's data and checks against the engine: `sim_data.py`, `sim_rules.py`, `sim_engine.py`, `sim_battery.py`,
    `sim_batch_bench.py`, `map_gen.py`, `map_modes.py`, `map_rampart.py`;
  * the sim as a game: `policy_sim_env.py`, `policy_opening.py`, `policy_variants.py`, and the scripts' tables (`policy_eco.py`,
    `policy_mix.py`, `policy_script.py`);
  * frames and the encoder: `frame_world.py`, `frame_data.py`, `frame_model.py`, `frame_vae.py`;
  * the policy: `policy_model.py`, `policy_latent.py`, `policy_clone_data.py`, `policy_clone_train.py`, `policy_clone_eval.py`,
    `policy_train.py`, `policy_pipeline.py`, `policy_golden.py`, `policy_parity.py`;
  * the trainer's play: `policy_graphs.py` (each round as a CUDA graph), `policy_collectors.py` (collector processes);
  * the run's dashboard: `policy_dashboard.py` (server), `dashboard/` (page, replay viewer), `policy_replay.py` (test games
    kept for it).
* **`results/`** (never committed; on this machine only): the runs' launch files and helpers.  For the run going:
  `start-rdnn.sh HOURS` (from nothing, with its guard, memory log, watchdog and dashboard), `launch-rdnn.sh` (the
  trainer's flags, each a variable with a default), `rdnn-status.sh` and `rdnn_terms.py N` (no torch: safe beside it),
  `stop-rdnn.sh`, `resume-rdnn.sh` (reads the run's `end`), `kill-rdnn.sh`, `rdnn-alive.sh`, `rdnn-dump.sh` (py-spy, as
  root), `check-rdnn.sh NAME N` (a copy of the last checkpoint carried on N iterations by the launch file as it is),
  `paired_tests.py`, `compare_tests.py`, `replay_summary.py`, `inspect_rdnn.py`, `rdnn_check.py`, `profile-run.sh`,
  `mem-guard.sh`, and `run-tests.sh` (the suite; it refuses beside a trainer).  The GRU run's are the same under `gru`
  names.  `edit_*.py` there are edits that made the code, as scripts.
* **Tests**: `tests/policy/`, `tests/test_*.py`, `sim/tests/`, run as `pytest.ini` says.  None needs the engine.
* **Engine games**: `zk-startup.ps1`, `zk-maps.ps1`, `maps.md`, `README.md`.

## Running

In WSL, from `/mnt/c/Users/Thoma/Documents/sources/zk-ArenaAI`, with `PYTHONPATH=$HOME/zksim/pt:bots`:

    ~/zkml/bin/python -m pytest -q                     # about a minute, on 8 processes (pytest.ini); all pass
    ~/zkml/bin/python -m pytest -q -n 0 tests/policy/test_train.py     # one process: for a debugger, or to time a test
    bash results/start-rdnn.sh 24                      # a run from nothing as rdnn-3 is; first give results/*rdnn* a new directory's name
    bash results/rdnn-status.sh                        # where the run going stands; no torch
    bash results/stop-rdnn.sh; bash results/resume-rdnn.sh     # stop it and wait; carry it on to its end (every game anew)
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

**The tests** (850 pass, 3 of them need CUDA and are skipped without it): `pytest.ini` runs them on 8 processes, a file's
tests in one process so that what they share is made once, and `tests/policy/conftest.py` keeps each process's maths to
one thread.  About 60 s, most of it each process importing torch and the tools; `--durations=20` shows what's slow.  The
user wants the suite lean: no redundant, sure-to-pass or slow tests.

In the engine (Windows), recording raw frames from a game: `$env:ZKBOT_FLAGS='rawframe'`, then
`.\zk-startup.ps1 ControllerAI CircuitAI:Normal -Headless -Map 'TitanDuel 2.2' -NoDownload -Client 'python bots\zkbot\main.py {team}'`.
README.md has the rules for engine games: headless at max speed, windowed only when the user asks.

## Next

0. **`rdnn-3`** is going till 2026-11-19 05:23, a check of it every hour scheduled in the session that started it: read
   its tests (500 and on) against its own iteration 0's, report to the user, change nothing in the middle unless the
   user says.  Open, put to the user: a site's status still says "enemy" of a spot from an extractor only remembered.
1. **If it learns no better than `gru-1`**: the GRU with the memory told as a vector and the same rates, which separates
   the vector and the rates from the RDNN; and the missing arms of the paper's task ("Earlier runs", `rdnn-1`).
2. **To set a run beside the older ones**, test their checkpoints anew against the script as it is (`gru-1` and
   `rdnn-1` keep one every 20 iterations, none of iteration 0); it needs the card, so no run going.
3. **Why the clone stops expanding** (mine, put to the user and not answered).  Held out, `full-4`'s clone gets the
   kind right for 83% of the script's builds (97-100% for fight, move, produce, assist), and a whole build (type,
   place, offset) costs 7.5 nats against 0.3-2.4 for the rest (`clone/log.jsonl`): split that by type, place and
   offset, and by what's built.  A clone would also meet real site statuses only in play (recordings read every spot
   free).
4. **The bot's inference in C++** (02-policy.md, "Deployment"; the nets in onnxruntime, nothing to install, the user).
   Built: libzkpolicy has the type table, the featurizer and the actions, bit for bit with the Python over 49,451
   frames of the four kinds, and the RL trainer featurizes through it (`--featurize-threads`).  Next: cloning's samples
   and frame data and the orders' encoding and decoding through it, then the map bundle.  The bits match on glibc's
   libm only, so the bot on Windows will need the same.
5. **The engine hand-off**: the rest of item 4; decoded orders sent as commands; the opening and the economy script in
   the engine; the checks listed in 02-policy.md, "Not yet checked live".
6. **The user's open questions** (`docs/model.md`, "Open questions").

## Traps

* **Nothing with torch beside a running stage**: no pytest, smoke run or measurement, on the CPU or the card.  The memory
  guard killed `full-3` and `full-4` that way, and three torch processes on the card left no room for a fourth's CUDA
  context.  A change is made and tested in a tree copy (its sim tests run without torch), then applied with the run
  stopped.  A check run must end inside the watchdog's 25 minutes, or it takes that trainer for the run's.
* **A launch file that's running**: bash reads a script as it goes, so an edit to `results/launch-*.sh` while its run is
  going is run from wherever bash had got to when the trainer ends; any script overwritten while a shell was still in
  it ran the new file's lines from the old one's place.  Stop the run (`results/stop-rdnn.sh` waits for the launch file
  too), then edit.  The Python is safe to edit beside a run only in that its modules are loaded already.
* **A restart starts every game anew**, so a run's games are young again for an hour and the KL bursts 10 to 14
  iterations after.  If libzksim changed, the map cache is made again first (about 2 minutes before the trainer's first
  line).  A copy carried on needs `--decisions` past the checkpoint's, or it ends at once.
* **Memory**: check `wsl free -g` before a launch.  Never run two jobs that load the map pool at once.  Every GPU trainer caps
  itself at a share of the card (`--gpu-memory`, `--gpu-share`): past the card, the Windows driver spills into system
  RAM instead of failing, and at about 11.3 GB of the 12.3 a process stalls.  With collectors, the learner's cap leaves
  each `--collector-gb` plus about 0.4 GB of CUDA's own.  The learner's first training step of the windowed actor on the
  CPU grows to 23 GB (dense attention with a mask); on the card it doesn't.
* **Leftover build directories**: `~/zksim/build` is the one in use.  The `build-*` and `agent*` directories beside it are
  leftovers of comparisons and of agents' tree copies, safe to delete, except `agent-maps-lupa` (above).  Another agent
  changes the sim and the scripts at times (the user): a long run plays the build it starts with.
* **A site's status** is free in the sim's own records (`batch.frame`, a recording): `SimEnv.frame` fills it.  And the sim
  builds an extractor wherever it's told and pays each one on a spot: whoever places one keeps off a held spot itself.
* **The map cache** (`~/zksim/map-cache`, 6.1 GB; `ZKSIM_MAP_CACHE` moves it, empty turns it off): `load_pool` makes a
  map's file again when the map file or libzksim changes.  A change to what's kept, or to how `zksim.Map` makes what it gives
  the sim, needs `zksim.MAP_CACHE_VERSION` raised, or the old files are read.  The tests keep theirs in pytest's temporary
  directory (`tests/policy/conftest.py`).
* **Forking with torch**: a child forked after the parent ran a parallel CPU op hangs at its first one (OpenMP), so collectors
  set one thread first.  A parent that has started CUDA can't fork children that use it: the trainer forks its collectors
  before the learner touches the GPU, and the tests of them run the trainer in a fresh process.
* **Pinned memory in WSL**: a collector's 20 MB `cuMemHostAlloc` failed as out of memory with 7 GB of the card free.  The
  graphs' inputs aren't pinned.
* **`pkill -f` / `pgrep -f` from `wsl bash -c '...'`** match that shell too, whose command line holds the pattern.  Write the
  pattern as `[p]olicy_train`.  A `kill -STOP` that way stopped the shell itself.
* **A background process started by `wsl bash -lc 'nohup X &'`** dies when that call returns.  `setsid nohup X > LOG 2>&1 <
  /dev/null &` outlives it.
* **A `$` in `wsl bash -c '...'`** is taken by the shell wsl starts first, where the variable isn't set: `F=x; grep y $F`
  reads stdin, and a `$(...)` runs once, before the loop it's in.  Put it in a script file.
* **A trainer started through `sudo -u` from a root shell** died at its first training step with "CUDA driver error:
  unknown error": that environment hasn't `/usr/lib/wsl/lib` on its path.  Start it as the user, and attach py-spy as
  root from another call.
* **`NAME` in WSL** is the PC's name, set for every shell: a launch script that reads `${NAME:-...}` gets it.  The launch
  files read `RUN_NAME`.
* **The dashboard's port**: a standalone `policy_dashboard.py` left on 8765 makes the pipeline run without its own (it says so
  and carries on).  Stop it before a launch, or give the pipeline another `--port`.
* **The pipeline plans a `frames` stage** whenever a run's `frames/manifest.json` is missing: keep it when deleting frames.
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
