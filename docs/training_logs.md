# Training logs

Every training run of the learned policy, newest first: what it was, its tests, what was seen in it, and each hourly
check of the run going.  This is the dated record, and anything dated is added here.  `AGENTS.md` has only where
things stand; `docs/learned-policy/03-training.md` has how training works and what the first runs showed; and
`AGENTS.md` as it was at `f8a156c` in git history has the longer record of every run to 2026-11-18 (each change made in
the middle of one, each hourly check of `gru-1`, `rdnn-1` and `rdnn-2`, the timings).

Each run is under `/mnt/d/zk-learned/runs/` unless said.

## Reading a test

Since `full-4` a test is 128 1v1s against the script, the same games each time, read as each game's change from
iteration 0's (`results/paired_tests.py`); the score is `eval_fight_score`.

* **A test compares only with tests on the same sim and script.**  The sim changed on 2026-11-12 (projectiles, the
  tactical AI, no side first), 2026-11-13 (the end of a game, the script's defences) and 2026-11-16 (water, sonar).  And
  on 2026-11-18 the script and the opening stopped standing several extractors on one spot, each of which the sim pays:
  the scripts' own games went from 9.6 extractors on 8.7 spots at 6 minutes and 23.8 on 16.0 at 12, for 30.6 and 75.6
  metal a second, to 8.5 on 8.5 and 12.6 on 12.6, for 27.0 and 43.8.  So every test, eval and recording before
  `rdnn-1`'s iteration 364 was against a richer script.  Against the script as it is: `rdnn-1`'s test at 500, and all of
  `rdnn-2`'s, `rdnn-3`'s, `rdnn-4`'s and `rdnn-5`'s.

## `code-2`: `code-1` again with 6 collectors and the faster collector (2026-11-19 18:17 to 2026-11-20 00:17; going)

The user, 2026-11-19, as the collector was being made faster ("A faster collector", below): "When that is all done, stop
the current training run and start a new one, from iteration 0, with 4 collectors for a 6-hour trial run.  parameters
should include `--economy-start 1 --economy-end 1` as usual"; then, the check with 4 in: "OK, since 4 collectors was
fine memory-wise try 6-8 collectors.  If all looks fine, then start the 6-hour training run."; and with 6 checked and
8 begun: "Actually, skip 8 collectors.  If 6 hours looks fine proceed with the training run.  Whilst that is ongoing,
commit+push your changes".

**What it is**: `code-1`'s flags (`results/launch-code.sh`: the code actor on `vae-3`, from nothing, against itself,
`--economy-start 1 --economy-end 1`, the spread's least from 0.1 to 0.03 by the run's clock) with `--collectors 6`, 16
of the 96 games each, on the collector as it is now.  Everything under `/mnt/d/zk-learned/runs/code-2`; started by
`results/start-code.sh 6` at 18:17.  Mine in it:

* **6 of the 6 to 8**: 96 games go into 6 collectors' groups of 8 and not into 8's, which would have needed 128
  games, a second change; the check of 8 on 128 was killed as it began, on the user's word.
* **`--collector-gb 0.8 --gpu-share 0.76`**: a collector's process costs the card 0.4 GB before its cap, so six caps of
  2.0 don't fit; a collector of 16 games held 0.25 GB in the check and the learner 1.35, which this leaves 2.0.
* **A checkpoint every 50 iterations**, not 20: one is 0.39 GB, D: had 44 GB free, and at 2.5 times `code-1`'s
  iterations an hour every 20 would not fit.  The tests are every 250, so each tested iteration has one.

### Its tests

* **Iteration 0** (18:17; the test took 16 s, its games over six collectors): no win of 128, the score 0.155 ± 0.003, an
  income share of 0.287, 2.2 extractors at 6 minutes for the script's 10.3.  An untrained actor, as `code-1`'s 0.163
  was: the two differ by their draws.

### The hourly checks

A cron job in the session that started it (at about 45 minutes past) runs `results/code-status.sh` and
`results/code_terms.py 5`, and adds a line here.

* **18:19, iteration 9**: running, 10 trainer processes, the guard, the dashboard, the memory log and the watchdog up.
  2,650 decisions a second overall so far (collect 3,350, train 3,780).  1.8 commands a decision and 0.69 units told;
  the spread 1, its least 0.1.  The learner 1.35 GB of the card and a collector 0.23 of its 0.8, the card 4.9 GB in
  all and 57% busy; WSL has 26 GB to spare, the CPUs 58% idle.

## A faster collector (2026-11-19, between `code-1` and `code-2`)

The user, of `code-1` at 1,250 decisions a second with its learner waiting 10 s of every 14 (`code-1`, 17:15, has the
profile): "Is there any way to get the collection rate to match the training rate (~5000 decisions/sec)?"  I put three
things to them, more collectors (1), fewer and bigger passes (2), and the Python after each step of the sim with what a
collector sends back (3); they said "Proceed with 2 and 3", then asked for the collectors too (`code-2`).

**What changed**, all of it the collector's own work, none of it what a game or a reward is:

* **Every seat's reward terms at once** (`SimEnv.seats_terms`, which `GroupCollector` asks after every step of the
  sim where it asked `reward_terms` a seat at a time), each count of steps' discounts made once.
* **The finished units' bookkeeping** (`SimEnv._diversity`): a team with no unit newly finished sorts nothing and
  keeps what it had, where every team's units were sorted and searched each decision; it and the leads read the
  teams' units once (`_own`); and the alliances' sums skip the slots with no team (`_by_alliance`, `_others`).
* **A code's step keeps no frame** (`Step.frame` is None): the update reads its latents, and a round's frames were
  most of what a collector pickled back.
* **A code's pass takes to the card only what the frame encoder reads** (`policy_model.collate(frame_only=True)`,
  `Graphs.frame_only`).  A batch's hidden units pad to their cap, 256 tokens a row, for a critic that sees through the
  fog, and its action tables are a row for every controllable unit: at a small frame they were most of the bytes a
  pass put together and copied, and the code actor's pass reads none of them.  Not asked for: found in the profile
  of the first version, below.
* **Wider passes for a code** (`policy_graphs.CODE_ROWS`, `MERGE_TOKENS`, `CODE_KEEP`; `Graphs.passes`): up to 128
  rows, and a level's few go up into the next level's pass where that saves a pass for no more than 4,096 unit tokens
  of padding.  The numbers are mine and unmeasured apart: with the rest of the first version it changed nothing I
  could measure.

**That the numbers are the same**: `results/faster/env_ab.py` plays 3,000 steps of 16 games (1v1s, team games and
chicken games on the training maps, games ending at a short limit and starting anew, a diversity bonus that pays) in
the tree before and the tree after: every step's terms for every seat, leads, bonus, counts of what was made, deltas
and what was paid are the same to the bit, as are `seats_terms` and `reward_terms` seat by seat.  The tests say the
last in a 1v1 with the leads, a team game and a free-for-all with an alliance out
(`tests/policy/test_sim_env.py`, `_every_seat`); two GRPO tests that stood their own numbers in for `reward_terms`
now stand them in for `seats_terms`.  The suite passes (859).

**What it's worth**, by `code-1`'s last checkpoint (iteration 252) carried on with every game anew
(`results/faster/check-code.sh`), each line the mean of iterations 255 to 261 or 263; two such runs play different
games within a few iterations, so a few percent between them is nothing:

| | Decisions a second overall | collecting | A collector's seconds an iteration |
|---|---|---|---|
| the collector as it was, 2 collectors | 1,480 | 1,660 | 10.4 |
| the first version, 2 (all but the frame-only pass) | 1,640 | 1,670 | 10.6 |
| as it is now, 2 | 1,920 | 2,010 | 8.6 |
| as it is now, 4 (24 games each) | 2,640 | 2,700 | 6.7 |
| as it is now, 6 (16 games each) | 3,060 | 3,160 | 5.9 |

* *The first version collected no faster.*  It was 17% ahead for three iterations and level after
  (1,685 decisions a collector second for 1,689), its passes a third fewer (386 an iteration for 530); what it gained
  overall was the learner's wait for a smaller pickle.  A profile of one of its collectors
  (`results/faster/spy-one.sh`) had the terms at 2% for 7% and the Python after a step at 11% for 17%, but 13% in
  loading a pass's batch and 7% in putting it together: the bytes, not the passes.  With only the encoder's fields
  sent, a collector second made 2,048 decisions (+21%).
* *With 6 collectors* a collector held 0.25 GB of the card (0.35 with 4) and the learner 1.35, the card 4.1 GB in all
  and 57% busy, WSL 12 GB used of 39, the CPUs 56% idle.  Collecting is then 5.9 s an iteration for training's 4.3.
* *In the env alone* (`env_ab.py`, 48 1v1s): a step and its terms 6.1 ms before and 4.2 after.

**Mine, and wrong on the way**: I told the user changes 2 and 3 might give 1.5 times a collector; they gave 1.2, most
of it from what I hadn't proposed.  An edit script meant for the tree copy ran once in the repo, beside `code-1`
(`tools/policy_sim_env.py`; put back at once from the copy's base, its hash HEAD's; the run's modules were loaded
already).  And the check of 8 collectors had started when the user said to skip it, and was killed.

**The tools** are under `results/faster/` (never committed): the scripts that made each change in a tree
(`fast_*.py`), the copy and its tests with torch stubbed out beside a run (`tree-fast.sh`, `fast-test.sh`), the env
before and after (`env_ab.sh`), the checks (`check-code.sh TREE NAME N`, `check_compare.py`, `check_work.py`) and
the profile (`spy-one.sh` as root, `profile_report.py`).

## `code-1`: a 6-hour trial of the actor that chooses a code (2026-11-19 16:38 to 17:46, stopped by the user at iteration 252)

The user, 2026-11-19 at about 16:20, with `vae-3` taken, the spread's least on a schedule and the suite passing: "OK,
commit+push then start a 6-hour trial run."  Committed and pushed as `d6488c9`; then `results/start-code.sh 6`,
everything under `/mnt/d/zk-learned/runs/code-1`.

**What it is**: `rdnn-5`'s flags (`results/launch-code.sh`) with `--size code` on `vae-3`: from nothing, against itself,
the policy running its own economy, bootstrapped GRPO with the memory's critics, 2 collectors of 48 games on the pool,
a checkpoint every 20 iterations and a test every 250.  Its sim is libzksim ABI 32 (several build one thing, a reclaim
is an area's) with a build that sticks, neither of which a run has had.  What it's about is in "After `rdnn-5`",
below.  Its spread's least falls from 0.1 at 16:38 to 0.03 at 22:38, by the clock (`--spread-floor 0:0.1 6:0.03
--spread-floor-began`, from the run's `spread-floor` and `begin` files).

**Started twice.**  First at 16:28, with the spread's least by iteration, `0:0.1 2100:0.03`: six hours at 350
iterations an hour, my estimate from the smoke's 2,000 decisions a second.  Its first 15 iterations showed the
estimate wrong: 1,850 to 1,970 decisions a second in iterations 2 and 3, then 1,040 to 1,290 by 8 to 12 and falling,
as its games grew old.  `rdnn-5` went the same way (1,040 to 1,130 in its iterations 2 and 3, 890 at 8 to 20, 600 to
670 at 40 to 100, 520 over its seven hours).  At that, I took six hours for 800 to 1,100 iterations and the least to
end near 0.07, not the 0.03 the user asked for.  (It has held at 1,200 to 1,300 since iteration 20, which is 1,500
iterations in six hours and a least of 0.05 at the end: the second estimate was out too, the other way.)  So I stopped it at iteration 15 (16:33), deleted it, and made the
schedule the run's clock's: `--spread-floor`'s points are hours since `--spread-floor-began` when that's given
(`Learner.set_floor`; `start-code.sh` writes the run's `begin`, `launch-code.sh` reads it, so a run carried on is where
its hours put it).  The code actor's two tests and the suite (859) pass with it, and a smoke told it began three hours
ago read a least of 0.065.  Started again at 16:38.  Mine, all of it; not committed: `--spread-floor-began` and
`--checkpoint-every` (tools/policy_train.py, the test), added after `d6488c9`.

**The rate isn't four times `rdnn-5`'s**, as I had written here, in 05 and to the user from the smokes: that set a
smoke's first three iterations beside `rdnn-5`'s seven hours.  At the same iterations it's 1.8 times at 2 and 3, 1.3
times at 8 to 12, and 1.7 to 2 times from 20 to 136 (1,190 to 1,300 decisions a second overall for 610 to 760).  The
update is quick (3.6 s an iteration); collecting is what takes the time (12 to 13 s by the two collectors).

### Its tests

* **Iteration 0** (16:38; the test took 34 s): 1 win of 128, the score 0.163 ± 0.006, an income share of 0.29, 2.6
  extractors at 6 minutes for the script's 10.3.  Beside `rdnn-5`'s iteration 0 in the same games, on the sim before
  ABI 32: the score 0.144 there (+0.019 ± 0.004), 3.3 extractors at 3 minutes for 2.3.  It's where codes drawn at
  random stood in the check by play (0.165).
* **Iteration 250** (17:46; 46 s): 1 win of 128, the score 0.164 ± 0.006, +0.001 ± 0.003 on iteration 0's in the same
  games (+0.3 SE), an income share of 0.286, 2.2 extractors at 6 minutes for the script's 10.4 (-0.31 ± 0.16 on its
  own).  Nothing learnt by the rule.  `rdnn-5` had 0.160 at 250, from 0.144.

### The hourly checks

A cron job in the session that started it (at about 54 minutes past) runs `results/code-status.sh` and
`results/code_terms.py 5`, and adds a line here.

* **16:40, iteration 5**: running, 6 trainer processes, the guard, the dashboard, the memory log and the watchdog up.
  1,520 decisions a second overall so far (collect 1,730, train 4,180).  1.8 commands a decision and 0.59 units told;
  the spread 1, its least 0.1.  The learner 1.34 GB of the card and a collector 0.37 of its 2.0; WSL has 23 GB free.
* **17:15, iteration 136, and where a collector's time goes** (the user: "The training run is now restricted by the
  collection rate at ~1250 decisions/sec.  Is there any way to get the collection rate to match the training rate
  (~5000 decisions/sec)?").  1,230 decisions a second overall, collecting at 1,370 and training at 4,620: the learner
  trains an iteration in 3.6 s and waits 10.  The spread 0.996 (its least number 0.977), the floor 0.093, the mean's
  size 0.019, the KL 0.001, the rate's scale 1.0; 1.9 commands a decision and 0.78 units told.
  * *The machine is mostly idle*: 16 logical CPUs (8 cores) 75 to 80% idle, the card 33% busy, 28 GB of WSL's memory
    free.  A collector is one Python thread at 100% with helpers (160% of a CPU), 2.7 GB of its own memory and 0.77 GB
    of the card.
  * *A collector's main thread, 30 s of py-spy* (`wsl -u root`, 50 samples a second; no torch):

    | Share | What |
    |---|---|
    | 32% | the policy's pass (`Graphs.forward`): 14% the replay and reading it back, 10% loading the pass's frames into the graph's tensors, 6% putting them together |
    | 30% | `SimEnv.step`: 8% the C++ sim, 17% what's done in Python after it (the leads 4%, diversity 2%, the rest), 6% the scripts |
    | 9% | the frames: `SimEnv.frame` 3%, featurizing 6% |
    | 8% | what the code says, and `selections.decode` (3%) |
    | 7% | the round's own bookkeeping (`collect_tally`) |
    | 5% | pickling the segments back to the learner |

    A pass is small: about 640 of them an iteration between the two collectors, 25 rows each of the 96 a pass could
    hold.  Its cost is the pass's, not the card's.
* **17:46, iteration 252: stopped** (`results/stop-code.sh`, its watchdog and memory log first), on the user's word,
  once its test at 250 was in: for the collector's tests and checks, which need the card, and `code-2`.  At the stop:
  1,220 decisions a second over its last 50 iterations; the spread 0.995, its least number 0.978, the floor 0.087, the
  mean's size 0.021; 1.7 commands a decision and 0.74 units told; the KL 0.001; the critics' explained variance 0.57
  for the result and 0.8 to 0.86 for the economy and the leads; 10% of ended seats in draws, the lead rule on.  No
  traceback, fallback or split in its 68 minutes.  Its last checkpoint is `ppo/ckpt/it-000252.pt`.

## After `rdnn-5`: orders as commands, an autoencoder of them, and an actor that chooses its code (2026-11-19; before `code-1`)

**Built and checked by play; no run has trained by any of it.**  The user's answer to what `rdnn-5`'s checks left open
(below, "The checks after it": the fighters waiting for their turn, a BUILD's blind choice), over that afternoon:

* "Starcraft's Pluto includes player-like unit selection before issuing group commands.  I think this is feasible and
  likely to be more efficient."  And of the orders that take one unit: "I think all the single-unit orders can be made
  to work with multiple units.  OK, write it up."
* "Proceed with the BUILD change.  Additionally, make RECLAIM an area reclaim order (default to a small radius to
  ensure compatibility).  The circle select could have a 'portion' parameter, to select say QUARTER, HALF or FULL.
  The orders autoencoder shouldn't have access to the frame's latent tokens (risk of representation collapse).  Rather
  than selecting or positional orders based on region, consider a spatial encoding or finer subregions.  Go straight
  to the autoencoder and remove unit cross-attention from the policy."
* While it was being built: "The training data for the autoencoder should also include randomly/synthetically
  generated actions to ensure coverage of cases/orders not seen in the script."

What each is and how it works: docs/learned-policy/06-select-then-command.md (the commands, the sim's two changes) and
05-order-autoencoder.md (the autoencoder, its data, the actor).  Here, what was measured.

### The scripts through commands (before any autoencoder)

`results/checks/language_check.py`, 128 1v1s, seed 7, `rdnn-5`'s flags with a build that sticks: team 0 a seat the
scripts only advise, what's carried out being their advice said as commands and read back; team 1 the scripts as they
are.

| | The advice as it is | As commands and back |
|---|---|---|
| team 0 won, lost, drew | 67, 57, 4 | 62, 64, 2 |
| its games' result (a draw half) | 0.539 ± 0.044 | 0.492 ± 0.044 |

Level within what 128 games show (-0.047 ± 0.062), not shown the same.  Of the 1.82M units the scripts told something
at a look, 71% were told the same by the commands; a quarter of the commands were past the 16 a decision, all in the 3%
of looks with 20 or more; 10% of the orders couldn't be said (4.6% to units the side's frame doesn't give it to command,
5.4% whose own command picked another or wasn't obeyed: not looked into).  A look is one command 28% of the time and 8
or fewer 90%.  Two faults of mine the check caught on the way: every MOVE and FIGHT read back at the map's corner
(team 0 then won 9 of 128), and a wide circle claiming units that had an order of their own that look.

### The recording and the autoencoder

* **`/mnt/d/zk-learned/orders/commands-1`** (`results/order-data.sh`; 19 minutes on 8 processes, 68 MB): 1,500,022
  decisions of the scripts' games played in commands, every kind of game on the pool's training maps, 5,564,234
  commands, 3.7 a decision.  Counted in one of its 8 parts: 29% of decisions are one command, 77% four or fewer, 3%
  the full 16; of the commands, FIGHT 56%, ASSIST 14%, PRODUCE 9%, BUILD 8%, MOVE 6%, RECLAIM 4%; 62% select one unit;
  82 sets of roles, ten of them 96% of the commands; 104 types.  In these games, with their bigger armies, a third of
  the commands were past the 16 and 21% of the scripts' orders were to units past the 512 a side commands.
* **`/mnt/d/zk-learned/orders/vae-1`** (`results/order-vae.sh`; 16,000 steps in 26 minutes, 3.3M parameters), three in
  ten of its training decisions not the scripts' own (half wholly made up, half a script's with three in ten of its
  commands swapped).  From a code's mean, held out: the scripts' decisions come back whole 77% of the time (kind 97%,
  roles 88%, type 87%, a place within 15 elmos at the median); a made-up decision of a few commands keeps its count,
  kinds, radii and places (25 elmos) but its set of roles 18% of the time and its type 39%; sixteen made-up commands
  don't fit.  05, "What it gives back", has the tables.
* **Two things in it that a run would meet** (05, the last two tables there):
  * a code is 174 nats from the unit Gaussian, and one that's off by a spread of 1 says another decision (the count
    right 18%, the kind 56%, places 1,300 elmos off); at 0.3 the count and kinds mostly hold (78%, 83%) and places are
    450 elmos off; at 0.1 places are 150 off;
  * zero says nothing: codes drawn round zero at a spread of 1 say nothing 72% of the time and 0.46 commands a
    decision; at 0.3, nothing 99% of the time.

### The scripts through the code, and codes drawn at random

The same 128 games (`language_check.py ... coded` and `prior`), with `vae-1`:

| | Team 0 won, lost, drew | Its games' result | The tests' score |
|---|---|---|---|
| The scripts' commands to their code's mean and back | 32, 89, 7 | 0.277 ± 0.044 | 0.361 ± 0.025 |
| A code drawn from the unit Gaussian at each look | 1, 126, 1 | 0.012 ± 0.044 | 0.165 ± 0.007 |

* **The scripts lose through the code what they kept through commands** (62 wins): 29% of the units they told were told
  the same by what the code said, for 71%.  By a decision's length, held out: with 1 to 4 commands (76% of decisions)
  97 to 99% come back right in everything, a place within 8 elmos; with 5 to 8, 18%; with 9 or more, 1 to 3%.  The code
  holds four commands, and the scripts' armies are told a unit a command.
* **Random codes hardly play**: their 88,681 commands gave 12,656 orders, a unit obeying one command in seven; a random
  command's roles and kind seldom go together.  The score 0.165 is roughly beside the 0.144 to 0.151 of the runs from
  nothing at iteration 0 (other games of the same kind).

**A second autoencoder, `vae-2`** (`--beta 0.1 --empty 0.02`, trained beside those checks; mine, to see what the two
faults were worth): the scripts' decisions back whole 74.7% for 77.0%, from 86 nats for 174; codes round zero at a
spread of 1 say nothing 5% of the time for 72%, and 1.9 commands a decision for 0.46; a code off by 0.3 keeps its kinds
and types 93% and 75% for 83% and 46%, and puts its places 630 elmos off for 440; a made-up command's set of roles
comes back 6% for 18%.  I'd take it for a run.  It hasn't been played through.

**The machine's memory, at about 14:20.**  The user stopped an eval of mine: "Memory was at 100%, GPU memory was leaking
into shared memory."  `order_vae.py eval`, written that hour with no cap, sent 100,000 decisions through the net in one
batch on the card (and a timing of mine had tried a batch of 8,192: 13.5 GB, 14 s a step).  Now `order_vae.cap_card`
holds the tool to 6 GB of the card, so it fails and doesn't spill, and nothing goes through in more than 4,096 at a
time: the same eval held 1.35 GB.


### The actor and the trainer

`tools/policy_code.py`, `policy_train.py --size code`: `rdnn-5`'s actor with everything between the memory and the
orders gone, 6.38M trained parameters for 9.87M.  A decision is one draw from a Gaussian over the code; the update
reads the frame's latents, the memory and the code, and featurizes nothing.  Tests: the suite, 859 with
`tests/policy/test_code_actor.py`'s two (played through the trainer's path with each code as likely in training as when
drawn, and the pass as a CUDA graph) and `test_selections.py`'s three.

**A smoke of the trainer** (`results/smoke-code.sh 3`, at 14:32, on `vae-2`, `rdnn-5`'s flags with `--size code`, in
`/mnt/d/zk-learned/runs/code-smoke`, since deleted): three iterations in 82 s with the start, no error, `masked_out` 0.
By the third: 2,340 decisions a second overall (collect 2,160, train 4,910; `rdnn-5`: 520), the learner 1.3 GB of the
card and a collector 0.27 of its 2.0, a minibatch about 530 decisions, 115 graphs captured.  The actor waits out the
critics' 10 iterations, so `kl` and `ratio_deviation_max` were 0: play and training agree on each code's probability
through the collectors' graphs.  Untrained it said 1.8 to 2.3 commands a decision and 0.4 to 0.5 units a decision
obeyed one (`actions_a_decision`, `orders_a_decision`); `state_abs` 0.21 to 0.27.  Nothing in it says whether it
learns.

### The user's choices, and the autoencoder after them

Put to the user after the checks above: which autoencoder, what to do about decisions longer than four commands, where
the actor starts, what a made-up command is, and a run.  Their answer: "1. vae-2 2. bigger code should be fine 3. either
is fine 4. commands should be plausible/valid".

* **Plausible commands** (`selections.able`, `order_vae.made_up`, `OrderVAE.say`).  By the type table, a role can be
  told a kind if some type of it ever can, and can make a type if one of its types builds or produces it.  A made-up
  command's roles are drawn among those its kind is for and its type among what they make, and the decoder says only
  such: the likeliest roles that can, the likeliest type they make.  All 60,738 of the scripts' held-out commands looked
  at are plausible by it.
* **`vae-3`** (`--beta 0.1 --empty 0.02 --codes 8`: `vae-2`'s settings, 128 numbers, plausible made-up commands; 32
  minutes).  From a code's mean, held out: the scripts' decisions right in everything 89.3% (`vae-2`: 74.7%), kind
  99.0%, roles 96.3%, type 95.7%, a place within 11 elmos.  By length: 99.6% of decisions of one command, 98.8% of 2 to
  4, 95.6% of 5 to 8 (`vae-1`: 18.4%), 5% of 9 and more, where two commands in three are right.  A made-up decision of
  a few commands comes back whole 64% of the time, its roles 82% and its type 95% (6 to 18% and 22 to 39% before).
  Codes round zero at a spread of 1 say nothing 1.5% of the time and 1.7 commands a decision.  A code off by 0.3 keeps
  its kinds 97.5%, roles 94%, types 90%, and puts its places 650 elmos off; by 0.1, 220 to 230; by 0.03, 70.
* **The scripts through its code** (`language_check.py ... coded`, the same 128 games): 60 won, 62 lost, 6 drawn, 0.492
  ± 0.044, the tests' score 0.495 ± 0.028.  Level with commands alone (62, 64, 2), where `vae-1`'s 64 numbers lost 32
  to 89.  38% of the units the scripts told were told the same by what the code said (commands alone 71%, `vae-1`
  29%), and 7% of looks had 20 commands or more for 3%: what differs is in the long looks, and it didn't cost games.
* **`vae-4`** (`--codes 16`, 256 numbers, all else as `vae-3`; 31 minutes; mine, to see whether the longest decisions
  would fit): worse than `vae-3` throughout.  Decisions right in everything 79.4%, those of 5 to 8 commands 30.9%, of 9
  and more 0.5 to 3%; a code off by 0.3 keeps its kinds 92%, roles 85%, types 74%.  It left more of its code at the
  Gaussian (the encoder's own spread 0.88 for 0.74).  Not looked into, and not played.
* **Taken: `vae-3`** (`policy_latent.ORDER_CODER`, `launch-code.sh`'s `CODER`, and `order_vae.py`'s defaults).  The
  actor has 6,414,992 trained parameters with it.  Where the actor starts is as it was built, the unit Gaussian.
* **A smoke of the trainer on it** (`results/smoke-code.sh 3`, at 15:52): three iterations in 78 s, no error,
  `masked_out` 0, `kl` and `ratio_deviation_max` 0.  2,020 to 2,120 decisions a second overall in the second and third
  (collect 1,800 to 1,920, train 3,280 to 5,120), the learner 1.34 GB and a collector 0.27 to 0.31, a minibatch about
  515 decisions.  Untrained: 1.7 to 2.1 commands a decision, 0.5 to 0.6 units a decision told (0.4 to 0.5 on `vae-2`,
  before the commands were plausible): most of what keeps a command from being obeyed is the frame's.
* **The actor's least spread**, put to the user (0.1 as built, at which a place is 220 elmos off; 70 at 0.03).  Their
  answer: "Start with 0.1, decreasing to 0.03 as the training run continues."  Built as `--spread-floor
  ITERATION:SPREAD ...` (`Config.spread_floor`, `Learner.set_floor`, the actor's `log_std_least`): linear between its
  points by iteration, as the trainer's other ramps, for the learner, the collectors (with the weights) and the
  average that plays the tests.  `results/start-code.sh HOURS` gives a run `0:0.1 N:0.03`, N its hours at 350
  iterations an hour (mine: the smoke's 2,000 decisions a second, less its tests; 8,400 for 24 hours).  A smoke with
  `0:0.1 4:0.03` (at 16:05): `code_spread_floor` 0.1, 0.0825 and 0.065 in its three iterations, no error, the
  collectors' weights as sent.
* **The helpers for a run** (`results/start-code.sh HOURS`, `code-status.sh`, `code_terms.py`, `stop-code.sh`,
  `resume-code.sh`, `kill-code.sh`, the watchdog and the memory log): the `rdnn` run's, for
  `/mnt/d/zk-learned/runs/code-1` and `launch-code.sh`; `rdnn_status.py` has two lines for a code actor (its Gaussian;
  commands and units told a decision).  Read on the smoke's directory; no run has used them.
* The suite: 859 pass.

## `rdnn-5`: `rdnn-4` again with one game in five kept clear of the lead rule (2026-11-19 01:55 to 08:54, stopped by the user for checks)

**Stopped at iteration 769** (13.21M decisions, `ppo/ckpt/it-000769.pt`) at 08:53:37, seven hours in, by the user ("OK- stop
the run and run checks"), told that it had stopped building and left its fighters at home ("What happened in it").  Its
tests against the script: 0.144, 0.160, 0.159 and 0.171 at iterations 0, 250, 500 and 750, with 1, 2, 1 and 2 wins of
128: level with `rdnn-3`'s at the same iterations (0.151, 0.160, 0.164, 0.166).  The checks run after it are under
"The checks after it".

The user, 2026-11-19 at about 01:55, told that `rdnn-4`'s lead rule had come on at iteration 28 and was in every game
from there, and that a set share of games without it is `--lead-fade`'s to give: "OK, proceed restarting the run with
`--lead-fade 0:0.8`".

So `rdnn-4` from nothing again (its three changes below), with `--lead-fade 0:0.8` where it had `0:1`: once fewer than
80% of the last 800 ended seats are draws (`--lead-draws 0.8`, as `rdnn-4`), 80% of the games that start have the lead
rule and 20% are played out to an elimination or the 80-minute limit.  Tests keep the rule in every game.
`results/start-rdnn.sh 24`, everything under `/mnt/d/zk-learned/runs/rdnn-5`; no code changed, so the suite wasn't run
again.  Mine: from iteration 0 in a directory of its own and not `rdnn-4` carried on from its iteration 42; 24 hours.

### Its tests

* **Iteration 0** (01:57): as `rdnn-4`'s, the same net and games: 1 win of 128, the score 0.144.
* **Iteration 250** (about 03:52): 2 wins of 128, the score 0.160 +- 0.008.  Paired with iteration 0: the score +0.016
  +- 0.006 (+2.6 SE), own value 4,361 to 4,997 (+2.7 SE), the damage share 0.351 to 0.412 (+4.2 SE), damage dealt
  +1.0 SE, income +0.7 SE (its share -2.0 SE), extractors 2.3 to 2.5 at 3 minutes (+1.1 SE) and 2.1 to 2.1 at 6 (the
  script's 10.5).  So learning by the rule, on the score.  Beside `rdnn-3`'s 250, the same games: the same score
  (0.160), 2 wins for its 1, and behind it on the economy: extractors 2.5 for its 3.1 at 3 minutes (-4.1 SE) and 2.15
  for 2.6 at 6 (-2.7 SE), own value 4,997 for 5,681 (-2.5 SE), income -1.4 SE; damage dealt +1.3 SE.
* **Iteration 500** (about 06:07): 1 win of 128, the score 0.159 +- 0.006.  Paired with iteration 0: the score +0.015
  +- 0.004 (+4.0 SE), own value 4,361 to 5,469 (+4.7 SE), income +2.8 SE, extractors 2.3 to 2.9 at 3 minutes (+3.9
  SE) and 2.1 to 2.5 at 6 (+2.8 SE; the script's 10.4), damage dealt +1.2 SE.  So learning by the rule on the score
  and on three of the four.  On its 250: the score the same (0.160 to 0.159), own value 4,997 to 5,469, extractors at
  6 minutes 2.1 to 2.5: the economy came on with the place and offset heads' entropy gone.  Beside `rdnn-3`'s 500,
  the same games: the score 0.159 for its 0.164 (-1.2 SE), own value 5,469 for 6,145 (-2.2 SE), extractors 2.9 for
  3.35 at 3 minutes (-3.4 SE) and 2.57 for 2.48 at 6 (+0.6 SE), income +0.3 SE, damage dealt +1.0 SE, a win each.
* **Iteration 750** (about 08:42, ten minutes before the stop): 2 wins of 128, the score 0.171 +- 0.008.  Paired with
  iteration 0: the score +0.027 +- 0.006 (+4.3 SE), own value 4,361 to 6,102 (+8.1 SE), income +4.6 SE, extractors
  2.3 to 3.0 at 3 minutes and 2.1 to 2.85 at 6 (+4.8 SE each; the script's 10.3), damage dealt +1.4 SE, the games 46 s
  longer (+2.8 SE).  Beside `rdnn-3`'s 750: the score 0.171 for its 0.166 (+0.9 SE), damage dealt +2.3 SE, extractors
  at 6 minutes 2.83 for 2.63 (+1.4 SE), own value -2.0 SE, 2 wins for its 1.  The best score of these runs, within 1 SE
  of `rdnn-3`'s; and with no BUILD or area mex among its actions, so its extractors are the opening's, kept longer.

### What happened in it

* **The lead rule came on at iteration 33** (about 02:10), as in `rdnn-4`: 800 seats had ended and few of them in
  draws (6% by iteration 40).  From there 80% of the games that start have it.
* **The place and offset heads lost their entropy in a burst of KL at iterations 114 to 116** (about 02:53), and
  haven't got it back.  The guard had let the scale up to the whole 2e-3 at 111; the KL was 0.016 at 113 and then
  0.28, 0.31, 0.20 and 0.09 at 114 to 117 (the target 0.01), the scale cut to 0.13 by 119 and whole again from 126.
  Over those iterations the place head's entropy went from 4.3 to 0.5, and the offset's from 3.2 to 0.6 by 119 and to
  0.00 by 134; at 300-319 they are 0.13 and 0.01 (`rdnn-3`'s at the same iterations: 4.9 and 3.2, and never under 4.6
  and 1.8 in its 1,149).  They are a BUILD's where: which unit or site, and where beside it.  So a BUILD now goes to
  much the same choice every time, and with no entropy bonus under GRPO nothing pushes them back.  What they chose
  isn't known (reading it needs the actor, so no run going).  Put to the user; nothing changed.
* **The policy stopped building, and "BUILD" in the checks was ASSIST** (found 08:55, on the user's question below).
  `share_kind` goes by rawframe's numbers: 10 is BUILD and 11 ASSIST.  Every check to 08:23 here, and `rdnn-3`'s,
  `rdnn-2`'s and `rdnn-1`'s, called `share_kind11` BUILD: read ASSIST there.  The BUILD and area-mex actions an
  iteration (of 31,000 to 49,000 actions in all): 1,150 and 1,000 at 100-113, before the burst; 630 and 610 at
  114-119; 170 and 280 at 120-139; 19 and 60 at 200-299; 5 and 19 at 300-499; 0 and 1 at 500-699; **0 and 0 from
  700**.  So the place and offset heads' lost entropy is this: the policy all but gave BUILD up in the burst and
  wholly by 500, and what its side builds is the opening's.  `rdnn-3` went the same way more slowly: BUILD 770 an
  iteration at first, 113 at 200-299, 2.5 over 700-1149 and none from 1000; the area mex 500 to 580 till 500, 36 at
  the end.  The constructors' actions went to ASSIST (2% of actions to 15%) and RECLAIM (2% to 11%).
* **The fighters stand at home more as it trains** (the user, 08:50, of the replay `it00692-game3`: "the policy
  doesn't seem to give many orders to combat units ... they're all clustered around the factory. Could this be caused
  by the frozen encoder?").  In the replays against the script (4 games an hour, `scratchpad/replay_targets.py`): a
  finished fighter had no order 17%, 18%, 38%, 30%, 31% and 65% of the time at 137, 282, 401, 502, 602 and 692, and
  stood within 800 elmos of its start 25%, 28%, 59%, 32%, 36% and 66% of it; 25, 15, 10, 24, 16 and 8 new orders to
  fighters a minute.  Of those orders' targets about 45% are within a third of the way to the enemy, 27% beyond two
  thirds.  `rdnn-3`'s fighters were within 800 elmos 43 to 99% of the time in all its replays.  In training the
  actions a decision went from 1.0 to 2.5 (`rdnn-3`: 0.8 to 1.65), 65% of them MOVE, FIGHT or ATTACK.  Not the
  encoder's doing by what can be seen without the card: whom to tell is scored from each unit's own features through
  the actor's own trained embedding (`hu`: its idle flag, place and order are there), the 40 frozen latents only the
  setting; the same encoder under `full-4`'s clone gave the script's orders (97 to 100% of the kinds right but for
  builds); and this run at 137, with the same encoder, left a fighter without an order 17% of the time.  Not checked:
  how well the latents keep a frame of the policy's own games (it needs the card).
* **A second burst of KL at iterations 673 to 676** (about 07:58): 0.06, 0.09, 0.44 and 0.26 an iteration after 0.0003
  to 0.003 for the 150 before, the actor's gradient norm no more than 0.8; the guard took the scale to 0.20 at 677 and
  it was whole again at 681, the KL back under 0.002.  No head's entropy moved with it this time (the region head's
  2.8 to 3.3), and BUILD's share of actions stayed 14 to 15%.  The recurrent weights, which the guard doesn't hold
  (the user), stepped 0.005% of their mean over 660-679 for 0.0003% either side.

### The checks after it

With the run stopped and the card free (the user, 08:50: "stop the run and run checks").  The scripts are in
`results/checks/` (`run.sh` there runs one and lists them): they play games as `policy_train.evaluate` does, from a
checkpoint, on the same maps and starts each time.  Each is one run of 32 to 128 games, none repeated.

* **The frozen encoder on the policy's own frames** (`encoder_check.py`, the checkpoint at 769): team 0's frame every
  30 s from minute 2 of 64 1v1s on the same maps and starts, three ways, through `full-4`'s encoder and its own
  decoder.  What it gives back of each region, as R^2 over the cells that hold something (over all cells it is 0.96 to
  0.98 everywhere), and its answers to its questions:

  | | scripts against each other | the policy against the script | the policy against itself |
  |---|---|---|---|
  | frames, unit tokens a frame | 1,893, 80 | 1,038, 30 | 2,232, 55 |
  | every column | 0.67 | 0.77 | 0.49 |
  | own ground units | 0.80 | 0.64 | 0.41 |
  | own builders | 0.69 | 0.68 | 0.39 |
  | own value | 0.81 | 0.77 | 0.62 |
  | enemy ground units | 0.80 | 0.74 | 0.59 |
  | enemy value in sight | 0.74 | 0.82 | 0.69 |
  | where a unit is, nats (1.39 knows nothing) | -3.08 | -4.14 | -2.64 |
  | what's at a place, nats | 0.76 | 0.39 | 0.62 |

  So in the games against the script, where the user saw the fighters at home, the encoder keeps a frame as well as
  it keeps the scripts' own.  In self-play it keeps less of how much stands where, the policy's own army most (0.41
  for 0.80): there the policy's ground units are in 24% of a map's regions, the scripts' in 9%, with fewer units.  It
  still places a unit 0.44 nats short of the scripts' frames and far from knowing nothing.  So the latents blur the
  policy's scattered army in training, and that isn't what leaves fighters without orders in the tests.
* **Does building pay, by the run's own reward?** (`build_check.py`, the checkpoint at 100, before the burst at 114.)
  128 1v1s of the policy against itself to 40 minutes, the lead rule on: in each, one team as it is and the other with
  BUILD and AREA_MEX out of its legal kinds, the team alternating.  The team that can build gave 31 BUILDs and 30 area
  mexes a game, and **won 31 and lost 35 of the 66 games that ended** (20 by elimination, 46 by the lead; 62 still
  going).  Its return a term (the other team's is this turned round): extractor lead +0.20 +- 0.04 (+4.8 SE), economy
  +0.10 +- 0.06, production lead +0.03, combat -0.035 +- 0.021 (-1.7 SE), result -0.03 +- 0.07, and the sum by the
  run's weights +0.06 +- 0.14 (+0.4 SE).  It had 3.3 extractors at 6 minutes for the other's 3.2, and 4.5 for 2.8 at the
  end (+5.1 SE), income 14.7 for 11.5 (+3.4 SE), value 11,060 for 11,470.  The team that can't gave twice the ASSISTs
  and RECLAIMs.  So 61 build orders a game bought 1.7 extractors and no more games: what building paid at 100 was
  0.06 of weighted return in a spread of 1.6 a game (0.14 x sqrt 128).  (Mine: too little for a branch of one to three
  minutes to show, so nothing held BUILD up when the burst at 114 moved the kinds.)
* **What became of its build orders** (`build_fate.py`, the same checkpoint, 32 self-play 1v1s to 20 minutes, both
  teams as they are; 14 BUILDs and 15 area mexes a team).  **What a BUILD would build is drawn evenly from every
  structure**: of 892, the ten most chosen were a gunship factory, a plane plate, a caretaker, a missile silo, a ship
  plate, a torpedo launcher, an airpad, a radar, a plane factory and a tank factory, 20 to 24 each.  The place is far:
  840 / 1,760 / 2,740 elmos from the unit (quartiles), the same for an area mex's middle.  3 s on, the unit is on a
  build after 92% of BUILDs and 61% of area mexes (no free spot in the rest's area).  And **the policy's next action to
  that unit came while it was still on the build in 53%**, a median 21 s after (9 / 21 / 42), as ASSIST, BUILD,
  AREA_MEX, MOVE or RECLAIM about evenly; in the other 47% a median 42 s after.  A team finished 9.8 economy buildings,
  0.3 defences and 0.6 factories in 20 minutes, 5.4, 0 and 0.1 of them by minute 3 (the opening's): about 5 buildings
  of its 29 build orders.  (Run again that afternoon: the first run took a unit for on a build by its record's order,
  which is the last it was given and stays when the build is over, and said 56%.  On a build is that and not idle.)
* **Would an area mex pay if the policy let it finish?** (`build_stick.py`: `build_check.py` again, the same 128
  games, with two things changed for the team that builds: it has AREA_MEX and no BUILD, so no type or exact place to
  get right, and none of its own orders reaches a unit in the 120 s after a BUILD the env gave it, done by then or
  not (the record's order, as above); 29 orders a game were held back so.)  It gave 38 area mexes a game and had **4.7 extractors at 6 minutes for the other's 3.0 (+8.4 SE), 9.8
  for 2.5 at the end (+9.4 SE), and an income of 26.8 for 10.7 (+8.0 SE)**.  Its return: economy +0.73 (+9.6 SE),
  extractor lead +0.85 (+10.7 SE), production lead +0.26 (+7.9 SE), the sum by the run's weights +0.58 +- 0.15 (+3.9
  SE, for +0.06 before).  **And it won no more: 34 won, 32 lost of 66 ended**, the result +0.02 +- 0.07, its value at
  the end 11,000 for 10,830, its PRODUCE actions 5,175 for 5,140, combat -0.04 (-1.9 SE).  So left to finish, the
  area mex alone gives 3 times the extractors and 2.5 times the income, which three of the reward's terms pay for;
  and at 100 the policy turns none of that metal into army, so the result pays nothing for it.  Not separated: how
  much is the builds sticking and how much is BUILD gone (a BUILD or another area mex was a third of what cut a build
  short).
* **The fighters** (`fighter_check.py`: team 0 of 64 1v1s from minute 2 to 20; a unit a group order covers counts as
  told; idle is the frame's own flag: no order, or the last one over, and nothing it's chasing):

  | | 140, against the script | 769, against the script | 769, against itself |
  |---|---|---|---|
  | fighters a decision | 8.3 | 15.0 | 28.6 |
  | fighter-time idle | 53% | 71% | 73% |
  | within 800 elmos of home | 28% | 43% | 37% |
  | both | 18% | 36% | 32% |
  | actions a decision | 0.73 | 0.87 | 1.63 |
  | group orders, of actions | 24% | 1% | 1% |
  | an idle fighter is given an order, of decisions | 7.7% (it waits 39 s) | 3.9% (76 s) | 4.0% (76 s) |
  | a fighter at work is given another | 7.8% | 3.8% | 3.8% |
  | decisions with an idle fighter where none of them was told | 78% | 74% | 61% |

  So what the user saw in the replay at 692 holds over 64 games: 71% of fighter-time idle at 769 for 53% at 140, half
  of it at home.  The policy gives under one action a decision against the script, to a unit chosen without regard to
  whether it's idle (the two rates are the same at both checkpoints); it has given up group orders (24% of actions to
  1%); and the army has doubled.  So each fighter waits twice as long for its turn, and one fresh from the factory
  stands there 76 s.  Those it does order go far: a MOVE or FIGHT sends one a median 1,650 elmos, to 2,300 from home,
  as at 140.  (Run again that afternoon: the first run counted a fighter "without an order" by its record, 21%, 39%
  and 40% of fighter-time, which misses one standing where its last order ended.)

**What they say together** (mine).  The frozen encoder is behind neither thing the user and I saw.  The policy stopped
building because building, as it played at 100, paid nothing: a BUILD was any structure at all, a long walk off, and
its own next order to the unit, 21 s on, ended more than half its builds.  Let alone, an area mex pays well in the
economy terms and still wins nothing, since the metal isn't spent.  And its fighters wait because it orders units one
at a time, at most one or two a decision, whichever they are, while its army grows.  Both are the same shape: one
decision every 3 s reaches each unit too seldom to keep an army busy and too often to let a constructor finish.

**Not checked**: any of this in `rdnn-3` or `gru-1`, or at a second checkpoint for the builds; the sticking and the
missing BUILD apart; why the choice of unit doesn't lean to the idle (`idle_lead` pays for it, and a unit's token has
the flag); why the group orders went; whether the encoder's blur in self-play slows what the
actor learns there; and why 2.5 times the income buys no more army.

**Put to the user, 2026-11-19** (what to change is theirs).  Their answer to the first, that afternoon: "I think I
have a solution: make builds stick, but also add a 'cancel build' order as well.  On 'cancel build', the builder's
order(s) are explicitly canceled."  It's made ("What changed under the runs", "A build sticks"; "Under the new rule"
below).  The other three aren't answered as put.  Later that day the user proposed something wider in their place: "I
think a transformer-based autoencoder for a frame's orders (for one side) would show improvements and simplify the RL
training process given that its parameters are frozen when training the policy; proceed with documenting the proposed
architecture."  Documented then, and built since ("After `rdnn-5`", above): docs/learned-policy/05-order-autoencoder.md.  (I had answered a first question
on it, "Would a small autoencoder for frame actions perform better?", with doubt: the scripts' games are all it could
learn from, the clone of them topped out at 8 to 17% of tests, and a decision is mostly keep.  Those are in the
document, under "What it can't do, and what may go wrong".)

And after it, for the fighters: "Starcraft's Pluto includes player-like unit selection before issuing group commands.
I think this is feasible and likely to be more efficient.  As an example (in token form): `<latent tokens>
<SELECT_CIRCLE CENTRE=(x,z) RADIUS=r TYPE=combat> <ATTACK_MOVE AT=(x,z)> <END_FRAME_ORDERS>`.  Could this work?"  I
said it could, and that I'd take it before the autoencoder: today's group order is the same parts in another order,
optional, and round a unit (a group took in 2.4 units at 140).  Of the orders that need one unit, the user: "I think all
the single-unit orders can be made to work with multiple units.  OK, write it up."  Documented then, and built since:
docs/learned-policy/06-select-then-command.md.  Pluto's own selection, by its README, is its own action round a unit it
points at, kept between its steps of 6 frames; here the selection and its command are one action in a decision.

1. *A build that sticks*: a unit on a BUILD or an area mex isn't offered for an order until it's over, to a limit (the
   commit lock, lengthened for these; the featurizer in Python and C++).  It's what the last check tried from outside.
2. *What a BUILD is, early*: the type head starts even over 40-odd structures.  A narrower offer, a start from
   `full-4`'s clone, or the economy script's advice as a cloning term for a while (`--clone-weight`, which the trainer
   has).
3. *The fighters*: a new unit takes an order without the policy's turn (its factory's last, as a rally point is), or
   more actions a decision (`DONE_START`), or the group orders kept from going (`ALONE_START`, an entropy floor).
4. *Or the constructors left to the economy script* (`--economy-start`/`--economy-end` under 1) while the army learns.

**Under the new rule** (2026-11-19 afternoon, once a build stuck and the cancel was a kind): the two build checks
again at the checkpoint at 100, the same games, its config given the cancel (`probe_common.config(run, sticky=True)`).
Its weights never trained by this rule: the cancel's own logit is as it began, and the 3 nats against picking a unit
on a build are the code's.

| | the old rule | the new |
|---|---|---|
| *`build_fate.py`, 32 games to 20 minutes, both teams as they are* | | |
| BUILDs and area mexes a team | 14 and 15 | 9 and 9 |
| the next action to the unit came while it was on the build | 53%, a median 21 s after | 17%, all cancels, 111 s after |
| cancels a team | | 2.4 |
| finished a team: economy, defences, factories | 9.8, 0.3, 0.6 | 16.1, 1.0, 1.7 |
| extractors standing at 20 minutes | 4.7 | 9.3 |
| *`build_check.py`, 128 games to 40 minutes, one team barred from building* | | |
| ended games the building team won and lost | 31 and 35 | 45 and 31 (the result +0.12 +- 0.07, +1.6 SE) |
| extractors at 6 minutes, it and the other | 3.3 and 3.2 | 5.2 and 3.5 (+8.5 SE) |
| extractors at the end | 4.5 and 2.8 | 8.4 and 3.0 (+7.9 SE) |
| metal income at the end | 14.7 and 11.5 | 23.3 and 11.7 (+6.4 SE) |
| value at the end | 11,060 and 11,470 | 14,000 and 11,540 (+2.8 SE) |
| its return by the run's weights | +0.06 +- 0.14 | +0.58 +- 0.16 (+3.6 SE) |
| PRODUCE actions, it and the other | 7,575 and 5,162 | 10,915 and 4,902 |

So by this rule building pays at 100, in the reward's economy terms (economy +0.55, extractor lead +0.73, production
lead +0.20, each past 5 SE) and in value, where the old rule paid nothing; the games won lean its way and aren't
shown (+1.6 SE).  Its BUILDs are still any structure at all, and the factories among them are finished now (1.7 a
team for 0.6): that's where the PRODUCE actions and the value come from, which the area mexes alone didn't give
(`build_stick.py`).  The team gave 5.4 cancels a game.  Not checked: any training by the rule.

### The hourly checks

A cron job in the session that started it (at about 53 minutes past) ran `results/rdnn-status.sh` and
`results/rdnn_terms.py 5`, and added a line or two here.

* **03:23, iteration 184** (3.15M decisions): running, 15 s since its report, 640 to 680 decisions a second (880 in
  its first 20 iterations, 600 to 730 since 40).  The lead rule on since 33, in 80% of games starting; of the seats
  ended in the last 50 iterations 12% drew (218 of 1,864), and 66 to 90% of ended seats by twenties were in games with
  the rule.  The KL 0.005 to 0.014 an iteration by twenties but for 0.052 over 100-119 (the guard took the scale to
  0.13 at 120; whole again from 130), 0.004-0.005 at 180-184.  The recurrent weights' step 0.006 to 0.012% of their
  mean, the state's mean size 0.33 at first and 0.047 now (largest 0.53).  BUILD 2.5% of actions at first and 6 to 9%
  from 120, the area mex 2.3-3.4% and 0.4% from 160; the place head's entropy 4.28 to 0.60, the offset's 3.20 to 0.73.
  The critics: idle lead 0.84, extractor lead 0.81, `production_lead` 0.81, economy 0.78, constructor lead 0.69,
  score 0.67, result 0.60, map 0.12, combat and experience 0.00.  No test since 0 (250 at about 03:50).  The replays
  at 137 (4 games against the script, all lost, 8.5 minutes a game): 3.8 extractors at 6 minutes for the script's
  14.2, 5.5 constructors at most, metal floated 43% of the time.  No other kinds before 500.  The learner 2.94 GB, a
  collector 1.26 of its 2.0 (0.38 at the start), the card 8.4 of 12.3 GB, WSL 27.0 GB available; the token budget
  138,782 after 33 splits (`rdnn-3`: 164 in 1,149 iterations).  No error, the watchdog only watching.
* **04:23, iteration 319** (5.46M decisions): running, 11 s since its report, 690 decisions a second over the last 50
  iterations and 750 over the 50 before (640 to 680 at the last check).  The test at 250 (above): learning by the
  score, level with `rdnn-3`'s and behind it on extractors.  The place and offset heads' entropies 0.13 and 0.01
  (above).  The lead rule in 80% of games starting; 13% of the seats ended in the last 50 iterations drew (204 of
  1,596), 82 to 90% of ended seats by twenties in games with the rule.  The KL 0.002 to 0.006 by twenties, the scale
  whole.  The recurrent weights' step 0.003 to 0.015%, the state's mean size 0.016 to 0.031 (largest 0.30).  BUILD 9
  to 12% of actions, the area mex 0.1 to 0.3%.  The critics: result 0.76, idle lead 0.80, economy 0.74, constructor
  lead 0.72, extractor lead 0.71, score 0.71, `production_lead` 0.70, map 0.43, experience 0.31, combat 0.20 (the
  last three were 0.12, 0 and 0).  The replays at 282 (4 games, all lost, 14.8 minutes a game): 3.0 extractors at 6
  minutes for the script's 8.2, 26 constructors at most, metal floated 22% of the time.  No other kinds before 500.
  The learner 2.96 GB, a collector 1.26 of its 2.0 (as an hour ago), the card 8.8 of 12.3 GB, WSL 27.0 GB available;
  the token budget 130,883 after 53 splits.  No error, the watchdog only watching.
* **05:23, iteration 426** (7.28M decisions): running, 3 s since its report, and slower: 460 decisions a second over
  the last 50 iterations and 590 over the 50 before (690 and 750 an hour ago), 107 iterations in the hour for 135.
  From about iteration 340 a decision is bigger, some 700 tokens for 575 (minibatches of 185 decisions for 240, a
  collector's round 33 s for 23, the game minutes an iteration the same): more on the map a frame, no fault.  More
  draws: 23% of the seats ended in the last 50 iterations (338 of 1,458; 33% at 380-399), 18% of the last 800; the
  lead rule in 80% of games starting, 70 to 100% of ended seats by twenties in games with it.  The KL 0.001 to 0.005
  by twenties, the scale whole.  The place and offset heads' entropies still 0.14 and 0.00 (the group head's 0.37,
  as `rdnn-3`'s went).  The recurrent weights' step 0.0005 to 0.005%, the state's mean size 0.005 to 0.014 (largest
  0.13).  BUILD 9 to 12% of actions, the area mex 0.0 to 0.1%.  The critics: idle lead 0.86, extractor lead 0.83,
  economy 0.79, `production_lead` 0.76, result 0.69, constructor lead 0.64, score 0.64, experience 0.43, combat 0.37,
  map 0.24.  No test since 250 (500 at about 06:05).  The replays at 401 (4 games, all lost, 8.9 minutes a game):
  3.0 extractors at 6 minutes for the script's 16.0, 3.2 constructors at most, metal floated 6% of the time.  No
  other kinds before 500.  The learner 2.67 GB, a collector 1.28 of its 2.0, the card 6.6 to 8.8 of 12.3 GB, WSL
  26.1 GB available (each of the three big processes 0.3 GB up at 04:56, to 3.6 GB); the token budget 137,987 after
  77 splits.  No error, the watchdog only watching.
* **06:23, iteration 528** (9.04M decisions): running, 31 s since its report, 510 decisions a second over the last 50
  iterations and the 50 before (460 and 590 an hour ago), 102 iterations in the hour.  The test at 500 (above):
  learning by the rule on the score, own value, income and extractors, and level with `rdnn-3`'s 500 within 1.2 SE on
  the score.  The KL 0.0003 to 0.004 by twenties and 0.0001 to 0.0003 at 524-528, the scale whole, the actor's
  gradient norm 0.26.  The recurrent weights' step 0.0003 to 0.007%, the state's mean size 0.003 to 0.013 (largest
  0.08).  BUILD 13 to 17% of actions, the area mex 0.0%.  The place, offset and group heads' entropies 0.22, 0.00 and
  0.20.  17% of the seats ended in the last 50 iterations drew (262 of 1,506), 18% of the last 800; the lead rule in
  80% of games starting, 72 to 86% of ended seats by twenties in games with it.  The critics: idle lead 0.82, result
  0.77, economy 0.76, constructor lead 0.72, extractor lead 0.72, `production_lead` 0.70, score 0.70, experience
  0.32, combat 0.28, map 0.26.  The replays at 502 (4 games, all lost, 10.5 minutes a game): 3.0 extractors at 6
  minutes for the script's 9.5, 7 constructors at most, metal floated 15% of the time.  Other kinds: none ended yet
  (the ramp is at 0.7% of games at 528).  The learner 2.97 GB, a collector 1.47 of its 2.0 (1.28 an hour ago, and
  the other kinds still to come), the card 7.8 of 12.3 GB, WSL 25.7 GB available; the token budget 138,553 after 99
  splits.  No error, the watchdog only watching.
* **07:23, iteration 620** (10.63M decisions): running, 17 s since its report, 480 decisions a second over the last 50
  iterations and 460 over the 50 before (510 an hour ago; 360 to 460 at 616-620), 92 iterations in the hour.  Other
  kinds 2% of games, one team game ended.  The KL 0.0005 to 0.002 by twenties, the scale whole.  The recurrent
  weights' step 0.0002 to 0.006%, the state's mean size 0.003 to 0.011 (largest 0.11).  BUILD 12 to 17% of actions,
  the area mex 0.0%.  The place, offset and group heads' entropies 0.17, 0.00 and 0.16.  23% of the seats ended in
  the last 50 iterations drew (288 of 1,276), 18% of the last 800; the lead rule in 80% of games starting, 84 to 93%
  of ended seats by twenties in games with it.  The critics: result 0.85, idle lead 0.85, extractor lead 0.81,
  economy 0.75, `production_lead` 0.73, constructor lead 0.62, score 0.59, experience 0.45, map 0.44, combat 0.33.
  No test since 500 (750 at about 08:45).  The replays at 602 (4 games, all lost, 10.7 minutes a game): 2.5
  extractors at 6 minutes for the script's 8.2, 9 constructors at most, metal floated 5% of the time.  The learner
  3.00 GB, a collector 1.50 of its 2.0, **the card 9.7 of 12.3 GB** (7.8 an hour ago; `rdnn-3` was at 9.6 at its
  iteration 1117, and `gru-1` stalled at 11.3), WSL 24.9 GB available (the learner and a collector 0.25 to 0.4 GB up
  in the hour, to 3.9 and 4.1 GB); the token budget 130,180 after 112 splits.  No error, the watchdog only watching.
* **08:23, iteration 713** (12.24M decisions): running, 13 s since its report, 510 decisions a second over the last 50
  iterations and 430 over the 50 before (480 and 460 an hour ago), 93 iterations in the hour.  Other kinds 5% of
  games (chicken and team games ending).  A second burst of KL at 673-676 (above); otherwise 0.0003 to 0.0013 by
  twenties and 0.0001 to 0.0004 at 709-713, the scale whole.  The recurrent weights' step 0.0003 to 0.005%, the
  state's mean size 0.003 to 0.009 (largest 0.06).  BUILD 14 to 15% of actions, the area mex 0.0%.  The place, offset
  and group heads' entropies 0.13, 0.00 and 0.12.  18% of the seats ended in the last 50 iterations drew (192 of
  1,074), 14% of the last 800; the lead rule in 80% of games starting, 73 to 91% of ended seats by twenties in games
  with it.  The critics: result 0.88, idle lead 0.87, extractor lead 0.85, score 0.83, `production_lead` 0.78,
  economy 0.77, constructor lead 0.69, map 0.53, combat 0.49, experience 0.40.  No test since 500 (750 at about
  08:45).  The replays at 692 (4 games, all lost, 8.9 minutes a game): 2.5 extractors at 6 minutes for the script's
  14.0, 10 constructors at most, metal floated 13% of the time.  The learner 2.86 GB, a collector 1.50 of its 2.0 (as
  an hour ago), the card 9.5 of 12.3 GB (9.7), WSL 24.4 GB available (the three big processes 4.2 to 4.3 GB each);
  the token budget 148,672 after 122 splits.  No error, the watchdog only watching.

## `rdnn-4`: `rdnn-3` again with no lead rule while games draw, production capped by income, and an ended attack out of the record (2026-11-19 01:33 to 01:55, stopped for `rdnn-5`)

**Stopped at iteration 42** (717,108 decisions, `ppo/ckpt/it-000042.pt`; 1.1 GB) at 01:55:25 by the user's restart with
`--lead-fade 0:0.8`, 22 minutes in.

The user, 2026-11-19 at about 01:15: "OK, stop the run for now- I don't think it's learning. I see two problems: first,
when an attack order with the parameter being a unit ('attack unit') is given but the unit to be attacked moves out of
range, the attacking unit just stops but the unit still counts as busy (non-idle); additionally, the factory metal spent
score term needs to be capped to around half of the side's income. The winning through 2.5x metal advantage should be
disabled until the policy stops drawing in at least 80% of games. After this, start another training run from iteration
0."  And then: "the attack order bug occurs when an enemy no longer appears on radar. This is why the policy is getting
'stuck' raider units, with those units being counted as non-idle."

So `rdnn-3` from nothing again, the same flags (`results/launch-rdnn.sh`), with three things changed:

* **An attack that's over leaves no order in the unit's record** (`World::end_attack`; 02-policy.md, "Nothing
  remembered").  What a probe found first (`sim/tests/test_unit_ai.py` has it as a test now): a raider told to attack a
  radar dot, the radar then killed.  Within 2 s the sim gave the attack up, the unit stood, **its frame said idle and
  `idle_lead` counted it**, with or without a turret, a riot, an assault or an extractor of the enemy's 450 to 700
  elmos off.  So the sim's flag and the idle term were right already.  What still said "attack" was the record's order
  (`ord_kind` ATTACK_UNIT and its target, kept till the next order as every order is): the nets read it as order memory
  and the viewer draws it, to wherever the target really is.  That is what's cleared now, in the sim and in
  `rawframe.from_host`.  It takes no column and no ABI number: libzksim stays 31, `FEATURE_VERSION` 10.
* **`production_lead` counts no more of a team's spending on units than half its metal income** at that step
  (`policy_sim_env.PRODUCTION_SHARE`; 03-training.md, "Reward"); it was no more than 30.  The lead that is 1 stays 30.
* **No lead ends a training game while they mostly draw** (`--lead-draws 0.8`; 03-training.md, "Reward"): the lead rule
  off until, of the last 800 seats whose game ended, fewer than 80% drew at the 80-minute limit; then on in every game
  that starts, for good.  Tests keep the rule, so they are the same 128 games as `rdnn-3`'s.

The suite passed with them (852 tests, 2 of them new) before the start.  `results/start-rdnn.sh 24`, everything under
`/mnt/d/zk-learned/runs/rdnn-4`; 24 hours is mine.

### Its tests

* **Iteration 0** (01:38): 1 win of 128, the score 0.144 +- 0.006, income share 0.285, 2.1 extractors at 6 minutes for
  the script's 10.3.  Beside `rdnn-3`'s iteration 0 (another draw of the same net's start): the score -0.007 +- 0.006,
  nothing else apart by 1 SE.

### What happened in it

* **From nothing, without the lead rule, the policy's games don't mostly draw**: of the 12 games it started with, 7
  reached the 80-minute limit at iteration 18 (112 seats drawn at once, 56 copies of 96) and 5 had lost all 8 copies
  to an elimination before; and copies ended by elimination all along, 2 to 62 seats an iteration from iteration 4
  (games 18 minutes old), 690 by iteration 28.  So when 800 seats had ended, at iteration 28 (01:49:48), 14% of them
  were draws (112), well under the 80%, and **the lead rule came on there**, for the games starting from then.  Put to
  the user: by their own measure the rule was off for the run's first quarter of an hour only.  The user: "If I recall
  correctly the lead rule has a set % chance of being disabled per game regardless. This is all fine."  Told that this
  run has no such chance (`--lead-fade 0:1`: every game starting since has the rule), they had it restarted with one
  (`rdnn-5`).  How those games ended isn't recorded beyond won or lost with no lead rule, which leaves an alliance out
  (no builder or factory left).
* **The games got old at once**: the token budget went from 160,000 to 116,640 by iteration 27 in 4 splits (`rdnn-3`
  had 164 in all, its budget between 95,000 and 144,000), the learner at 3.0 GB and a collector at 0.66 of its 2.0,
  the card at 6.3 to 7.0 of 12.3 GB; 600 to 900 decisions a second.
* The KL was 0.02 to 0.08 an iteration at 11 to 16, when the actor began (the critics' 10 iterations over), and the
  guard took the scale from 0.15 to 0.01 and back to 0.07 by 24.

## `rdnn-3`: RL from nothing with an RDNN and nothing remembered among the tokens (2026-11-18 05:23 to 2026-11-19 01:18, stopped by the user)

**Stopped at iteration 1149** (19.80M decisions, `ppo/ckpt/it-001149.pt`) at 01:17:58 on the 19th, four hours short of
its 24, by the user: "OK, stop the run for now- I don't think it's learning."  Its tests against the script: 0.151,
0.160, 0.164, 0.166 and 0.166 at iterations 0, 250, 500, 750 and 1000, with 2, 1, 1, 1 and 1 wins of 128 and 2.1 to 2.6
extractors at 6 minutes for the script's 11; what it gained it had by 750.  Its 48 replays against the script: 43 lost
to the lead rule at 10 minutes on average, 2 to an elimination, 3 drawn at the limit.

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
* **The memory** (the user, 03:15, of `rdnn-1`: "The fading memory is to be expected early on; most games end early and
  there's not much to remember"): a small state early isn't a fault.  The checks give its numbers as facts and act on
  neither.  And (the user, 2026-11-18 00:45): "No need to put recurrent weights under a KL guard".
* **No change in the middle of it** (mine, from the user's question about `rdnn-1`; this run exists to have none), and
  a check of it every hour: `AGENTS.md`, "The run going".
* **To watch**: the tests; the shares of BUILD and of the area mex; the replays' extractors and constructors; the KL
  an iteration and the lr scale; the state's size and `memory_step`; the share of other kinds from 500; the card (a
  collector's share grows with the games' age: `rdnn-1` reached 9.6 GB of 12.3 with games an hour old, and `gru-1`
  stalled at 11.3); `collect_rate`, `wait_s`.

### Its tests

* iteration 0: 2 wins of 128, the score 0.151 +- 0.008, extractors at 6 minutes 2.1 (the script's 10.4); within 1.4
  SE of `rdnn-2`'s iteration 0 on every measure;
* iteration 250 (about 07:20), against its iteration 0: 1 win, the score 0.160 (+0.009 +- 0.007, 1.4 SE), own value at the
  end 4,374 to 5,681 (+4.8), income over the game +3.0, damage dealt -0.1, extractors at 3 minutes 2.4 to 3.1 (+5.2)
  and at 6 2.1 to 2.6 (+3.0, the script's 11.0), games 28 s longer.  Learning by the rule (own value, income,
  extractors).  Against `rdnn-2`'s 250 over the same games: the score -0.003 +- 0.004 (-0.9 SE), income -1.8 SE,
  extractors at 6 minutes the same, nothing past 2 SE.  So at 250 the remembered buildings make no difference that
  128 games show.
* iteration 500 (16:48, 140 iterations after the restart at 360), against its iteration 0: 1 win, the score 0.164
  (+0.013 +- 0.007, 1.8 SE), own value at the end 4,374 to 6,145 (+5.7), income over the game +3.3, damage dealt
  +0.1 (taken -2.7), extractors at 3 minutes 2.4 to 3.4 (+6.7) and at 6 2.2 to 2.4 (+1.8, the script's 11.2), games
  34 s longer.  Learning by the rule (own value, income), not by wins or the score.  On its 250, by the means: the
  score 0.160 to 0.164 and own value 5,681 to 6,145, but extractors at 6 minutes 2.6 to 2.4 and income at 6 minutes
  12.4 to 11.9, so more held at a game's end and no more economy.  Beside `rdnn-1`'s 500 (0.165, 2 wins,
  extractors 2.5; not paired): the same.  Past 500 there is no outside test against this script.
* iteration 750 (about 20:05), against its iteration 0: 1 win, the score 0.166 (+0.014 +- 0.007, 2.1 SE), own value
  at the end 4,374 to 6,613 (+7.6), income over the game +4.6, damage dealt +0.2 (taken -4.0), extractors at 3
  minutes 2.4 to 3.3 (+6.3) and at 6 2.1 to 2.6 (+3.5, the script's 11.3), games 62 s longer.  Learning by the
  rule: on the score for the first time, and on own value, income and extractors.  Its share of income is down
  (0.286 to 0.275, -2.9 SE), the script's own extractors at 6 minutes being up 0.9 on iteration 0's games.  On its
  500, by the means: the score 0.164 to 0.166, own value 6,145 to 6,613, extractors at 6 minutes 2.4 to 2.6.  No
  outside test against this script at 750.
* iteration 1000 (about 23:17), against its iteration 0: 1 win, the score 0.166 (+0.015 +- 0.007, 2.3 SE), own value
  at the end 4,374 to 6,662 (+9.8), income over the game +5.3, damage dealt -0.0 (taken -2.7), extractors at 3
  minutes 2.4 to 3.4 (+7.7) and at 6 2.1 to 2.5 (+2.4, the script's 11.2), games 62 s longer.  Learning by the
  rule (the score, own value, income, extractors).  On its 750, by the means: nothing gained, the score 0.1655 to
  0.1664, own value 6,613 to 6,662, income 14,860 to 14,800, extractors at 6 minutes 2.6 to 2.5.  So 0.160,
  0.164, 0.166 and 0.166 at 250 to 1000, with 1 win each: what it gained it had by 750.

### What happened in it

* **Down with WSL from 08:37 to 15:08, and carried on from iteration 360** (the user, 15:05: "The training run has
  stopped and PC has been restarted. Resume the run").  Every process in WSL stopped within 5 s at 08:37:37, the run at
  iteration 372 with no error in its log, 25 GB available and nothing from the watchdog; Windows took the VM's network
  adapter down at 08:38:40 and logged no crash or power event then.  So WSL's VM went down under the run; by what, the
  logs don't say.  Carried on by `results/resume-rdnn.sh` from `ckpt/it-000360.pt` (12 iterations lost), with the
  guard, the dashboard, the memory log and the watchdog started again as `start-rdnn.sh` starts them.  Its end is
  unchanged, 2026-11-19 05:23 (mine: the user said "resume"), so it trains about 17.5 hours of its 24.  The logs were
  cut at 360 first, the trainer appending to them (`ppo/metrics.jsonl`, `ppo/minibatches.jsonl`, `logs/ppo.out`; the
  originals beside them as `*.before-resume`).  It is this run's one restart: every game anew at 360, so the test at
  500 has young games 140 iterations behind it, and `rdnn-1`'s bursts of KL came 10 to 14 iterations after one.

### The hourly checks

A line or two each, from 2026-11-18 05:23; the times are local.

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
* 07:49, iteration 299 (5.12M decisions): 469 decisions a second over the last 50 iterations (544 the 50 before;
  minibatches of 166 decisions as frames grow); the KL 0.006 an iteration with the scale whole; the recurrent
  weights' step 0.007%, the state's mean size 0.028; BUILD 12-21% of actions over 200-279, 26% at 280-289 and 32% at
  290-299, the area mex 3-4% throughout; the critics at result 0.66, economy 0.82-0.89, the three older leads
  0.76-0.90, `production_lead` 0.73-0.75, score 0.57, map 0.35-0.39, combat and experience under 0.1; the replays
  at 263 (4 games) with 3.2 extractors at 6 minutes (the script's 7.5), 16 constructors at most and metal floated
  21% of the time (3% at 145); the card 7.4 GB, a collector 1.49 GB of its 2.0, WSL 25 GB available (in use up 1.1
  GB an hour, the learner and each collector 0.3 GB of it); no error, the watchdog only watching.
* 15:11, iterations 361-363, the first after the restart: 182, 308 and 642 decisions a second as the games begin
  (0.2 to 0.8 actions a decision), the KL 0.0002-0.0008 an iteration, the state's mean size 0.017 (at 360, before
  the stop: 0.009, the recurrent weights' step 0.001%, 373 decisions a second); the card 3.7 GB and rising, WSL 28
  GB available; no error.
* 15:49, iteration 420 (7.20M decisions), 60 iterations after the restart: 507-556 decisions a second at the last
  five (446 over the last 50, as over the 50 before); no burst of KL after the restart, 0.002-0.007 an iteration by
  tens and 0.010 at most, the scale whole; the recurrent weights' step 0.007%, the state's mean size 0.018; BUILD
  27-36% of actions over 300-359, 22% in the ten after the restart and 31-40% since, the area mex 2-3%; the critics
  at result 0.65-0.73, economy 0.76-0.87, the three older leads 0.77-0.91, `production_lead` 0.69-0.73, score
  0.55-0.67, map 0.38-0.45, combat 0.21-0.42 (under 0.1 at 299), experience 0; the replays at 358 (4 games, before
  the stop) with 1.8 extractors at 6 minutes (the script's 10.0), 8 constructors at most, metal floated 24% of the
  time; no test since 250 (500 at about 16:35); the card 9.0 GB, a collector 1.47 GB, WSL 25.6 GB available; no
  error, the watchdog only watching.
* 16:49, iteration 500 (8.57M decisions), its test above: 401 decisions a second over the last 50 (443 the 50
  before; 271-433 at the last twelve, minibatches of 98-161 decisions, a collection 30-56 s); the KL 0.003 an
  iteration over the last 50 (0.0005-0.005 by tens, 0.018 at most, at 491), the scale whole; the recurrent weights'
  step 0.0014% over the last 20 (0.0006% at the last five), the state's mean size 0.006 (largest 0.10); BUILD
  34-44% of actions by tens over 420-499, the area mex 2.1-2.6%; the critics at result 0.74-0.78, economy
  0.87-0.91, the extractor and idle leads 0.86-0.92, the constructor lead 0.64-0.72, `production_lead` 0.66-0.77,
  score 0.55-0.59, map 0.43-0.47, combat 0.33, experience 0.09-0.26; other kinds 0 to 500, ramped in from here;
  the replays at 450 (4 games) with 2.8 extractors at 6 minutes (the script's 9.8), 6 constructors at most, metal
  floated 27% of the time; the card 9.3 GB, a collector 1.71 GB of its 2.0, WSL 24.7 GB available (in use up 0.9
  GB in the hour); no error, the watchdog only watching.  The test at 750 at about 20:00.
* 17:49, iteration 578 (9.91M decisions): 375 decisions a second over the last 50 (401 at the last check; 318-390
  at the last six, minibatches of 102-126 decisions); the KL 0.0008 an iteration over the last 50 with the scale
  whole: 0.007 over 510-519 (0.012 at most), then 0.001 and under, 0.0000-0.0004 from 560 (`gru-1`'s was
  0.0005-0.001 where it stood still), the actor's gradient norm 0.48 (1.26 at 500); the recurrent weights' step
  0.0003%, the state's mean size 0.004 (largest 0.12); BUILD 36-45% of actions by tens, the area mex 2.2% to 519
  and 1.2-1.4% from 520; the critics at result 0.76-0.82, economy 0.80-0.87, the extractor and idle leads
  0.82-0.89, the constructor lead 0.74-0.77, `production_lead` 0.72-0.83, score 0.39-0.55, map 0.27-0.38, combat
  0.31-0.38, experience 0.18; other kinds 1% of games (free-for-alls ended in 5 of the last 50 iterations); the
  replays at 531 (4 games) with 1.5 extractors at 6 minutes (the script's 9.8), 16 constructors at most, metal
  floated 0% of the time; no test since 500 (750 at about 20:00); the card 8.9 GB, a collector 1.71 GB of its 2.0
  (39 splits), WSL 24.5 GB available (in use up 0.3 GB in the hour); no error, the watchdog only watching.
* 18:49, iteration 656 (11.25M decisions): 370 decisions a second over the last 50 (375 at the last check); **a
  burst of KL with no restart before it**: 0.0000-0.0005 an iteration over 560-622, then 0.005 at 623-625, 0.05 at
  626, 0.10 at 627 and 0.034 to 0.010 over 628-632, the guard taking the scale to 0.30 over 629-633 and giving it
  all back by 636; 0.001-0.003 since.  The recurrent weights' step went from 0.0003% to 0.008% at 629 and is
  0.0009% now; the state's mean size 0.004 to 0.010 and 0.005 now (largest 0.07-0.09).  BUILD 39-45% of actions by
  tens, the area mex 1.1-1.3%; the critics at result 0.71-0.83, economy 0.86-0.89, the three older leads
  0.84-0.93, `production_lead` 0.84-0.89, score 0.73-0.79 (0.39-0.55 at 578), map 0.63 (0.27-0.38), combat
  0.42-0.53, experience 0.37-0.46 (0.18); other kinds 3% of games; the replays at 613 (4 games) with 2.0
  extractors at 6 minutes (the script's 12.2), 15 constructors at most, metal floated 14% of the time; no test
  since 500 (750 at about 20:00); the card 9.4 GB, a collector 1.71 GB of its 2.0, WSL 23.5-24.4 GB available; no
  error, the watchdog only watching.
* 19:49, iteration 731 (12.54M decisions): 396 decisions a second over the last 50 (370 at the last check; 244-339
  at the last six, a collection 37-65 s); the KL 0.0005-0.002 an iteration by tens over 650-719 and 0.0001-0.0003
  from 720, as before the burst at 626, the scale whole; the recurrent weights' step 0.0002%, the state's mean
  size 0.003 (largest 0.05); BUILD 31-40% of actions by tens, the area mex 1.1-1.2% to 679 and 0.7-0.9% since;
  the critics at result 0.81-0.84, economy 0.85, the extractor and idle leads 0.86-0.93, the constructor lead
  0.59-0.70, `production_lead` 0.82-0.85, score 0.72, map 0.57-0.59, combat 0.72-0.78 and experience 0.73-0.76
  (0.42-0.53 and 0.37-0.46 at 656); other kinds 5% of games; the replays at 688 (4 games) with 2.8 extractors at
  6 minutes (the script's 11.2), 2.5 constructors at most, metal floated 29% of the time; no test since 500 (750
  at about 20:05); the card 9.2 GB, a collector 1.71 GB of its 2.0, WSL 24.1 GB available (in use steady at 16
  GB); no error, the watchdog only watching.
* 20:49, iteration 808 (13.86M decisions), its test at 750 above: 403 decisions a second over the last 50 (396 at
  the last check; 288-424 at the last six); the KL 0.0000-0.0001 an iteration over 720-799, the policy standing
  still for 80 iterations, and 0.002-0.004 from 800, the scale whole, the actor's gradient norm 0.58; the
  recurrent weights' step 0.0005% (0.002% at 808), the state's mean size 0.003 (largest 0.06); BUILD 36-47% of
  actions by tens, the area mex 0.6-0.8%; the critics at result 0.82-0.84, economy 0.90-0.95, the three older
  leads 0.91-0.94, `production_lead` 0.85-0.91, score 0.69-0.71, map 0.51-0.55, combat 0.47-0.52 and experience
  0.38-0.42 (0.72-0.78 and 0.73-0.76 at 731); other kinds 7% of games (a chicken game, a free-for-all and a team
  game ended in the last 50 iterations); the replays at 770: three lost by the lead rule in 6 to 8 minutes, and
  one on Acidic Quarry drawn at the 80 minutes' limit, the first not lost (the script with 1 extractor all game,
  the policy 3, 159 constructors and 359 units, value 67,911 to 47,871); the card 9.1 GB, a collector 1.71 GB of
  its 2.0, WSL 23.9 GB available; no error in the run's log, the watchdog only watching.  The status's
  `results/replay_summary.py` (no torch) stopped on that game, a long one being kept as pages (`NAME.pN.json.gz`
  with the frames' units): it reads them now.  The test at 1000 at about 23:20.
* 21:49, iteration 884 (15.16M decisions): 386 decisions a second over the last 50 (403 at the last check; 315-401
  at the last six); the KL 0.001-0.002 an iteration by tens over 800-829 and 850-869, 0.0001-0.0005 between and
  since (0.005 at most), the scale whole; the recurrent weights' step 0.005% over the last 20 (0.0004-0.003% at
  the last six), the state's mean size 0.008 (largest 0.08); BUILD 33-36% of actions over 800-859, then 29%, 21%
  and 26% by tens, the area mex 0.7% down to 0.2-0.3%; the critics at result 0.60-0.75 and score 0.43 (0.82-0.84
  and 0.69-0.71 at 808), economy 0.79-0.92, the three older leads 0.81-0.95, `production_lead` 0.81-0.87, map
  0.53-0.67, combat 0.63, experience 0.78-0.82; other kinds 9% of games; the replays at 852 (4 games, 28 minutes
  a game) with 1.8 extractors at 6 minutes (the script's 7.2), 21 constructors at most, metal floated 21% of the
  time; no test since 750 (1000 at about 23:20); the card 9.1 GB, a collector 1.71 GB of its 2.0, WSL 23.9 GB
  available (in use steady at 16 GB); no error, the watchdog only watching.
* 22:49, iteration 963 (16.54M decisions): 406 decisions a second over the last 50 (386 at the last check; 265-403
  at the last six, minibatches of 76-120 decisions); the KL 0.0003-0.002 an iteration by tens (0.005 at most) and
  0.0000-0.0003 at the last five, the scale whole, the actor's gradient norm 0.30; the recurrent weights' step
  0.0001%, the state's mean size 0.002 (largest 0.03), its smallest yet; BUILD 15-29% of actions by tens over
  870-963 (33-45% before 860), the area mex 0.1-0.4%; the critics at result 0.85-0.94, economy 0.94-0.96, the
  three older leads 0.92-0.95, `production_lead` 0.93, score 0.67-0.78, map 0.49-0.61, combat 0.18-0.31 and
  experience 0.37-0.43 (0.63 and 0.78-0.82 at 884); other kinds 11% of games (team games ended in 4 of the last
  50 iterations, free-for-alls in 5, chicken games in 2); the replays at 935 (4 games) with 1.5 extractors at 6
  minutes (the script's 7.2), 7.5 constructors at most, metal floated 15% of the time; no test since 750 (1000 at
  about 23:20); the card 8.4 GB, a collector 1.71 GB of its 2.0, WSL 22.4 GB available (in use up 1.5 GB in the
  hour to 17.7, each collector's own memory up 0.4 GB as the other kinds come in); no error, the watchdog only
  watching.
* 23:49, iteration 1042 (17.91M decisions), its test at 1000 above: 460 decisions a second over the last 50 (406
  at the last check; 419-506 at the last six); the KL 0.0000 an iteration over 960-999, the policy standing still
  again, and 0.0001-0.0015 by tens since (0.007 at most), the scale whole; the recurrent weights' step 0.0002%,
  the state's mean size 0.002-0.003 (largest 0.04); BUILD 25-37% of actions by tens over 950-1009 and 21-22%
  since, the area mex 0.2-0.4%; the critics at result 0.74-0.86, economy 0.79-0.85, the three older leads
  0.80-0.93, `production_lead` 0.78-0.85, score 0.64-0.71, map 0.52, combat 0.44-0.53, experience 0.61-0.64;
  other kinds 13% of games; the replays at 1012 (4 games) with 2.0 extractors at 6 minutes (the script's 9.2), 6
  constructors at most, metal floated 5% of the time; the card 9.4 GB, a collector 1.71 GB of its 2.0, WSL 22.0
  GB available (in use 18.1 GB, up 0.3 GB in the hour; the learner's own memory 4.3 to 5.3 GB at about 23:00); no
  error, the watchdog only watching.  The test at 1250 at about 02:25, its last: the run ends near iteration
  1,470.
* 00:49 on the 19th, iteration 1117 (19.22M decisions): 390 decisions a second over the last 50 (460 at the last
  check; 290-437 at the last six); the KL 0.0000 an iteration over 1070-1089 and 0.0002-0.0016 by tens otherwise
  (0.004 at most), the scale whole; the recurrent weights' step 0.0007%, the state's mean size 0.003-0.005
  (largest 0.05); BUILD 39% of actions at 1050-1059 and 19-31% otherwise, the area mex 0.1-0.4%; the critics at
  result 0.74-0.88, economy 0.84-0.90, the extractor and idle leads 0.85-0.96, the constructor lead 0.80-0.82,
  `production_lead` 0.77-0.84, score 0.61-0.72, map 0.45-0.47, combat 0.27-0.42, experience 0.39-0.59; other
  kinds 15% of games; the replays at 1100 (4 games, 27 minutes a game) with 1.8 extractors at 6 minutes (the
  script's 10.0), 5.5 constructors at most, metal floated 11% of the time; no test since 1000 (1250 at about
  02:35); the card 9.6 GB, a collector 1.71 GB of its 2.0, WSL 21.9 GB available (in use steady at 18.2 GB); no
  error, the watchdog only watching.

## Earlier runs

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

## What changed under the runs

* **What the sim models** is in 01-simulator.md, and since the first runs: shots as projectiles aimed ahead and the whole
  of Zero-K's tactical AI ("Weapons and damage", "Unit AI"); no side going first ("Unit AI", "Validation"); Zero-K's end
  of a game, an alliance with no builder or factory left ("Teams and alliances"); plates and idle caretakers, and an
  economy script that builds them, a second factory at 40 metal a second and defences ("Economy", "The economy
  script"); water weapons, floats and sonar ("Water", "Sensing"); air transports ("Transports"); chicken defense; 12
  stats a team ("Teams and alliances").  The libraries: libzksim ABI 31, libzkscripts 17, libzkrecord 2, libzkpolicy 4.
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
* **An attack that's over leaves no order in a unit's record** (the user, 2026-11-19; from `rdnn-4`):
  `World::end_attack` where `run_commands` and a bomber's `used_ammo` ended one, and `emit_view` for a target dead
  before the unit's own update; `rawframe.from_host` the same.  Till then the record kept ATTACK_UNIT and its target
  till the next order, while the unit's flag said idle.  Frames and recordings from before read the old way; no
  column moved and no ABI number changed.
* **A build sticks, and a cancel ends one** (the user, 2026-11-19, after `rdnn-5`'s checks: "make builds stick, but
  also add a 'cancel build' order as well.  On 'cancel build', the builder's order(s) are explicitly canceled").  The
  policy's kind 12, `rawframe.CANCEL_BUILD`, the one kept in reserve for a repair: no tensor or head changed shape,
  `FEATURE_VERSION` stays 10 and no ABI number changed.  A unit on a build (its record's order a BUILD, and not idle)
  is offered the cancel alone, and no other unit is offered it: `featurize_kernels.masks` and libzkpolicy's `masks`,
  only where the cancel is among the kinds (`policy_sim_env.KINDS` has it now; a saved run's own list hasn't).  The
  env sends the cancel as the sim's stop and ends the unit's area mex (`sim_orders`, `_take_areas`), and keeps the
  policy's group orders off units on a build (`_spare_builds`, `SimEnv.builds_stick`).  In the sim a stop leaves no
  order in the record (`World::order_unit`; till then it left kind 0, and nothing sent one).  The actor's score for
  picking a unit on a build starts 3 nats down (`policy_model.CANCEL_START`).  02-policy.md, "A build sticks".
  Checked: the suite (the area mex's test has the rule, the group order and the cancel), and libzkpolicy against the
  Python on the corpus's 49,451 frames, no mismatch (`/mnt/d/zk-learned/parity/check-sticky.json`).
* **Several told to build one thing build one, and a RECLAIM is an area's** (the user, 2026-11-19; libzksim ABI 32; no
  run has had either).  A BUILD that arrives where an unfinished frame of its type and allyteam stands, within the
  type's footprint of the order's place, works on that frame (`World::frame_to_join`), and BUILDs of one type told to
  one point in a decision are given one place, an extractor's spot among them (`SimEnv._placed`): before, each
  constructor placed its own, and two extractors told to one point went to two spots.  A RECLAIM takes what lies in
  its area, the nearest to the unit first, and ends when nothing's left: 32 elmos round a feature named
  (`ZK_RECLAIM_RADIUS`, mine), which is how the scripts and the old actor name theirs, or a radius round a place
  (`zksim.reclaim_area`).  Before, it was the one feature.  01-simulator.md, "Economy".
* **`production_lead`'s cap is half a team's metal income** (the user, 2026-11-19; from `rdnn-4`), where it was 30 metal
  a second: `policy_sim_env.PRODUCTION_SHARE`.  The term's numbers before and after don't compare.
* **The lead rule can wait for the draws to go** (the user, 2026-11-19; `--lead-draws`, `rdnn-4` at 0.8):
  `Learner.follow_draws`, `policy_train.DRAW_WINDOW`; `metrics.jsonl` has `draw_share`, `draw_window`, `lead_open`,
  `ended_seats` and `drawn_seats`.

## My choices the user hasn't confirmed

Each is a constant or a flag, easy to change.

* **In what was built after `rdnn-5`** (2026-11-19; docs/learned-policy/06-select-then-command.md and
  05-order-autoencoder.md, each ending with a list for the user to decide).  The user's: selection before a command,
  every order working on several units, the portion, the shared BUILD, the area RECLAIM, places as coordinates, the
  autoencoder frozen and reading no frame, made-up decisions in its data, a policy that reads no unit.  Mine:
  * **in a command**: the roles as its type, any set of the table's 19; 11 radii, the first "the nearest one"; a
    selection that's never empty (the nearest of the roles when the circle has none); a unit that obeyed in no later
    selection; one target for a command that's aimed (whoever's nearest its place), and one transport for a LOAD; a
    MOVE or FIGHT going wherever it's told; 16 commands a decision; 32 elmos as the least area;
  * **in putting the scripts' orders into commands**: the halving of a group that no circle takes exactly, and a
    circle allowed to take in units already on that very order;
  * **in the autoencoder**: a nat for a place 1% of the map off; width 192 and 3 blocks each way; three in ten made up,
    half wholly and half a script's with three in ten of its commands swapped; a made-up command's kind any of the 21
    alike.  The user's since ("The user's choices, and the autoencoder after them"): the pull of 0.1 and a fiftieth of
    decisions emptied (`vae-2`'s), a bigger code, plausible commands.  Mine in those: the code's 128 numbers as 8
    tokens of 16; what's plausible taken role by role from the type table; the decoder held to it as well as the data;
  * **in the collector** ("A faster collector"): a code's pass 128 rows at most, a level's few into the next level's
    for 4,096 unit tokens of padding a pass saved, 20 shapes kept; only the encoder's fields sent, where no critic
    plays; and for `code-2` 6 of the 6 to 8 collectors, their 0.8 GB and the trainer's 0.76 of the card, and a
    checkpoint every 50 iterations;
  * **in the actor**: one spread a number, the same at every decision, 1 at most; its least from 0.1 to 0.03 over the
    run is the user's, and that it falls evenly, by the run's clock, is mine (`code-1`);
    the mean zero at first ("either is fine", the user, of this or a start cloned from the
    scripts' codes); the memory told the code through a linear layer; the KL estimated from the codes drawn; the token
    budget counting a decision as its 64 latents.

* **A build that sticks** (2026-11-19; the user's are the sticking and the cancel):
  * **the start against a cancel**, 3 nats on the choice of unit (`CANCEL_START`).  To pick a unit on a build is to
    cancel it, and an untrained actor picks every unit alike, so without it a build would be cancelled about as often
    as any order used to cut one short (53% at `rdnn-5`'s 100); with it, 17% in the same games;
  * **on a build** is the record's order being a BUILD and the unit not idle, for a unit that moves: so the
    opening's and a script's builds stick too once the policy has the unit, and a unit that's chasing something after
    its build reads as on one till it stops;
  * **nothing else reaches it**: no cloak's order either, and no group order of the policy's (the others in the group
    get theirs); a script's order still does;
  * **the cancel waits out the commit lock**: none in the 3 s after the build order;
  * **no limit of time**: a build that can't be reached or paid for holds its unit till the policy cancels;
  * **the area mex's gap** between two extractors (a step, when the unit is idle and free) is left as it was;
  * **the reserved kind 12** and not a 25th, and its engine command a `stop`.

* **`rdnn-5`** (2026-11-19): from iteration 0 in its own directory, not `rdnn-4` carried on; `--lead-draws 0.8` kept
  beside the user's `--lead-fade 0:0.8`; 24 hours.
* **`rdnn-4`'s three changes** (2026-11-19; the user, on the report that listed them: "This is all fine"):
  * the attack: that the fix is the record's order, the unit's idle flag and the idle term having been right already;
    that it's cleared for a dead target and a bomber out of ammunition too; that the live side does the same;
  * the production cap: half of the income the frame says at that step (extractors, reclaim, what's shared), on each
    team's rate a step; 30 kept as the lead that is 1, which makes the term smaller than it was;
  * the lead rule's wait: draws counted by seat, of each copy of a game that ends (a GRPO group's copies end apart, and
    a game whose every copy but one was eliminated still goes on), over the last 800, nothing said before 800;
    chicken games not counted; on for good once on; tests and the replays against the script keep the rule;
  * 24 hours, as the runs before it.
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
