# zk-ArenaAI, the learned policy: where it stands

Updated 2026-11-16.  Branch `learned-policy-plan`.  Repo: `C:/Users/Thoma/Documents/sources/zk-ArenaAI`.  This file was
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
* **The economy script builds defences** (the user, 2026-11-13; libzkscripts ABI 14): LLTs by contested spots from 2:30
  (before the extractor), Pickets among them against tanks, AA once the enemy flies, and from 30 metal a second a Stinger
  by an extractor with 700 metal of wrecks near it, or a Gauss where turrets are being lost.  Before, it built none but
  Cerberus and Lucifer, so no clone built a turret.  01-simulator.md, "The economy script" ("Defences").  Since 2026-11-16 (the
  user; libzkscripts ABI 17), by a spot or extractor in water it builds an Urchin in place of an LLT or Picket once an enemy
  has been seen under water, which neither can fire at.
* **Weapons know water** (the user, 2026-11-16; libzksim ABI 29), as the engine has them: only a water weapon (torpedoes and
  depth charges, the Dagger's, Gauss's and Ultimatum's guns, the Hunter's, Siren's and submarines') aims at what's under water,
  or from under water at what's in it; a torpedo only at what's in water; other weapons don't fire from under the surface, and
  their shots go off at it.  The amphibious units with a float (Buoy, Grizzly, Bulkhead, Scallop, Archer, Angler) rise to the
  surface to fire, in 1-3 s, by `unit_impulsefloat_toggle.lua`'s arithmetic and the engine's drag in water, and sink when
  they've nothing to aim at.  Checked against the engine's cases and float traces.  A frame's `y` is now a unit's base, so a
  ship's or submarine's is below the surface.  On a map with no ground below 0 the sim plays as before.  01-simulator.md,
  "Water"; "Known gaps" for sonar, which isn't modelled.
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

## Self-play (2026-11-16): a run of 16 hours, stopped at iteration 275

The user, after Pluto's README (self-play alone, 1v1s) and `full-4`'s RL not improving: train against itself, a tenth of
the games against the scripts, 1v1s first and the other kinds mixed in later.  03-training.md, "Opponents".

* **The run** (the user, 2026-11-16: 24 hours, the policy's own economy in every game, after a look for easy speed):
  `results/launch-full4.sh`, the pipeline's `ppo` stage, so `full-4/ppo`, `full-4/logs/ppo.out`, the dashboard at
  `http://localhost:8765`, `full-4/pss.log` and `memguard.log`.  From the DAgger actor, GRPO as `full-4` ended (unbootstrapped,
  the seven terms, `--group-step 2048 --lr-start 1e-5 --kl-target 0.005`), `--self-play 0.9`, `--kinds-ramp 500:0 2500:0.5`,
  `--economy-start 1 --economy-end 1`, `--hours 24`, a test of 128 1v1s against the script every 50 iterations.  Three
  collectors of 16 games at `--collector-gb 1.4`.  Started 14:17; the user stopped it at iteration 275 (2026-11-17 06:21,
  4.80M decisions, `ppo/ckpt/it-000275.pt`, `ppo` not marked done) as not learning.  To go on: remove `full-4/ppo/stop` and
  run the launch file again (with `--ppo-hours` what's left of the 24, about 8).  The script-only GRPO
  is `full-4/ppo-script` now, its replays `replays/ppo-script`.  Its test at iteration 0 (9.4 minutes): 10.9% wins, 1.6%
  draws, a score of 0.245 +- 0.019, extractors at 6 minutes 4.8 against the script's 9.2.  At 50 (16:59, 875,000
  decisions): 10.2% wins, the score's change -0.006 +- 0.021, extractors 5.1 against 9.5.  119 decisions a second over
  iterations 2-50 (`selfplay-test`'s 2-16 made 98, its own 136), about 2.5 minutes an iteration and 15 a test, so 24
  hours end at about iteration 500, all of it 1v1s.  A collector 1.05 GB at most, WSL 15-16 GB, the KL past 0.01 in 10
  iterations of 50, no errors.
* **At iteration 268** (2026-11-17 05:56, 4.67M decisions, 15.6 hours; the user: "I don't think it's learning").  Tests at
  100, 150, 200 and 250: wins 11.7%, 17.2%, 13.3%, 13.3%, the score's change +0.017, +0.045, +0.017, +0.028, each +- 0.02.
  Game by game (`ppo/tests.jsonl`), the mean of those four less iteration 0's: the score +0.027 +- 0.017, the result +0.036
  +- 0.025, income share +0.019 +- 0.007, extractors at 6 minutes +0.55 +- 0.20 (4.77 to 5.63 at 250, the script's
  9.2-9.5), games 166 s +- 61 longer.  So a little economy, about 2.7 SE on the best of several measures looked at, and no
  wins a test can show.  The first RL run here whose tests are all above its start, and far too slow to matter.  In
  training nothing trends: the KL 0.006-0.008 an iteration with the guard slowing the rate in 201 iterations of 268 (so the
  step isn't what limits it), about 450 branches (56 groups) an iteration, 1.3-2 new games.
* **Where the time goes** (py-spy on a run's processes with `results/profile-run.sh`, as root; 03-training.md, "Playing"):
  the GPU, about 90% busy with the learner and the collectors on it.  A collector of 16 games waited on its passes for
  about 75% of its time, with the sim 2% and featurizing 4% (one of 48 games: 60%, and featurizing 20%); the learner
  waited on its data workers for 20%, and copied every gradient before each minibatch for 12%.
  * Kept: the learner asks for the next minibatch's decisions while the GPU is on this one's, and takes a minibatch's
    gradient apart from the parameters' (no copy).  With them the run made 163 and 156 decisions a second overall at
    iterations 2 and 3, where `selfplay-test`, the same but for them, made 124 and 124: training at 166 against 129-132,
    and collecting at 171 and 154 against 128 and 121, with the GPU the learner leaves.
  * Not kept: fewer, bigger collectors.  Runs of 5 iterations from the DAgger actor, no test, decisions a second overall
    at iterations 4 and 5: three of 16 as it was 119 and about 105; one of 48 128 and 102; two of 24 with the learner's
    changes 138 and 136.  So the run started with two of 24 at `--collector-gb 2.0` (13:42, now `full-4/ppo-2x24`).
    But those runs had no test, and a test's long games widen every pass after it: with it, two collected 108 decisions
    a second at iteration 2 where three had made 128, the learner waiting 70 s of the iteration's 158, and the test took
    24 minutes, not 13.  Collecting is the limit, and three processes get more of the GPU and of the CPUs than two.
    Restarted at 14:17 with three of 16.
  * Tried and taken out: a pass takes about 0.25 s by iteration 5 whatever its rows (16, 24 or 48) and however many
    collectors.  A `Graphs` for each class of frame size, small frames in wide passes and wide ones in narrow, made each
    pass no faster and a round more of them (98 decisions a second collecting against 147).  No stops among the action
    steps (one graph for all 16) made it no faster either.  So more rows a pass is what collects faster, and a collector's
    memory, which grows with rows times the widest frame it has played, is what limits that.
  * `full-4/selfplay-profile`, `selfplay-ab-*` and `ppo-2x24` are those runs: scratch.

* **`--self-play 0.9`** (the pipeline's `ppo` stage now): nine new training games in ten the policy plays every player's
  team; the rest it plays team 0 against the scripts, as before.  The gate it replaces (self-play once two tests averaged
  80% wins) never opened.  Under GRPO each side's branches are judged against the same side's in the group's other copies.
* **`--kinds-ramp 500:0 2500:0.5`** (the pipeline's): the share of new games that aren't 1v1s, by iteration: none till 500,
  then up to half at 2,500, split among team games, free-for-alls and chicken games as `--kinds` weighs them.  It counts
  the learner's iterations, which start at 0 with `--init` and carry on with a resume, as the lead rule's fade does.
* **A pass is a seat a game at most**: against itself a round has 2 seats a 1v1 and up to 8 a team game, and a collector's
  graphs grow with a pass's rows, so a round goes in passes of the collector's games (16).
* **The test** (`results/launch-selfplay-test.sh`, `full-4/selfplay-test`): 16 iterations of 16,384 from `full-4`'s DAgger
  actor, with `full-4`'s last GRPO flags (unbootstrapped, the seven terms, `--group-step 2048 --lr-start 1e-5 --kl-target
  0.005`).  It ran through.  Of the games ended, 266 copies were against itself and 61 against the scripts (all lost).
  82-124 decisions a second overall (the script-only run's first 16: 82-145), a collector 0.95-1.05 GB of its 1.4, WSL
  17.2 GB at most, the KL 0.002-0.025 an iteration with the guard slowing the rate from iteration 8 to 15.  Tests of the
  same 128 1v1s against the script: 14.8% wins at 0 and at 16, the score's change -0.009 +- 0.024.  16 iterations say
  nothing about learning, and weren't meant to.
* **About 3 minutes an iteration**, so iteration 500 is about a day in and 2,500 about 5 days, tests (13 minutes each) on top.
  A game is played through as one group of 8 copies, about 2 new games an iteration.
* **The other kinds' check** (`full-4/selfplay-mixed`: `KINDS_RAMP=0:0.5 ITERATIONS=3`, no test): it ran through, with
  free-for-alls ended and judged (`wins_mirror_ffa` a third), at 131-201 decisions a second.  Its collectors held 0.5-0.6
  GB, but it had no test's late frames: `full-4`'s passes of the same 16 rows, of every kind, reached 1.07 GB.
* **Seen in both, and in `full-4`'s script-only GRPO** (13 of its 141 iterations): `adv_abs_max` in the thousands.  In an
  iteration where no group's copies differ in who won, the result term's scale (its mean spread over the groups where it
  varied) is about 1e-4, and a decision taken after other copies of its game had ended is a whole result from its
  group's mean from that moment (`_judge`, reward-to-go).  The clip at ±3 bounds it.  Not changed: my reading of the
  code, not reproduced in a test.
* **The scripts and the sim are being changed** by another agent (the user, 2026-11-16): `~/zksim/build` was rebuilt at
  11:24, after the test and before the other kinds' check.  A long run plays the build it starts with.
* **My choices, listed for the user before they asked for the run, and not changed by them**: the method (GRPO as `full-4`
  ended, not PPO with the critic) and the start (the DAgger actor, not `ppo-script/ckpt/it-000141.pt`); the lead rule
  fading as before (`--lead-fade 0:0.9 100:0`); the ramp's share split by `--kinds`' weights (two fifths team games, two
  fifths free-for-alls, a fifth chickens); both flags on by default in the pipeline's `ppo` stage.

## `runs/full-3` (started 2026-11-13 21:44)

The user's next whole run, `results/launch-full3.sh` (the same command again carries on):

    policy_pipeline.py --root /mnt/d/zk-learned/runs/full-3 --train-encoder --data /mnt/d/zk-learned/clone/army-8         --encoder-steps 75000 --clone-steps 10000 --dagger-rounds 1 --ppo-hours 10 --ppo-decisions 100000000         --ppo-iteration 16384 --record-args "--workers 4 --threads 16"         --ppo-args "--group 8 --collectors 4 --concurrent --economy-start 1 --economy-end 1 --eval-every 50"

* Recorded anew (`army-8`) on the sim as it is now, with the economy script's defences: the encoder 3x its default steps,
  the clone 2x `full-2`'s, one DAgger round (the user), then GRPO for 10 hours with the policy's own economy in every game.
* 16,384 decisions an iteration, 4x the trial's: 55 groups scored an update (03-training.md, "GRPO from branched games",
  "Memory, and bigger batches").  To change GRPO's flags, stop the pipeline before its `ppo` stage and run it again with
  them: the stages done are kept.
* Deleted for room, the user asked: `clone/army-7`, and `full-2`'s `frames`, `ppo`, `ppo-kl-127` and `dagger`; on
  2026-11-14, during GRPO, `full-3/frames`' shards (118 GB).  Its `manifest.json` is kept so that the pipeline, run again,
  doesn't plan a `frames` stage (it's planned whenever the manifest is missing, and no longer in `pipeline.json`'s done).
* Its recording thrashed WSL at 100% of its memory at first (the user saw it, 2026-11-13 22:05): a recorder worker of 64 games now holds about
  2.5 GB beside the map pool's 14.7 GB, against about 1 GB in `full-2`, and 12 of them don't fit in 40 GB.  Killed, and
  relaunched at 22:17 with 4 workers of 16 threads (`--record-args`, new in the pipeline), and
  `results/mem-guard.sh` watching: it kills the run when less than 4 GB is left (`full-3/memguard.log`; 8 GB at first, but
  recording leaves about 15 and GRPO's measurement 12, so 8 risked killing a healthy stage overnight).
* Recorded in 89 minutes (8,800 games or so, 100M decisions, about 1,775 bytes a decision against `full-2`'s 1,420), frames
  in 29.  The encoder levelled off from about step 10,000 (loss 0.02-0.04, the probes' R² flat), so the user had it
  stopped at step 20,600 of 75,000 (2026-11-14 02:16, 2 hours in, the rate still 4e-4): its `stop` file, then
  `encoder` marked done in `pipeline.json` by hand and the same command run again.
* GRPO started at 13:42 on 2026-11-14 and ran to iteration 105 (1.79M decisions), tests winning 17.2% of 64 1v1s at 50 and
  7.8% at 100.  WSL's memory went from 26 GB to 32 in its first hour, then about 1 GB an hour more as the lead rule faded
  (none from iteration 100), to 35.5 of 40, and `mem-guard.sh` killed it at 17:34 under 4 GB free.  Its rates fell as games
  grew: collecting from 415 decisions a second to about 150, training from 340 to about 140.  Resumed at about 18:40 from
  iteration 100 (the logs cut there, the originals kept as `*.before-resume`) with 3 featurizing workers, not 6 (each about
  1.5 GB, and far from the bottleneck), and `--hours 6.3`, 10 in all; `results/pss-log.sh` logs each process's memory
  every 10 minutes to `full-3/pss.log`.
* Minutes after, WSL held 33.5 GB: the collectors 4-6 GB each (3 GB in the measurement), with games no longer ended by a
  lead, and training the limit (60-85 decisions a second against collecting's 125; the 3 featurizing workers at 13% CPU).
  Stopped at iteration 111 by its `stop` file (a checkpoint saved) and resumed with 3 collectors of 16 games (48; games
  must be a multiple of `--group` times `--collectors`) and `--hours 5.85`, the 10 hours in all kept.
* GRPO moved the policy about 3x less a decision than the trial's PPO (KL 0.0006 per 1,000 decisions against 0.0019): 0.8
  optimizer steps per 1,000 decisions against 9.7, at the pipeline's rate of 1e-5, and the cloning loss (weight 0.095 at
  iteration 122, about 0.31 of the loss against the policy loss's 0.06) and the KL to the clone (about 0.04) holding it to
  the scripts' play.  So, the user (2026-11-14, after cameronrwolfe.substack.com/p/grpo-tricks, whose methods drop the
  KL penalty): from iteration 123 or so `--kl-start 0 --clone-weight 0 --kl-target 0 --group-step 256`, no KL loss, no
  cloning loss (the scripts no longer advise), no KL guard, and an optimizer step a group or so.  About 40 optimizer steps an
  iteration, but the KL an iteration fell (0.002-0.008): the cloning loss had been what moved it.  Twice as fast (170-385
  decisions a second), as the clone's forward pass and the scripts' advice went.  The test at 150 won 10.9%, as the DAgger
  actor had.
* `mem-guard.sh` killed it at 20:15 when I ran tests and a parameter count beside it (resumed from 140), and the user then
  stopped it: iteration 141 (2.41M decisions) is its last checkpoint, `ppo` not marked done.  The user asked for a critic-free
  method's review (arXiv 2508.08221) before the next RL stage.

## `runs/full-4` (2026-11-14)

The user's next run, with a bigger model, `results/launch-full4.sh`, started 2026-11-14 20:56: the encoder 192 wide (was 144)
with 64 latent tokens (was 40), and the actor `policy_latent.WIDE`, 384 wide in 12 heads (was 256 in 8) with two layers of
self-attention among the controllable units (`Size.unit_layers`, `LatentActor.among`): 27.0M trained parameters, was 10.5M.
On `army-8`, frames collected anew, the encoder at its default 25,000 steps, the clone 10,000 (`--clone-args "--size wide
--pass-gb 4.5"`), one DAgger round, then GRPO.  The pipeline's `--encoder-args` is new for it.
* `--pass-gb 4.5` for the clone and DAgger (`results/wide-smoke.sh`, 60 steps from a random encoder): at the default 7.0 the
  wide actor's passes split 19 times and the allocator retried 258, at 382 decisions a second; at 4.5 none split, 2
  retries, 410-585 a second.  So its clone takes about 5-7 hours more than `full-3`'s 553 minutes.
* `results/stop-before-ppo.sh` puts a `stop` file in its root when the eval stage starts, so it halts before GRPO.  Then,
  before GRPO: `policy_train.py --grad-check 16384` from the DAgger actor (03-training.md, "GRPO from branched games",
  "Signal or noise"; without `--bootstrap`, as the critic is untrained then), which decides `--group-step` and `--lr-start`,
  and a short GRPO run to size the wide actor's memory with 3 collectors and a test of 128 games (below).  GRPO then starts with what the review of arXiv 2508.08221 brought (the user,
  2026-11-14): tests of the same 128 games each time, scored and paired; the economy as the change from each decision;
  no entropy bonus; advantages clipped at ±3 and SPO's pull bounded.  And bootstrapped (`--bootstrap`, the user's choice
  over alternating GRPO and PPO by the hour): each branch's return to its stretch's end plus the critic's value of where it
  ended, the critic learning in the learner only, the actor waiting 10 iterations for it; the actor's and the critic's
  gradients clipped apart; and resignation as AlphaGo Zero's (`--resign`), with how much play goes to lost games logged
  (`hopeless80`).  `full-3`'s GRPO had none of it.
* The memory guard killed its encoder at step 5,100 (2026-11-14 22:01): I ran a CPU smoke run of GRPO through collectors
  beside it, whose main process grew to 31 GB.  Relaunched at 22:02, the encoder from step 0 (it saves only at its end; the
  killed one's log is `encoder-killed-5100/`).  Run again alone at the stop (2026-11-15), with stacks dumped as it grew: it's
  the learner's first training step on the CPU, the time transformer's dense attention (`policy_latent.py`,
  `F.scaled_dot_product_attention` with a mask, which on the CPU keeps every score), 1 to 23 GB in 6 s.  CPU-only: on the
  GPU it's a fused kernel in the card's memory.
* Done in 637 minutes, the clone (held out: unit kind 0.971, anchor 0.910) played as `full-3`'s: record-1 15.6% / 4.7% /
  79.7% of 64 mixed games (`full-3` 15.6% / 6.3% / 78.1%), extractors at 6 minutes 5.3 against the script's 8.8.  DAgger's
  actor, 64 1v1s: 9.4% / 3.1% / 87.5% (`full-3`'s 10.9% / 1.6% / 87.5%), extractors 4.36 against 9.89.  The wide actor
  bought nothing in play by cloning.  The clone's memory spiked by 4-6 GB for a minute twice (02:09, 10:05), to 4.0 GB
  available; the guard's floor is 3 GB since 07:48 (the user: 3-4 GB available is fine).
* Before GRPO (2026-11-15): the gradient check from the DAgger actor, unbootstrapped, found noise (03-training.md, "Signal
  or noise": all -0.03 against shuffled -0.06), so `--group-step 4096 --lr-start 3e-6`.  The A/B, 3 iterations each:
  bootstrapped with resignation, 125-130 decisions a second overall (training 128-141, the learner's 4.8 GB of the card
  full: minibatches split, the token budget down to 72,900); unbootstrapped 235-245.  WSL at 29.0 GB at most with a test.
  Its first try ran a collector out of its 1 GB in the iteration-0 test (128 games, 43 a collector): tests through
  collectors now go in chunks of a collector's training share (16; 03-training.md, "Opponents"), and `--collector-gb 1.4`
  (a collector held 1.07 GB).  A test of 128 takes 13-16 minutes.
* GRPO started 2026-11-15 15:08 (`results/launch-full4.sh`, `full-4/pss.log` every 10 minutes): 10 hours, bootstrapped,
  resigning, the actor waiting 10 iterations for the critic.  `full-4/ab-*` and `full-4/grad-check` are the checks'.
* Stopped at iteration 120 (2026-11-15 22:06, 2.03M decisions, about 6.9 of its 10 hours; `ppo` not marked done), the user
  asked, for a bootstrapped gradient check (`full-4/grad-check-120`).  By then the critic had learnt (explained variance
  0.7-0.8 from iteration 60), but the policy barely moved (KL 0.0006-0.0008 an iteration, 4 optimizer steps of 3e-6), and
  tests of 128 games won 15.6%, 10.9% and 14.8% at 0, 50 and 100, the score's change from iteration 0 -0.029 +- 0.020 and
  -0.003 +- 0.023.  The check found noise still (03-training.md, "Signal or noise").  The user's next idea: more reward
  terms, closer to the decisions.  Built the same night: `--terms` `extractors`, `map` (the shares of the map in sight
  and radar) and `experience` (the army's trades), from five new team stats (libzksim ABI 27, `ZK_STATS` 11; 03-training.md,
  "Reward"), and `--grad-splits`.  An unbootstrapped gradient check from iteration 120's actor (`full-4/grad-check-terms`,
  8 splits) found each new term noise alone, but the mix of all seven above the floor in every split (0.12 against 0.01),
  where the bootstrapped check of that actor wasn't.
* So GRPO began again at 23:34 (the user asked for the RL phase to go on, and for adjustments if it still didn't learn):
  unbootstrapped from iteration 120's actor (`--init`), the seven terms (`result:1 score:0.5 economy:0.3 combat:0.3
  extractors:0.3 map:0.3 experience:0.3`), `--group-step 2048 --lr-start 1e-5 --kl-target 0.005`, 10 hours
  (`results/launch-full4.sh`).  Its iterations count from 0 again, so the lead rule fades anew.  The bootstrapped run is
  `full-4/ppo-boot` (its replays `replays/ppo-boot`): to resume it, put it back as `ppo` and run the launch file with its old
  ppo-args (in its comments).
* It moved (KL 0.002-0.018 an iteration, the guard slowing it at times), but tests of the same 128 games didn't rise: the
  score's change from iteration 0 -0.022 +- 0.028 at 50 and -0.026 +- 0.023 at 100.  At 50 its economy was up 2-3 SE
  (extractors at 6 minutes 5.27 to 5.92, income at 3 minutes +1.4), but back to 5.06 at 100.  The user stopped it at
  iteration 141 (2026-11-16 07:41, 2.38M decisions, `ppo/ckpt/it-000141.pt`, `ppo` not marked done): WSL held 33-34.5 GB
  of its 40, and Windows, on 48 GB, ran short.  Where it went (smaps, iteration 140): the map pool 13.8 GB, shared by
  all 7 processes through fork; the learner's own 5.7 GB; each collector's own 3.5-4.1 GB, growing with games' length;
  each featurizing worker's own 0.2 GB (their 2.2 GB in `pss.log` is their share of the pool).  A pool map was about 20 MB
  as `zksim.Map`, which kept its numpy arrays (height, speed, slope, segments, routes: about 8 MB) beside the C++ copies.
  So, the user asked (2026-11-16): `zksim.Map` keeps none, the C++ scripts copy no ground, and the sim reads each map from
  a file of the map cache mapped into memory (01-simulator.md, "The map cache"): the pool now takes 1.36 GB of the
  processes' own memory, and the cache's pages are shared and droppable.
* `results/mem-guard.sh` watches it (`full-4/memguard.log`).
* Its frames (118 GB, D: down to 31 GB free) are deleted once the encoder is done, the user asked (2026-11-14), but
  `manifest.json`, as `full-3`'s: `results/delete-frames-after-encoder.sh`, which says what it did in
  `full-4/frames-deleted.log`.

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
  and scripts play its allies and enemies (in self-play, since 2026-11-16, it plays every team).  The headline test stays
  a 1v1.
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
  | `maps/` | the map pool: 204 real (18 of them Violet Rampart) and 200 made-up maps, 31 held out, each with its `modes`.  It loads through the map cache in 27 s, to 1.4 GB of a process's own memory (01-simulator.md, "The map cache") |
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

**The tests** (831 pass, 3 of them need CUDA and are skipped without it): `pytest.ini` runs them on 8 processes, a file's tests in
one process so that what they share is made once, and `tests/policy/conftest.py` keeps each process's maths to one thread.
That took the suite from 211 s to about 60 (2026-11-11), with the slowest learning runs cut down and those of paths the
pipeline doesn't use removed: the variational encoder's fits, checkpoints from before feature version 7, the trainer's
in-process concurrent collection.  `--durations=20` shows what's slow; most of what's left is each process importing torch
and the tools (about 15 s).

In the engine (Windows), recording raw frames from a game: `$env:ZKBOT_FLAGS='rawframe'`, then
`.\zk-startup.ps1 ControllerAI CircuitAI:Normal -Headless -Map 'TitanDuel 2.2' -NoDownload -Client 'python bots\zkbot\main.py {team}'`.
README.md has the rules for engine games: headless at max speed, windowed only when the user asks.

## Next

0. **After the self-play run** ("Self-play" above; stopped 2026-11-17, nothing is running).  The user's idea then: an
   RL-only run again (no clone), with GRU memory, as Pluto has; not built, and not agreed how.  Mine, put to the user and
   not answered: before more RL, find why the clone stops expanding.  Held out, `full-4`'s clone gets the kind right for
   83% of the script's builds (97-100% for fight, move, produce, assist) and a whole build (type, place, offset) costs 7.5
   nats against 0.3-2.4 for the rest (`clone/log.jsonl`); split that by type, place and offset, and by what's built
   (extractors and energy against defences, whose rules look back 3-5 minutes, past the actor's 36 s).  If it's what the
   actor can't see: memory in the features or older frames in the window before a GRU, and the encoder unfrozen for 1,000
   cloning steps against frozen.  For RL: PPO with the per-term critic, a few maps with the lead rule on, a paired test
   against the DAgger actor.
1. **GRPO from branched games** (the user, 2026-11-13, after `full-2`'s PPO got worse), in place of PPO's critic: a game
   copies in the sim and the scripts (libzksim ABI 26, libzkscripts ABI 13) and in `SimEnv` (01-simulator.md, "The batch"),
   and `policy_train.py --group 8` trains by GRPO (03-training.md, "GRPO from branched games").  **The trial**
   (`/mnt/d/zk-learned/runs/grpo-trial`, at FEATURE_VERSION 9), from `full-2`'s DAgger actor, which won 9.4% of 64 1v1s on
   the sim as it is now: GRPO won 14.1%, 15.6% and 14.1% at iterations 25, 50 and 75, PPO 6.2%, 7.8% and 20.3%.  The user
   stopped it at PPO's 75th, its results not needed.  At 64 games a test it says no more than that neither fell; GRPO
   played 1.5-2.5x as fast (no critic).  But GRPO's orders swung from FIGHT to MOVE at iterations 72-79 (03-training.md,
   "It broke at the end"): the per-minibatch scaling in `_update_loss` and the average over a decision's actions the likely
   causes.  Fixed (the user, 2026-11-13, after a survey of GRPO's variants), with reward-to-go within a stretch, each term
   over its spread, and an optimizer step over whole groups (03-training.md, "Advantages", "The update").  Before `full-3`
   reaches GRPO, nothing to do: it starts with them, 16 optimizer steps an iteration of 16,384 (`--group-step` 1,024).
   Watch its `kl` (the trial's was 0.001-0.01 an iteration) and the share of MOVE orders on the dashboard.  Since
   `full-3`, the changes from the review of arXiv 2508.08221 (the user, 2026-11-14): built and tested, for `full-4`'s GRPO
   (above).  Judge it by `eval_fight_score_change` and its `_se`, not the win rate alone.
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
3. **`runs/full-3`** is the next whole run (above), started 2026-11-13 21:44.
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

* **Leftover build directories**: `~/zksim/build` is the one in use (libzksim ABI 29 since water and libzkscripts ABI 17
  since the Urchin, both 2026-11-16; libzkrecord ABI 2).  One built before that needs rebuilding.  The `build-*` and `agent*` directories beside it are leftovers of comparisons and of agents' tree
  copies, safe to delete, except `agent-maps-lupa` (above).
* **The map cache** (`~/zksim/map-cache`, 6.1 GB; `ZKSIM_MAP_CACHE` moves it, empty turns it off): `load_pool` makes a
  map's file again when the map file or libzksim changes.  A change to what's kept, or to how `zksim.Map` makes what it gives
  the sim, needs `zksim.MAP_CACHE_VERSION` raised, or the old files are read.  The tests keep theirs in pytest's temporary
  directory (`tests/policy/conftest.py`).
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
* **A `$` in `wsl bash -c '...'`** is taken by the shell wsl starts first, where the variable isn't set: `F=x; grep y $F`
  reads stdin, and a `$(...)` runs once, before the loop it's in.  Put it in a script file (`results/wait-lines.sh`).
* **`NAME` in WSL** is the PC's name, set for every shell: a launch script that reads `${NAME:-...}` gets it (a run started
  in `full-4/DESKTOP-DS1JCNE`, 2026-11-16).  `results/launch-selfplay-test.sh` reads `RUN_NAME`.
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
