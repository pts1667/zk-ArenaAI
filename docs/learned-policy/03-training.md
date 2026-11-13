# Training

How the policy is trained, from frames to PPO, with the settings in use and the numbers behind them.  The network is in
`docs/model.md` and the sim in 01-simulator.md.  Paths under `/mnt/d/zk-learned` are on D:, outside git.

## The pipeline

`tools/policy_pipeline.py --root DIR [--train-encoder] [--dry-run]` runs every stage in order, each as the tool's own command in
a process of its own.  So no two stages load the map pool at once, and each hands back the GPU when it ends.

| Stage | What | Settings from |
|---|---|---|
| `record` | if `--data` holds no finished recording: the scripts' games recorded there, `--record-decisions` (100M) of them, in the kinds of game `--kinds` says | `army-6` |
| `frames` | with `--train-encoder`, if `--frames` (default `ROOT/frames`) has no `manifest.json`: `frame_data.py collect`, 1M frames each from the frame games, scenarios and the C++ scripts' games, the last in the kinds `--kinds` says | `pool5` |
| `encoder` | with `--train-encoder`: `frame_vae.py train`, 25,000 steps, to `ROOT/encoder`.  Later stages read its `vae-last.pt`; otherwise they read `--encoder` (default `policy_latent.ENCODER`) | `encoder-big` |
| `zindex` | the data's `z.npz`, if it has none | |
| `clone` | `policy_clone_train.py` on `--data` (`army-6`), 10,000 steps, checked and saved every 500 | `clone-8`: `--z --z-dropout 0.2 --economy-weight 2.5 --finish-winners 0.1`, lr 6e-4, Muon 0.008 |
| `record-N` | the last clone plays the army script, told its variant, with the scripts advising it, in the kinds `--kinds` says, as CUDA graphs, until 256 of its games are kept (see "DAgger").  Its first 64 are an eval (`evals/record-N.json`, by kind where there's more than 1v1s) | DAgger |
| `zindex-N` | `z.npz` over every round's games | |
| `dagger-N` | 500 steps from the last clone, half the samples from every round's games and half from `army-6`; lr 2e-4, Muon 0.003 | DAgger |
| `eval` | the last clone, 64 1v1s, the headline: `evals/final.json` | |
| `eval-KIND` | the same in games of each of `--test-kinds` alone: `evals/final-KIND.json` | |
| `ppo` | `policy_train.py --init` the last clone, 64 games of `--kinds`, 4,096 decisions an iteration, 2M decisions (`--ppo-decisions`; `--ppo-hours` caps it), the critic alone for 10 iterations at its own rate; tested in 1v1s and in each of `--test-kinds` | the PPO smoke run, `full-1` |

* **Options**: `--dagger-rounds` (2).  `--clone-args`, `--dagger-args` and `--ppo-args` go at the end of their commands, so they
  override what's set.
* **Kinds of game** (the user, 2026-11-10): every stage that plays games plays what `--kinds` says, by default
  `1v1:0.5 teams:0.2 ffa:0.2 chicken:0.1` (01-simulator.md, "The game in the sim"), in batches with `--slots` teams a game
  (0: what the biggest kind needs) and chicken games at `--chicken-difficulties`.  The tests' headline stays a 1v1 against
  the script, so that runs stay comparable; `--test-kinds` adds a test of each kind it names.
* **State**: logs go to `ROOT/logs/STAGE.out`, and `ROOT/pipeline.json` lists finished stages with their times and exit codes.
* **Re-running**: the same command again carries on from the first stage not done.  `--redo STAGE` runs it and every stage after
  it again.  The encoder and PPO resume from their checkpoints.  A cloning stage starts over, and sets its old `log.jsonl` aside.
  A round recorded again first deletes its own games.
* **Stopping**: a `stop` file in `ROOT` halts the pipeline once the running stage ends.  A `stop` file in `ROOT/encoder` or
  `ROOT/ppo` stops that training at its next report, and the pipeline with it.
* **DAgger rounds** all record into `ROOT/dagger`, each under its own prefix (`--record-prefix game-daggerN`), so each
  fine-tuning draws from every round so far.
* **Dashboard**: while it runs, the pipeline serves a page at `http://localhost:8765` (`--port`, 0 for none), which Windows
  reaches through WSL's localhost forwarding.  It shows the stages, the log as it's written (server-sent events; the
  pipeline's own lines are in `ROOT/logs/pipeline.out`), each stage's results and charts, and replays.  The tests keep their
  first 4 games for it (`--replay-games`), and PPO plays 4 more against the army script every hour (`--replay-hours`), all
  in `ROOT/replays/STAGE`.  The page dies with the pipeline; `tools/policy_dashboard.py --root ROOT` serves it again, during
  the run or after.  Server: `tools/policy_dashboard.py` (its docstring lists the endpoints); page and viewer:
  `tools/dashboard/`; replay format: `tools/policy_replay.py`.  Unit icons come from `../Zero-K` (`--zero-k`), map pictures
  from the minimaps in the archives in `zero-k-portable/maps` (`--map-archives`), or for a made map from its heights.
* **Replays before format 2** named each unit after the def one past its own, so the viewer showed the wrong icons (missile
  silos for extractors, and so on).  `policy_replay.py repair DIR...` rewrites them from their def ids, and leaves format 2 alone.
  The dashboard serves replays uncached, so a repaired one shows at once.
* **Times** (`full-1`, 2026-11-10): frames 15 minutes, the encoder 74, cloning 6.2 hours (1,528 decisions a second), each round
  25-32 minutes of recording and 18 of fine-tuning, the eval 7.  PPO: see "Speed".

**Until a new encoder exists, run with `--train-encoder`.**  `FEATURE_VERSION` 10 (even DPS sectors and the frame over 80
minutes, 2026-11-13; 9 was air transports, 8 the raised caps, 7 64 players and 8 allyteams) made every encoder, clone and
frame pool unloadable, `full-1`'s and `full-2`'s among them, so the pipeline has to collect fresh
frames (51 GB at the old caps, by the same recipe) and train an encoder on them.  The scripts changed as well, and play new
kinds of game, so give `--data` a new directory: the `record` stage fills it.

## Frame data (`tools/frame_data.py`)

The encoder learns from frames of many kinds of game, on the map pool with its held-out maps left out.  Every player's seat of a
game is used, under fog, with the economy on.  Frames are stored featurized (about 13 kB each at the caps before
`FEATURE_VERSION` 8), with probe values: the truth the seat can't see, and each game's outcome, its own team's.  A shard's
`kind` column says which kind of game each frame is from.  No game here ends on a lead (the user, 2026-11-10): the lead
rule is off in all three sources, so that frames of lopsided and late games are there to learn from.

* **`script`**: whole games of Python scripts (`frame_world.FrameGames`).  The opening's factory suits the map: water factories
  are 3x likelier on wet maps.  A base script changes production every 30-120 s and keeps idle builders busy.  A fight script
  (`FrameFight`) keeps orders on reachable ground, jumps, throws, pushes and pulls, and uses beacons and silos.
* **`scenario`**: random set-ups played for 24 s.  Each team has 150-40,000 metal (log-uniform) in 1-3 clusters, a base, wear,
  wrecks, and partial memory of its enemies' bases.  What stands where follows the unit data (water depth, floating), and
  buildings the placement rule (01-simulator.md, "Where a building can go").  60% are themed: air battles 20%, air raids
  against AA 10%, fleets 10%, heavies 10%, swarms 10%.
  * **Its kinds** (`ScenarioRules.shapes`): 1v1s 40%, team games, free-for-alls and chicken games 20% each, as far as the
    maps hold them.  Teams, alliances and starts are drawn as `SimEnv` draws a game's, and an alliance shares its theme.
  * **Against chickens**: 1 to 12 burrows away from the players, half of them defended, an army of 1 to 3 kinds worth
    300-60,000, and the queen or a White Dragon on the board in 15% each.  The sim's spawner orders them.
  * **Caps**: a team has at most 480 units, and all the teams together 1,216, so that a frame's tokens hold them.
  * The shape weights, the 480 and the chicken numbers are first guesses, not measured against real games.
* **`army`**: the C++ army and economy scripts' games, with variants: what the actor is cloned from.  In the kinds `--kinds`
  says, every player's seat of each.

`pool5` (1M frames of each, 1v1s only, deleted since) was the last collected.  Aircraft were in 23% of `script` frames, 48% of `scenario` frames and 0.2% of
`army` frames, so the encoder's knowledge of air comes from scenarios.  `pool4` (script and scenario only) is the held-out set
the encoders were compared on.

## The frame encoder (`tools/frame_vae.py`, `frame_model.py`)

A Perceiver-style reader (`docs/model.md`, "The frame encoder") trained on its own, then frozen.  The recipe is `encoder-big`'s:

    frame_vae.py train --data FRAMES --run RUN --steps 25000 --autoencoder --region-pool 32 --two-part --questions \
        --muon-lr 0.008 --queries 40 --d 144 --rounds 3

* **Autoencoder**: no noise and no KL.  Each latent token gets a LayerNorm with no weights.
* **What the latent must give back**:
  * for each region, what's in it now: value by side and class, DPS, wreck metal, remembered structures;
  * the global numbers;
  * 64 sampled questions a frame.  Twenty ask "where is a unit like this?" (8 places with weights, since identical units make
    it ambiguous).  Forty-four ask "what's at this exact place?": whether a unit or wreck is there, and its type, side and state.
* **Region pooling** (`--region-pool`): each unit's numbers are summed into its region's token before the queries read it,
  because attention averages and can't count.
* **Two-part cells** (`--two-part`): each cell answers whether anything is there, and how much, on filled cells only.
* **Muon** on the hidden matrices and AdamW on the rest.  At 5,000 steps, Muon 0.008 beat AdamW 5e-4 on every measure
  (region R2 0.843 against 0.590).
* 1.65M parameters, 82 minutes on the 4070.

`encoder-big` (`runs/encoder-big/final`; unloadable since feature version 7), on 8,000 held-out frames of `pool5`:

| Measure | Value |
|---|---|
| Region reconstruction R2 | 0.970 |
| Own ground units by region, R2 | 0.722 |
| Own / enemy aircraft by region, R2 | 0.50 / 0.62 |
| "Where": any of its places within 0.05 of the map / its likeliest | 0.781 / 0.601 |
| "What's here": unit found / false alarms | 0.890 / 0.043 |
| Its type / side right | 0.818 / 0.988 |
| Health / x velocity R2 | 0.764 / 0.740 |
| Aircraft asked about: found / type right | 0.871 / 0.722 |

It was better on every measure than the encoder before it, which had 24 tokens, 0.92M parameters and no army frames (for
example aircraft type 0.58 to 0.76).  What part was size, tokens or data wasn't separated.

How the recipe was reached, each a short run on held-out maps:

* LeJEPA left most probes below an untrained encoder's.
* A VAE did far better (own air probe 0.17 to 0.82), and so did the autoencoder with pooling and two-part cells on regions
  (region R2 0.989 against the VAE's 0.921).
* Questions added fine position: a fresh reader's "enemy value within 0.05 of a unit" went from 0.59 to 0.76.
* A 3x bigger VAE did no better at the same steps.

## Script data (`tools/policy_clone_data.py`)

    policy_clone_data.py record --out /mnt/d/zk-learned/clone/NAME --games 32 --decisions N --workers 12 --threads 12 --seed S \
        --variants --maps /mnt/d/zk-learned/maps [--kinds 1v1:0.5 teams:0.2 ffa:0.2 chicken:0.1]
    policy_clone_data.py check --data DIR --maps /mnt/d/zk-learned/maps      # kinds, orders that can't be said, bytes a decision
    policy_clone_data.py zindex --data DIR --workers 15                     # z.npz, for --z

* **Games**: the C++ army and economy scripts against themselves, all looking every 6 slots, on the pool's training maps, in
  the kinds `--kinds` says (1v1s if not given).  They keep the lead rule.
* **Recording**: the recorder (`sim/record`, `zkrecord.h`, files headed `ZKREC002`) takes the batch's arrays by pointer and
  writes one `.zkr` file a game, every player's side (data version 7).  A file's header lists its sides and each one's team;
  its meta has the game's teams, alliances, kind, how it ended and when each side went out; a chicken game has its players'
  sides alone, with its difficulty.  Version 6 files (`army-6`, two sides) still read.
  * A unit is 40 bytes.  Positions are 16-bit fractions of the map, and floats are halved where that's lossless enough.
  * Each record is stored as a change from the unit's last record.  Columns go a byte plane at a time, zstd'd in chunks of 64
    frames.
  * That makes about 240-370 bytes a decision, 2.7x smaller than the npz it replaced.  A reader can stop after the chunk it needs
    and decode one side only.
* **Only whole games** are recorded: once the budget is reached, running games are played out.  Workers share the map pool and
  each map's derived data, loaded before they fork.  One process records 62,600 decisions a second.
* **Memory**: a writer holds each running game's compressed chunks until it ends.  Since 2026-11-11 each chunk is its own buffer
  of just its size, a game's sides are freed when it's written, the uncompressed chunk is one buffer a thread, and the heap
  is trimmed after each file.  Before that, 12 workers of the default kinds (8-team, 80-minute games) grew without end and
  filled 40 GB in 5 minutes.  A worker of 64 games on 60 maps went from +2.3 GB to about +1 GB, following the games it holds.
  On the full pool, 12 workers leave about 6 GB of 39 available.
* **Labels**: the army script's orders, then the economy script's, and `GROUP_MARK` records for group orders.  They are kept
  raw, not as choices, so the data outlives featurizer changes.  `actions.encode` makes the choices when the data is read.
* **z** (`Z_*`, 32 numbers a side):
  * whether the knobs are known, and whether the counts are;
  * the army script's 7 role weights (logs) and 5 knobs, and its preset;
  * in hindsight, the side's extractors, energy, factories, defences and army at minutes 3, 6 and 10.
* `sides.npz` caches each file's side and frame counts, so a run starts in seconds.  `z.npz` has a row a side.
* **Known**: `policy_clone_train` holds out one file in 50 by hash, so a recording of under about 50 games can have none held
  out, and fails.

**`army-6`** (`/mnt/d/zk-learned/clone/army-6`): 83,820 games, 300.6M decisions (both sides), 105 GB.  It was stopped at 75% of
the 400M asked.  Per look, the economy script gives 1.13 orders and the army script 1.11 actions.  Together they exceed 16 in
1.1% of looks, and 2.3% of orders are cut there.  Of the army's labels, keep is 98%, fight 1.2%, produce 0.6% and move 0.2%.

## Cloning (`tools/policy_clone_train.py`)

    policy_clone_train.py --data /mnt/d/zk-learned/clone/army-6 --maps /mnt/d/zk-learned/maps --run RUN \
        --decide-every 6 --samples 8 --accumulate 8 --lr 6e-4 --muon-lr 0.008 --workers 15 --steps 10000 \
        --z --z-dropout 0.2 --economy-weight 2.5 --finish-winners 0.1

* **A sample** is a stretch of 48 of the actor's decisions, with the frames of its reach before it.  The actor decides every 6
  slots, at the frames where the scripts looked.  A step is `--samples` x `--accumulate` stretches (64 in the pipeline), drawn
  at random from `--mix-pool` made (256).
* **Passes by size** (`--pass-gb 7`, 2026-11-11): a step's stretches go into forward passes smallest first, as many to a pass
  as `pass_gb` says fit in 7 GB of the card.  That model, fitted to 200 passes of `army-7`, is 0.14 GB off on average and
  at most 1.0 under.  A pass costs about the same few thousand kernel launches whatever it holds, and half the stretches
  have under 171 unit tokens and 40 controllable units, against caps of 1,280 and 512: passes hold about 8 on `army-7`.
  Each pass's loss is the mean of its labels', weighed by its share of the step's stretches.  `--pass-gb 0` is `--samples` a
  pass, as before.
* **Labels**: at each decision, the scripts' orders become actions in a random order, at most 16, then done.  The orders have
  no order of their own.  So an action's anchor scores as the sum over every unit still to be told (the set loss), and done
  scores once none is left.  The action tokens are built from all the orders.
* **Losses**:
  * the negative log-likelihood of each action, done included;
  * BUILD, ASSIST and RECLAIM weighted `--economy-weight`;
  * a side loss, `head_ordered`: for every unit, whether it's told anything this decision.
* **`--z`**: the side's z is added to every token.  With `--z-dropout 0.2` the whole z is unknown (zeros) in that share of
  samples, and the counts alone in as many again.
* **`--finish-winners 0.1`**: the last 10% of steps draw only the winning sides of decided games.
* **Workers** featurize ahead of the GPU.  Each cuts 8 stretches one after another from a game it decodes (`--per-read`; it
  was 4), and sends them as one tensor of shared memory each.  The time transformer uses flex attention on CUDA
  (`--attention flex`).
* **A pass past the card's memory** is undone, the gradients as they were before it, and its stretches go in two halves, each
  weighed by its share.  One too big alone is left out, and the report counts both (`split_passes`, `too_big`); a held-out
  chunk is measured a stretch at a time.  On `army-7`, at 4 stretches a pass, a pass took 3.3 GB of the card at the median and
  8.6 GB at most in 80, padded to every cap (2026-11-11), and one stretch at most 1.9 GB.  `full-2`'s first clone ran out
  of its 10.2 GB within 5 minutes.
* **A pass whose loss isn't finite** skips its step, and its stretches' games are written to `nonfinite.jsonl`; so does a
  gradient that isn't finite (`nonfinite_passes`, `skipped_steps`).  The losses stay on the GPU and are read once a step, so
  that the host never waits for a pass to finish.  `full-2`'s first clone went NaN between steps 500 and 1000.  The cause
  was found when two later attempts logged the same stretch: a summary token's health, summed over its units, came below
  -1 with units in the frame they die (health below zero), and its `log1p` was NaN.  Featurizing now counts such a unit's
  health as 0, as a unit token does.
* **`--compile dynamic`** (the pipeline's) compiles the encoder, the action tokens, each time block and the heads, with
  symbolic shapes.  A graph has a few versions, each compiled once: with gradients and without, a pass of one stretch, and
  sizes past Inductor's thresholds.  It's about 5 minutes of compiling, the first time on a machine (Inductor caches in
  `/tmp`), and 13% faster after.
* **Held-out checks** use 96 stretches from 50 held-out files:
  * `anchor_right`: the likeliest anchor is a unit still to be told;
  * `kind_right`;
  * `done_right`;
  * `stops_early`: done likelier than any unit still to be told;
  * `nll_act`;
  * `ordered_bce`, against guessing the share told;
  * per-kind `right_KIND` and `nll_KIND`.

Results at 1,000 steps (one cosine schedule, 3.1M looks, 44 minutes):

| | `clone-7` (baseline) | `clone-8` (z, economy 2.5, winners) | `clone-7s` (1.9M actor) | `clone-7` blind |
|---|---|---|---|---|
| NLL of actions | **2.840** | 2.857 | 3.080 | 3.122 |
| Anchor right | 0.856 | **0.860** | 0.843 | 0.840 |
| Kind right | 0.938 | **0.942** | 0.925 | 0.924 |
| Done right | **0.958** | 0.945 | 0.954 | |
| Stops early | 0.114 | **0.108** | 0.126 | 0.141 |
| Ordered BCE (guessing: 0.331) | **0.148** | 0.149 | 0.156 | |
| NLL build / reclaim | 7.81 / 3.80 | **7.56 / 3.56** | 8.05 / 3.99 | |
| NLL fight / move | **4.44** / 3.64 | 4.67 / **3.62** | 4.89 / 4.22 | 5.02 / 4.35 |

* **Builds are the weak part.**  A build's 7.8 nats are in what it builds and where.  Reclaim's kind is right only 57% of the
  time.  Every curve was still falling at 1,000 steps.
* **The economy weight** did what it was for, and fights paid for it.  z is worth 0.03 nats held out.
* **The small actor** saturated on these looks, and the gap widened through the run: width matters.
* **Blind** (action tokens zeroed) is worse everywhere, so the actor reads its past actions without leaning on them alone.

**In closed loop** (`policy_clone_eval.py`): the clone plays side 0 and the army script side 1, in 64 games on the training maps
with variants on both sides.  Each slot's first game counts.

| Eval | Won / drew / lost | Extractors at 3 / 6 / 10 min, clone (script) | Income at 10 min, clone (script) |
|---|---|---|---|
| `clone-7`, T 1.0 | 3% / 17% / 80% | 4.0 / 4.2 / 5.7 (5.0 / 9.3 / 17.3) | 24.6 (55.2) |
| `clone-8`, T 1.0 | 2% / 19% / 80% | 4.1 / 3.8 / 5.1 (5.0 / 9.8 / 18.3) | 24.2 (56.6) |
| `clone-7`, T 0.8 | 2% / 20% / 78% | 4.1 / 4.3 / 5.2 (5.1 / 9.2 / 15.4) | 24.8 (50.7) |
| `clone-8`, T 0.8 | 3% / 19% / 78% | 4.0 / 4.5 / 5.6 (4.9 / 9.4 / 17.0) | 25.3 (55.4) |
| **`clone-8`, told its variant** | **5% / 19% / 77%** | 4.0 / 4.7 / 6.2 (5.0 / 9.9 / 18.5) | 26.6 (59.4) |
| `clone-7s`, T 1.0 | 5% / 16% / 80% | 3.9 / 3.8 / 5.0 (5.0 / 9.6 / 18.0) | 22.0 (57.3) |

* **Every clone keeps up for 3 minutes, then stops expanding.**  None of temperature, z, the economy weight or width moves
  that.  So what holds it back in play is drift into states the scripts' games never show, which is what DAgger labels.
* The user goes on with `clone-8` told its variant.
* The sim has to step every slot, with the actor deciding every 6th, as the data was recorded.  Stepping at the actor's rate
  changes what happens between decisions: a clone played that way fell from 2.75 extractors to 1.
* The evals at T 1.0 and 0.8 without z, and the z-unknown NLL of 2.886, were measured while an unknown z skipped `z_in`'s bias.
  That's not the zeros cloning showed it, and it's fixed now.  The z-variant eval is unaffected.

## DAgger

`policy_clone_eval.py --record DIR --record-prefix game-daggerN` keeps the clone's games, recorded as the scripts' are.  On the
clone's side the scripts only advise: libzkscripts' label seats get the army script's orders, and the economy script's where the
policy runs its economy, as labels that are never sent.  `policy_clone_train.py --mix DIR --mix-share 0.5 --init RUN/actor.pt`
fine-tunes on them mixed with the original data.

**Speed** (2026-11-12).  `full-2`'s `record-1` took 97 minutes for 320 games.  Its one busy thread spent 77% of its time choosing
actions eagerly, a kernel at a time (each launch an `ioctl` in WSL), 15% in the recorder's step and the sim, and its last 38
minutes playing out the 64 games still running when the 256th was written.  Since then:
* it plays as CUDA graphs (`--graphs`, on by default, as PPO plays): on 19 maps it reached 20,000 decisions in 194 s against
  312 eagerly (1.6x), with 3.6 GB of the card against 2.6;
* the recorder packs and compresses games' frames on 4 threads (`WRITE_THREADS`), 5.7% of the thread before;
* it stops starting games to keep once those written and those running come to `--record-games` (`Recorder.close_when`), so
  a round writes 256 games, not 320.  Every game started till then is still played out, so the data isn't biased towards
  short games;
* it prints a line of progress every 5 minutes.

**In `full-1`** (2026-11-10), each eval 64 games against the army script, told its variant:

| Actor | Wins / draws / losses | Extractors at 6 min (script's) | Ended by lead / time / elimination |
|---|---|---|---|
| clone (10,000 steps) | 5% / 19% / 77% | 4.1 (9.7) | 49 / 12 / 3 |
| after round 1 | 13% / 28% / 59% | 4.7 (9.0) | 44 / 18 / 2 |
| after round 2 | 13% / 23% / 64% | 4.8 (8.8) | 49 / 15 / 0 |

* **DAgger helped once**: wins went from 3 to 8 of 64 and losses from 49 to 38.  The second round added nothing.
* **It didn't fix the stall in expansion**: at 6 minutes the actor has about half the script's extractors, as every clone did.
* Held out, the clone picks a unit the scripts told 91% of the time, and the right kind 97%.

## PPO (`tools/policy_train.py`)

### Reward

In a 1v1, for side A against side B at decision t:

    V(t)   = the side's value: cost x build progress over its real units and structures (not blips or memories)
    raw(t) = (V_A - V_B) / (V_A + V_B + 1000)
    r_A(t) = (1 - beta) * [raw(t) - raw(t-1) + bonus_A - bonus_B + income_weight * (inc(t) - inc(t-1))] + beta * [game over] * z
    r_B(t) = -r_A(t)

* **z** is +1 for a win, -1 for a loss and 0 for a draw at the time limit.  A kill lowers the victim's value, so there's no
  separate kill term.
* **beta**, the share of the result, is 0 for the first 10% of the run's decisions, rises to 0.9 by 60%, and stays there.
* **The diversity bonus** (Pluto's): each finished unit of a factory-made mobile type is worth `diversity` (0.05) x its rarity.
  Rarity is 0 for a type the policy's sides make at least as often as the average type, rising on a log scale to 1, from
  decayed counts over the batch.  At the default it's about 2% of the shaped reward a game, and 14% at most.
* **The income term** is `inc = (a - b) / (a + b + 5)` on metal income, overdrive included.  `--income-weight` 0.5 in the
  pipeline.
* **With A alliances** the same terms are zero-sum over them, and every team of an alliance gets its alliance's reward:
  * `raw` is an alliance's value less the mean of the other live alliances', over all of theirs and 1,000.  That's
    (A share - 1) / (A - 1), where share is its part of the total, and at two the formula above.  Income goes the same way;
  * the result: an alliance that goes out gets -1 / (A - 1) at once, and its seats finish.  At the game's end the S
    alliances still in share what the others lost, (A - S) / ((A - 1) S) each: 1 for a sole winner, 0 if all are in at the
    time limit.  In a chicken game the players win when the queen dies, lose when they're out, and draw at the limit.
* **The speed term** (`--speed-weight`, 0.1 in the pipeline; the user, 2026-11-10): an alliance that goes out also pays the
  weight x the share of the time limit left / (A - 1), and those still in at the end share what was paid, as the result.
  It isn't faded by beta, so a sooner win is worth a little more all through a run.
* **The lead rule fades** (`--lead-fade`, default `0:0.9 100:0`; the user, 2026-11-10): the share of training games a lead
  can end (01-simulator.md, "The game in the sim") goes from 90% at iteration 0 to none from iteration 100, linear
  between, so that the policy learns to finish games.  Tests keep the rule in every game.  `metrics.jsonl` has
  `lead_share` and `lead_rule_games`.
* **Discounting**: gamma is 1 and GAE lambda 0.95.  With gamma below 1 the sum of potential changes would stop meaning "who's
  ahead".  Returns are bounded, so there's no reward or value normalisation; advantages are normalised per minibatch.
* **The terms apart** (`--terms`; the user, 2026-11-12, after Pluto's and AlphaStar's several value heads): by default the
  reward is the one number above (`blend`).  Given terms, each has its own reward, its own head on the critic, its own
  GAE, and a weight; nothing fades by beta.  The pipeline gives `result:1 score:0.5 economy:0.3 combat:0.3`:
  * `result`: z and the speed term;
  * `score`: the value balance's change and the diversity bonus, the shaping above without income;
  * `economy`: the income balance `inc` x (1 - gamma) each step, gamma a step being exp(-seconds / 600), so its return is a
    discounted average of the income balance over about 10 minutes.  At the alliance's end it pays the whole level, as if
    it lasted, so a game won early isn't worth less of it than one that drags on;
  * `combat`: the change in the balance of damage dealt, in the metal it cost (libzksim's stat 4, ABI 24): at two,
    (dealt - taken) / (dealt + taken + 1000).  That's a side's trades, as Zero-K's experience counts a unit's.
  
  A decision's advantage is each term's weighed over its returns' spread that iteration (5th to 95th percentile) where
  that's over 1, as DreamerV3 scales returns: a wide-spread term doesn't drown the rest, and a near-constant one isn't
  blown up (`return_scale_TERM` in `metrics.jsonl`).  Then it's normalised per minibatch as before.  The heads' losses add
  up.  `metrics.jsonl` has the first term's `explained_variance` and returns' range as before, and each other's with its
  name after (`explained_variance_economy`).  Why: in `full-2` explained variance reached 0.99-1.0 by iteration 550.  The
  reward was mostly the result, beta at 0.85, and the policy lost about 95% of its games, so the critic knew the
  outcome and the advantages were mostly noise.  Income and trades say more about a decision's worth than a game it was
  always going to lose.
  * **Resuming with other terms** than the checkpoint's keeps all of the critic but its value heads, which start at zero
    with Adam's moments of them, and the actor stays as it is for `--critic-rewarm` iterations (10) while they learn.

### The update

* **Segments** are 32 decisions.  Each runs in one forward pass from the frames kept at collection.
* **The policy loss is SPO's** (Simple Policy Optimization, `docs/research/simple-policy-optimization.md`; the user,
  2026-11-10), in place of PPO's clip: each ratio r scores r A - |A| (r - 1)^2 / (2 eps).  That's greatest at r = 1 + sign(A)
  eps, and pulls every ratio back toward it, where the clip left a ratio past 1 ± eps with no gradient at all.  eps falls from
  0.15 to 0.1 over the run, as the clip did.  Each action, done included, has its own ratio, averaged within the decision,
  and there's one advantage per decision.  `clip_fraction` is still the share of ratios outside 1 ± eps, and
  `ratio_deviation` the mean of |r - 1|, the paper's measure.  Dual-clip went with the clip: it capped the loss of a ratio
  far past the region, which is the gradient SPO keeps.  `spo-1` ran it for 140 iterations ("What the earlier runs showed"):
  its tests can't be told from `full-1`'s PPO.
* **Entropy**: a bonus per head (0.01), raised for any head below 30% of its maximum entropy (`--entropy-floor`; 0 from a
  clone).
* **One pass** over each sample (`passes` 1).  An EMA of the weights (0.99) plays the tests and is saved.
* **The critic** is a separate network over the privileged frame, with the enemies the seat can't see.  It's never in the bot.
  With `--critic-bins` its value is a distribution over bins from -span to span, trained by HL-Gauss.  `--critic-lr` gives it a
  rate of its own, falling in step with the actor's: at the actor's 1e-5 a new critic learned nothing (explained variance
  0.0000 for 19 iterations in `full-1`).  The pipeline gives it 3e-4, and a span of 2, since `score` and `combat` returns
  are differences of balances (it was 1.5 with the reward as one number).  With `--terms` it has a value head for each,
  reading the same global token.
* **The critic's shape** (the user, 2026-11-12): 32 learnt queries and the global token read every token, as the frame encoder
  reads, in 4 rounds, each a reading and then the queries attending to each other.  So its cost grows with the tokens, not their
  square, as it did when it was self-attention over every token.  At the caps (16 frames of 2,193 tokens), a forward and
  backward pass takes 36 ms and 0.54 GB against 95 ms and 1.42 GB, with 6.67M parameters against 3.24M.
* **Muon on the critic** (`--critic-muon-lr`, the user, 2026-11-12): its hidden matrices, all its linear layers' weights but the
  value's, at 0.004 falling as its Adam rate does, with weight decay 0.01; Adam for the rest of it.  The critic learns from
  nothing here, as the encoder did where Muon beat AdamW.  The actor, fine-tuned from the clone at small rates, keeps Adam
  unless `--actor-muon-lr` puts its hidden matrices on Muon too (all its linear layers' weights but the heads' answers),
  at that rate's share of Adam's schedule and nothing during the critic's warm-up.  Adam won the A/B (2026-11-12, 50
  iterations each from `full-2`'s DAgger actor, the pipeline's flags, Muon at 1.3e-4).  Over iterations 11-50:

  | | Adam | Muon on the actor |
  |---|---|---|
  | Tests at 20 and 40 (64 1v1s each) | 14%, 11% | 8%, 14% |
  | Training games won against the script | 12.7% of 165 | 11.6% of 190 |
  | KL an iteration / ratios past 1 ± eps | 0.0095 / 16% | 0.021 / 26% |
  | Gradient norm | 13 | 26-30 |
  | Cloning loss, last 10 | 2.51 | 2.67 |
  | Decisions a second, overall | 102 | 91 |

  Muon moved the policy twice as far an iteration, and its KL rose to 0.032 by iteration 40, for no more wins.
* **The actor's rate follows the KL** (`--kl-target`, 0.01; the user, 2026-11-12): after an iteration whose KL passed
  twice the target, the actor's rate (Adam's and Muon's) is 1.5x slower; after one under half of it, 1.5x faster, never
  past the schedule.  The critic's isn't touched.  The scale is checkpointed and in `metrics.jsonl` (`lr_scale`).  In
  `full-2` the fade took the KL to the clone's weight from 0.48 to 0.094 by iteration 126, and PPO's KL an iteration rose
  with it from about 0.009 to 0.052, ratios past 1 ± eps from 15% to 36%, the policy loss from 0.04 to 0.2-0.3; the tests
  at 100 and 120 won 8% and 9%, against 11-16% from 20 to 80.  It was rolled back to iteration 80 and went on with this.
* **Minibatches** are filled to a token budget (`--token-budget`, 100,000 at most): a decision counts its padded tokens and
  200 more (`DECISION_TOKENS`).  A step takes about 0.5 s whatever its size (on `full-2`'s first segments, 2026-11-12, 32
  decisions took 0.25 s and 384 took 0.50, at 14.1-16.7 MiB a decision), so minibatches are as big as the learner's memory
  allows.  It used to weigh a token by `1 + tokens / 1200`, fitted when attention paired every token: frames grew to about
  730 tokens by `full-2`'s 18th iteration, minibatches fell from 141 decisions to 34, and training from 252 decisions a
  second to 70, with most of its memory unused.
  * A minibatch that doesn't fit is done again in 2, then 4... runs of its segments, each weighed by its decisions.  The
    budget then shrinks by a tenth, once in the iteration, as its minibatches were all planned before.  After an iteration
    without a split whose training's tensors peaked under 85% of the learner's limit, it grows by a twentieth, up to
    `--token-budget` (what the allocator reserves stays near the limit, so it's no measure).
  * The 200: with 800, from the critic's 14 KiB a token and the first segments' 14 MiB a decision, 241 decisions of about 280
    tokens fit the learner's 4.6 GB, but 151 of about 590 and 103 of about 860 didn't, and the budget shrank to fit them, from
    260,000 to 154,000 by iteration 7.  The actor's reading of the units grows with the tokens too.  Shrunk at every split,
    before the once-an-iteration rule, it had gone from 260,000 to 112,000 in one iteration.  Fitted on the 2,042
    minibatches of `full-2`'s 50 iterations with Adam, memory was 0.52 GB + 9.3 MiB a decision + 30.9 KiB a token, give or
    take 0.36 GB.  The worst (95th percentile) per unit of cost was level between small frames and big at 200, 4.0 and 4.1 GB
    a 100,000, so 100,000 fits the learner's 4.6 GB.  That run settled at 85,000-95,000 from 130,000, which split 12
    minibatches in its first iteration, and played 97-233 decisions a second overall.
  * Even so, those 7 iterations played 145-222 decisions a second overall, against 124-131 at the same point before.
  * Each minibatch's decisions, padded tokens, peak memory and whether it split are in the run's `minibatches.jsonl`, to fit
    the weight to.  `metrics.jsonl` has `splits`, `token_budget`, and `learner_reserved_gb` and `learner_allocated_gb`, the
    iteration's training peaks.

### From a clone

`--init RUN/actor.pt` starts the actor as the clone, at its size, and keeps a frozen copy:

* `--kl-start`: a k3 KL to the clone, at this weight at iteration 0.  `--kl-fade` takes its share down by iteration: by
  default to 20% at iteration 100 and to nothing at 500, linear between (the user, 2026-11-10).  At 0 the clone's forward
  pass is skipped.  Before, it fell linearly over the run's decisions from 0.5 to 0.1, and in `full-1` was still 0.48 at
  iteration 88: weighed, the KL was about 0.14 of the loss, against the policy gradient's 0.002-0.004.
* `--clone-weight`: a cloning loss on what the scripts advise on the actor's own frames (`SimEnv.advise`): DAgger alongside
  PPO.  It fades as the KL does (`--clone-fade`, the same points by default; the user, 2026-11-10), and at 0 isn't
  computed.  Kept whole, it was 0.5 x about 2.4 of `full-1`'s loss.  The army script stops advising once the weight is 0
  for this iteration and the next (the next is when concurrent collection trains), and never advises in tests.
* `--critic-warmup` iterations with the actor frozen.
* `--economy-start` / `--economy-end`: the share of games where the policy runs its own economy.
* `--fine-sim --decide-every 6`: the sim steps every slot and the actor decides every 6th, as cloning's data was recorded.
* `--variants --z variant`: each seat is told its side's knobs and preset, with the counts unknown.  Each step keeps its z, and
  the update and the KL read the segment's first step's z.

**The smoke run** (16 games, 3 iterations of 512 decisions, from `clone-7`): at a learning rate of 1e-4 the actor drifted too far
from the clone.  At 1e-5 the KL to the clone went from 0.003 to 0.023, PPO's own KL per iteration stayed at 0.005-0.008, and 10%
of ratios clipped.  The pipeline's PPO flags are this run's: lr 1e-5 to 3e-6, KL 0.5, cloning loss 0.5, 51 bins over ±1.5,
economy share 0.25 to 1, income 0.5, token budget 40,000 (as it was counted then), not concurrent; since `full-1`, the critic at 3e-4 alone for 10
iterations; and since 2026-11-10 the KL and the cloning loss fading by iteration, and SPO's loss in place of the clip and
dual clip 3, as above.

### Opponents

Training starts against the scripts.  Every `--eval-every` iterations, 64 test games against the script count wins, with draws
and losses logged (`metrics.jsonl`, `tests.jsonl`).  `SelfPlay` then moves to self-play:

* it **starts** when the last two tests average 80% wins;
* it **adds** 1% of new games each iteration;
* it **pauses** while those two tests are under 70%.

The test stays against the script throughout, so it stays comparable.

**Kinds of game** (`--kinds`, `--slots`, `--test-kinds`, `--chicken-difficulties`; the flag for action kinds is now
`--action-kinds`).  Against the scripts the policy plays team 0, and scripts play every other team, its allies too; in
self-play it plays every team.  The headline test stays the 1v1; each of `--test-kinds` is tested as well, on team 0.
`metrics.jsonl` has `wins_{mode}_{kind}`, `games_{mode}_{kind}` and `eval_fight_{kind}`, and each line of `tests.jsonl` its
`kind`.  No run has trained on the new kinds yet.

### Checkpoints and stopping

* Checkpoints are written beside their name and renamed.  `latest.json` names the last complete one, and was loadable after
  each of 12 kills mid-save (`--check-kill`).
* `--check-repro` runs one iteration twice: 0 difference with `use_deterministic_algorithms`, and 6e-4 without (scatter adds
  in no fixed order, which Adam magnifies).  Long runs use the fast mode.
* A `stop` file in the run's directory ends it at the next report, and the same command resumes it.
* Before torch starts, the allocator is set to `expandable_segments` with `garbage_collection_threshold:0.8`, and the cache is
  emptied after each iteration.  Reserved memory had grown by fragmentation until the card spilled.

### Playing

A decision round is the sim stepping, the scripts and featurizing on the CPU, and one forward pass of the actor and critic over
every seat that decides.

* **Aligned decisions** (`SimEnv.align`, on with `--fine-sim`): every game starts on a slot `--decide-every` divides, so all
  seats decide together, in one pass.  A game that ends between such slots waits, untouched, for the next.  Each game's own
  slot count, which the scripts and the opening go by, runs as before.  Unaligned, games drifted into all 6 phases, and the
  sampler ran on 79% of slots at 13.6 seats each.
* **CUDA graphs** (`tools/policy_graphs.py`, on unless `--no-graphs`, or `--concurrent` in this process): the whole round's
  forward pass (encoder, time transformer, action steps, critic, action tokens) is captured once per padded shape and
  replayed.  Eager, its thousands of small kernels took about 0.25 s a round from one core, whatever the batch.  Shapes pad
  to a ladder of widths, and each width only grows: padded per batch, a new shape came one round in five.  Padding changes
  no decision's own outputs.  The graph reads the weights where they are, so it follows training.
* **The action steps stop** at the first of steps 4, 6, 8, 10, 12 and 14 where every decision of the round is done: a graph
  for the decision and its first 4 steps, one for each 2 steps after, and one to end the round, with the host asking between
  them whether any decision is still going.  With `full-1`'s PPO actor at iteration 88, a decision took 3.2 steps on average,
  but the longest of a round of 64 took 14 at the median, and 1.3% of decisions took all 16.  A step costs about 1 ms
  however few its decisions (hundreds of small kernels), 1.6 ms at 64, beside 41 ms for the decision itself.  So the steps
  come to about 21 ms a round instead of 25 at 64 decisions, and 12 instead of 19 at 16, a collector's share.  Stepping only
  the decisions still going after each cut came to 17 and 11 ms, but it needed a graph for each count of them: 4 collectors
  then held 1.8 GB more of the card, outside PyTorch's count and each one's cap.
* **Collector processes** (`tools/policy_collectors.py`, `--collectors N`): each plays its share of the games and of the
  tests, with its own copy of the nets and its own graphs.  They fork before the GPU is touched, after the map pool is
  loaded, and share its maps.  Weights reach them through shared memory before each collection, checked by a checksum.
  Each may hold `--collector-gb` (1.0) of the card, which the learner's own limit leaves room for.  At 0.75, a collector ran
  out in `full-2`'s 8th iteration (2026-11-12): mixed games' frames grow past 1v1s', and a graph's shape only grows.  At the
  caps, a collector's 16 games reserve 0.65 GB with one shape and 0.83 with two; at 1.0 one reached 0.96 by iteration 16,
  holding the shape before while capturing the next.  So a shape narrower than the widest now goes before the next capture.
  With 4 at 1.0 the learner keeps about 4.6.  With `--concurrent` they collect the next iteration while this one trains.

### GRPO from branched games (`--group`)

The user asked for it on 2026-11-13, after `full-2`'s PPO got worse over its 4M decisions ("What the earlier runs showed").
A critic of a long, noisy game has to learn the whole game's value before its advantages mean anything.  GRPO (Shao et al.
2024) has none: it plays several futures of the same moment and compares them, so the moment's value cancels out.  For that
the sim copies games (01-simulator.md, "The batch").

* **Groups.**  With `--group G` the games go in groups of G (8), each group's games copies of one game, with what the actor
  kept of each seat (its history and its name map), which the policy then plays on apart.  Copies keep the world's random
  state, so only the policy's choices differ between them, and the scripts', which follow from those.
* **Stretches.**  A group plays a stretch of game time drawn evenly from `--horizon` (60 to 180 s: 2 minutes on average, and
  where in a game the stretches start varies).  Each seat's stretch in each game is a branch.  Then one of the group's games
  still going for the policy, drawn at random, is copied into the rest for the next stretch; when none is, a new game is
  started and copied.  So every decision played is learnt from, the games go on as the policy plays them, and no game in a
  group waits but one that ended (`SimEnv.hold_ended`), until its stretch is over.  This is the plan's "a trunk copied into
  a group" with the trunk a branch drawn after each stretch, rather than a ninth game that's only played on.
* **Scores.**  A branch's score is its terms over the stretch (`--terms`, as PPO's): `result` and `score` and `combat` their
  sums over it (the result only if its game or its alliance ended in it), and `economy` the mean of the income balance over
  it, what it earned against the other branches.  They're weighed by `--terms`' weights.
* **Advantages.**  A branch's advantage is its score less its group's mean, over its group's spread (Dr. GRPO, Liu et al.
  2025, drops the spread: `--no-group-std`), and each of its decisions takes it.  The branches of each seat are compared
  only with the same seat's in the group's other games.  A seat whose branches all scored the same is dropped
  (`branches_dropped`).  The update is PPO's (SPO's loss, the KL to the clone, the cloning loss), with no value loss.
* **A collection** stores decisions only of branches that have been judged: a branch still being played at its end goes to
  a later one, its decisions played by the weights of their time, as concurrent collection's are.
* **The learner has no critic**, nor do the collectors, the checkpoints or the moving average.  A run can't go between PPO and
  GRPO; another can start from its actor (`--init`).
* **Metrics**: `branches`, `branches_dropped`, `new_games`, `branch_decisions` (a branch's on average), and `group_spread`,
  a group's spread of weighed scores on average, with each term's (`group_spread_result` and so on).  No explained variance.
* **Checked** by `test_grpo_judges_each_branch_of_a_copied_game_against_the_other_and_trains_with_no_critic`: in a group of
  two, each stretch's branches start from the same frame and the same kept history, and every advantage is +1 or -1.
  Leaving out the copy of the game or of the actor's memory fails it.
* **An iteration** ends when its decisions are in and its groups' stretches have ended, so it holds more than `--iteration`
  (4,768 at first against 4,096), and a run reaches `--decisions` in fewer iterations than PPO.
* **The trial** (2026-11-13, `runs/grpo-trial`, at FEATURE_VERSION 9): from `full-2`'s DAgger actor, which won 9.4% of 64
  1v1s on the sim as it is now, with the pipeline's PPO flags and `full-2`'s, 8 groups of 8 in 4 collectors, tested every
  25 iterations on 64 1v1s:

  | Iteration | GRPO | PPO |
  |---|---|---|
  | 25 | 14.1% | 6.2% |
  | 50 | 15.6% | 7.8% |
  | 75 | 14.1% | 20.3% |

  The user stopped it there.  At 64 games a test (a standard error of about 4.5 points) it says that neither fell, not which
  learns faster.  GRPO played 150-265 decisions a second overall against PPO's 105, having no critic to run.  In GRPO's
  first minutes the groups' scores spread most in the damage balance (0.062, against 0.011 for value and 0.006 for
  income), and its orders moved from FIGHT towards MOVE (MOVE 6% of them at iteration 25, 47% at 75).

## Speed

* **Cloning** runs at 1,326-1,462 decisions a second with 15 workers.  The GPU alone would take 2,450: 157 ms for 8 stretches
  with flex attention, against 230 ms dense.  Making samples is the limit:
  * decoding takes 0.67 ms a decision;
  * featurizing takes 0.4 ms a frame, since numba (it was 1.6);
  * the main process needs most of a core to issue a step's kernels, and shares the 8 cores with the workers;
  * `/mnt/d` is about 15% slower than WSL's own disk.
  
  With numba, `full-1`'s cloning ran at 1,528 a second.
* **On the variety's games** (`army-7`: teams, free-for-alls, chickens, 80 minutes, the raised caps) `full-2`'s first clone made
  323 decisions a second, 9.6 s a step.  Measured on 2026-11-11 and changed, to about 750 (steady, 80 steps):
  * **decoding held the workers to 268 a second**: a late stretch of a long game decodes every frame of its side before it.
    `zkrecord.read` now reads a file only as far as it needs (it read all of it, up to 140 MB), undoes the deltas in one
    numba pass (the chains' sort was a third of decoding), and makes only the frames a stretch reads (`keep`: a quarter).
    Together a side decodes in about half the time, byte for byte the same; and 8 stretches a read, not 4.  12 a read
    measured 742 a second from the workers alone, and about 7% more in training than 8, with 9.9 GB of samples in shared
    memory;
  * **receiving a sample took the main process 85 ms**: through the pool's pipe, 64 KB at a time, while a second pool
    thread spun on the readable pipe.  As one shared tensor, a sample is a handle;
  * **4 stretches a pass**: a pass's kernel launches, about 3,000, cost the CPU about 85 ms in WSL whatever it holds.  Passes
    by size hold about 8;
  * **the frame encoder's attention** took PyTorch's math kernel, its heads being 36 wide: padded to 40 (`policy_model.attention`),
    5 times faster at a pass's sizes, the same to bf16's rounding.
  
  Then, the same day, to about 1,070 (steps 60-140 of 140):
  * **the host waited for the GPU** at every pass: the loss read back, and two small copies from pageable memory (the
    strategy z, and a table of kinds).  Without them nothing waits but the step's end, which alone gained nothing: the main
    process was idle 35% of its time for other reasons;
  * **reads were taken in order**: with a slow one first (a late stretch of an 80-minute game), the trainer waited 14% of its
    time while the reads after it were done and the workers idle.  Taken as they're done, about 940 a second;
  * **compiled** (`--compile dynamic`), 13% more.  Three things had to change for it: `kind_in` built a table on the host
    inside the graph, `F.one_hot` fixed the count of units (a graph for every count), and each call's new layout recompiled
    a time block every call.

  What's left: the main process waits a quarter of its time while the backward pass is launched, and unmaps the last
  step's samples (5%).
* **What made cloning 18x faster** (52 to about 950 decisions a second):
  * deciding every 3 s, not every slot;
  * decoding only the side and the part of a game a sample needs;
  * indexing the age bias by frame (its backward had been 77% of the GPU's time);
  * one encoder call a step over all the samples' frames.
* **PPO** at the pipeline's settings, from `full-1`'s `dagger-2`, all 64 games starting together (2026-11-10).  Rates are means
  over iterations 2-11, the tests apart:

  | | Decisions a second | 64-game test | Card at peak |
  |---|---|---|---|
  | Eager, unaligned (`full-1`'s first 35 iterations) | about 33 | 373-422 s | |
  | Eager, aligned | 92 | 162 s | about 5 GB |
  | Graphs, in this process (the default) | 158 | 83-86 s | 9.5 GB |
  | Graphs, 4 collectors, `--concurrent` | 178 | 81-89 s | 8.9 GB |

  * With a test every 20 iterations that's 78, 136 and 151 a second: graphs give 1.75x, collectors 1.9x.
  * **The GPU is now the limit**: Task Manager shows it 80-100% busy.  4 collectors not concurrent collected more slowly than
    one process, and concurrent ones slow training from 10-17 s an iteration to 18-27 s.
  * **Stopping the action steps** (above), from `full-1`'s PPO actor at iteration 88 with 4 concurrent collectors, means
    over iterations 2-20: 186 decisions a second against 175 running all 16 steps, with the card's peak the same (8.9 GB).
    Collection went from 14.3 s an iteration to 12.4, and training, sharing the GPU, from 23.3 s to 22.0.  Training is now
    the longer of the two, so a faster collection gains little more here.  In this process the GPU is about a quarter of
    a round, the sim most of the rest: there the saving should be about 2% (not measured).
  * **Memory**: the learner reserves about 4.9 GB to train, 7.1 GB with its own graphs; a collector of 16 games 0.59 GB.
  * Concurrent, the update reads data played by weights two updates old.  Its explained variance grew more slowly than in
    this process (0.004 against 0.05 by iteration 11), within what separates runs: eager `full-1` had 0.0035 there.

  The sim's own rates are in 01-simulator.md, "The batch".
* **The sim's budget** was 0.5 ms of wall time per game decision at 400 units on 16 threads, 2x allowed
  (`tools/sim_batch_bench.py`).  It measured 0.34 ms.

## What the earlier runs showed

* **`full-1`** (2026-11-10), the first whole pipeline: its DAgger rounds are in "DAgger".  PPO from `dagger-2` was stopped by the
  user at iteration 88 (360,552 decisions), eager throughout, still against the script only:

  | Iteration | Test wins / draws / losses | Explained variance |
  |---|---|---|
  | 20 | 16% / 27% / 58% | 0.34 |
  | 40 | 31% / 19% / 50% | 0.80 |
  | 60 | 22% / 27% / 52% | 0.93 |
  | 80 | 27% / 23% / 50% | 0.96 |

  From 13% wins after DAgger to 22-31%.  The first attempt, with the critic at the actor's rate, is kept in `full-1/ppo-critic-1e-5`.
  In a `record-2` game the actor rushed a singularity reactor to win on the metal ratio; by minute 10 it had won anyway, so the
  rule was left alone.
* **`spo-1`** (2026-11-10): SPO's loss and the fading KL and cloning loss, from the same `dagger-2` actor, with 4 concurrent
  collectors.  The user stopped it at iteration 140 (573,440 decisions):

  | Iteration | Test wins / draws / losses | Explained variance |
  |---|---|---|
  | 20 | 23% / 33% / 44% | 0.24 |
  | 40 | 14% / 27% / 59% | 0.61 |
  | 60 | 12% / 30% / 58% | 0.75 |
  | 80 | 28% / 27% / 45% | 0.81 |
  | 100 | 28% / 25% / 47% | 0.82 |
  | 120 | 14% / 23% / 62% | 0.89 |
  | 140 | 20% / 28% / 52% | 0.90 |

  The same range as `full-1`'s PPO, and no trend in either: a test is 64 games, so one is good to about 5 points.  The KL
  and the cloning loss were at 0.09 of their weights by the end, and the mean of |ratio - 1| rose from 0.07 to 0.14-0.19.
* **RL from scratch didn't learn.**  These runs were against the raider-only script, with an earlier actor:
  * `selfplay-6h` (7.2M decisions): 25% test wins in the first half and 28% in the second;
  * `latent-1` and `latent-2` (2M each): 13-15% early, 21-23% late.
  
  KL per iteration was 0.001-0.002 with 1 pass, so the policy barely moved.  The factory drawn decided most games (01-simulator.md,
  "What decides script games"), and in same-factory games the policy won 8% where the script wins 37%.  Hence cloning first.
* **A toy game** (`toy-focus`) showed the pointer heads can learn focus fire.  But beating one script says little about how a
  policy wins: an exploit gets there first.  So tests should use several scripts.
* **Action tokens** gained nothing at 3,000 cloning steps on an undertrained actor.  On `clone-7`, blinding them costs 0.28 nats,
  so they stay.

## Dead ends

| Tried | Why it went | Where it is |
|---|---|---|
| The rules bot | Rule changes stopped moving the win rate: 2 of 18 against CircuitAI Hard at 3x, with and without the latest flags | tag `rules-bot-final` |
| The first actor: a recurrent transformer over every token, with pair-distance biases, and its numpy forward for the bot | Trained at 360-400 decisions a second, a quarter of the plan.  Its live forward did fit: 17 ms mean, game held at 2.74x | git history before the trim (2026-11-08) |
| The toy game | It had served its check | git history |
| LeJEPA for the frame encoder | Most probes ended below an untrained encoder | git history (`frame_jepa.py`) |
| A VAE for the encoder | The autoencoder with pooling, two-part cells and questions kept more | `frame_vae.py` without `--autoencoder` |
| The latent actor with GRU memory; then with no memory and two past frames | The windowed time transformer learns whole stretches in one pass | git `bcea3be` for the GRU |
| Per-unit actions: every unit choosing at once | Zero-K's unit AI needs little micro, and a player gives a few orders to groups.  Actions in turn, as Pluto's | git history |
| The raider-only fight script as RL opponent | The factory drawn decided games | `frame_world.FrameFight` (frames only) |
| The Python army and economy scripts | The C++ ones are 17x faster with the same orders | git `8e9bca7` |
| `torch.compile` in PPO | No speed-up, and a hang after a recompile | (in cloning, `--compile dynamic` adds 13%: "Speed") |
| The age bias as flex attention's score_mod | 2.4x slower than dense: its gradient came a pair of tokens at a time | `TimeBlock.bias_dims` instead |
| Linear attention (DeltaNet and others) for memory | The gain was from dropping memory, not from linear attention | |
| GaLore | Optimizer state was 0.6% of memory: activations dominate | |
