# Training

How the policy is trained, from frames to PPO, with the settings in use and the numbers behind them.  The network is in
`docs/model.md` and the sim in 01-simulator.md.  Paths under `/mnt/d/zk-learned` are on D:, outside git.

## The pipeline

`tools/policy_pipeline.py --root DIR [--train-encoder] [--dry-run]` runs every stage in order, each as the tool's own command in
a process of its own.  So no two stages load the map pool at once, and each hands back the GPU when it ends.

| Stage | What | Settings from |
|---|---|---|
| `frames` | with `--train-encoder`, if `--frames` (default `ROOT/frames`) has no `manifest.json`: `frame_data.py collect`, 1M frames each from the frame games, scenarios and the C++ scripts' games | `pool5` |
| `encoder` | with `--train-encoder`: `frame_vae.py train`, 25,000 steps, to `ROOT/encoder`.  Later stages read its `vae-last.pt`; otherwise they read `--encoder` (default `policy_latent.ENCODER`) | `encoder-big` |
| `zindex` | the data's `z.npz`, if it has none | |
| `clone` | `policy_clone_train.py` on `--data` (`army-6`), 10,000 steps, checked and saved every 500 | `clone-8`: `--z --z-dropout 0.2 --economy-weight 2.5 --finish-winners 0.1`, lr 6e-4, Muon 0.008 |
| `record-N` | the last clone plays the army script, told its variant, with the scripts advising it, until 256 of its games are kept.  Its first 64 are an eval (`evals/record-N.json`) | DAgger |
| `zindex-N` | `z.npz` over every round's games | |
| `dagger-N` | 500 steps from the last clone, half the samples from every round's games and half from `army-6`; lr 2e-4, Muon 0.003 | DAgger |
| `eval` | the last clone, 64 games: `evals/final.json` | |
| `ppo` | `policy_train.py --init` the last clone, 64 games, 4,096 decisions an iteration, 2M decisions (`--ppo-hours` caps it), the critic alone for 5 iterations | the PPO smoke run |

* **Options**: `--dagger-rounds` (2).  `--clone-args`, `--dagger-args` and `--ppo-args` go at the end of their commands, so they
  override what's set.
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
* **Expected times**: the encoder about 82 minutes after the collection; cloning about 7 hours (at `clone-7`'s rate, before
  numba); a DAgger round about 45 minutes; PPO's rate in the fine sim is unknown.  None of this was measured on the pipeline
  itself.

**Until a new encoder exists, run with `--train-encoder`.**  `FEATURE_VERSION` 7 (64 players, 8 allyteams) made `encoder-big`,
every clone and `frames/pool5` unloadable, so the pipeline has to collect fresh frames (about 51 GB, as `pool5`, the same recipe, took).

## Frame data (`tools/frame_data.py`)

The encoder learns from frames of many kinds of game, on the map pool with its held-out maps left out.  Both seats of a game are
used, under fog, with the economy on.  Frames are stored featurized (about 13 kB each), with probe values: the truth the seat
can't see, and each game's outcome.

* **`script`**: whole games of Python scripts (`frame_world.FrameGames`).  The opening's factory suits the map: water factories
  are 3x likelier on wet maps.  A base script changes production every 30-120 s and keeps idle builders busy.  A fight script
  (`FrameFight`) keeps orders on reachable ground, jumps, throws, pushes and pulls, and uses beacons and silos.
* **`scenario`**: random set-ups played for 24 s.  Each side has 150-40,000 metal (log-uniform) in 1-3 clusters, a base, wear,
  wrecks, and partial memory of the enemy's base.  What stands where follows the unit data (water depth, floating).  60% are
  themed: air battles 20%, air raids against AA 10%, fleets 10%, heavies 10%, swarms 10%.
* **`army`**: the C++ army and economy scripts' games, with variants: what the actor is cloned from.

`pool5` (1M frames of each) is the newest.  Aircraft are in 23% of `script` frames, 48% of `scenario` frames and 0.2% of
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
        --variants --maps /mnt/d/zk-learned/maps
    policy_clone_data.py check --data DIR --maps /mnt/d/zk-learned/maps      # kinds, orders that can't be said, bytes a decision
    policy_clone_data.py zindex --data DIR --workers 15                     # z.npz, for --z

* **Games**: the C++ army and economy scripts against themselves, both looking every 6 slots, on the pool's training maps.
* **Recording**: the recorder (`sim/record`, `zkrecord.h`) takes the batch's arrays by pointer and writes one `.zkr` file a
  game, both sides (data version 6).
  * A unit is 40 bytes.  Positions are 16-bit fractions of the map, and floats are halved where that's lossless enough.
  * Each record is stored as a change from the unit's last record.  Columns go a byte plane at a time, zstd'd in chunks of 64
    frames.
  * That makes about 240-370 bytes a decision, 2.7x smaller than the npz it replaced.  A reader can stop after the chunk it needs
    and decode one side only.
* **Only whole games** are recorded: once the budget is reached, running games are played out.  Workers share the map pool and
  each map's derived data, loaded before they fork.  One process records 62,600 decisions a second.
* **Labels**: the army script's orders, then the economy script's, and `GROUP_MARK` records for group orders.  They are kept
  raw, not as choices, so the data outlives featurizer changes.  `actions.encode` makes the choices when the data is read.
* **z** (`Z_*`, 32 numbers a side):
  * whether the knobs are known, and whether the counts are;
  * the army script's 7 role weights (logs) and 5 knobs, and its preset;
  * in hindsight, the side's extractors, energy, factories, defences and army at minutes 3, 6 and 10.
* `sides.npz` caches each file's side and frame counts, so a run starts in seconds.

**`army-6`** (`/mnt/d/zk-learned/clone/army-6`): 83,820 games, 300.6M decisions (both sides), 105 GB.  It was stopped at 75% of
the 400M asked.  Per look, the economy script gives 1.13 orders and the army script 1.11 actions.  Together they exceed 16 in
1.1% of looks, and 2.3% of orders are cut there.  Of the army's labels, keep is 98%, fight 1.2%, produce 0.6% and move 0.2%.

## Cloning (`tools/policy_clone_train.py`)

    policy_clone_train.py --data /mnt/d/zk-learned/clone/army-6 --maps /mnt/d/zk-learned/maps --run RUN \
        --decide-every 6 --samples 8 --accumulate 8 --lr 6e-4 --muon-lr 0.008 --workers 15 --steps 10000 \
        --z --z-dropout 0.2 --economy-weight 2.5 --finish-winners 0.1

* **A sample** is a stretch of 48 of the actor's decisions, with the frames of its reach before it.  The actor decides every 6
  slots, at the frames where the scripts looked.  A step is `--samples` stretches a pass, over `--accumulate` passes.
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
* **Workers** featurize ahead of the GPU.  Each cuts 4 stretches one after another from a game it decodes (`--per-read`).  The
  time transformer uses flex attention on CUDA (`--attention flex`).
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
fine-tunes on them mixed with the original data.  Built and tested; no round has been run yet.

## PPO (`tools/policy_train.py`)

### Reward

For side A against side B at decision t:

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
* **Discounting**: gamma is 1 and GAE lambda 0.95.  With gamma below 1 the sum of potential changes would stop meaning "who's
  ahead".  Returns are bounded, so there's no reward or value normalisation; advantages are normalised per minibatch.

### The update

* **Segments** are 32 decisions.  Each runs in one forward pass from the frames kept at collection.
* **Clipping**: each action, done included, has its own clipped ratio, averaged within the decision.  There's one advantage per
  decision.
* **Entropy**: a bonus per head (0.01), raised for any head below 30% of its maximum entropy (`--entropy-floor`; 0 from a
  clone).
* **One pass** over each sample (`passes` 1).  An EMA of the weights (0.99) plays the tests and is saved.
* **The critic** is a separate network over the privileged frame, with the enemies the seat can't see.  It's never in the bot.
  With `--critic-bins` its value is a distribution over bins from -span to span, trained by HL-Gauss.
* **Minibatches** are filled to a token budget (`--token-budget`): 40,000 fits the 10M actor in 85% of the card, where 165,000
  didn't.

### From a clone

`--init RUN/actor.pt` starts the actor as the clone, at its size, and keeps a frozen copy:

* `--kl-start` / `--kl-end`: an annealed k3 KL to the clone.
* `--clone-weight`: a cloning loss on what the scripts advise on the actor's own frames (`SimEnv.advise`).
* `--dual-clip 3` and `--critic-warmup` iterations with the actor frozen.
* `--economy-start` / `--economy-end`: the share of games where the policy runs its own economy.
* `--fine-sim --decide-every 6`: the sim steps every slot and the actor decides every 6th, as cloning's data was recorded.
* `--variants --z variant`: each seat is told its side's knobs and preset, with the counts unknown.  Each step keeps its z, and
  the update and the KL read the segment's first step's z.

**The smoke run** (16 games, 3 iterations of 512 decisions, from `clone-7`): at a learning rate of 1e-4 the actor drifted too far
from the clone.  At 1e-5 the KL to the clone went from 0.003 to 0.023, PPO's own KL per iteration stayed at 0.005-0.008, and 10%
of ratios clipped.  The pipeline's PPO flags are this run's: lr 1e-5 to 3e-6, KL 0.5 to 0.1, cloning loss 0.5, dual clip 3, 51
bins over ±1.5, economy share 0.25 to 1, income 0.5, token budget 40,000, not concurrent.

### Opponents

Training starts against the scripts.  Every `--eval-every` iterations, 64 test games against the script count wins, with draws
and losses logged (`metrics.jsonl`, `tests.jsonl`).  `SelfPlay` then moves to self-play:

* it **starts** when the last two tests average 80% wins;
* it **adds** 1% of new games each iteration;
* it **pauses** while those two tests are under 70%.

The test stays against the script throughout, so it stays comparable.

### Checkpoints and stopping

* Checkpoints are written beside their name and renamed.  `latest.json` names the last complete one, and was loadable after
  each of 12 kills mid-save (`--check-kill`).
* `--check-repro` runs one iteration twice: 0 difference with `use_deterministic_algorithms`, and 6e-4 without (scatter adds
  in no fixed order, which Adam magnifies).  Long runs use the fast mode.
* A `stop` file in the run's directory ends it at the next report, and the same command resumes it.
* Before torch starts, the allocator is set to `expandable_segments` with `garbage_collection_threshold:0.8`, and the cache is
  emptied after each iteration.  Reserved memory had grown by fragmentation until the card spilled.

## Speed

* **Cloning** runs at 1,326-1,462 decisions a second with 15 workers.  The GPU alone would take 2,450: 157 ms for 8 stretches
  with flex attention, against 230 ms dense.  Making samples is the limit:
  * decoding takes 0.67 ms a decision;
  * featurizing takes 0.4 ms a frame, since numba (it was 1.6);
  * the main process needs most of a core to issue a step's kernels, and shares the 8 cores with the workers;
  * `/mnt/d` is about 15% slower than WSL's own disk.
  
  Cloning's rate with numba hasn't been measured.
* **What made cloning 18x faster** (52 to about 950 decisions a second):
  * deciding every 3 s, not every slot;
  * decoding only the side and the part of a game a sample needs;
  * indexing the age bias by frame (its backward had been 77% of the GPU's time);
  * one encoder call a step over all the samples' frames.
* **PPO** collection was limited by one thread issuing small calls, not by the GPU or the sim (measured on the 1M actor).  The
  sim's own rates are in 01-simulator.md, "The batch".
* **The sim's budget** was 0.5 ms of wall time per game decision at 400 units on 16 threads, 2x allowed
  (`tools/sim_batch_bench.py`).  It measured 0.34 ms.

## What the earlier runs showed

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
| `torch.compile` in PPO | No speed-up, and a hang after a recompile | (in cloning, `--compile` exists and adds little) |
| The age bias as flex attention's score_mod | 2.4x slower than dense: its gradient came a pair of tokens at a time | `TimeBlock.bias_dims` instead |
| Linear attention (DeltaNet and others) for memory | The gain was from dropping memory, not from linear attention | |
| GRPO (no critic) | It needs branching the sim, costs the group size in collection, and loses the privileged critic | |
| GaLore | Optimizer state was 0.6% of memory: activations dominate | |
