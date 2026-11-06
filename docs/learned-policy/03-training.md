# Training pipeline: self-play PPO, rollout, evaluation

**Reconciled with the other plans:** [00-overview.md](00-overview.md) lists what changed and wins any remaining disagreement.

Status: T-M0 and T-M1 built (section 10).  **The learner is about 4x slower than section 2.6 assumes.**  Follows [00-overview.md](00-overview.md).  Where
[01-simulator.md](01-simulator.md) and [02-policy.md](02-policy.md) (written in parallel) agree with each
other against the overview, this plan follows them and says so.  Their open questions to this plan are
answered in section "Needed from 01 and 02".

Labels: **[M]** measured here by a throwaway script (`C:\Users\Thoma\.claude\jobs\9ba24cc0\tmp\plan03`,
not in the repo); **[O]** the overview's measurement, not re-run; **[V]** verified in repo or engine
source (path given); **[E]** estimate, arithmetic shown; **[U]** unverified.

## 0. Decisions most worth review

1. **Timing.**  Sim tick 5 frames, decisions every 3 ticks = 15 frames = 2 Hz, live `block_n_frames` 5 in
   policy mode.  01 and 02 both propose this.  The overview's 6-frame tick cannot carry a 15-frame
   decision (objection 1).  All numbers below use 2 Hz: 2,400 decisions per side per 20-minute game.
2. **Reward:** a bounded potential difference of own value, zero-sum by construction; the terminal
   win/loss term is blended in by `beta`.  Kills are inside the difference, so no separate kill term
   (objection 2).
3. **Discount gamma = 1, GAE lambda = 0.95.**  At gamma 0.99 a win/loss at minute 20 is discounted to
   `0.99^2400`, about 1e-10.  Finite episodes and a telescoping reward need no discount.
4. **Per-unit PPO ratios, one advantage per decision.**  A joint ratio over up to 256 units is a sum of
   up to 256 log-ratios; per-unit clipping keeps the trust region meaningful.
5. **Budget.**  With 02's measured training rate and two passes, about 45-50M decisions/day at 70% of the
   GPU.  100M/day needs one pass or a faster model; it is not the planning number (section 2.6).  The
   real engine gives 2.6-4.4M learnable decisions per day, so it fine-tunes and does not train from
   scratch.

## 1. Algorithm

### 1.1 Factorised actions, clipping, losses, critic

A **choice unit** is a unit or factory whose order may change at this decision.  Units whose only legal
action is "keep current order" are not choice units (02 defines the masks).  Its log-probability is the
sum over the heads it uses (act, target, region, offset, site).  Illegal entries are masked to -1e9 before
the softmax and are excluded from log-probabilities and entropies.

* Ratio per choice unit: `r[d,u] = exp(logp_new - logp_old)` over its own heads.
* Advantage `A[d]` per decision (section 1.2), shared by the decision's units.
* Clipped surrogate per unit, averaged within a decision, then averaged over decisions, so a 300-unit army
  does not outweigh a 3-unit skirmish.
* Epsilon 0.15, annealed to 0.1.

Why not one ratio per decision: the decision log-ratio is a sum over `U` units.  At `U = 100` a per-unit
KL of 0.001 becomes a joint KL of about 0.1, and a joint clip then stops almost every update.  [E]  The
cost of per-unit clipping is that units in one decision are correlated, so the update is optimistic.
The KL monitor (section 7) and the entropy floor cover that.

Entropy: per head, averaged over choice units.  Coefficient 0.01 per head, adjusted by a controller so that
each head's entropy stays above 30% of its maximum.

Critic: 02's design (3.6): a separate encoder over the privileged frame (all enemies, hidden or not) plus
the actor's GRU state.  The actor never sees privileged inputs (02 section 11 requires this; 01 provides
the hidden records).  Its 0.48M parameters are about 16% of the actor's 2.94M [measured].

### 1.2 Discount and advantages

Effective horizon `1/(1-gamma)` at 2 Hz:

| gamma | horizon | at game end (2,400 decisions) |
|---|---|---|
| 0.99 | 100 decisions = 50 s | a win/loss term is multiplied by 1e-10: lost |
| 0.999 | 1,000 decisions = 8 min | multiplied by 0.09: weak |
| 1.0 (chosen) | whole game | not discounted; GAE lambda 0.95 gives an advantage horizon of about 20 decisions (10 s) |

Why 1.0: with gamma < 1 the discounted sum of potential differences is no longer the final difference, so
the shaping would stop meaning "who is ahead".  Episodes are finite (about 2,400 decisions per side), and
truncation at the time limit bootstraps from the critic.  Fallback if explained variance stays low: gamma
0.999, measured, not assumed.

Advantage normalisation per minibatch.  Returns are bounded (1.3), so no reward or value normalisation.

### 1.3 Reward

For side `s` at decision `t`:

    V_s(t)  = sum over s's tracked units and buildings of cost(u) * buildProgress(u)
    raw(t)  = (V_A(t) - V_B(t)) / (V_A(t) + V_B(t) + C)          C = 1000 metal
    Phi(t)  = raw(t) - raw(0)                                     Phi(0) = 0
    z       = +1 win, -1 loss, 0 timeout

    r_A(t)  = (1 - beta) * (Phi(t) - Phi(t-1)) + beta * [t = T] * z
    r_B(t)  = -r_A(t)                                             zero-sum by construction

* "Tracked" is the rule the stage 1 gate uses: real units only, not blips and not remembered structures
  (01 asks for this, so the reward and the gate measure the same thing; `fightnet.py` MEMORY flag, 32).
* `raw(0)` is subtracted because TitanDuel's start is slightly asymmetric; otherwise one side starts with an
  offset.
* `Phi` lies in (-1, 1).  Late-game exchanges move it less, because it is a ratio.  If that blunts the
  signal in the logs, switch to `(V_A - V_B)/S` with `S` = 500 metal and normalise the value head.
* Kills: a kill lowers the victim's V, so the difference rises by the victim's cost.  A separate killed-value
  term would count each kill twice (objection 2).
* Timeout: `z = 0`, so a time-limited game is judged only by `Phi`.

Annealing with `p` = fraction of the stage's decision budget used:

    beta(p) = 0                              p < 0.10
    beta(p) = 0.9 * (p - 0.10) / 0.50        0.10 <= p < 0.60
    beta(p) = 0.9                            p >= 0.60

Keeping 0.1 of the shaped term gives a dense signal when wins are rare.  If the win rate against anchors is
under 5% at p = 0.6, hold beta at 0.5 (section 3).

Opening bonus: none (user exclusion).  Diversity bonus: none in v1.  It is added only if the unit-mix trigger
in 4.4 fires, at most 5% of the shaped magnitude.

Scale [E]: `V` is perhaps 10-30k metal per side at 20 minutes, so `Phi` moves 0.1-0.3 per game, about 1e-3
to 1e-2 per decision.

### 1.4 Hyperparameters (starting point)

| item | value | source |
|---|---|---|
| gamma / lambda | 1.0 / 0.95 | 1.2 |
| clip epsilon | 0.15 to 0.1 | Gym-muRTS 0.1 [O] |
| passes per iteration K | 2 (measure 1 and 4) | 02 objection 1 proposes one or two |
| minibatch | token budget 115k tokens | 1.6 |
| learning rate | 3e-4 to 1e-4, linear | Gym-muRTS 2.5e-4 to 0 [O] |
| Adam eps / grad-norm clip | 1e-5 / 0.5 | Gym-muRTS [O] |
| value loss weight | 0.5 | Gym-muRTS [O] |
| entropy coefficient | 0.01 per head, floor-controlled | 1.1 |
| iteration | 65,536 decisions = 256 games x 256 decisions | 2.4 |

Gym-muRTS: PPO, 300M environment steps, 24 envs, 256-512 step rollouts, gamma 0.99, lambda 0.95, about 60
hours on one GPU, one small map, 91% win rate against its 11 bots with action masking and shaped rewards
(arXiv 2105.13807, Abstract, Section V, Table III, Section VI; read from the ar5iv HTML, the PDF was not
readable).  Its episodes are 2,000 ticks, so 300M steps is about 150k episodes, or 0.6-0.7B player-decisions
at 2,400 per side per 20-minute game.  That is the only external cross-check of the full-game budget.
Pluto's hyperparameters and compute are not published; its README gives the method and the 44 ms per-frame
budget only [U; checked against the online copy].

### 1.5 Recurrence

Per 02 (3.4): segments of **32 decisions** (16 s of game time), each starting from the **stored GRU state** at
its first decision (fp16), **no burn-in** by default.  Segments never cross a game boundary; a game that ends
mid-segment is padded and masked.  Burn-in of 8 decisions is the fallback if stale stored states hurt the
value fit (cost 1.25x).  This is a measurement to make in M2, not an assumption.

Pluto keeps a GRU and a slower transformer memory [O].  This plan follows 02, which has a global GRU and
per-region GRUs (02 section 3.4).

### 1.6 Minibatching, precision, compile

Token counts vary by decision (about 110-220 typical, about 750 worst case under 02's caps; 2.4).  Minibatches are built by token budget:
segments are sorted by size and each minibatch takes whole segments up to a budget of padded tokens, with big frames weighed
more for attention's pairs (`memory_tokens` in `tools/policy_train.py`).  While playing, a forward pass is limited to `ACT_PAIRS`
padded pairs, and seats past that go in another pass.  Peak memory stayed under the 10.2 GB limit at 256 units a side, every enemy seen, with concurrent collection.
Padding is masked.  The step is run in bf16 autocast (no GradScaler), with softmax and log-probabilities in
fp32.  `torch.compile` was planned here and isn't used: it gained nothing and could hang (10.3).  The table's compile rows are from a plain transformer.

Measured on the RTX 4070 in WSL, PyTorch 2.15 cu132 [M, `gpu_train.py`, `gpu_rnn.py`]:

| configuration | shape | step time | decisions/s | tokens/s |
|---|---|---|---|---|
| fp32 eager, plain 4-layer d256 transformer | 64 x 200 tokens | 33 ms | 1,946 | 0.39M |
| bf16 eager | same | 18 ms | 3,509 | 0.70M |
| bf16 + compile | same | 12 ms | 5,528 | 1.11M |
| bf16 + compile + GRU over 16-decision chunks | 32 x 16 x 200 | 108 ms | 4,753 | 0.95M |
| bf16 + compile + GRU over 8-decision chunks | 64 x 8 x 200 | 102 ms | 5,028 | 1.01M |

Peak memory 3.5 GB for 512 decisions at 200 tokens, about 6.8 MB per decision in training.  These are a
plain transformer with no critic and no region GRUs.  02 measured 1,500-1,720 decisions/s for its own
actor-plus-critic at d=192-256 (02 section 10, objection 1).  That is a third of the plain figure, and this
plan uses 02's number for planning until M2 measures the real model.

## 2. System architecture

### 2.1 Loop

    one machine: WSL, 16 threads, RTX 4070 12 GB, 40 GB RAM cap
    +-----------------------------------------------------------------------------------+
    | env pool (01, C ABI via ctypes, 16 workers, each owns whole games)                 |
    |    | raw frames (compact, decision ticks only)                                     |
    |    v                                                                               |
    | featurizer (02, numpy, data workers) --> fp16 tokens (RAM, one iteration)          |
    |    |                                                                               |
    |    v                                                                               |
    | actor (bf16, batched over games) --> actions --> env pool (next decision)          |
    |    | per decision: raw frame, action, logp, value, GRU state, reward parts          |
    |    v                                                                               |
    | rollout buffer (RAM <= 5 GB, spill to D:) --> GAE --> K passes, token-budget batches|
    |    v                                                                               |
    | learner (bf16, compiled) --> checkpoint (atomic, D:)                              |
    +-----------------------------------------------------------------------------------+

### 2.2 Binding and threads

01 chose a C ABI loaded with ctypes and rejected pybind11 for now (01 section 4.4).  This plan accepts that.
Two points for 03: `ctypes.CDLL` calls release the GIL, so a step can run in a thread while the GPU works;
and the batch step takes preallocated numpy buffers, so raw frames are views, not copies.  Per decision a
raw frame is at most about 19 KB compact (2.4), so a pipe would also work, but the C ABI is simpler.

### 2.3 Sync versus async

Double-buffered, **one iteration of lag**: games are split into two groups, one steps in C++ while the GPU
infers for the other.  Behaviour comes from the policy one iteration old; PPO's ratio uses the stored
behaviour log-probability.  The lag KL between behaviour and learner is logged.  Fall back to lag 0 if that
KL averages above 0.02 [E].

### 2.4 Buffer size

**Reconciled decision: raw frames only, in RAM for one iteration, featurized per minibatch by data workers.**
A record in 02's schema is 80 bytes, so about 12-20 KB per decision and
about 1 GB per iteration, plus stored GRU states.  Storing tensors isn't viable: 02's tokens are 140-d with an
N x N bucket table and pointer masks, about 140 KB per decision and 9 GB per iteration [E], against the 5 GB
ceiling.  At 1.3 ms per decision (02), with one featurization for inference and K=2 for training, 50M/day takes
about 2-3 threads [E].  The table and rule below are from the earlier 64-d estimate, and are superseded.

| tokens | fp16 tokens per iteration (65,536 decisions) | raw frames per iteration |
|---|---|---|
| 215 (median: entities + regions; sites and globals add a few) | 1.8 GB | 0.45 GB |
| 400 | 3.4 GB | 0.8 GB |
| 600 (maximum: 460 entities + 120 regions) | 5.0 GB | 1.3 GB |

Rule: raw frames stay in RAM (5 GB ceiling) and are not spilled.  Nothing is kept after its iteration except
1% of decisions for analysis, written to **D:**.  At 50M decisions a day, raw frames alone would be about
0.75 TB, so they are not archived.

### 2.5 Featurizer cost (measured)

[M, `featurize.py`]: 460 entities plus 120 regions into a 580 x 64 array, one thread: 1.16 ms (pure
Python loop over dict records), 0.38 ms (numpy).  That has no rays or pair buckets, so plan on 02's 1.3 ms.  In the live bot the cost is
the JSON observation parse, not featurizing [V, `bots/zkbot/client.py:18-31`: orjson when installed].  The
golden test (overview) compares sim and bot frames through the same featurizer.

### 2.6 Budget per decision and per day

Per decision:

| item | cost | basis |
|---|---|---|
| sim step, 400 units, 16 threads | 0.5 ms wall | [O] 370 us per 5 Hz tick; 2.2 s per 20-minute game over about 4,000 decisions (a floor: the benchmark has no full sim) [E] |
| featurize | 0.4 ms CPU (1 thread) | [M] |
| inference, 150 tokens, 5.1M | 0.09 ms GPU | [O] 11,400/s |
| inference, 400 tokens, 5.1M | 0.37 ms GPU | [O] 2,700/s |
| training pass, planning basis | 1 / 1,700 s = 0.59 ms GPU | 02's measured rate, actor plus critic |
| training pass, plain transformer | 0.2 ms GPU | [M] 5,028/s |

Both training rates are at about 200-250 tokens; at 400 tokens assume half [E].

GPU-limited decisions per day, 65,536-decision iterations, collection overlapped [E].  Iteration time =
65,536 x K / rate + 6.6 s of inference:

| training rate | K | iteration | decisions per day at 100% GPU | at 70% GPU |
|---|---|---|---|---|
| 1,700/s (02) | 1 | 45 s | 126M | 88M |
| 1,700/s (02) | 2 | 84 s | 68M | 47M |
| 1,700/s (02) | 4 | 161 s | 35M | 25M |
| 5,000/s (plain) | 2 | 33 s | 173M | 121M |
| 5,000/s (plain) | 4 | 59 s | 96M | 67M |

Read: with 02's rate and two passes, 45-50M per day leaves headroom for evaluation and for the desktop.
100M per day needs one pass, the plain-transformer rate, or a smaller model.  The 2.3M model measured at
33,800/s inference [O] is a candidate; its training rate is not measured [U].

CPU: 100M decisions at 0.5 ms sim is about 14 hours of wall with all 16 threads, plus about 11 thread-hours
of featurizing.  It fits, with no slack for the desktop or engine batches.

### 2.7 Live inference during play (measured)

The bot is synchronous: the engine waits for each observation reply [V, `bots/zkbot/client.py:1-6`].  At
3x a 15-frame decision period is 0.5 s of game time, which is 167 ms of wall.  The policy call adds to wall
time, and the speed must stay at 3x for the comparison to hold (HANDOFF section 3).  Budget: mean 35 ms wall per
decision, 20% of the period; p99 60 ms; the game holds at least 2.7x (overview).

Current bot compute at 3x [V, profile lines in `results/*/zkbot-team0.log`, three one-minute samples of 900
observations]: 4.7-6.6 ms per observation.  Engine share 77-82% of wall (HANDOFF 6b.5).

Numpy forward, one decision, transformer with the stated shape, feed-forward width 4d [M, `np_forward.py`,
naive, no fused kernels, so an upper bound]:

| model | tokens | 1 thread | 2 threads | within 35 ms? |
|---|---|---|---|---|
| d=128, 3 layers (about 0.6M without GRU) | 150 | - | 12.5 ms | yes |
| d=128, 3 layers | 400 | - | 36 ms | just |
| d=128, 3 layers | 600 | - | 55 ms | no |
| d=256, 4 layers (about 3.1M without GRU) | 150 | 46 ms | 40 ms | no |
| d=256, 4 layers | 400 | - | 98 ms | no |

PyTorch CPU at 2 threads for 5.1M: 9-20 ms [O].  So the 5M-class model needs either a GPU server or
PyTorch in the bot.

Reconciled: v1 uses numpy with 02's d=192 (30 ms for the encoder at 243 tokens, measured by 02), and P-M2 measures
the whole forward in shadow mode.  ONNX Runtime and then d=128 are the fallbacks, and the server below is the last
resort.  The options as first written:

* **Numpy in the bot, small model** (the overview's rule): d about 128, 0.6-2M parameters, up to about 400
  tokens.  02 chooses the size.
* **GPU inference server** in WSL, a separate process.  The bot sends fp16 tokens over localhost and receives
  logits; requests from the concurrent games are batched.  Estimated round trip 2-5 ms [E].  The bot does
  not import PyTorch.  Needs a Windows-to-WSL localhost path [U: expected to work, not tested here].  This
  is the path for stage 4 with the 5M-class model.

### 2.8 Resources on this PC

[V, [M] this session]:

| resource | total | note |
|---|---|---|
| CPU | 16 threads (Ryzen 7 5800X3D) | engine 77-82% of wall in bot games |
| RAM | 48 GB to Windows; WSL capped at 40 GB (`.wslconfig`) | one engine commits 3.3-4.7 GB; 8-9 games fit in 48 GB (`tools/prun.py:73`, HANDOFF 3) |
| GPU | RTX 4070 12 GB | 1.7 GB used at idle |
| disk C: | 14.8 GB free; WSL `ext4.vhdx` (17.8 GB) lives here | **not usable for buffers** |
| disk D: | 110 GB free | runs, buffers, checkpoints |

Rules:

* **Never run a training pass and an 8-game engine batch together.**  8 games need about 26-38 GB of commit
  [E: 8 x 3.3-4.7 GB]; the trainer with a 5 GB buffer and CUDA host memory needs about 10 GB [E].  Together
  they reach the 48 GB ceiling, the failure seen before.  Schedule in phases (section 6.4).
* Trainer: 2 torch threads, `nice 10`, GPU memory fraction 0.45 (about 5.4 GB).  About 5 GB of VRAM stays free
  for the desktop and a spectated game.
* Sim workers: 10 while training, 16 when the trainer is paused.
* The `.wslconfig` cap (40 GB) exceeds what is needed.  Lowering it is the user's decision; not proposed
  here beyond noting it.

### 2.9 Pause, resume, checkpoints

* `D:\zk-learned\control.json` holds `{"state": "run" | "pause" | "stop"}`.  The learner reads it at each
  iteration start.  `pause` finishes collection and parks; `stop` checkpoints and exits.  `tools/learn_gate.py`
  (new, M4) writes `pause` before an engine batch and `run` after; the user writes `pause` by hand before
  spectating.
* Checkpoints go to `ckpt-<step>.tmp`, are fsynced, then renamed.  `latest.json` names the last complete one,
  so a kill mid-write leaves the previous one valid.
* A checkpoint holds model, optimiser, beta position, K-controller state, RNG states, iteration counter, and the
  config hash.  The rollout is not saved; resume starts a new collection.
* Keep the last 5 plus one per 10% of stage budget.  At 5.1M parameters fp32 a model is 20 MB and with Adam
  about 60 MB, so about 0.5 GB per run.

## 3. Curriculum across stages 1-4

Stage 1 trains no policy.  It is the fidelity gate: the sim's combat must predict logged fights as well as
`fightnet` does (overview: AUC 0.59-0.62 [O]).  Logged data: 46,389 fight snapshots in 184
`fights-team0.jsonl` files [V, counted].  Stages 2-4 train the policy.

| stage | sides | scenarios | policy controls | economy | maps | episode | decisions per side | budget [E] | stuck when |
|---|---|---|---|---|---|---|---|---|---|
| 2a | 1v1 to 4v4 | flat ground, random starts | army | none | flat test map | 60-120 s | 120-240 | 1-5M | win rate vs scripted attack-nearest < 80% after 5M |
| 2b | 4v4 to 12v12 | initial states from logged fights (real positions, HP) | army | none | TitanDuel 2.2 | 2-5 min | 240-600 | 10-50M | < 60% vs scripted after 50M |
| 2c | up to about 100 units a side | from the start of a logged game | army only; a scripted production schedule, from the rules bot's logged spend mix, supplies reinforcements (factories are stage 3, 02 section 4.1) | fixed income: rules bot's `m_inc` at minutes 5-10 (`bots/zkbot/statelog.py:170`) | TitanDuel 2.2 plus 3 | 10 min | 1,200 | 50-300M | exchange X10 vs scripted <= 1.0 after 150M |
| 3 | both | full game | army, factories, builders, static defence (02) | policy | 16 maps (`maps.json`, 111 entries) | 20 min or win | 2,400 | 0.5-2B | < 60% vs anchors after 1B, or value explained variance < 0.3 |
| 4 | both | randomised (section 5) | as 3 | as 3 | as 3 | engine, 11 min mean | 1,300 per seat | 2.6-4.4M per day | section 5 |

Notes:

* **Maps, for now.**  Five from the 1v1 matchmaker pool (`maps.json`), `TRAINING_MAPS` in `tools/policy_sim_env.py`: TitanDuel 2.2 (24 regions,
  182 of the 188 games with logged fights), Red Comet Remake 1.7 (22, narrow), Agalta v1.0 (16, open), FrostyCove v1.13 (30, hills and
  chokes), Faultline v1.0 (31, 17% water).  Left out: harmful liquid, mostly water, heights generated in-game, and more than about 35 regions, since
  playing pads every seat to the map with the most regions, and a big map's frames also cost more themselves.  Carabanossi stays exported for the gate only.
* **Early end.**  From 2 minutes of game time, a side with more than 2.5x the other's metal value (cost x build progress, as in 1.3) wins
  there and then.  The engine doesn't have this rule, so win rates in the sim aren't comparable with the engine's without it.
* **Seats and starts.**  One set of weights controls both sides.  Each game draws a seat and a start for each
  side, so TitanDuel's asymmetry averages out.  Report by seat.  Sanity check (M2): self-play with the
  start-A seat wins 45-55% over 200 games.
* **Scripted baseline.**  Stage 2 needs a fixed scripted army in the sim (attack nearest, focus weakest in
  range, no kiting), written in 01's sim package.  The rules bot's army code does not port.
* **Factories.**  Sampled per game as prun does (`--enemy-factories`).  Stage 2 uses the factories a scenario
  names.
* **Maps.**  2b is TitanDuel only, so terrain tokens stay fixed.  From stage 3, 16 maps spanning the region
  count (small to 120).  01 must confirm it loads any `maps.json` terrain [U].
* **Budgets** are estimates with wide ranges.  The full-game range rests on Gym-muRTS's 0.6-0.7B
  player-decisions [O]; the action space here is larger, so the top of the range is more likely than the
  bottom.
* **Splits (01 asks).**  Games are split by game, never by fight or decision.  Five folds over the 184 games,
  fixed seed, written to a committed `tools/splits.json` (small).  Stage 1's fightnet CV and stage 2's
  train/validation use the same file.  No test-fold game is used by any stage's training.
* **Stuck** means the stopping rule fired: stop and rethink (section 9), do not add decisions.

## 4. Self-play stability

4.1 **Current weights only, first**, as Pluto does [O].  No opponents, no league.

4.2 **Anchor pool, evaluation only.**  Every 5% of a stage's budget, snapshot the policy.  Keep 10 anchors (the
last 5 plus 5 spread over the run).  Anchors are never trained on.  Each evaluation: 200 sim games per seat
pairing against the last 3 anchors and the first anchor, 800 games at about 4,800 decisions each, about 3.8M
decisions, roughly 5-6% of a day at 60M.  [E]

4.3 **Rating and cycles.**  Bradley-Terry over anchor games, refit each evaluation; keep the full pairwise
matrix.  A cycle is three checkpoints A, B, C with P(A>B), P(B>C), P(C>A) all above 0.6: non-transitive play.

4.4 **Unit-mix drift.**  For each checkpoint, a histogram of spend by role over its own games (the categories of
`bots/zkbot/statelog.py:53` `spend_category`).  Compare checkpoints by Jensen-Shannon divergence.  Collapse: one
role above 70% of spend for three evaluations running, or entropy below 0.5 nats.

4.5 **League trigger.**  Start a league (30-50% of games against a sampled older checkpoint, weighted toward
those near 50% win rate, PFSP-style) when any holds: a cycle (4.3); win rate against anchors 15 points below its
peak over two evaluations; JS mix change above 0.3 between evaluations without a win-rate gain.  Once started it
stays on.  [E]

## 5. Domain randomisation and fine-tuning

### 5.1 What to randomise (stage 4, in sim)

| dimension | range | reason |
|---|---|---|
| map | the 16 of stage 3, then every `maps.json` entry with valid terrain | terrain generalisation |
| seat and start | uniform | side bias |
| unit cost and HP | +-10% per type, per game | the catalog may be off |
| weapon damage and reload | +-15% per weapon | `weapons.json` and balance errors |
| projectile flight time | +-20% | 01's approximate weapon model |
| command latency | 0-1 extra tick | the live latency is not exact; 01 defines the tick |
| income | +-20% | rules-versus-policy economy |
| radar blip type hidden | 30% of blips | fog realism |

Validation (do not assume a dimension helps):

1. Train two stage 3 policies, one randomised and one not, and evaluate both in the engine with the 6.3 gate
   (100 pairs).  The randomised one must not lose more than 10% on M15 or X15, and its sim-engine gap (6.6)
   must be smaller.
2. Evaluate on the widest ranges held out.  Win rate should fall smoothly.  A cliff means the range is beyond
   what the policy handles; narrow it.

### 5.2 Engine trajectories from `bots/zkbot`

New module `bots/zkbot/policy/trajectory.py` (M4), beside 02's `runtime.py`, enabled by `ZKBOT_FLAGS` containing
`policy`, with `ZKBOT_POLICY` giving the exported `.npz`.  At each decision frame:

* the bot builds the raw frame from its observation, in 01's record layout (golden test);
* it calls the policy (numpy or server, 2.7) and applies the actions through its existing command path
  (`brain.take_commands`, `bots/zkbot/main.py`);
* it appends to `D:\zk-learned\traj\<run>\game-<n>.bin`: compact raw frame, decoded actions, behaviour
  log-probability, value, GRU state, reward parts.  Own-side value uses the bot's own unit costs
  (`bots/zkbot/statelog.py:131`); the enemy side uses its memory (`enemy_memory`, `statelog.py:144`), so it is
  visible-only.  Files are written per game and atomically; a crashed game is dropped.

Engine rewards are noisier on the enemy side, so run engine training with beta near 0.9, where the win term
dominates.  Exact end-of-game values come from the demo team statistics (`tools/demostats.py`, `summarize`).

### 5.3 Engine throughput and CircuitAI in the loop

Measured: 672 recorded games, mean length 658 game-seconds [M, `results/*/results.csv`, column `seconds`],
about 11 minutes.  A 16-game batch at 3x (8 in parallel, rev6) took about 23 minutes [M, demo timestamps].
So about 1,000 games per day, against the overview's 1,700 [O].

Per game, a seat makes about 1,300 decisions (2 Hz x 658 s).  Self-play gives 2,600 learnable decisions per
game, CircuitAI 1,300.  So about **2.6-4.4M learnable decisions per day for self-play** and 1.3-2.2M against
CircuitAI.  Against CircuitAI only one side learns, and at about 10% wins that is about 100 wins per 1,000
games, too few to learn from scratch but enough to check.

Proposed mix: 75% engine self-play at max speed (no CircuitAI, so no speed cap), 25% CircuitAI Hard at 3x.
At max speed, per-decision latency must be under about 60 ms of wall per decision (0.5 s of game time at 10x)
[E]; unmeasured.  HANDOFF records about 10x with 8 parallel games early on [V].  CircuitAI games are also the
gate (6.2).

### 5.4 Off-policy correction

Engine data come from the sim checkpoint and from the current engine policy; the lag is one iteration (2.3), and
PPO's ratio handles it.  Two checks:

* **Action-distribution gap.**  The same checkpoint, sim versus engine: per-head action frequencies, matched
  on army size and tech.  Report KL per head.
* **Fallback.**  If a head's engine KL is above 0.5 nats per decision, use V-trace with truncation rho = 1
  for engine data and log both.

### 5.5 Fallback: imitation from existing logs

What the logs record [V]:

| log | per record | orders? |
|---|---|---|
| `state-team<N>.jsonl` (statelog, every 10 s, `statelog.py:25`) | economy, force by role, structures, value grid | no |
| `fights-team<N>.jsonl` (every 2 s, `fightnet.py`) | entity snapshots and measured losses after 10 and 20 s | no |
| `deaths-team<N>.jsonl`, `engagements-team<N>.jsonl` | death causes, branches | no |
| `observe-team<N>.jsonl` (`bots/observer/main.py`) | CircuitAI's new units with positions, snapshots every 10 s | no |
| demo `.sdfz` (Recoil) | team statistics and **AI command packets** | yes, partly |

The demo does hold commands.  The server broadcasts AI commands (`RecoilEngine/rts/Net/GameServer.cpp:1252-1265`)
and `Broadcast` writes every packet to the demo (`GameServer.cpp:559-560`; `DemoRecorder.cpp:129`).  Confirmed on
`results/d1/game-001` (a 20-minute ControllerAI-vs-CircuitAI game) [M, `demo_scan.py`]: 208,356 type-14
(NETMSG_AICOMMAND) packets, 161,887 for AI id 1 and 46,469 for AI id 0.  Field offsets from `DemoTool.cpp:219-232`:
playerNum byte 3, aiID byte 4, unitID int16 at byte 5, command id int32 at byte 7, options byte 11, floats after.

Not decoded yet [U]: the command-id values I read do not look like the small engine ids (CMD_MOVE = 10,
CMD_ATTACK = 20, `RecoilEngine/rts/Sim/Units/CommandAI/Command.h:22,25`).  The layout needs checking against
DemoTool before use.  Unit ids map to types only through build commands or a replay; units from factories are
not typed.  Estimated 1-2 days of decoding [E].  The use is a warm start for the heads by behaviour cloning on
the rules bot's own demos, only if stage 2 stalls.

## 6. Evaluation

### 6.1 Sim metrics, every checkpoint

| metric | definition |
|---|---|
| win rate vs anchors | by seat |
| Bradley-Terry rating | over anchor games (4.3) |
| exchange X | value killed / value lost, sim, same definition as the engine's X |
| metal ratio M | our / their produced metal at 10 and 15 min (sim) |
| decisions to first win | sample efficiency |
| masked-action count | must be zero (a bug if not) |
| decisions per second | collection and training, against 2.6 |

### 6.2 Engine gates: how a checkpoint enters an arm

Verified in `tools/prun.py`: `--arm NAME=FLAGS` sets `ZKBOT_FLAGS` per arm (line 130); `--arm-env NAME:K=V` sets an
environment variable for one arm (lines 131, 196); the bot code is snapshotted at batch start (line 226).  The
weights are read from `ZKBOT_POLICY` when each game starts, so a policy arm copies its checkpoint into the batch
directory first.  Training keeps writing new checkpoints without changing a running arm.

    BASE=hitrun,mexpatrol,nearbuild,raidfirst,raidprobe,seeturrets,spearhead   # frozen rules baseline
    python tools/prun.py p1 --games 100 --jobs 8 --diff Hard \
      --arm rules=$BASE --arm policy=$BASE,policy \
      --arm-env policy:ZKBOT_POLICY=D:/zk-learned/gate/p1/ckpt-000123.npz \
      --enemy-factories cloak,shield,veh,tank,hover,amph,spider,jump

`--games 100` is 100 games per arm, 200 engine games, paired by game index (map, seat, factory).  At about
1,000 games per day, about 5 hours.  The policy arm controls the army units with the rules economy (stage 2).
The integration point is `bots/zkbot/military.py` and `bots/zkbot/armyplan.py` [V, both exist]; M4 chooses.

### 6.3 Game counts (computed)

Win rate at a base of 10%, sign test on discordant pairs, two-sided alpha 0.05, power 0.8 [M, `power.py`]:

| alternative win rate | pairs needed |
|---|---|
| 30% | about 70 |
| 25% | about 110 |
| 20% | about 215 |

Continuous metrics, paired log-ratio, normal approximation, alpha 0.05, power 0.8 [M, `power.py`]:

| per-game SD of paired log-ratio | detect +22% | detect +35% | detect +11% |
|---|---|---|---|
| 0.3 | 18 pairs | 8 | 71 |
| 0.5 | 50 | 22 | 197 |
| 0.7 | 97 | 43 | 385 |

Measured SDs of paired log-ratios from two existing batches [M, `pairsd.py`, via `tools/abstat.py`]:

| batch (arms) | M10 SD | M15 SD | X10 SD | X15 SD | games with M15 |
|---|---|---|---|---|---|
| `tuned-ab` base vs tuned, 20 per arm | 0.29 (n=18) | 0.51 (n=14) | 0.53 (n=20) | 0.51 (n=16) | 14 of 20 |
| `rp-ab` base vs raidfirst, 12 per arm | 0.25 (n=11) | 0.32 (n=10) | 0.57 (n=11) | 0.69 (n=10) | 10 of 12 |

Reading: X15 needs about 50-100 pairs to see +22%; M15 about 20-50.  Up to 30% of games never reach 15 minutes,
so use M10 and X10 when M15 is missing, and say so.

Pre-registered gate, per checkpoint, written before the batch: paired log X15 and log M15, effect 22%, 100 pairs.
Win rate is reported, not gated: at 10% a 10-to-20 move needs about 215 pairs.  The last A/B (18 per arm) could
only show about +40% at SD 0.5, which is why it showed nothing.

### 6.4 When evaluation runs

* Every checkpoint: sim anchor evaluation (6.1, 4.2).
* Before a checkpoint enters a gate: 2 engine smoke games at 3x for crashes, speed (at least 2.7x in the profile
  lines) and sane action counts.
* Full engine gate (100 pairs): at the end of each stage, and at a stage's midpoint if its sim curve is flat.
* Engine work is phased: while a batch runs, the trainer is paused (2.9).  Not during the user's own games.

### 6.5 Spectating the latest checkpoint on TitanDuel

Use `zk-startup.ps1`.  From its help: a `Player` member is you playing, and `-Headless` makes spectators.  A
windowed match with no `Player` should let the user watch [U: not run here; check once].  Pause the trainer first,
because the windowed engine and the trainer share the GPU and RAM.

    $env:ZKBOT_POLICY = 'D:\zk-learned\latest.npz'   # exported by tools/policy_export.py
    $env:ZKBOT_FLAGS  = 'hitrun,mexpatrol,nearbuild,raidfirst,raidprobe,seeturrets,spearhead,policy'
    .\zk-startup.ps1 ControllerAI CircuitAI:Hard -Map 'TitanDuel 2.2' -NoDownload `
      -Speed 3 -Client 'python bots\zkbot\main.py {team}' -OutDir 'results\watch-latest'

`-Speed 3` keeps the comparison at the 1x-equivalent speed (HANDOFF section 3).

### 6.6 Sim overfitting

Gap = (change in sim win rate or rating between two checkpoints) minus (change in engine paired log X15 or M15
between the same two).  After three checkpoints have engine gates, a positive gap in two successive pairs means the
sim rewards something the engine does not.  Check first: blip and memory handling (02), turrets and line of sight
(01 section 0 items), catalog values (`bots/zkbot/catalog.py`).  Also run stage 1's evaluator on sim and logged
fights whenever 01 changes; its AUC must not fall.

## 7. Experiment tracking

Runs live in `D:\zk-learned\runs\<run-id>\`.  C: has 14.8 GB free [M].  `results/` is gitignored
(`.gitignore:1`) but sits on C:, so runs go to D:.

    config.json          hyperparameters, passes K, map list, sim commit, code commit, seeds, config hash
    metrics.jsonl        one line per iteration (collection, training, diagnostics)
    evals/<ckpt>.json    anchor and engine results
    ckpt/                step-numbered checkpoints, immutable
    traj/                rollouts and engine trajectories, deleted after use
    control.json         run / pause / stop
    logs/                stdout, sim logs

Logging: JSONL with the standard library.  TensorBoard is not installed in `~/zkml` [V, `find_spec`]; a
plotting script that reads `metrics.jsonl` is enough.

Diagnostics:

| name | definition | healthy | trigger |
|---|---|---|---|
| entropy per head | mean over choice units | above 30% of max | K-controller raises the coefficient |
| explained variance | 1 - Var(return - V) / Var(return) | above 0.5 by mid-stage | below 0.3 for 20 iterations |
| KL, behaviour to learner | per-decision, batch mean | below 0.02 | above: lower lr or lag 0 |
| clip fraction | unit ratios clipped | 0.05-0.2 | above 0.3: lower lr |
| masked-action count | chosen actions outside mask | 0 | any: bug |
| action frequencies | per head, per action type | none above 60% | above: collapse |
| unit-mix JS | 4.4, per evaluation | below 0.3 | above: league (4.5) |
| return range | distribution of returns | inside [-1, 1] | outside: scaling bug |
| throughput | decisions/s, collection and training | 2.6 | below 70% of estimate: profile |

Reproducibility: seeds for sim and torch in `config.json`.  01 must make the sim deterministic per seed (its
determinism test: 1 vs 16 workers, identical frame hashes).  CUDA is not bit-deterministic; M0 checks that one
iteration reproduces within 1e-4 on the same machine.

What is committed:

| item | committed | rule |
|---|---|---|
| code, configs, scripts | yes | normal git |
| `tools/splits.json` | yes | small, fixed split (3.) |
| evaluation summaries (`evals/*.json`, gate outcomes) | yes, as `docs/learned-policy/runs/` JSON | small, the record of decisions |
| checkpoints | no | size rule below |
| shipped policy in `bots/zkbot/data/` | yes if at most 4 MB | fp16 2M parameters is 4 MB; 5M is 10 MB; larger needs the user's go-ahead |
| rollouts, trajectories | never | size |

Precedent: `bots/zkbot/data/fightnet.npz` is 643 KB and `valuenet.pt` 537 KB [V].  The 4 MB limit is this plan's
proposal.

## 8. Sequencing

| milestone | deliverable | acceptance | size |
|---|---|---|---|
| M0 toy env | PPO core (1.1-1.6) on a numpy toy env: 8 vs 8 units on a plane, same decision period and token interface; checkpoints; resume | learns focus fire: beats random 90% and "attack nearest" 60% within 2M decisions; one iteration reproduces within 1e-4; a kill mid-checkpoint keeps `latest.json` valid | medium |
| M1 binding | 01's C ABI wrapper; step, reset, raw frame views | 256 games step within 2x of 2.6; featurizer golden test on toy and 01 frames | medium (needs 01 M5) |
| M2 sanity, stage 2a | PPO on 1v1 to 4v4 | seat bias 45-55%; beats scripted attack-nearest 80% at 3v3 within 5M decisions; mirror self-play 50% +-5%; burn-in test (1.5) | medium |
| M3 stage 2b, 2c | logged-fight scenarios; fixed-economy army; anchors; cycle detection | gates in 3 | large |
| M4 engine arm | `policy/trajectory.py`; numpy runtime (02); `learn_gate.py`; prun arm | 2 smoke games at 3x, no crash, speed at least 2.7x; 100-pair gate runs | medium |
| M5 stage 3 | full game with economy and construction | 60% vs anchors at 1B, or stop | large |
| M6 stage 4 | randomisation (5.1) validated; engine fine-tuning (5.2-5.4) | engine gate passes at the pre-registered effect; sim-engine gap within bound | large |

M0 needs no sim and tests the PPO core, masks, segments and checkpoint logic.  M1 starts when 01 provides the
C ABI.  Until then, M0 and M2's toy scenarios are the only environments.

## 9. Risks and stop rules

**Budget, bluntly:** 50M decisions per day may not be enough for full-game play.  The full game is 0.5-2B
decisions by estimate, which is 10-40 days at 50M per day, and the rate falls to about 25-40M per day at 400 tokens.  Measure
these before committing to stage 3:

1. Sample efficiency in stage 2b: win rate vs scripted at 1M, 5M, 20M and 50M decisions.  Flat by 20M means
   architecture or reward, not compute.
2. The training rate of 02's real model at M2 (basis A, 1,700/s, or better), and the rate at 400 tokens.
3. Per-head entropy and KL over the first 1M decisions.  One collapsed head means the action space is wrong for
   PPO; 02 must change.
4. Lag KL (2.3) and engine decision latency at 3x (2.7).

| trigger | action |
|---|---|
| stage 1 gate fails | stop; fix 01's combat before any policy training |
| stage 2b below 60% after 50M, or 2c exchange not above 1.0 | rethink architecture or reward; do not add decisions |
| engine gate: 100 pairs, no effect at 22% on X15 or M15, though the sim showed one | stop engine work; fix the sim gap (6.6) |
| stage 3 below 60% vs anchors at 1B | stop the full game; keep the rules economy with the policy as army controller (the stage 2 product) |
| engine arm below 2.7x | the policy is too slow for the sync loop: smaller model or server path |
| memory kill during a batch | do not restart on your own (user rule) |
| gain disappears when the league starts | revert to current-only and report |

Other risks:

* **Unit-count dominance** of gradients even with per-decision weighting: log gradient norm against decision count.
* **Reward hacking** (building and trading without winning): watch unit-mix JS and X15; the 0.1 shaped floor and
  the engine gate catch it.
* **C: disk**: 14.8 GB free, and the WSL disk lives on C:.  A full disk can corrupt it.  Everything large goes
  to D:.
* **Bot cadence**: the bot's adaptive `block_n_frames` (`bots/zkbot/intel.py:259-265`) must be pinned to 5 in
  policy games, or decisions drift off the 15-frame grid.

## Objections to the shared contract

All are accepted in the overview, with the live model set to d=192 numpy (objection 3) and K=2 at 50M/day (objection 6).

1. **Timing (overview, Shared decisions, Timing).**  "Sim tick 6 frames. Policy decision every 15 frames."  15 is
   not a multiple of 6 (the least common multiple is 30).  The bot's blocks are 6 or 10 frames
   (`bots/zkbot/intel.py:259-265`; `brain.py:179`).  01 and 02 both propose tick 5 and decisions every 3 ticks;
   this plan adopts that.  Live bot block 5 in policy mode.  Lead to accept or reject; if 6 stays, the decision
   period becomes 12 frames (2.5 Hz) or 18 frames (1.67 Hz), and every number in sections 1.2, 2.6 and 5.3 changes.
2. **Reward (overview, Shared decisions, Reward).**  "metal value of own units and buildings plus metal value killed."
   Taking the difference of two such sums counts each kill twice: the kill raises the killer's killed term and
   lowers the victim's own value.  Proposed: the difference of own values only (1.3).  A separate kill term needs
   its own weight.
3. **Live play (overview, Shared decisions, Live play).**  "numpy or another light CPU runtime".  Measured numpy:
   40 ms for d=256, 4 layers at 150 tokens, 98 ms at 400 (2.7), over the 35 ms budget.  So either the policy is
   d about 128 (0.6-2M parameters, numpy in the bot) or stage 4 uses a GPU server.  The server is outside the bot
   but arguably breaks "no PyTorch dependency in the bot"; the lead decides.
4. **Token counts (overview, Measured so far).**  The 150 and 400-token benchmarks undercount: 460 entities plus
   120 regions is 580 tokens, and the median before sites and globals is about 215.  Benchmark at 250 and 600.
5. **Arithmetic (overview, Measured so far).**  100M decisions per day is about 20,800 games at 4,800 each, not
   24,000.  The real-engine rate of about 1,700 games per day is not what the latest batch showed: about 1,000 per
   day at 11-minute games [M].  Real learnable decisions are about 2.6-4.4M per day, not the 8M the 1,700 x 4,800
   figure suggests.
6. **Budget basis (overview, Shared decisions and Measured so far).**  The overview's 5M model with two passes at
   100M per day contradicts 02's measured 1,500-1,720 decisions per second for its own actor and critic.  This plan
   plans on 45-50M per day with two passes (2.6).  Lead to choose the pass count; 02 asked for the same.
7. **Stage numbering.**  Stage 1 trains no policy.  This plan treats it as the sim gate, and stage 2 as the first RL
   stage.

## Needed from 01 and 02

From 01 (names as in 01 section 4 where given; 03 adds nothing new):

* The batch step signature (4.4) and the reset scenario format (map id, start setup, seed): this plan calls
  `reset(scenario)` and `step(actions)` over 256 games.
* Decision tick: 5-frame tick, decisions every 3 ticks (01's proposal).  Command latency in ticks (overview's 4
  frames is Pluto's, not ours).
* Reward inputs: the tracked-unit rule of 1.3 (real units, no blips, no memory); `buildProgress`; cost in metal.
* Determinism per seed (01's test).
* Confirmation that any `maps.json` terrain loads (3, [U]).
* The scripted attack-nearest baseline lives in 01's package.
* The hidden (critic-only) records must not reach the actor's inputs (02 section 11).

From 02 (names as in 02 sections 2 and 11):

* Token feature dimension `D` (this plan assumes 64, fp16), the GRU sizes (global 384, per-region 32, 02 section 3.4),
  and the critic's privileged tokens.
* Choice-unit definition and mask format (for per-unit ratios, 1.1); head list with action counts (for entropy
  floors).
* The live path: numpy model size (up to about 400 tokens within 35 ms) or the GPU server.
* Segment length 32 and stored state, no burn-in (1.5, as 02 proposes).
* Measured training rate of the real model (2.6 basis A) and the pass count in the config.

## Not verified

* Pluto's compute, hyperparameters and reward weights: none in the README, online or local [U].
* Gym-muRTS figures are from the ar5iv HTML (Table III, Section V and VI).  The PDF text was not readable; the
  abstract says 60 hours and Section VI gives 37-117 hours.
* The overview's inference rates (11,400 and 2,700 decisions/s) and 3,400 samples/s: not re-run here.  02's
  training rates were measured by 02, not here.
* The 0.5 ms sim cost per decision is derived from a benchmark with no full sim.
* Numpy speeds are for a naive forward; the real one could be faster or slower.
* Demo command-id decoding (5.5): packets are present; the id field is not decoded.
* Windows-to-WSL localhost for a GPU server; the windowed spectating behaviour (6.5).
* Self-play throughput at max speed and the 60 ms latency it needs (5.3).
* Whether the trainer stays under its memory and VRAM caps in a real run.
* Burn-in value and whether stored GRU states are good enough without it (1.5).

## 10. As built (T-M0, T-M1; 2026-11-05)

`tools/policy_train.py`, with `policy_model.py`, the toy in `policy_toy.py` and `policy_toy_env.py`, and the sim in `policy_sim_env.py`.

### 10.1 What changed from the plan

* Featurizing runs in forked worker processes, which run ahead of the GPU during training.  Threads couldn't help under the GIL.
* Games carry on across iterations with their recurrent state.  Real games last far longer than an iteration collects per seat,
  so restarting them would never reach the late game.
* Minibatches group segments of similar size, jittered by 15%, so padding stays small, and then go in random order.
* Evaluation counts each slot's first game only.  Otherwise games that end soonest count most.
* `--token-budget` is a setting.  A toy decision has a fifth of the tokens, so 16,000 gives it the real game's decisions per minibatch.
* The toy's tick is whole-array and decided from the tick's starting positions.  The earlier per-unit loop moved one side first,
  which leaned 60-40 in a mirror; now 516-481 over 1,000 games.

### 10.2 Checks

| Check | Result |
|---|---|
| One iteration twice | 0 difference with `use_deterministic_algorithms`; 6e-04 without, from scatter adds accumulating in no fixed order, which Adam magnifies on near-zero gradients.  Long runs use the fast mode. |
| Killed while saving, 12 times | `latest.json` named a loadable checkpoint every time; leftover `.tmp` files are removed on the next save |
| Toy scripts | focus beats nearest 1000-0, nearest beats random 994-6 |
| Toy learning | The win marks are met, but the policy didn't find the tactic the toy was built to teach; see below |
| One iteration on sim frames | 4,131 decisions, no masked action chosen, collection 1,085/s at 64 games |

**What the toy showed.**  Two runs, 8 against 8.  Wins are from 64 games each; the scripts are 200 games, and focus beats nearest in all of them.

| | Beats nearest | Beats focus | Enemies sharing its fire | Its first kill | Its first loss |
|---|---|---|---|---|---|
| Script: nearest | 50% | 0% | 2.78 | decision 7.3 | 7.5 |
| Script: focus | 100% | | 1.30 | 3.0 | 8.5 |
| `toy-1`: all kinds, 848k decisions | 95% | 36-41% | 2.80 | 7.7 | 12.3 |
| `toy-focus`: KEEP and ATTACK_UNIT only, 1.0M decisions | 89% | 16-22% | 1.90 | 5.6 | 7.0 |

- **The pointer head learns.**  Given only targets to choose, `toy-focus` concentrates its fire and kills sooner, without being told to.
  It got less than halfway to the script's concentration in 1M decisions, and reached 84-89% against nearest rather than 90%.
- **`toy-1` won another way.**  It kills no sooner than nearest and spreads its fire just as widely, but loses its first unit 5 decisions later.
  ATTACK_UNIT is 2% of its orders, with STOP 27% and ATTACK_GROUND 21%.  That points to dodging the script's fire, perhaps by pulling hurt
  units back so the script retargets, but this wasn't measured directly.
- **What this means for later stages.**  Winning against a fixed script says little about the tactic behind it: an exploit can get there
  first, and `toy-1` still loses to focus fire.  So Gate 2 should test against several scripts, including one that focuses fire, as well as
  measuring how the policy wins.  Self-play ought to punish script exploits, but nothing here shows it does.

**Learning, on the model with one pool and attack-target inputs (2026-11-05).**  One seed can't rank settings.  Targets only, wins against nearest during training at about 490k decisions:

| Run | Pair table trained | Minibatch | Wins |
|---|---|---|---|
| `toy-focus`, earlier model, seed 1 | every step | 749 | about 55% |
| `toy-aim-focus`, seed 1 | 1 in 8 steps | 548 | 17% |
| Same settings, seed 2 | 1 in 8 steps | 551 | 55% |
| Pair table every step, seed 1 | every step | 550 | 25% |
| Bigger minibatch, seed 1 | 1 in 8 steps | 751 | 37% |
| Both, seed 1 | every step | 749 | 65% |

To 850k-1M decisions, `toy-aim` beat nearest 73% and focus 17-22%, with 2.70 enemies sharing its fire; `toy-aim-focus` beat nearest 55-61% and focus 0-3%, with 2.10-2.27.
Neither found focus fire.  Whether the attack-target inputs help isn't known: a comparison with them on and off, four seeds each, ran out of memory with seven trainings at once, and its partial curves show no clear difference.

**First sim run, `sim-short` (2026-11-05).**  2M decisions on the five maps, 256 games, concurrent, one pass, 40 units a side, fog on, 978 decisions/s, no masked action chosen.  Wins against the fight script rose from about 3% to about 15-20% during training; tests of 64 games
gave 5%, 8% and 14% at 660k, 1.3M and 2.0M.  Explained variance stayed between 0.06 and 0.56 with no steady rise, and entropy rose on the kind, enemy and region heads.  It isn't tested against anything else, or in the engine.

### 10.3 Training rate: the miss

Real frames from live games (about 160 tokens, 33 controllable), 24 segments of 32 steps, 6.3 GB of GPU memory:

| Part of a minibatch | Seconds |
|---|---|
| Backward | 0.94 |
| Encoder | 0.43 |
| Heads | 0.28 |
| Critic | 0.24 |
| Transfers and waiting on workers | 0.19 |
| **Rate** | **about 360-400 decisions/s per pass** (plan: 1,700/s) |

What helped: backward through `index_select`, whose backward is an `index_add`, where plain indexing's took 1.65 s; a lookup in a
small pair table.  What didn't: `torch.compile` on the attention blocks (405/s against 400/s, though it may have run uncompiled past the recompile limit; see below).  Bigger minibatches don't either:
230k tokens needs 15.8 GB against the card's 12 GB, and fell to 139/s.

The cost is mostly per recurrent step, not per frame, because the toy's far smaller frames cost nearly the same per step:

| Segments x steps | Decisions/s per pass | Peak GPU |
|---|---|---|
| 24 x 32 (the plan) | 359 | 6.3 GB |
| **48 x 16** | **597** | 6.5 GB |
| 96 x 8 | 655 | 6.5 GB |

Estimated decisions per day with 65,536-decision iterations, using the measured 1,085/s collection and no evaluation.  Collection
at 256 games isn't measured and should be faster, so these are floors:

| Setting | Iteration | Per day |
|---|---|---|
| 32 steps, two passes (the plan) | about 400 s | **about 15M** (plan: 45-50M) |
| 32 steps, one pass | about 225 s | about 25M |
| 16 steps, two passes | about 280 s | about 20M |
| 16 steps, one pass | about 170 s | about 33M |

Shorter segments cut how far gradients reach through memory: 16 steps is 8 seconds of game instead of 16.  Whether that costs
anything isn't known yet.  Stage 2 can test it, because the toy has no fog and can't.

Where it stands (2026-11-05, 256 sim games, concurrent, 32 steps, one pass, 40 units a side, measured over a few iterations; the sim's rates, not the real frames' above):

| Change | Decisions/s, overall | Per day |
|---|---|---|
| Caps at 256 and 256, all maps | 679 | 58.7M |
| The five training maps (`TRAINING_MAPS`) | 841 | 72.7M |
| One pool with per-unit ids | 831 | 71.8M |

These frames have fog, so few enemies are in them.  With every enemy seen, the same setup gave 584-644/s at 40 a side and 58-61/s at 256 a side (02 section 2.4).

**`torch.compile`, tried properly and removed (2026-11-06).**  On the sim, `--concurrent`, 12 iterations each, last six:

| Setup | Decisions/s | Training | Collecting | Recompiles | Past the limit |
|---|---|---|---|---|---|
| No compile | 866 | 37.2 s | 32.2 s | 0 | |
| Attention blocks, one shared compile | 817 | 39 s | 34.8 s | 8 | yes, so mostly uncompiled |
| Blocks, compiled apart for training and playing, each with its own limit | 799 (8 iterations, then hung) | 40.5 s | 33.0 s | 17 | no |
| The same, `dynamic=True` | 760 | 41.7 s | 31.5 s | 22 | no |

* Recompiles came from grad mode, the actor and critic sharing a `forward`, `bias` needing gradients on only some steps, frames of one decision, token counts divisible by 16, and sizes.  `dynamic=True` traded the last for more of the others.
* Compiled and plain blocks agree to within rounding.  `expandable_segments` and `garbage_collection_threshold` work with it.
* It didn't speed up training, one run froze just after a recompile (probably two threads, no stack trace), and with `dynamic=True` the main process grew from 3.3 to 4.6 GB in 12 iterations.

**Memory between runs.**  Nothing leaks: after an iteration a run holds 51 MB (toy) or 67 MB (sim), with live tensors flat.  What grew was reserved memory, by fragmentation, as batch shapes changed: 7.9 to 9.3 GB in 4 iterations on the sim, against a peak in use of 6.7 GB.
Past the card, memory spills into system memory at about a fifth of the speed, and training took about 10x longer per iteration.  `expandable_segments`, `garbage_collection_threshold:0.8` and `empty_cache` after each iteration are now defaults: peak reserved stays within 7.5 GB.  The limit is per process, so runs sharing the card must pass `--gpu-runs`.
They cost no speed: over the same 12 iterations on the sim, 866 decisions/s with them and 865 without, with peak reserved at 7.5 and 9.8 GB.  So they don't explain why `sim-short` ran at 978/s; that difference is still unexplained.

### 10.4 Considered and not adopted (2026-11-05)

Measured on the same real frames, 24 x 32:

| | Decisions/s per pass | Peak GPU |
|---|---|---|
| Actor and critic, as now | 405-413 | 6.28 GB |
| No critic at all, the most a critic-free method can save | 461-469 | 5.01 GB |
| Critic trained after the recurrent loop, in flat batches | 283-289 | 5.01 GB |

* **GaLore 2:** gradients are 13 MB and Adam's state 26 MB, together 0.6% of the 6.28 GB peak, and `optimizer.step` takes 0.003 s of
  a 1.9 s minibatch.  The memory is the graph of activations kept for backward through 32 steps, which low-rank gradients don't touch.
  It targets models whose optimizer state dominates, and it would add subspace updates here.
* **GRPO:** dropping the critic gains about 14% speed and 1.3 GB.  In exchange it needs a group of samples from the same state with
  comparable rewards.  Here that means branching the sim, which it can't yet do, and rolling each branch out, so collection costs
  the group size times over for the same number of distinct states.  One advantage would then cover every action of every unit in
  a rollout.  It would also lose the privileged critic, which is how training handles fog (section 3.6 of 02).  16-step segments
  gain 66% with no such cost.
* **The flat critic** was my own idea for removing the critic's per-step overhead.  It's slower, because it featurizes every frame
  again and pads to the largest.

### 10.5 Linear attention, for memory between frames (2026-11-05)

Not adopted.  Attention within a frame stays: tokens are an unordered set and the pair bias needs a per-pair score.  Swapping the
GRU in place gains nothing, because each frame's attention reads the previous memory, so fused kernels can't span frames.  A flat
design with no memory input to attention reached 784-844 decisions/s against about 495 (actor only, heads without memory), with
the plain GRU.  That gain comes from going flat and sorting frames by size, not from linear attention.

One layer each, d=192, 4 heads, random input, forward + backward + Adam under bf16, `flash-linear-attention` 0.6.0 in chunk mode.
Memory is peak above the weights and Adam state.

| Model | 24 x 32 | 24 x 160 | 8 x 1024 | 2 x 4096 | Memory at 8,192 tokens |
|---|---|---|---|---|---|
| Self-attention, causal, torch SDPA | 474k tokens/s | 2,097k | 4,618k | 4,081k | 47 MB |
| MLA, ours on torch SDPA, not linear | 384k | 1,535k | 3,123k | 3,514k | 51 MB |
| GRU, torch | 226k | 358k | 229k | 51k | 62 MB |
| **DeltaNet** | **129k** | **555k** | **886k** | **1,138k** | **76 MB** |
| Gated DeltaNet | 87k | 445k | 955k | 1,027k | 86 MB |
| Kimi Delta Attention | 70k | 374k | 748k | 770k | 139 MB |
| RWKV-7 | 72k | 208k | 792k | 783k | 237 MB |
| Mamba2, slow fallback only | 17k | 75k | 117k | 116k | 3,432 MB |

Mamba2's fast path and `fla`'s own MLA need `mamba-ssm`, `causal-conv1d` or `flash-attn`.  None has a prebuilt wheel, and there's no
CUDA toolkit here to compile them.  If long memory becomes the limit, try Gated DeltaNet first: it's close to DeltaNet and can
forget stale sightings.  Self-attention over a window is faster in training, but its live cost grows with the window.
