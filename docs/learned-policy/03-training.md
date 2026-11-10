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

Opening bonus: none (user exclusion).  Diversity bonus: none in v1, it said here; added at the user's request
on 2026-11-08 (11.19), inside the shaped term, zero-sum: `r_A` gains (1 - beta) times A's bonus less B's.  At
its default its share of the shaped term's magnitude was about 2% a game on script games, 14% at most.

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
* **Early end.**  From 2 minutes of game time, a side with more than 2.5x the other's metal value (cost x build progress, as in 1.3), and at least 10,000 metal ahead, wins
  there and then.  The engine doesn't have this rule, so win rates in the sim aren't comparable with the engine's without it.
* **Seats and starts.**  One set of weights controls both sides.  Each game draws a seat and a start for each
  side, so TitanDuel's asymmetry averages out.  Report by seat.  Sanity check (M2): self-play with the
  start-A seat wins 45-55% over 200 games.
* **Scripted baseline.**  Built, differently from the plan (2026-11-06, 01 section 8.9): the same opening on both sides, then on the script's side every finished fighter
  attack-moves to the enemy's start, re-ordered every 3 s; it has since learned to finish, retreat and group (10.2).  It lives in `tools/policy_script.py`, not the sim.  The rules bot's army code does not port, so the script isn't the rules bot, and nothing has played the rules bot in the sim.
* **Factories.**  Sampled per game as prun does (`--enemy-factories`).  Stage 2 uses the factories a scenario
  names.  As built, each seat draws evenly from the factories that suit its map (`Ground.suits`, 11.1).
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

**The opening game, `opening-smoke` (2026-11-06).**  The fixed armies are gone: both sides play the scripted opening of 01 section 8.9, and the policy controls only the raiders.  726k decisions, 256 games, concurrent, one pass, no masked action chosen.

| Side 0 against the fight script, 256 games unless noted | Wins | Draws | Losses | Median game |
|---|---|---|---|---|
| Does nothing: its raiders stand where they come out | 25% | 14% | 61% | 462 s |
| The script itself | 30% | 40% | 30% | 910 s |
| The policy at 660k decisions, one test of 64 games, before Kodachis joined the opening | 64% | not logged | not logged | not logged |

So the policy passed doing nothing quickly: for comparison, the fixed armies had 8% at 1.3M.  **How it wins hasn't been looked at.**  It kept its order 84% of the time, and from 2 minutes a side wins at 2.5x the metal value, so holding raiders back while the script trickles in may be most of it.  One test of 64 games is about +/- 12 points.  Explained variance reached 0.96, higher than `sim-short`, probably because much of the value is now in a slow, predictable economy, so it says less about fights.

Draws matter more now: two scripts draw 40% of games at the 20-minute limit.  A long match could be set up differently, but nothing has changed there yet.

`sim-1v1` was stopped at iteration 20 of about 4,600 when this replaced its game, and its checkpoints don't apply to it.

The table above was on the old starts, with half of Faultline's metal missing, no overdrive and the old early end (01 section 8.9).  Baselines on real start boxes with overdrive, 256 games:

| Side 0 against the fight script | Early end | Wins | Draws | Losses | Median game |
|---|---|---|---|---|---|
| Does nothing | 2.5x | 18% | 7% | 75% | 360 s |
| The script itself | 2.5x | 39% | 30% | 31% | 613 s |
| Does nothing | 2.5x and 10,000 ahead | 15% | 20% | 65% | 782 s |
| The script itself | 2.5x and 10,000 ahead | 24% | 52% | 24% | 1,200 s (the limit) |
| Does nothing, against the script below | 2.5x and 10,000 ahead | 2% | 15% | 83% | 680 s |
| That script itself | 2.5x and 10,000 ahead | 43% | 20% | 37% | 534 s |

**Stalemates (2026-11-06).**  Under the new early end, half of script-against-script games reach the 20-minute limit.  No game of 512 ended by elimination, so every win was the 10,000 lead, at a median of 766 s.  Two things cause it, and neither is units being stuck: of the 4,811 fighter sightings that stood still for a minute from 10 minutes on, every one was at the enemy's start.

1. **Even trickles, in most draws.**  Both factories were alive in 83 of 117 draws and both commanders in 89.  Each side makes about 280 units at about 17 metal per second and they meet half way: the furthest side's front stays at 0.5-0.6 of the way across, and a side's value stays near 3,300 from 5 minutes to 20.  Matchups decide it: mirrors draw 83% and mixed factories 40%.  Of decided games, tank won 89%, spider 85%, shield 66%, amph 60%, veh 56%, hover 6%, cloak and jump 3%.  Whether that spread is the game's or the sim's isn't known, since the replay gate failed on fights (01 section 8.1).
2. **The script stops at the enemy's start.**  An army that gets there stands at the point it was sent to.  A side loses by elimination only when every unit is dead, as in Zero-K (`game_over.lua`), and the extractors and solars up to 1,300 elmos away are left alone.  So the winner can only win by piling up a 10,000 lead, which takes at least 10 minutes at this income.

An experiment sent fighters that reach the start on to the enemy's nearest building: draws fell from 52% to 38%, and 67 of 256 games then ended by elimination.

**The script now (`tools/policy_script.py`).**

* **Finishing.**  Once any of a side's fighters is within 500 elmos of the enemy's start, they all attack-move to the enemy's nearest building, and to whatever remains when no building does.  The script knows where buildings are without scouting: it's a training opponent.
* **Retreat.**  A fighter whose def regenerates moves to the rally point below 40% health and stays until 90%.  By the data that's every raider, at very different cost: from 40% to 90% takes a Glaive about 11 s, a Kodachi 76 s, and most others 90-134 s.
* **Groups.**  From 3:00 a new or recovered fighter attack-moves to a rally point past the factory, and those waiting go together once they're worth 400 metal at 3:00, 200 more each minute, 3,000 at most.  Metal rather than units, so groups take as long to gather whichever factory was drawn: at 17 metal a second, 25 s at first and about 3 minutes at the largest.  Nobody waits over 3 minutes, so that a side without income doesn't hold its last fighters for ever.

With it, 167 of 256 script games end by elimination, 38 on the lead and 51 at the limit.  Mirrors draw 44% and mixed factories 13%.  Of decided games spider won 91%, tank 77%, shield 64%, amph 53%, veh 43%, jump 28%, cloak 13% and hover 8%, so regeneration helped Glaives (3% before) and the spread is still wide.  Whether it's the game's or the sim's isn't known.

`script-smoke`, the first 725k decisions on this script (before section 01 8.10): 33% wins in one 64-game test at iteration 20, against 2% wins for doing nothing and 43% for the script itself.  It was 632 decisions a second late on, with no masked-out actions.

`m8-smoke`, the same run after 01 8.10, 725k decisions: **16%** wins in the same single test, against 1% for doing nothing and 36% for the script itself.  Rate 524 a second late on, train 62 s and collect 47 s per iteration, 887 decisions a minibatch against 1,065, and no masked-out actions.  Explained variance reached 0.93.  Its share of finished games against the script rose from nothing to 0.2-0.75 over the last ten iterations, on few games each.

Neither test says much alone: one test of 64 games has a standard error of about 5 points, and each run is about 0.5% of the 150 million decisions a real run would have.  What does hold is the cost: 17% fewer decisions a second, in both training and collecting.  The likely cause, not measured apart, is up to 32 more tokens a side where wrecks lie.  FrostyCove starts with 138 features.

Faultline's start boxes differ from one load to the next, and every run before `m8-smoke` saw one draw of them (01 section 8.9).

### 10.2a Against the script, then against itself

`SelfPlay` in `tools/policy_train.py`.  Training starts wholly against the script.  Every 10 iterations a test of 64 games counts wins alone, with draws and losses logged beside it (`metrics.jsonl`) and each game's map, factories, length and ending in `tests.jsonl`.

* **Starts** when the last two tests average 80% wins.  Two because one test has a standard error of about 5 points.
* **Grows** by 1% of new games each iteration, so about 100 iterations from none to all.  Games under way stay as they are.  A self-play game gives twice the decisions of one against the script, so the share of decisions rises faster than the share of games.
* **Pauses** while those two tests average under 70%.  It never goes back.
* The test stays against the script throughout, so it stays comparable, but once self-play dominates it no longer measures everything the policy is learning.

`--hours` stops a run after that long, as well as at `--decisions`.

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

On the opening game the rate is 728 decisions/s, about 63M a day: 44.6 s training and 37.0 s collecting per iteration, against 866, 37.2 s and 32.2 s on fixed armies.  The sim itself isn't the cost: 70,000 seat steps/s, script against script.

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

### 10.6 `selfplay-6h`: no measurable learning (2026-11-06)

`--games 256 --passes 1 --iteration 32768`, GRU, against the script, on the 5 training maps.  Stopped by hand at iteration 221 after 4 h 52 min: 7,241,728 decisions, about 413 a second overall.  Self-play never began, as no test came near 80%.  The run used ABI 8, from before shields and the unit AI, so its numbers don't describe the present sim.  `/mnt/d/zk-learned/runs/selfplay-6h`.

**Tests, 64 games every 10 iterations:**

| Iterations | Wins | Draws | Losses |
|---|---|---|---|
| 10-50 | 23.1% | 21.6% | 55.3% |
| 60-100 | 22.5% | 23.7% | 53.7% |
| 110-150 | 34.4% | 16.3% | 49.4% |
| 160-220 | 25.0% | 21.2% | 53.8% |

The best single test was 28 of 64 at iteration 110, and the last 10 of 64.  The first and second halves, 704 games each, are 24.7% and 27.6% wins, which is within noise.  Training-game wins rose from 19% to 46% at about iteration 110 and fell back to 28%.

**The factories decided the games**, as they do for the script (`01-simulator.md` 8.12):

| Factories | Games | Wins | Draws | Losses |
|---|---|---|---|---|
| Own Spider or Tank, enemy neither | 273 | 89.4% | 8.8% | 1.8% |
| Enemy Spider or Tank, own neither | 302 | 0.0% | 1.3% | 98.7% |
| Both Spider or Tank | 84 | 21.4% | 35.7% | 42.9% |
| Neither | 749 | 14.2% | 31.2% | 54.6% |
| Same factory on both sides, of those above | 167 | 8.4% | 33.5% | 58.1% |

In same-factory games the first and second halves are 8.4% and 8.3% wins, where the script against itself wins 37% (on the present sim).  So the policy is well below the script in a fair matchup, and didn't improve.

**The policy barely moved:**

* KL per iteration stayed at 0.001-0.002 and clip fraction at 1-3%, with 1 pass, a learning rate falling from 3e-4 and gradient clip 0.5.
* The fine-position head stayed uniform: entropy 2.195 against ln 9 = 2.197.
* The kind head's entropy rose from 0.91 to 1.20, and KEEP's share fell only from 87% to 83%.
* Explained variance reached 0.995, and stayed there while the policy didn't improve, so it says nothing about skill.
* `beta` went from 0 to 0.9, so reward moved from value change to the win itself as training went on.  It may account for the rise and fall in training wins, but I didn't test that.

**What this suggests, none of it tested:**

* 80% against the script can't be reached while the factory is drawn at random for both sides, unless the policy far outplays it.  A fair test holds the factory equal, or scores each matchup against the script's own record.
* Check the raider matchups against the engine before training on them, since Spider's 92% may be the sim's.
* More updates per decision: 1 pass at this KL is a very small step.
* Behaviour cloning from the script before PPO, as the user proposed, would start the policy near the script instead of far below it.

## 11. The frame model (2026-11-06): data collected, long run not started

`tools/frame_data.py`, `tools/frame_jepa.py`, `tests/policy/test_frame_jepa.py`.  It compresses what a seat sees, from up to several hundred tokens, into one vector for the main model.  It follows LeJEPA (Balestriero and LeCun, [arXiv 2511.08544](https://arxiv.org/abs/2511.08544); the Epps-Pulley and slicing code follows [the authors' repository](https://github.com/rbalestr-lab/lejepa) and [stable_pretraining](https://galilai-group.github.io/stable-pretraining/_modules/stable_pretraining/methods/lejepa)).  Two million frames are collected on 298 maps.  Only 5,000-step checks have been trained, and **those don't beat an untrained encoder on most probes** (11.3).

### 11.1 Data

    python tools/map_gen.py real --facts D:/zk-learned/maps/real_facts.json --out D:/zk-learned/maps    # Windows Python: reads map archives
    python tools/map_gen.py generate --count 200 --seed 1 --out D:/zk-learned/maps
    python tools/map_gen.py stats --dir D:/zk-learned/maps
    wsl ~/zkml/bin/python tools/frame_data.py collect --maps /mnt/d/zk-learned/maps --out /mnt/d/zk-learned/frames/pool2 --script-frames 1000000 --scenario-frames 1000000 --games 256 --threads 16 --workers 8

Both sources take both seats and see under fog, with `economy=True`.

**The collection in `/mnt/d/zk-learned/frames/pool3`** (2026-11-07, 25 GB, 18 minutes beside a training run): as below, with everything that can stand on the board let in, and a scenario's structures up to 6,500 metal, which takes in a Singularity and the big guns and no superweapon.  Scripts now throw with Lobsters, flip Newtons and Sumos between pushing and pulling, send a Djinn ahead of a beacon with others waiting at it, and build missiles in silos (`FrameFight`, `BaseScript`; `01-simulator.md` 8.17).  Feature version 5, as the token layout hasn't changed, so `pool2` and `pool3` are the same shape.  Share of frames holding one, from scripts and from scenarios, of 1M each:

| Unit | Script | Scenario |
|---|---|---|
| Sumo | 1.7% | 4.8% |
| Lobster | 2.9% | 8.1% |
| Djinn / its beacon | 2.1% / 1.9% | 7.4% / 5.1% |
| Newton | 11.6% | 11.4% |
| Missile silo | 0.6% | 6.1% |
| Each kind of missile | 0.01-0.03% | about 0.9% |
| Fusion | 11.2% | 10.1% |
| Singularity | 0% | 2.5% |
| Cerberus | 0.2% | 3.8% |
| Heavy artillery (`staticheavyarty`) | 0% | 1.4% |
| `turretheavy`, `turretantiheavy` | 1.8%, 0.4% | 5.9%, 4.6% |
| Those with unmodelled weapons (`jumpblackhole`, `veharty`, `vehcapture`, `bomberassault`) | 1.2-3.6% | 5.8-7.6% |

`turretsunlance`, `zenith`, `dronecarry` and `hoverminer` didn't appear.  Missiles are rare from scripts as games are short: 64 script games over 12.5 minutes finished 43, with no draws, at a median of 8.7 minutes, and a silo costs what a side doesn't often have by then.  `pool3` hasn't been trained on as of this note.

**The collection in `/mnt/d/zk-learned/frames/pool2`,** feature version 5, replacing `pool1`, since deleted, which the trainer would no longer accept because unit tokens gained jump charge: 1M frames from each source, 25 GB, 15 minutes on 8 processes at about 2,250 and 2,630 frames a second (one process manages about 420).  A frame takes about 13 kB.  At the 50th, 90th and 99th percentile a frame has 27, 70 and 129 unit tokens, up to 236, and 44, 94 and 134 regions, up to 148.

| Share of frames with any | Script | Scenario |
|---|---|---|
| Own aircraft | 23% | 36% |
| Own ships | 3% | 7% |
| A map with water | 49% | 47% |
| Wreck metal in sight | 85% | 84% |

**The maps, `D:/zk-learned/maps`, 1.3 GB, 7.3 GB once loaded, 37 s to load.**  Forked processes share them.

* **98 real maps,** made from the height maps of every archive that has them, with start boxes and spots from a game (`sim_engine.py facts --out`).  Nine are left out: six make their terrain in the game, and three have no start boxes or spots.
* **200 made-up maps** (`tools/map_gen.py`, `Rules`): rolling, plateaus, ridges, craters or canyons, then no water, lakes, a river, a central sea, a coast, islands or open sea, and rot180, mirror, diagonal or no symmetry.  Starts are flattened, ramps join what walkers couldn't reach, and maps that still fail are drawn again.  Open-sea and island maps need only hovers and amphibians to connect.
* **Drawn half real and half made,** with maps of 3% water or more 1.5 times as likely.  52% of frames are on those.
* **21 maps are held out** (`held_out_maps`): TitanDuel, and a tenth by name's hash, of which 5 are real.  No latent or probe fits on them.

| 10th, 50th, 90th percentile | Real | Made |
|---|---|---|
| Relief | 192, 392, 704 | 216, 464, 992 |
| Water share | 0, 0.05, 0.46 | 0, 0, 0.47 |
| Spots per area | 0.16, 0.28, 0.49 | 0.20, 0.36, 0.48 |
| Regions | 18, 36, 67 | 27, 58, 101 |
| Sea regions | 0, 0, 13 | 0, 0, 5 |
| Kbots can go on | 0.54, 0.92, 0.99 | 0.72, 0.94, 1.00 |

Made maps are a little rougher, with more regions but fewer sea regions and walls.  That's left as it is: it widens what the encoder sees, and half the frames are on real maps anyway.

**Script games** (`FrameGames`) are whole games on any of those maps.  `WideOpening` opens with any factory that suits the map, and factories that use the water are 3 times as likely where 15% of the map or more is water, so ships, hovers, amphibians and aircraft all come up.  Ships are half as common as they were, since a shipyard now needs a real sea at both bases.  `BaseScript` is no plan, only so frames look played: factories change what they make every 30-120 s, and idle builders build, assist or reclaim within 1000 elmos, with more expensive things as the clock runs.  `FrameFight` keeps each unit's orders on ground it can reach, so ships stay at sea.  96 games over 2400 decisions: 126 ended, median 639 s, 85% with a winner.

**Scenarios** are built by `draw_side` in `ScenarioRules`, and play for 24 s, taken from 2 s in, every 6 s, and again 2 s later.
  * **Value:** each side gets 150-40,000 metal, log-uniform, up to 240 units.
  * **Army:** 1-3 clusters, at home, at the enemy's start, between them, or anywhere.  70% of sides keep to one factory's roster, the rest draw from all.
  * **Base:** a factory 80% of the time, a second likelier from 3,000 metal and a third from 12,000, 0-8 extractors on spots, other structures up to 1500 metal each, none on each other or on a spot.  Turrets may stand at home, by a spot, or on the way to the enemy.
  * **Wear:** 40% of units hurt, 10% of structures part built, 85% of sides with a commander.
  * **Wrecks:** 60% of sides have lost things, up to half their value in up to 40 wrecks, valued from the catalog, a quarter of them shot down to heaps.
  * **Memory:** each side has seen 0-100% of the enemy's structures, and those read as remembered until seen again.
  * **What stands where comes from the unit data** (`Ground`, `Footing`): water depth limits and `floatOnWater`.  So a shipyard goes on deep water, however far that is from the base, a solar never does, and a floating turret can be either.  A factory that doesn't suit the map isn't drawn.
  * **Which factories suit a map** (`Ground.suits`, `tools/policy_opening.py`), from what the units can reach, not from how much water there is: Akilon is a third water and land still reaches all its metal, and on Sailaway land reaches 2%.  The factory has to have room to stand on the map.  Air then always suits.  Ground units need 4% of the map to move on, a path to the enemy, and at least half the metal the best class could reach from that start.  Ships need a sea of 20% of the map within 1500 elmos of both bases.  Over the pool's 596 starts: a median of 10 factories offered, at least 3, and always one on the ground or water.  Sailaway, Expanded Tropics and Shimmershore get amphibians, gunships, hovers, planes and ships.  Hotstepper, Deserted Third and Painted Redux get no tanks or vehicles.  Hover and Amph still count on dry maps, since they work there.  The training maps keep all 8 land factories, drawn evenly.
  * **Jumping:** `FrameFight` turns a charged jumper's order into a jump when it's 150 elmos or more away, so frames show it.  `FightScript`, which the policy trains against, doesn't jump.

Frames also carry probe values, including the truth the seat can't see, and `outcomes.npz` holds how each game ended.

**Scenarios aren't quite what a game gives:**

* The clock is invented from the side's value, at 15 metal a second, because nothing in the sim sets it.
* No unit is part way out of a factory at the start, and there's little radar history.
* The fight script gathers as if at the start of a game.

**Two fixes this needed:**

* The sim's records leave every metal spot "free".  `with_site_status` works out each seat's view from the extractors it knows of, as the live bot's Brain does.  Training the policy still has economy features off, so this applies only here.
* Extractors added before `set_economy` earned nothing.  `set_economy` now works their income out again.  Ordinary games are unaffected since they start with commanders only.

### 11.2 The model

`FrameEncoder`, 1.65 million parameters at the defaults.  Each token group goes through its own input layer, with sines of its map position, then 24 learned queries read all tokens twice, talking among themselves after each read.  Those queries, flattened, go to a 256-number `z`.

* **Cost is tokens x queries**, not tokens squared.  Self-attention over tokens is there as `token_layers`, off by default.
* **No pair bias.**  Positions come from the token features, not the main net's distance buckets, so the data needn't store them.
* **Same output whatever the token order or padding**, tested.
* **Reads a `policy_model.Batch` as it stands**, with the same answer as its own batches, tested.

The projector (1024, 1024, 128) is used only in training, and only the encoder is saved.

**The loss** is `(1 - lambda) invariance + lambda SIGReg`, with lambda 0.02, 1024 random directions, and 17 points to t = 3.  There's no stop-gradient, teacher network or decoder.

**Views**, in `GLOBAL_VIEWS` and `LOCAL_VIEWS`, each its own forward pass:

| View | Keeps units / regions, sites / features | Else |
|---|---|---|
| Global: the whole frame | all | |
| Global: the same seat 2 s later | all | |
| 2 local | 70% / 80% / 70% | 10% of names unknown, positions jittered 0.5% of the map |
| 2 local | 40% / 50% / 40% | 10% of names unknown, positions jittered 1% of the map |

**Considered and left out: a fully informed view with hidden units.**  Invariance can be met simply by ignoring the hidden tokens, so it wouldn't teach the latent to infer them, and the live model never gets them.  The critic has its own route to them.

### 11.3 How we'd know it works

`collapse_report` gives the latents' spread and effective rank.  `probe_report` fits a linear probe on maps not held out and scores it on those that are, overall and for real, made and wet maps.  It covers own and enemy value and fighters, what's known of the enemy, where each side is, commander health, metal, the clock, air, ships, sea, wreck metal and, on finished script games, the outcome from that seat's side.  A column that hardly varies in a group is left out: a dry map has no ships to score.

**Three numbers to read each probe against:** what an untrained encoder gets (the run's step 0), what the run gets, and the ceiling.  `python tools/frame_jepa.py ceiling --data DIR --out FILE` trains the same encoder directly on the probes.  What it can't reach isn't in the tokens or doesn't carry over to unseen maps; what it can, and a run doesn't, the run's objective left out.

| Held-out R^2 | Untrained | 5,000 steps, lambda 0.02 | 5,000 steps, lambda 0.1 | Ceiling, 6,000 steps |
|---|---|---|---|---|
| Own value | 0.77 | 0.72 | 0.62 | 0.99 |
| Own fighters | 0.77 | 0.72 | 0.67 | 0.99 |
| What's known of the enemy | 0.59 | 0.47 | 0.53 | 0.99 |
| Own x, z | 0.77, 0.74 | 0.77, 0.74 | 0.17, 0.41 | 1.00, 1.00 |
| Commander health | 0.18 | 0.28 | 0.11 | 1.00 |
| Metal | 0.14 | 0.13 | 0.08 | 1.00 |
| Clock | 0.56 | 0.36 | 0.40 | 1.00 |
| Own air | 0.12 | 0.09 | 0.05 | 1.00 |
| Own ships | 0.15 | 0.10 | 0.07 | 1.00 |
| Sea share | 0.74 | 0.49 | 0.19 | 0.97 |
| Wreck metal | 0.79 | 0.76 | 0.88 | 1.00 |
| Enemy value (partly hidden) | 0.27 | 0.18 | 0.21 | 0.74 |
| Enemy air (partly hidden) | 0.01 | 0.01 | 0.00 | 0.66 |
| Outcome (not in the tokens) | 0.09 | 0.20 | 0.12 | 0.33 |

**What that says:**

* **The pipeline works.**  5.8 steps a second at batch 256, including probes, peak 3.2 GB allocated.  The batch goes from 64 to 256 at almost the same steps a second, so 256 is now the default.
* **The encoder isn't the limit.**  Directly trained, it reads all that's visible on maps it never saw, so it can count and carries over.  This ruled out the first suspect, that attention can't count.
* **The objective hasn't got there.**  Most probes fall below the untrained encoder early, then a few climb back; only commander health and outcome finish clearly above.
* **5,000 steps is short:** 1.28M samples, under one pass of the data.  The learning rate reached zero there, while the loss was still falling (0.96, 0.67, 0.42, 0.34 at 500, 2,000, 3,500 and 5,000).  SIGReg's statistic stayed about 15 where a real normal gives about 1.  **So this doesn't show whether a long run would learn.**
* **Lambda 0.1 is worse**, so it stays at 0.02.  Rank rises to 26 against 17, but what fills it isn't what the probes ask for.
* **Trained latents have most of their variance in a few directions:** 82% in 6 directions at lambda 0.02.
* **Map identity is not the explanation so far.**  Every view shares its map, so a latent could fingerprint maps and pass both losses.  Maps hold 37% of the trained latent's variance, but 38% of an untrained one's, against 3% of own value's.  It's high, but training hasn't raised it.

**The run** (the checks above were on `pool1`; on `pool2` it's only been seen to load and take 30 steps):

    wsl ~/zkml/bin/python tools/frame_jepa.py train --data /mnt/d/zk-learned/frames/pool2 --run /mnt/d/zk-learned/runs/frame-1 --steps 100000

About 4.8 hours and 100 MB.  A `stop` file in the run's directory ends it at the next report, and starting it again goes on from the last state, every 10,000 steps.  Its step 0 records the untrained scores.

**When to stop it:** by 20,000 steps, about 1 hour, own air, metal, the clock and sea share should be clearly above their untrained scores.  If they aren't, more of the same is unlikely to help and the objective needs changing.

**What to try then**, none of it tried yet: views that drop whole groups of tokens rather than a share of each, so that regions alone can't pass; a later view further ahead than 2 s, so that the slowly changing state counts for more than the constant; or a small probe loss added to the objective, which gives up on being purely self-supervised.

### 11.4 Not done, for the user to decide

* **How `z` reaches the main model.**  A single vector can't drive per-unit orders and target pointers, because those need a token per unit.  So `z` can replace the GRU's frame input and give the critic and global context, with unit tokens kept for the heads, but it can't replace them.  `tools/policy_model.py` isn't changed.
* **The live bot has no forward for it.**  `bots/zkbot/policy/net.py` runs in numpy and the encoder is torch only, so a numpy port with a parity test comes first.
* **Whether to start the long run**, with the stopping rule in 11.3.  `selfplay-6h` (10.6) points at the policy's updates and the factory matchups, rather than the frame input, as the first problems, so this may not fix what that run showed.

### 11.5 The frame VAE (2026-11-07)

`tools/frame_vae.py`, `tests/policy/test_frame_vae.py`.  It replaces the LeJEPA objective, whose long run wasn't started.  **Why:** LeJEPA asks a frame's latent to match the same frame with up to 60% of units dropped, and 2 s later, so it's taught to leave out which units are where, and parts of a frame mostly can't be predicted from other parts anyway.  The goal is a small latent the actor reads as tokens, in place of all its tokens, unit tokens included, so the latent should keep what's there.

**The model:** `FrameEncoder` as before, with its 24 queries kept apart rather than flattened.  Each gives a mean and variance of 32 numbers, 768 in all.  The decoder is 2 rounds of cross-attention with `d` 96.

**The decoder is asked about regions.**  A list of units has no order and no set length, and rebuilding one needs matching.  So each region's static columns and a sine code of its place make a query, and it answers with what's in that region now:

* reg_x's columns that change: own, seen enemy and radar value, own and seen enemy DPS, remembered structures, wreck metal.
* Value for each of own and enemy by structure, builder, air, ship and other ground units, worked out from the unit tokens, each counted for the nearest region centre.  Summary tokens add nothing here, though they do to the columns above.
* The global token, by one more query.

No new frames were needed.  It's asked only what the seat sees.  Each column is scaled to variance 1 first.

**The loss** is a Gaussian ELBO with variance `beta` on those scaled targets, summed over a frame's cells and divided by the mean cells in a frame.  So beta is roughly how much of a column may stay unexplained.  Beta rises from 0 over 2,000 steps.

**5,000-step checks on `pool2`, on held-out maps.**  The learning rate falls to 0 by the end, which accounts for some of the flattening from step 3,000.  3 at once ran 9 steps a second each, including probes.

| | Untrained | beta 0.03 | beta 0.1 | beta 0.3 |
|---|---|---|---|---|
| Latent size, bits a frame | | 637 | 231 | 111 |
| Reconstruction R^2, regions | -0.31 | 0.69 | 0.38 | 0.27 |
| Reconstruction R^2, global | -0.34 | 0.48 | 0.36 | 0.25 |
| Probe: own value | 0.72 | 0.86 | 0.86 | 0.85 |
| Probe: own air | 0.17 | 0.82 | 0.82 | 0.81 |
| Probe: own ships | -0.16 | 0.78 | 0.87 | 0.82 |
| Probe: what's known of the enemy | 0.70 | 0.87 | 0.90 | 0.88 |
| Probe: commander health | 0.26 | 0.92 | 0.87 | 0.85 |
| Probe: clock | 0.51 | 0.65 | 0.71 | 0.65 |
| Probe: metal | 0.21 | 0.41 | 0.39 | 0.37 |
| Probe: enemy value (partly hidden) | 0.27 | 0.45 | 0.47 | 0.45 |
| Probe: enemy air (partly hidden) | 0.04 | 0.49 | 0.49 | 0.49 |
| Probe: outcome | 0.15 | 0.28 | 0.38 | 0.40 |
| Probe: sea share | 0.77 | 0.54 | 0.42 | 0.53 |

Knowing only each region's mean on that map, over those same frames, gets a reconstruction R^2 of 0.15.

**What that says:**

* **The objective works.**  Almost every probe ends well above untrained, where LeJEPA's mostly ended below (11.3), though a linear probe has 768 numbers to read here against 256 there, so compare each against its own untrained column.
* **Not good enough yet on where things are.**  Over cells that hold something, R^2 is negative at every column but remembered structures: at beta 0.03, -2.0 for seen enemy value and -2.7 for own ground.  So the predictions are worse there than that column's mean over occupied cells; I'd guess too low, from blurring over regions, but haven't measured the sign.  The map baseline is -10 and -11, so this is much better than nothing, but blurred.
* **Metal and the clock are well short** of the ceiling's 1.00, as the global token is 1 query's answer among many cells.
* **Sea share falls below untrained**, as expected: it isn't asked for, and the actor would have the map another way.
* **Whether 768 numbers carry enough for the actor isn't shown by any of this.**

**The long run,** `/mnt/d/zk-learned/runs/frame-vae-b001`, beta 0.01, 100,000 steps, started 2026-11-07.  One at beta 0.03, `frame-vae-b003`, was stopped at step 8,500 once 0.01 was clearly ahead.  At 0.03 the command was:

    wsl ~/zkml/bin/python tools/frame_vae.py train --data /mnt/d/zk-learned/frames/pool2 --run /mnt/d/zk-learned/runs/frame-vae-b003 --steps 100000 --beta 0.03

A `stop` file in a run's directory ends it at the next report.

**How `frame-vae-b001` ended,** at step 99,999 on `pool2`, 1.9 hours, on held-out maps.  The marks set beforehand were a region R^2 of about 0.9 or better, and clearly positive R^2 over occupied cells for seen enemy value and own ground units.  It met the first two and not the third:

| | Step 80,000 | 85,000 | 90,000 | 95,000 | 99,999 |
|---|---|---|---|---|---|
| Region R^2 (the map alone: 0.14) | 0.942 | 0.928 | 0.938 | 0.949 | 0.945 |
| Global R^2 | 0.942 | 0.929 | 0.952 | 0.947 | 0.946 |
| Occupied: seen enemy value | 0.842 | 0.816 | 0.793 | 0.879 | 0.865 |
| Occupied: own value | 0.760 | 0.774 | 0.773 | 0.774 | 0.774 |
| Occupied: remembered | 0.932 | 0.935 | 0.948 | 0.934 | 0.940 |
| Occupied: own structures | 0.310 | 0.209 | 0.348 | 0.498 | 0.243 |
| Occupied: own ground | 0.021 | -0.041 | -0.062 | -0.014 | 0.199 |
| Occupied: enemy ground | 0.286 | 0.174 | 0.115 | 0.332 | 0.368 |
| Occupied: own builders | -0.898 | -1.661 | -0.713 | -0.614 | -0.930 |
| Occupied: enemy builders | -0.531 | -1.248 | -0.081 | 0.206 | -0.095 |
| Occupied: own air | -0.042 | -0.345 | -0.054 | -0.115 | 0.064 |
| Linear probe: metal | 0.797 | 0.801 | 0.762 | 0.763 | 0.790 |
| Linear probe: own x | 0.687 | 0.628 | 0.648 | 0.711 | 0.665 |
| Linear probe: enemy x | 0.401 | 0.293 | 0.299 | 0.329 | 0.343 |
| Linear probe: outcome | 0.109 | 0.108 | 0.146 | 0.044 | 0.062 |

The learning rate was all but gone by then, so these swings are what one check differs from another, not progress: own ground's 0.199 is the highest of the five, and the mean of the five is 0.02.

So value, firepower and memory come through well, and counts of a particular kind of unit, spread thin over the map, do not.  Linear probes of position fell through the run (own x 0.85 at step 20,000, enemy x 0.58, outcome 0.24) while reconstruction rose.  That doesn't show the information has gone, as the actor needn't read it linearly, nor that it can be used.  Own units matter less here, as the actor would read their raw features; enemy units are the concern.

**The larger run,** `/mnt/d/zk-learned/runs/frame-vae-large`, started 2026-11-07 on the user's standing go-ahead for a run that missed, on `pool3`: 3.05M parameters from 1.05M or so, a latent of 32 x 40 = 1280 numbers from 768.  It changes the size of all three parts and the frames at once, so it can't say which helped, and its held-out maps have new kinds of unit on them, so the numbers above aren't quite the same test.

    wsl ~/zkml/bin/python tools/frame_vae.py train --data /mnt/d/zk-learned/frames/pool3 --run /mnt/d/zk-learned/runs/frame-vae-large --steps 100000 --beta 0.01 --latent 40 --queries 32 --d 184 --rounds 3 --decoder-d 136

**How it went: stopped at the user's request at step 57,400,** since more parameters weren't helping.  Its learning rate had 0.00018 left to fall, and `frame-vae-b001` gained most of its last gains over that part, so this is a comparison at the same steps, on harder frames.

| Held-out maps | b001, 30k | large, 30k | b001, 55k | large, 55k | b001, end |
|---|---|---|---|---|---|
| Region R^2 | 0.901 | 0.879 | 0.921 | 0.912 | 0.945 |
| KL, nats | 656 | 659 | 550 | 610 | 488 |
| Numbers active, of 768 and 1280 | 682 | 969 | 600 | 928 | 600 |
| Effective rank | 211 | 218 | 214 | 220 | 216 |
| Seen enemy value, occupied | 0.636 | 0.686 | 0.772 | 0.741 | 0.865 |
| Own ground, occupied | -0.46 | -0.79 | -0.17 | -0.39 | 0.20 |

KL and rank hardly moved for 1.67 times the numbers, which suggests beta sets how much gets through, not the size.

**Where its error is,** `vae-last.pt`, 1,536 held-out frames, 62,850 region cells.  Deterministic R^2 0.929, with the latent's noise 0.925.  KL 585 nats a frame; tokens carry 1.2 to 29.4 each.  `where` is the share that are filled of as many cells as are, taken from those it rates highest; `how much` is R^2 over those it got right.

| Column | In the encoder's input? | Filled | R^2 | Occupied R^2 | Where | How much |
|---|---|---|---|---|---|---|
| seen_enemy_value | yes | 3.1% | 0.990 | 0.863 | 0.990 | 0.950 |
| seen_enemy_dps | yes | 2.1% | 0.992 | 0.899 | 0.993 | 0.949 |
| own_value | yes | 13.7% | 0.971 | 0.685 | 0.970 | 0.856 |
| own_dps | yes | 7.3% | 0.974 | 0.707 | 0.975 | 0.875 |
| enemy_ground | no | 1.8% | 0.917 | 0.412 | 0.929 | 0.687 |
| own_ground | no | 6.2% | 0.861 | -0.107 | 0.890 | 0.433 |
| own_structure | no | 9.1% | 0.908 | 0.344 | 0.922 | 0.524 |
| own_builder | no | 5.4% | 0.875 | -0.807 | 0.908 | 0.106 |
| enemy_builder | no | 0.8% | 0.916 | -0.393 | 0.951 | 0.457 |

Two things follow.  **The occupied-cell mark was harsher than it looked:** filled cells differ little (spread about 0.15 about a mean near 0.55), so own builders score -0.81 with 91% of its top cells right.  It does know roughly where things are.  **What it has to work out does worse than what it's handed:** `reg_x` gives the encoder value and DPS by region, and the split by class it can only get from the units.  The encoder's queries read by softmax attention with no layers over the tokens, which averages and can't simply count.  That split also needs more information through the same latent, so this doesn't separate the two causes; only a run can.

`--autoencoder`: no noise, no KL, a LayerNorm with no weights on each latent token.

**Summing units into regions and two answers for each cell** (`--region-pool`, `--two-part`, 2026-11-07).  The columns worked out from units did worst, and attention averages, so it can't simply count.  `FrameEncoder.pooled` sums what each unit says into its nearest region's token before the queries read, each number squashed by asinh.  And since most cells are empty and filled ones hardly differ, a cell can answer twice: a Bernoulli for whether anything's there, and a Gaussian for how much, on filled cells only.  `recon` and every R^2 use the expected value, so runs compare.  `how_much_` uses the second answer, which flatters a two-part run a little.

**The short runs,** `/mnt/d/zk-learned/runs/encoder-grid`, on `pool3`, 25,000 steps, the same cosine.  Nine were planned, to separate the three changes; the user stopped the rest after the first two as it doesn't matter which change did what, only that the encoder is slim, usable by the actor, and quick to train.  So **the table can't say which of the three helped.**  `ae-plain` and `vae-pool` were stopped within minutes and mean nothing.

    wsl ~/zkml/bin/python tools/frame_vae.py train --data /mnt/d/zk-learned/frames/pool3 --run /mnt/d/zk-learned/runs/encoder-grid/ae-pool-two --steps 25000 --autoencoder --region-pool 32 --two-part

| Held-out maps, step 25,000 | `vae-plain`, beta 0.01 | `ae-pool-two` |
|---|---|---|
| Region R^2 | 0.921 | **0.989** |
| Own ground R^2, where, how much | 0.858, 0.89, 0.46 | **0.993, 0.996, 0.96** |
| Own builders R^2, where, how much | 0.853, 0.89, 0.12 | **0.991, 0.99, 0.96** |
| Enemy ground where, how much | 0.90, 0.58 | **0.99, 0.97** |
| Effective rank | 209 | 215 |
| Encoder parameters (decoder another 237k or so) | 871k | 916k |
| CPU for a frame, one thread, torch, 28 to 112 units | 1.6 to 1.8 ms | 1.7 to 2.1 ms |
| Time, sharing the GPU with one other run | 33 minutes | 33 minutes |

About 20 steps a second alone.

**Could an actor use it?**  Frozen latents, 2,048 held-out frames.  A fresh decoder of one design, two-part, 8,000 steps, gets region R^2 0.966 from `ae-pool-two` and 0.908 from `vae-plain`, with per-class columns 0.99 against 0.87 to 0.91, so the autoencoder's latents really hold more about regions and it isn't its decoder's doing.  A fresh reader, as the planned actor would read (own unit's features reading the latents), was asked about circles around the unit, which no decoder ever was:

| R^2, own units, held-out | No latents | Untrained encoder | `ae-pool-two` | `vae-plain` |
|---|---|---|---|---|
| Enemy value within 0.15 of the unit | 0.55 | 0.57 | 0.78 | **0.82** |
| Own value within 0.15 | 0.55 | 0.58 | 0.81 | **0.84** |
| Nearest enemy | 0.46 | 0.46 | 0.94 | **0.95** |
| Enemy value within 0.05 | 0.47 | 0.48 | 0.64 | **0.67** |
| Own value within 0.05 | 0.37 | 0.40 | 0.57 | **0.59** |

The first two columns are 3,000 steps; the last two 15,000, which added about 0.03 to each.  The VAE's small lead held, so it isn't only a short reader.  Why isn't measured.  **Neither holds fine position:** the latent's only asked about regions, of which there are 44 in a median frame, so very roughly 0.3 across in map coordinates against 0.05.

Noise on the latents, by each number's own spread: `ae-pool-two` 0.989 at 0, 0.982 at 0.2, 0.841 at 0.5; `vae-plain` 0.928, 0.921, 0.865.  Taking away one unit moves a latent by 0.076 (autoencoder) or 0.040 (VAE) of the median distance between two frames.  Cutting the autoencoder's latent to its top directions, with no retraining: 512 numbers 0.974, 384 0.927, 256 0.796.  A latent trained at that size might do better; that isn't measured.  These figures are in `encoder-grid/usable.json` and `usable2.json`, from scripts not in the repo.

**Questions about units and wrecks** (`--questions`, 2026-11-07; the idea of asking "where's unit X" is the user's).  To make it keep per-unit detail and fine position, each frame also gets 64 sampled questions.  Twenty ask where a thing of this description is: a unit's type, side, knowledge, velocity, health, build and flags, or a wreck's metal, with nothing that gives away a place, so no position, altitude, orders, rays, or a wreck's distances to base and army.  Identical units make that ambiguous, so it answers with 8 places and a weight for each.  Forty-four ask what's at this exact place, within 0.02 of the map's -1..1: 16 at units, 4 at wrecks, 16 just beside units and 8 anywhere; it says whether a unit and a wreck are there, and if so its type, side and state or metal.  No slots, no matching, and the cost is fixed.  Each question's cost is added in nats, as a region cell's is.  Unit IDs were considered and left out: raw ones mean nothing, and random tags spend latent on telling apart units the actor treats alike and put noise in the latents.

**Muon** (`--muon-lr`, `Optimisers`, `torch.optim.Muon`) takes the weight matrices of hidden layers; AdamW keeps embeddings, learned queries, biases, norms and the `_out` layers.  Both follow one schedule.

**The short test,** `/mnt/d/zk-learned/runs/encoder-questions`, on `pool3`: autoencoder, pooling, two-part and questions, 5,000 steps with the cosine that long, so a warmup of 2,000.  The user judged this enough to compare optimisers, so no 25,000-step run followed.  One seed; **AdamW's rate wasn't tuned, and Muon's best is the highest tried.**

    wsl ~/zkml/bin/python tools/frame_vae.py train --data /mnt/d/zk-learned/frames/pool3 --run /mnt/d/zk-learned/runs/encoder-questions/sweep-muon-008 --steps 5000 --autoencoder --region-pool 32 --two-part --questions --muon-lr 0.008

| Held-out maps | AdamW 5e-4 | Muon 0.0005 | Muon 0.002 | Muon 0.008 |
|---|---|---|---|---|
| Region R^2 | 0.590 | 0.488 | 0.801 | **0.843** |
| "Where" cost, nats (uniform on the map 1.39) | -2.21 | -1.32 | -2.90 | **-4.63** |
| Any place within 0.05 | 0.461 | 0.200 | 0.544 | **0.710** |
| Likeliest place within 0.05 | 0.291 | 0.137 | 0.399 | **0.585** |
| A unit at a place: found, false alarms | 0.76, 0.32 | 0.67, 0.34 | 0.77, 0.15 | **0.84, 0.07** |
| Type right, side right | 0.40, 0.96 | 0.33, 0.94 | 0.59, 0.98 | **0.73, 0.98** |
| Health R^2, x velocity R^2 | 0.15, 0.17 | 0.09, 0.08 | 0.32, 0.49 | **0.58, 0.66** |
| Wreck at a place: found | 0.38 | 0.34 | 0.62 | 0.62 |
| Wreck metal R^2 | -0.14 | -0.01 | **0.13** | -0.03 |
| Time, sharing the GPU with one other run | **8.8 min** | 11.2 | 11.6 | 11.9 |

Muon 0.008's distance to its nearest place, median, by named units in the frame: 0.006 up to 30, 0.021 for 31 to 70, 0.052 above 70.  To its likeliest: 0.010, 0.041, 0.085.  So it does run out of room in crowded frames, as a latent of 768 numbers should.

The same 3,000-step reader as above, frozen latents, held-out frames.  Muon 0.008's encoder is only 5,000 steps old, the other two 25,000:

| R^2 | `ae-pool-two` | `vae-plain` | Questions, Muon 0.008 |
|---|---|---|---|
| Enemy value within 0.05 of a unit | 0.59 | 0.63 | **0.76** |
| Enemy value within 0.15 | 0.75 | 0.80 | **0.86** |
| Own value within 0.05 | 0.54 | 0.56 | **0.62** |
| Own value within 0.15 | 0.79 | 0.81 | **0.84** |
| Nearest enemy | 0.92 | 0.94 | **0.95** |

So the questions did put finer position where a reader finds it.  Region R^2 at this step is lower than the 0.891 `ae-pool-two` had at its step 5,000, on a schedule five times as long, so what the questions cost regions when trained out isn't known.  Reader figures: `encoder-questions/usable3.json`.

Muon 0.03 on the same 5,000 steps came out a little under 0.008 (region R^2 0.821, any place within 0.05 0.673, found 0.74, type 0.705), so the rate was left at 0.008.

**The final encoder,** `/mnt/d/zk-learned/runs/encoder-questions/final/vae-last.pt`, 25,000 steps, 50.5 minutes alone, with the GPU about a quarter busy, so what limits its speed isn't the GPU:

    wsl ~/zkml/bin/python tools/frame_vae.py train --data /mnt/d/zk-learned/frames/pool3 --run /mnt/d/zk-learned/runs/encoder-questions/final --steps 25000 --autoencoder --region-pool 32 --two-part --questions --muon-lr 0.008

| Held-out maps | 5,000 steps (`sweep-muon-008`) | 25,000 steps (`final`) | No questions, AdamW (`ae-pool-two`) |
|---|---|---|---|
| Region R^2 | 0.843 | **0.948** | 0.989 |
| Own ground, builders, enemy ground R^2 | 0.921, 0.928, 0.902 | 0.953, 0.967, 0.942 | 0.993, 0.991, 0.987 or so |
| "Where" cost, nats | -4.63 | **-5.28** | not asked |
| Any place within 0.02, 0.05, 0.15 | 0.565, 0.710, 0.901 | **0.622, 0.748, 0.918** | |
| Likeliest place within 0.02, 0.05, 0.15 | 0.471, 0.585, 0.794 | **0.527, 0.625, 0.816** | |
| A unit at a place: found, false alarms | 0.84, 0.07 | 0.83, 0.07 | |
| Type, side | 0.73, 0.98 | 0.74, 0.97 | |
| Health, x velocity, build R^2 | 0.58, 0.66, 0.60 | 0.60, 0.63, 0.63 | |
| Wreck at a place found, metal R^2 | 0.62, -0.03 | 0.68, 0.19 | |
| Fresh reader: enemy within 0.05, 0.15 | **0.76, 0.86** | 0.71, 0.83 | 0.59, 0.75 |
| Fresh reader: own within 0.05, 0.15 | **0.62, 0.84** | 0.60, 0.83 | 0.54, 0.79 |
| Fresh reader: nearest enemy | **0.95** | 0.94 | 0.92 |

Nearest place, median, by named units in a frame: 0.003 up to 30, 0.017 for 31 to 70, 0.036 above 70; likeliest place, 0.003, 0.033, 0.061.  **More steps went mostly to regions and to position; a unit's other state hardly moved after 5,000 steps,** and a fresh reader did a little worse on the final encoder.  Why isn't measured; the signs are of a latent that's full.  Its numbers matter more: noise at 0.2 of each number's spread takes R^2 from 0.949 to 0.886 (without questions 0.989 to 0.982), losing one unit moves a latent by 0.129 of the distance between frames (0.076), and keeping 512 directions of 768 leaves 0.772 (0.974).  A larger latent might help; that isn't measured.  Encoder 916k parameters, 320k more in training only; 1.6 to 1.9 ms a frame on one CPU thread with torch.  `final/usable.json`.

### 11.6 The actor on latents (2026-11-07)

`tools/policy_latent.py`, `tests/policy/test_latent_actor.py`, `policy_train.py --latent-encoder`.  It reads a frozen encoder's latent tokens in place of every token: built to the user's design.  It went through three versions, each at the user's request: with memory (a GRU), then with no memory and two past frames, and now **a transformer over time, causal and windowed, so that training sees a whole stretch of a game in one forward pass,** as a GPT does.  In 2M decisions against the raider script both later versions went from about 13 to 15% test wins to about 21 to 23%.

**The windowed actor (current).**

* **Each frame's 24 latent tokens attend to every frame in the 8 s before** (`WINDOW_SECONDS`; W = 17 frames at `--decide-every 1`), their own included, in each of 2 pre-LN blocks (`TimeBlock`, d=128, 4 heads).  A learned bias per head and age is added to the scores; frames later than the query or older than the window are masked out.  So the second layer reaches 32 frames, 16 s, back (`reach = LAYERS x (W - 1)`).  Frames from before the game read a learned `before` embedding.
* **Play keeps the raw latents of the last 32 frames per seat**, plus whether each was seen, and recomputes the window over them at each decision, which is exactly what the model would compute over the whole game.  **Training runs each segment from the latents stored at its start, in one forward pass** (`LatentActor.segments`, `Learner._loss_window`): the segment's frames attend causally to each other and to the stored history, so no step is unrolled.  `test_segments_score_in_training_as_their_decisions_did_in_play` checks that stored decisions score within 2e-3 in log-probability of what they did in play.
* Units, pointers, the critic and PPO are as in the no-memory version below.  Units cross-attend to their own frame's final tokens only.  The critic has no actor state.  `Actor.recurrent` picks the trainer's path: the first actor is `True` and trains as before.
* Parameters: 0.80M trained and 0.92M frozen.

**The no-memory version** (`/mnt/d/zk-learned/runs/latent-1/code.patch`):

* **A small transformer reads only the latents** of this frame and of those 2 s and 8 s before (4 and 16 decisions at `--decide-every 1`; `policy_latent.PAST_SECONDS`): 72 tokens, each with a learned embedding of which latent token it is and one of its frame's age, with frames from before the game's start masked out.  2 blocks, d=128, 4 heads, with no pair bias and no gate layer.  The encoder isn't trained further, and reads true names whatever names this game has dropped for the actor.
* **Each own unit reads what that made,** one cross-attention, with a query of its own raw features and sine codes of its place.  Its decision comes from that.
* **Pointers score against embeddings of the raw features** of what they point at.  The user chose sine codes of position in those embeddings, since the dot product of two sine codes depends on the difference of the positions, which a linear embedding of raw coordinates can't give.  So a key ends in a code of its place, and a query in the unit's own code, each band scaled by a gain the unit chooses, which makes that part of the score a sum over bands of gain x cos(band x difference).
* **No memory.**  A region's key is from its raw features alone.  A seat keeps the latents of its last 16 decisions, which is all a decision needs; the trainer stores each decision's 3 frames of latents (9 KB) with it.  So training is a plain transformer's: decisions are shuffled singly into minibatches and each minibatch is one forward pass, with nothing unrolled and the encoder not run, since its latents are stored (`Learner._loss_flat`).  Stored decisions score the same in training as in play, in any order (`test_a_stored_decision_scores_in_training_as_it_did_in_play`).
* The critic reads every privileged token, and with no memory isn't given the actor's state (`pm.Critic(vocabulary, 0)`).  Masks, decoding and PPO are unchanged; `pm.collate(pairs=False)` skips pair buckets.  The first actor has `recurrent = True` and trains as before.
* The version with memory (a global GRU and one per region) is in git at `bcea3be`.  It ran at 933 decisions a second, used 1.65M trained parameters, and was never trained for long.

No-memory parameters: 0.80M trained and 0.92M frozen, against the first actor's 2.99M.  The critic is 0.43M.

**Speed,** 256 games, one pass, 57,344 decisions on the sim, CPU and GPU otherwise idle, iterations from the third on:

| | First actor | On latents, memory | No memory | **Windowed** |
|---|---|---|---|---|
| Decisions a second, overall | 950 | 933 | 1,424 | **1,156** |
| Training, per pass | 1,009 | 1,013 | 2,140 | 1,678 |
| Collecting | 1,108 | 1,137 | 1,567 | 1,320 |
| Peak GPU memory | 8.8 GB | 5.1 GB | 5.7 GB | 6.1 GB |
| GPU busy, mean | 21% | 17% | 17% | 33% |

The last two columns are 102,400 decisions, otherwise the same.  The windowed actor is about a fifth slower than the no-memory one, and keeps the GPU twice as busy; the CPU still sets the rate.  Without memory, training per pass doubled, not more, and the GPU is still mostly idle, so the CPU still sets the rate.  Why isn't measured.  The likely reasons are featurizing each decision again in training, and collecting and training sharing one Python thread.  0 chosen actions were outside their mask.

**Learning,** `/mnt/d/zk-learned/runs/latent-1`: no memory, 256 games, one pass, 2.0M decisions against the script only (self-play never started), 64 test games every 20 iterations.  It ran in 30 minutes, 1,140 decisions a second counting the tests.

| Iteration | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 | 220 | 240 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Test wins | 12.5% | 12.5% | 12.5% | 28% | 31% | 16% | 31% | 23% | 25% | 16% | 20% | 16% |
| Draws | 19% | 12.5% | 16% | 19% | 9% | 9% | 5% | 8% | 9% | 11% | 9% | 14% |

The first three tests won 24 games of 192, the last nine 132 of 576 (23%).  With 64 games a test is ±5 points or so, so the rise is likely real, and the last three (17%) are within noise of the middle.  Wins in its training games against the script went from 0 to about 25%.  The value head's explained variance rose from 0.02 to 0.96.  KL fell from 0.002 to 0.0006 a pass and clip fraction from 0.05 to 0.005, while entropy rose (kinds 0.96 to 1.23, enemy 0.72 to 1.68).  So by the end it was barely changing the policy, and becoming more random.  That and the flat late tests suggest it was stalling, not still climbing.  Not checked: a longer run, two passes or a higher learning rate, self-play, and how the first actor does over the same 2M with these settings.

**Learning, windowed,** `/mnt/d/zk-learned/runs/latent-2`: the same settings, 2.0M decisions, against the raider script (`fight`).  974 decisions a second counting the tests, with the CPU shared with other work for part of it.

| Iteration | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 | 220 | 240 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Test wins | 19% | 14% | 12% | 28% | 23% | 19% | 30% | 16% | 22% | 14% | 16% | 22% |
| Draws | 19% | 17% | 19% | 19% | 14% | 9% | 11% | 19% | 12% | 11% | 16% | 9% |

The first three tests won 15%, the last nine 21%: the same rise as the no-memory version, within noise of it.  Explained variance ended at 0.94 to 0.96; KL 0.0003 to 0.0007 and clip fraction 0.002 to 0.005, so like that version it had nearly stopped changing.  So 2M decisions of RL from scratch don't show the window helping or hurting; the point of it is that the same model can learn from whole recorded games first (11.8).

**Why the GPU is so idle,** measured on the latent actor, collecting and training in turn over 40,960 decisions, with the GPU waited for around each part.  **One thread issuing small calls sets the rate, not the GPU or featurizing.**

| Collecting, 21.2 s, GPU 14% | | Training, 29.8 s, GPU 22% | |
|---|---|---|---|
| Waiting for featurizing | 52% | `actor.encode`, 928 calls | 45% |
| `actor.encode`, 165 calls | 14% | `backward`, 29 calls | 37% |
| Sim steps | 6% | Waiting for featurizing | 8% |
| Choosing actions, critic | 11% | Critic | 3% |

The main process used about one core, the workers 2 to 2.5 of their 6, and the machine 3 to 3.5 of 16.  Timed alone, `encode` costs 9 to 30 ms a call from 16 frames to 2,048, in no clear order, so 0.58 ms a decision at 16 and 0.011 at 2,048; a GPU setting the rate would make that grow with the batch.  Training calls it 928 times at about 44 frames, as the memory has to go step by step.  The frozen encoder is about half to two-thirds of each call, and isn't recurrent, so it needn't go step by step.

Pair buckets aren't the cause: 9.5% of featurizing at 1.01 ms a frame, so about 2% of the iteration.  Featurizing waits 67 ms a call, against 41 ms if six workers took 1 ms a frame with nothing else to pay, so sending frames between processes is likely the rest.  Collecting during training ran 933 a second, against 803 for these two phases in turn, though they alone would go at 1,929 and 1,377; sharing one thread's lock is the likely reason, not measured.

**Not built:** the numpy forward, export and parity for the bot, so this actor can't run in `bots/zkbot/policy/net.py` yet.  Muon isn't used for PPO.  Not checked: the token budget, which still allows for pair memory it doesn't use (819 decisions a minibatch in the run).

### 11.7 The army script, now the default for script-controlled sides (2026-11-07)

`tools/policy_army.py` (fighting), `tools/policy_mix.py` (production), `tests/policy/test_army.py`.  `SimEnv` now gives script-controlled sides the army script by default.  The raider-only script of 10.2 is still there as `"fight"`, and the script-against-script numbers from 10.2 onward were measured with it.

* **What a factory builds** (`policy_mix.py`).  Each unit is the cheapest modelled unit of its catalog role on the factory's roster, so nothing is picked by name.  A role is started when its share lags furthest behind its target share of the units its side has started.  Raiders are 100% of the mix to 4 minutes, then 50% at 10, 40% at 20, 35% from 60.  Riots are 30% from 10 minutes; assaults 25% from 12 (none before 6); artillery 10% from 18.  Skirmishers take 25% once a riot or assault has been seen, AA 12% once air has been seen, snipers 10% from 8 minutes once a slow heavy has been seen.  Scouts: one from 2 minutes, a second from 10.
* **PRODUCE orders** go on the factory's uid, and only when its next type differs from the one it's on, so each order is a real change.  `SimEnv.script_production(seat)` lists them for each decision.  No queue is involved: a switch only affects the next unit, as the sim does it.
* **Raiders** (`policy_army.py`).  Squads of three.  A squad goes for extractors first, then other buildings, one target at a time, and takes a target only when its metal is at least the cover on it: defences and armed units count, outranging ones four times over, and a riot heavy within 800 elmos rules a target out.  A squad with no target probes towards the enemy's start, stops before ground it can't cover, and holds if none is clear.  Probing squads within 900 elmos merge, up to six.  Past the first three raiders a quarter patrol their own extractors, and from 1:30 at least two do; a patroller answers a threat within 900 elmos of its extractor and no other.  A squad gives up on a target that takes it longer than 60 s plus distance/20 s, or 300 s more once it's there, and nobody goes at that target for 300 s.
* **The rest of the army.**  Assaults lead, to the enemy's start and then the nearest building.  Riots and the others hold 150 elmos behind the assaults' middle, or behind the goal when there are no assaults.  Skirmishers back off 250 elmos at once from an armed unit within 0.6 of their range.  Artillery stands at 0.75 of its range from the building nearest the goal.  AA follows the army's centre.  Snipers go for the nearest slow heavy.  Scouts move between four waypoints.  A unit is given an order only when it's new, its job has changed, its goal has moved more than 200 elmos, or it's idle more than 150 from its goal.
* **A bug found in testing and fixed.**  With no assaults, the other roles held behind the middle of the army itself, so the army walked back to the rally.  They now hold behind the goal.  `test_riots_hold_behind_the_goal_when_no_assault_leads` covers it.

**Tests.**  `test_army.py` has 29, all passing: the mix by role and time, skirmishers and AA only after the enemy has shown the right unit, scouts from 2 minutes, one PRODUCE per type change, per-role orders, probing that stops at defences, patience, merging and re-orders.  One sim check runs 8 games for 400 decisions and asserts that each PRODUCE is on the seat's factory and changes the type; another runs 8 games for 60 decisions on 1 and on 4 threads and asserts the same records.  `test_script.py` and `test_sim_env.py` pin the tests of the raider script to `("fight", "fight")`, where they check that script's own state.  Full suite: 613 passed, 9 failed.  Eight of the failures also happen with the committed `tools` files, so they come from uncommitted changes in `bots/` (`tables.py` and others): `test_golden`, `test_policy_actions`, and `test_the_policy_is_given_only_the_raiders`.  The ninth, `test_frame_vae::test_two_part_learns_and_reports_comparably`, passes when run alone and was not looked into.

**Measured,** 64 games, one per slot, each run to its own end, army against fight with sides drawn at random (`SimEnv(64, 2026, ["scripts"], threads=4, scripts=("army","fight"))`).  The harness is a script in the job's scratch folder, not committed.

| Army against fight | Won | Drew | Lost |
|---|---|---|---|
| 64 games | **57 (89%)** | 0 | 7 (11%) |

* Endings: 49 eliminated, 15 by the early end (2.5 times the value and 10,000 metal ahead).  None reached the time limit, so draws were impossible here.  Length: mean 648 s, median 645 s, range 224 to 1,084 s.
* **Unit mix by role per factory** (units the army side started; the fight side records no mix and builds its opening's raider type only).

| Factory | Started | Raid | Riot | Assault | Skirm | Scout | Arty |
|---|---|---|---|---|---|---|---|
| Shield | 932 | 50% | 23% | 17% | 5% | 3% | 2% |
| Cloak | 668 | 64% | 17% | 12% | 6% | | 1% |
| Hover | 518 | 65% | 20% | 11% | 4% | | |
| Veh | 482 | 57% | 22% | 16% | | 4% | 2% |
| Amph | 385 | 76% | 12% | 3% | 9% | | |
| Spider | 193 | 65% | 15% | 5% | 11% | 5% | |
| Tank | 189 | 72% | 17% | 11% | | | |
| Jump | 161 | 72% | | 18% | 4% | 6% | |

  No AA was started in any factory.  The fight script never fields air, so the enemy never showed any, and the AA rule never fired in this batch.  No snipers were started either; I didn't check why.
* **The drawn factory decides the result.**  Each factory's score is wins plus half draws over all its sides in the 64 games.  Of 64 games, 49 had different factories and a decided result, and the better factory's side won 31 of them (63%).  Five more were between the same factory, and ten had tied scores, so neither is counted.  The earlier figure of 91% (HANDOFF 6g: Spider 0.92, Veh 0.21) was on fight against fight, with draws and a different score pool.  This batch is lower, and the score here includes these same games, so it isn't a clean test.

**Not checked:** other seeds; a fight-against-fight run on this seed, to compare with the 46/8/46 in HANDOFF 6g; the army's results against the real engine; how much of the lead comes from production and how much from fighting, since the two weren't separated; whether the factory score is the right test when the two sides run different scripts; why no snipers were started.

### 11.8 Factories for the policy, labels from the script, and a dataset of script games (2026-11-07)

The user's plan: learn from whole recorded games before RL, which the windowed actor (11.6) allows, from the army script (11.7) playing itself, with the script's production labelled as well as its fighting.

**Factories are the policy's.**

* In the sim, a PRODUCE on a factory sets the type it keeps building, and is remembered as the factory's order like any other (`world.cpp` `order_unit`).  In the frame a PRODUCE's `ord_target` is the def id of what it makes, not a unit (`views.cpp`, `rawframe.py`).
* `tables.py`: the makers are the silos and the factories, each with its catalog roster; `producible` is the 127 types they make, sorted.  The PRODUCE mask leaves out the type a factory is already making, so a PRODUCE is always a change.  A PRODUCE's target isn't a unit, so it gets no target pair (`featurize.py`).
* Factories are controllable (`acts_in_place`), so the policy is given its fighters and its factories, still not its constructors.  The goldens were regenerated with `tools/policy_golden.py sim`; only factory rows differ from before.  `test_a_factory_chooses_what_to_build_among_its_roster_but_not_what_it_builds_already`, `test_the_sim_builds_what_the_policy_chose_and_the_frame_says_so`.

**Labels.**  `actions.encode` is the inverse of `actions.decode`: given orders, it finds the kind, the enemy by uid, the nearest legal region and the nearest of its fine options, and a factory's row in `producible`.  It also counts the orders it can't say.  `policy_sim_env.policy_orders` is the inverse of `sim_orders`, so the script's orders, as the sim took them, become the policy's.  `tests/policy/test_labels.py`: decode then encode gives back the decision, places go to the nearest legal option, and what can't be said is counted.

**The recorder,** `tools/policy_clone_data.py`, `tests/policy/test_clone_data.py`.

* `record` runs the army script against itself in batches of games in the sim.  `--workers` processes each run a batch with their own seed and share the sim threads.  The map pool is loaded once before they fork, so they share it: 12 to 14 GB at 8 to 12 workers, where loading it in each (7.5 GB a copy) ran WSL out of memory at 8.
* One file per side of a game: its frames (`rf.save_frames`), the orders its script gave at each decision, how the game ended for it, the map and the script.  Frames and orders are raw, not features or choices, so the data outlives featurizer changes: `labels` turns them into the policy's choices when read.  The commander's opening isn't the policy's, so it isn't labelled.
* **Whole games only.**  When the budget is reached, no new games are recorded and those running are played out.  Dropping them, as the first version did, kept only games that ended early: 0 of 63 had reached the time limit in a short run.
* `check` reports label kind shares, how many orders couldn't be said, and bytes a decision.
* About 340 bytes a decision compressed (12.7x).  `rf.join` replaced `np.concatenate` for records, which had cost more than the compression.

**Speed,** 32 games a worker, the five training maps, written to WSL's disk, decisions recorded a second.  Results don't depend on the thread count.

| Workers, sim threads | 1, 16 | 8, 8 | 8, 16 | 14, 14 | 8, 8, written to D: |
|---|---|---|---|---|---|
| Decisions a second | 7,490 | 30,922 | 33,700 | 38,309 | 28,997 |

Profiled at one worker before `rf.join`: the script was 41% of the time, writing 30%, the sim 12%.  Not profiled since.

**The first dataset,** `/mnt/d/zk-learned/clone/army-1`: the army script against itself on the map pool, held-out maps excluded, seed 1, 12 workers.

* 3,097 games, 11.56M decisions (both sides), 4.9 GB, about 400 bytes a decision.  5.3 minutes in all: 34 s loading maps, then 40,800 decisions a second.
* Endings: 1,136 by elimination, 885 by the early end on a lead, 1,076 at the time limit (35%, drawn).
* Labels, read from 60 files at every fifth decision (22,274 decisions): of controllable units' choices, keep 97.4%, fight 2.25%, PRODUCE 0.2%, move 0.15%, attack a unit 0.01%.  The script gave 0.32 orders a decision, and 2.5% of them couldn't be said in the policy's actions.

**What that means for cloning.**  Nearly every label is keep, since the script looks only every 6 decisions and then re-orders only what has changed, mostly with attack-moves, so a plain loss would teach keep.  The loss will need weighting by kind, or a split between whether to act and what to do.  The draws are a third of the games, which matters if results are used to weight games.

**Not checked:** which orders couldn't be said; how the script's games compare with the engine; whether labels for a unit should count only decisions where its script looked at it (the script looks every 6 decisions, `SCRIPT_EVERY`, so in 5 of 6 keep is a default, not a judgement).  **Not built:** the cloning trainer.

**Tests:** `sim/tests` and `tests/policy`, 624 passed.  The whole `tests` folder also has 2 failures that need `requests`, which isn't installed in `~/zkml`; they're in the rules bot, untouched here.

### 11.9 The economy script, missiles for the policy, action tokens and a cloning trainer (2026-11-08)

The user asked for more of the game in the script, for metal to be spent almost all the time ("metal stored is an inefficiency"), for missiles and the Trinity to be the policy's units, and for the actor to read what was done between frames: `[frame 1 latent tokens] [actions at frame 1] [frame 2 latent tokens] ...`.  Then for a cloning trainer in which keep counts for less.

**`tools/policy_eco.py`, the economy script.**  It runs for every side in the sim, script or policy, every `ECO_EVERY` decisions.  Its orders aren't labels: they aren't the policy's to give, as the opening's aren't (`SimEnv(economy=False)` turns it off).  A free constructor (idle, helping, or stuck) takes the first job due:

1. finish anything of ours being built nearby, or the big project wherever it is;
2. the big project, once due;
3. energy while energy made is below `ENERGY_FIRST` = 1.5 x metal income.  Building costs as much energy as metal, and overdrive takes only what's left, so short of energy everything slows and metal piles up;
4. expand to the nearest free metal spot on our half that no enemy is near (while metal floats, only while fewer than `FLOAT_EXPANDERS` = 4 are on the way to one);
5. **spend**, while metal floats (above `FLOATING` = 50% of storage, until it's below `FLOATING_UNTIL` = 15%): energy if that's what's short, else another factory like the first while income is above `FACTORY_PER_INCOME` = 12 a factory (at most 6), else help the nearest factory;
6. energy up to `ENERGY_SHARE` x income, which grows by the minute from 1.25 to 3.0: small generators by the extractors, fusions from 4 minutes and 15 metal a second, a Singularity from 12 minutes and 45;
7. a radar behind the front from 3 minutes, and a second once there's a silo;
8. a pylon joining an extractor or turret to the base's grid (overdrive is capped per grid; Cerberus and Lucifer fire only on a grid making 50);
9. help the factory.

* **Big projects.**  Cerberus or Lucifer, and the silo, from `BIG_FROM` = 16 minutes and 55 metal a second (the user raised it from 8 minutes and 40), when map control is stalled: neither side's extractor count has changed, net, by more than max(2, 10% of the map's spots) over the last two minutes.  At most 3 turrets, one silo.  The Trinity from `NUKE_FROM` = 18 minutes and 60 metal a second, by a 30% chance drawn once.
* **Constructors** come from the factory (`policy_mix.constructor_quota`): the larger of a schedule by the minute and one per `INCOME_PER_CONSTRUCTOR` = 3 metal a second of income, at most 20.  The army script's factory change waits out the PRODUCE commit lock, as the policy's must (`policy_army._produce`): otherwise its first constructor order, 3 s after the opening's PRODUCE, couldn't be said, and half the factory labels in the first decisions were lost.
* **Stuck constructors**: one that hasn't got 50 elmos nearer its goal in 30 s is given something else, and the place is avoided for 3 minutes.
* **A sim fix** (`sim/src/economy.cpp`, ASSIST): help given to a factory still being built went to what it would make, so the factory was never finished.  Help to an unfinished factory now builds the factory.

**Measured,** 32 games on the five training maps, army script against itself.  Metal lost at the storage cap, as a share of income:

| Minutes | 0-4 | 4-8 | 8-12 | 12-16 | 16-20 |
|---|---|---|---|---|---|
| Lost at the cap | 2% | 11% | 12% | 6% | 5% |

Before the spend step and the income-scaled constructors, 62% of income was lost at the cap from 6 minutes on.  Median metal income: 26, 34, 58 and 45 metal a second at 4, 8, 12 and 18 minutes.  Silos are finished and make missiles.

**Missiles and the Trinity are the policy's** (`tables.py`).  `missile` is the silo's missiles (`empmissile`, `missileslow`, `napalmmissile`, `seismic`, `tacnuke`) and `launcher` the Trinity (`staticnuke`, manual fire and not mobile); both are in `acts_in_place`.  A missile can ATTACK_UNIT any enemy token on the ground within its range, and the Trinity MANUAL_FIRE any within its manual range (`featurize._masks`).  Enemy tokens are what's in sight or on radar, so that's what they can aim at; remembered buildings aren't tokens.  `policy_eco.Aimer` aims them for the army script, so its aims are labels: each missile at the most valuable building in reach that is seen, the Trinity at the building with the most enemy value within its blast.  What the silo makes is the army script's PRODUCE (`ScriptData.best_missile`).

**Action tokens** (`tools/policy_latent.py`).  Each frame is now its `T` latent tokens followed by `ACTION_TOKENS` = 4 tokens saying what the policy did at that frame.

* Each own unit told something other than keep becomes a vector: the kind's embedding, the unit's raw embedding, the key of the enemy or region it was pointed at, the fine option, and the type a factory was told to make.  4 learned queries, given the log count of units that acted, read those and a learned "nothing", so a frame where everything kept has action tokens too, which say so.
* Attention: a frame's latent tokens don't see its own action tokens; its action tokens see its latent tokens and everything before.  So a decision never sees its own choice and the next one does.  Each head has a learned bias for action keys (`act_key`) beside the bias by age.  The window and reach are as before.
* What a seat keeps between decisions is each frame's latents and action tokens, flat.  `decide` leaves the newest frame's empty; `acted` fills it once actions are sampled (`policy_train.Player._forward`).  Training builds a segment's action tokens from the stored actions (`segment_actions`).
* They're the policy's choices only.  The economy script's and the opening's orders show only in the frames, as each unit's order.

**The cloning trainer,** `tools/policy_clone_train.py`.

* Reads recorded games (`policy_clone_data.py`), featurizes a stretch of `--length` decisions plus the actor's reach before it, and turns the script's orders into the policy's choices for every frame: those are the action tokens, and the ones where the script looked (every `SCRIPT_EVERY`th decision) are the labels.
* Loss: negative log-likelihood of the script's choice for every unit that can say it, with keep weighted by `--keep-weight` (0.1).  The saved actor adds -log w to keep's logit, so that it plays at the true odds of keep.  Units whose label is keep while they're idle can't say it (keep is illegal for an idle mobile unit), and are left out: they're the factory's new units, mostly.
* Workers featurize ahead of the GPU into a bounded queue.  Only the map bundles are loaded, not the map pool (which is 7.5 GB a copy, and thrashed memory with a recorder running).  50 files by name hash are held out for checks.
* `--blind` gives the actor zeros for action tokens, to measure what they're worth.

**`army-3`,** `/mnt/d/zk-learned/clone/army-3` (`DATA_VERSION` 3): the army script against itself with this economy, on the map pool, 12 workers.

* 3,233 games, 11.5M decisions, 13 GB: 913 bytes a decision, twice army-2's, from more units.  13 minutes at 15,200 decisions a second.
* Endings: 1,939 on a lead, 1,179 at the time limit, 115 by elimination.
* Labels from 60 files at every fifth decision: keep 96.5%, fight 2.7%, PRODUCE 0.69%, move 0.09%.  0.8% of orders couldn't be said.

**Cloning, with and without action tokens,** army-3, 3,000 steps of 16 stretches of 48 decisions, keep weight 0.1, checked on 96 held-out stretches:

| Step | Actor | NLL of acts | Act's kind right | Fight right | PRODUCE right | Move right | NLL of keeps |
|---|---|---|---|---|---|---|---|
| 1,500 | action tokens | 3.584 | 0.313 | 0.365 | 0.146 | 0.000 | 1.088 |
| 1,500 | blind | 3.587 | 0.217 | 0.269 | 0.036 | 0.018 | 1.119 |
| 2,000 | action tokens | 3.570 | 0.294 | 0.319 | 0.224 | 0.101 | 1.045 |
| 2,000 | blind | 3.565 | 0.210 | 0.229 | 0.156 | 0.064 | 1.072 |
| 3,000 | action tokens | 3.506 | 0.240 | 0.263 | 0.165 | 0.128 | 1.028 |
| 3,000 | blind | 3.494 | 0.225 | 0.225 | 0.249 | 0.083 | 1.050 |

Runs `/mnt/d/zk-learned/runs/clone-act` and `clone-blind`.  The checks hold 3,649 acts (2,849 fight, 691 PRODUCE, 109 move) and 9,720 keeps.  "Right" is the actor's most likely choice matching the script's; NLL is per unit.

* **Action tokens made no measurable difference at this length.**  The likelihood of the script's acts, the measure trained on, is within 0.015 at every check, and slightly better blind at the end.  The tokens run picked the right kind of act 9 points more often at 1,500 and 2,000 steps, but by 3,000 the gap is 1.5 points: fight better with tokens, PRODUCE better blind.  With 109 move and 691 PRODUCE labels, a check moves by 5 to 10 points on those from one to the next, so only the NLL and the fight column mean much.
* **Neither has learned much.**  An act's kind is right a quarter of the time and its NLL is still falling slowly when the learning rate reaches zero.  So this says the tokens don't help an actor this undertrained, not that they wouldn't help one that has learned the script.
* The tokens cost 19%: 133 decisions a second against 164.

**Not checked:** a Trinity firing in a sim game; whether Cerberus and Lucifer are powered when finished; the economy on the map pool rather than the training maps; the cloned actor playing.

**Tests:** `sim/tests` and `tests/policy`, 637 passed.  New: `test_eco.py` (the economy script and the aimer); in `test_policy_actions.py`, missiles and the Trinity are the policy's to aim; in `test_latent_actor.py`, a frame reads what was done before it but not its own choice, action tokens reach no further than frames, the newest frame's are blank when it decides, keeping everything reads alike, and what a unit was told and where reaches the tokens.  `test_frame_vae.py`'s two-part fit takes 250 steps, from 150: the sim's games are busier and it reached 0.708 against the 0.7 bar.

### 11.10 The trim (2026-11-08)

At the user's request, what the learned policy no longer uses is gone, kept at the local git tag `rules-bot-final` (00-overview.md has the note).

* **The rules bot.**  Every behaviour module of `bots/zkbot` (`brain.py` and the 32 around it), `bots/probe`, `bots/observer`, ML stages 1 to 4 (`fightnet`, the value net, the observer dataset, `tune.py`), `prun.py`, its reports and `tools/run.sh`, with their tests and `docs/ARCHITECTURE.md` and `docs/ML.md`.  What's left of `bots/zkbot` is the policy's host: `main.py` drives `host.py`, which keeps what `rawframe.from_host` reads (units, the commander, remembered enemy buildings, blip speeds, who holds each spot) and sends no commands.  `hazards.py` builds the live terrain and nothing else, `mapdata.py` reads the map's spots and liquid, `params.py` is the flags, and `units.py` lost the rules' hand tables (`FACTORY_UNITS`, `role_of`).  No unit is `RESERVED` now: that kept the rules' Gremlins and Phantoms from the policy.  Engine A/B baselines against the rules bot can't be run without checking out the tag.
* **The first actor** (`policy_model.Actor`, recurrent over every token with pair biases), its numpy forward (`bots/zkbot/policy/net.py`), shadow mode (`policyshadow`), `tools/policy_parity.py` and `tools/policy_export.py`.  The featurizer no longer makes pair buckets (its biggest array, N x N), and batches carry no gates, since nothing reads them.  `policy_model.py` keeps batches, sampling and scoring, and the critic, which no longer reads an actor's memory.  `actions.Decision` replaces `net.Decision` and the trainer's copy.  `FEATURE_VERSION` is in `featurize.py`, the embedding sizes in `policy_model.py`; neither changed, so saved encoders and actors still load.
* **The trainer** trains the actor on latents only: no `toy`/`sim` argument, `--latent-encoder` defaults to `policy_latent.ENCODER`, concurrent collection is on by default, and the step-by-step loss and its sampled pair gradient are gone.  A segment stores `history` and `seen`, what the actor keeps, in place of the first actor's memory.
* **The toy** (`policy_toy.py`, `policy_toy_env.py`).  Tests that played it read frames of a script game in the sim instead (`conftest.sim_frames`), and share a small untrained frame encoder (`conftest.encoder_path`).
* **The JEPA frame model** (`frame_jepa.py`: LeJEPA's loss, views and training, and the probe ceiling).  What the VAE shares with it, the encoder, batches, shards and probes, is `frame_model.py`; `EncoderConfig` keeps its unused projector fields so saved configs load.
* **A fix:** `policy_train.DataWorkers` without worker processes kept its bundles in a module global, so a second learner in the same process replaced the first's (a test's learner failed only in the full suite).  Each now keeps its own.
* **The raider-only fight script** is out of the sim's scripts (`make_script` has "army" only) and lives in `frame_world.py` as `FrameFight`'s base, which the frame data still uses.  `policy_script.py` is the scripts' shared data.  The opening's income test now runs beside the army script.

**Not checked:** an engine game with the host (it imports on Windows and its tests pass offline); `--check-repro` and `--check-kill`, which now run on the sim.

**Tests:** `sim/tests` and `tests/policy`, 599 passed; `tests/test_host.py`, `test_hazards.py`, `test_terrain.py` with `tests/policy/test_policy.py`, 88 passed.  Tests of what's gone went with it; tests that played the toy now play the sim (`test_labels.py` reads frames from 4 minutes in, where there's an army to order; the opening's income test plays 40 games, as the army script's raids leave fewer to check).  `tests/test_hazards.py` is rewritten for what `hazards.py` still does.

### 11.11 A faster sim, and the scripts in C++ (2026-11-08)

The user wants far more cloning data than `army-3`'s 11.5M decisions.  Profiled on 32 mid-game games with both sides scripted, a step's time went 51% to the economy script, 37% to the army script and 7.5% to the sim.  At the user's request, two agents (Opus 5.5, each in a git worktree of its own) made the sim's batch faster and ported the economy and army scripts to C++.

**The sim** (`sim/src`, `sim/include/zksim`).  It plays the same games bit for bit: frame hashes match the old library at 32 and 128 games and at 1, 8 and 16 minutes, and no golden changed.

* A batch keeps a pool of threads; the caller works too, and games are handed out costliest first by the last step's time.  Before, each step started a thread per slice and slices were fixed, so 8 threads did little better than 4.
* Fog and wreck visibility go out from each side's own units, rather than searching the longest sight range around every target.  The spatial grid is split by side, so a search for enemies reads only theirs, in the same order as before.  Frames are emitted in one pass, and a tick allocates nothing.
* Several batches can step at once in one process (profile counters are per batch), and a forked process with a live batch starts its own threads.  `zk_batch_step_begin` and `zk_batch_step_wait` let the caller work while the sim steps; `zk_batch_set_threads` changes the pool.
* Game-steps a second, `tools/sim_step_bench.py`, which times only `zk_batch_step`:

| 128 games, 8 threads | Before | After |
|---|---|---|
| 1 minute | 82,132 | 142,943 |
| 8 minutes | 26,946 | 57,889 |
| 16 minutes | 12,688 | 33,213 |

At 32 games, 1 thread, 16 minutes, the phases took 387 us a game-step before and 198 after: fog 53 to 13, unit AI 31 to 16, emitting frames and features 134 to 38; movement, now the largest, 120 to 96.  Left out: dividing by a step as a multiply in two line-of-sight loops (4%, not bit-identical).

**The scripts** (`sim/scripts/`, `libzkscripts.so`; `tools/policy_scripts_cpp.py`).  The economy script, the aimer, the army script, the role mix and the constructor quota, with per-seat state and a thread pool of their own.  They read the records the batch writes, so they don't depend on the sim's insides.  `SimEnv(script_impl="cpp")` is the default, and `policy_clone_data.py record --script-impl` chooses.  The opening is still Python.

* Given the same frames and Python's random draws, the C++ scripts give the same orders: 420,621 of 420,621 army orders and 253,856 of 253,857 economy orders (`tools/policy_scripts_parity.py`).  The one that differs is a build at another site: numpy's sort isn't stable, and C++ breaks ties by index.
* Their own random draws make their games differ from Python's after the economy's first draw.  Over 128 games for each, on two seeds, wins, length, metal lost at the cap (1.8 to 13.3% by phase, against Python's 1.9 to 14.5%), production mix, constructors, income and big-project timing agree within the seeds' spread.
* Labels from a 64-game recording each: keep 96.70%, fight 2.29%, PRODUCE 0.88%, move 0.13% (Python 96.71, 2.31, 0.85, 0.13).  Orders the policy can't say: 0.88% (Python 0.54%), all of them a sniper's attack, so it follows how many snipers were made.

**Together**, `SimEnv.step` with both sides scripted, 128 games, 8 threads, 8 minutes in: 92,443 seat decisions a second (2.8 ms a step), against 5,423 with the Python scripts (47.2 ms): 17x.  Recording is now held up by Python copying and writing frames: one worker made 6,697 decisions a second with the C++ scripts on the old sim, against 2,298 with Python.

**Not checked:** the new sim at 256 games or past 8 threads; the map pool; memory; the C++ scripts in modes other than army against army, beyond one test of policy seats.

**Tests:** `sim/tests` and `tests/policy`, 611 passed.  New: `sim/tests/test_batch_threads.py` (any thread count and schedule give the same games, two batches at once, a forked child) and `tests/policy/test_scripts_cpp.py` (the same orders as Python, any thread count, two at once, the recorder).

### 11.12 The Python scripts gone, and recording in C++ (2026-11-08)

**The Python scripts are gone**, at the user's request: the economy script, the aimer, the army script, the role mix, their parity tool and their tests (about 35 unit tests of what the scripts do), and `SimEnv`'s `script_impl` switch.  They're in git at `8e9bca7`.  `policy_eco.py` and `policy_mix.py` keep the tables the C++ is given.  The C++ scripts are tested only through games now (`tests/policy/test_scripts.py`).

Making C++ the default in `8e9bca7` broke the encoder's frame games (`frame_world.FrameGames`, `frame_data.ScenarioEnv`): they play their own Python scripts through `SimEnv`, which then stepped them with the C++ scripts on games it never reset.  No frames were collected in between.  A subclass now sets `own_scripts`, and a test checks they play their own.

**The recorder** (`sim/record/`, `libzkrecord.so`; `sim/py/zkrecord.py`; `zkrecord.h` describes the file).  `policy_clone_data.Recorder` hands it the batch's arrays by pointer each step, so Python copies nothing, and it writes a `.zkr` file a game (data version 4):

* Both sides of a game in one file, compressed with zstd a chunk of 64 frames at a time, so a game in progress holds at most one chunk raw.
* A unit is 40 bytes, not 80.  x and z are 16-bit fractions of the map, and order targets of [-0.25, 1.25] of it, since targets lie off the map.  y is 16 bits of the game's range, taken from the map's heights.  Velocities, health, experience and a blip's speed are half floats; build progress is 16 bits; an order's age is whole frames.  Everything else is exact, and so are features' metal, the globals, the sites (stored once a game) and the orders.
* A unit or feature that was in the side's previous frame is stored as each field minus its value there (wrapping), with how many records back that was.  Columns are stored a byte plane at a time.  The reader resolves every chain of records at once, by pointer jumping and a cumulative sum per chain, with no loop over frames.
* zstd level 3 by default (`--level`).  On the 6 `army-3` games of the study below, 64-frame chunks with both sides came out no larger than whole-game streams: units only, 414 bytes a decision at level 3 and 374 at level 9.  Level 3 compresses at about 1 GB/s, level 9 at 180 MB/s.
* Files are written on a thread of the recorder's own, beside their name and then renamed, so a reader never sees half a file.

How it was chosen, from 6 `army-3` games (units only unless said): the npz's deflate, 483 bytes a decision; zstd 19 on the records as they are, 816 (4 games, whole files); narrow records with the residuals, 147 at zstd 19.  Rounding positions to 1/16 elmo before the residuals gave 344, still lossless otherwise; the narrow layout is the user's suggestion.

| The same 6 `army-3` games, whole files | npz (data version 3) | .zkr (4) |
|---|---|---|
| Bytes a decision | 516 | 192 (2.7x smaller) |
| Reading, us a decision | 23 | 32 |

Values clamped to fit: none, on those games (map-pool maps) or in the recordings below.

Recording 64 games at once on the training maps, 8 threads of 16 CPUs, until 64 have finished (168,764 decisions): 62,606 decisions a second in one process, 239 bytes a decision.  The sim and the scripts take 68% of the time.  The recorder takes about 30%: narrowing frames 10%, compressing chunks 20%; all games fill a chunk on the same step.  Writing files on the caller's thread had cost 13% (WSL's `/mnt` is slow); on the recorder's own thread it costs 1%.  For comparison, one worker of the Python recorder made 6,697 decisions a second (11.11).

**Not checked:** a full multi-worker recording with the map pool; a training run on `.zkr` data (its data path was: it lists the sides of a game, splits them and makes a sample from one); memory per worker at 64 or more games.

**Tests:** 579 passed.  `tests/policy/test_clone_data.py` records games keeping a Python copy of every frame and order, and checks the file reads back the same: exactly where exact, else within each field's rounding.

### 11.13 Army script variants, and the start of `army-4` (2026-11-08)

**`army-4`** (`/mnt/d/zk-learned/clone/army-4`, data version 4) was started to be about 100 GB of script games.  The user stopped it once it was clear it would be one script mirroring itself, and asked for variants first.  It holds 3,243 finished games (3.8 GB), and has no `summary.json`.

How the army script's orders group, on 24 of those games at the 15,030 decisions where it looks: units act at 8,710.  At those, a median of 3 units act (mean 7.1, 90th percentile 17, max 149), sharing a median of 1 distinct order (mean 1.8, max 9).  Fight is 78% of acting units, produce 20%.  This was measured for the user's question about group ("area") actions.

**Variants** (`tools/policy_variants.py`, `zks_style` in `sim/scripts/zkscripts.h`, ABI 2).  Made by an agent at the user's request.  Each side's army script draws its own knobs at game start, from the environment's generator: `SimEnv(variants=True)`, `policy_clone_data.py record --variants`.  Each game's meta records both sides' knobs.  Without it nothing is drawn and play is the same as before: a seeded 32-game run hashes the same.

* **Knobs.** All were constants the C++ already had:
  * a weight per role in the factory mix;
  * `commit`: how much metal over the cover a squad wants before it goes;
  * `gather`: what the army waits to gather;
  * `patrol_share`: raiders guarding extractors;
  * `patience`: give-up timers;
  * `retreat_below`: the health share at which a fighter goes back to the rally, until 90%.
* **Presets.** `standard`, `rush`, `turtle`, `skirmish`.  Each side draws one, then each knob is jittered within bounds.

One check, 64 games on the training maps, one seed (128 sides):

| Variant | Sides | Won / drew / lost | Raid | Riot | Assault | Skirm | Arty |
|---|---|---|---|---|---|---|---|
| rush | 39 | 17 / 13 / 9 | 48% | 14% | 24% | 10% | 2% |
| skirmish | 26 | 9 / 8 / 9 | 42% | 15% | 14% | 17% | 12% |
| standard | 30 | 8 / 12 / 10 | 39% | 19% | 18% | 16% | 5% |
| turtle | 33 | 9 / 9 / 15 | 29% | 30% | 14% | 19% | 7% |

The mixes differ as meant, and every variant wins and loses.  The win rates aren't separated at this size.

What limits variety in the script itself:
* Every fighter retreats below 40% health: all 159 fighting types regenerate, so the rule isn't a rare one.  It's now `retreat_below`.
* Artillery starts at 12 minutes and games last about 13, so it's about 5% of units.
* Snipers need a slow heavy unit seen after 8 minutes, so they're 0.3%.
* AA never appears, since nothing flies.
* The role schedule is a fixed function of time.

**Not checked:** more than one seed, the map pool, a full recording with variants, training on it.  There are no tests of the variant path beyond the default's hash.

### 11.14 Air in the sim, and a second factory for the scripts (2026-11-08)

At the user's request: the sim flies every aircraft (`01-simulator.md` 8.18, ABI 16), and the scripts build a second factory.

**The second factory** (`second_factory_job` in `sim/scripts/src/eco.cpp`, scripts ABI 3).  Once it's both 18 minutes in and 50 metal a second not counting reclaim (the user's rule; it was first either 18 minutes or 45, which income alone always met first, at 4 to 13 minutes), an idle constructor starts a plane plant, a gunship plant or a strider hub, drawn once per side when it's first due (0.5, 0.3, 0.2: planes most often, then gunships, as the user plays).  A hub needs a grid of 50 energy and is built on the main grid; with less there, it's planes instead.  Reclaim's share of income comes from the batch's third stat (`ZK_STATS` is 3).  The floating rule that adds factories of the first kind is unchanged.

**What they make** (`choose` in `army.cpp`).  An air plant has its own mix, of whatever's on its menu: raiders 0.35, riots 0.2, assaults 0.25, skirmishers 0.2, bombers 0.4, and AA 0.42 once the enemy has been seen with air.  It counts only what flies, and a ground plant only what doesn't, so neither's mix moves the other's.  A plane plant makes Ravens, Phoenixes, and Swifts once it has seen air; a gunship plant Locusts, Revenants and Harpies.  A strider hub, not an air plant, follows the ground schedule: Scorpions, and Merlins once artillery's share comes in (none were seen).  Menus leave out ships and units that blow themselves up (`policy_mix.menu`): the hub had picked a Reef for artillery, and a gunship plant would have made Blastwings.  New role `bomb` (`R_BOMB`): a flying bomber goes for the defence nearest the army's lead, else the nearest building, and is ordered again when it's back and idle.

**One check**: 16 script games on the training maps, one seed, run to 25 minutes with no early end.
* 17 of 32 sides built one: 11 plane plants, 5 gunship plants, 1 hub.  Four games, 8 sides, were over before 18 minutes; of the 24 sides still playing then, 7 never reached 50 metal a second or lost first.
* 12 started between 18.0 and 18.8 minutes, with income from 50 to 119; the other 5 at 19.5 to 24.5 minutes, when income reached 50.  No reclaim was seen: the scripts don't reclaim.
* With the earlier rule (either trigger) on the same games, 29 of 32 built one: 17 plane plants, 8 gunship plants, 4 hubs.  The production counts below and the AA count are from that rule.
* On a second seed, what the second plants were told to make: plane plants Ravens 162, Swifts 110, Phoenixes 108, constructors 106, Sparrows 20; gunship plants constructors 22, Locusts 14, Tridents 14, Revenants 12, transports 9, Harpies 7; hubs Scorpions 21, Athenas 19.  Bombers were given 820 attack orders.
* Ground AA now appears: up to 52 a side.

**Not checked:** whether air is any good in these games (no win rates by plant), the map pool, more seeds, the policy with air, or the frame encoder on aircraft, which it never saw.  Constructors are a large share of what air plants make, since the constructor quota is filled by whichever plant asks first.  The gunship plant's scout is a transport (`gunshiptrans`), which carries nothing here.

### 11.15 Group orders: the army script's, and the policy's (2026-11-08)

At the user's request, after weighing FlexAttention with a token per order: group ("area") commands, in the sim (`01-simulator.md` 8.19), the C++ army script, and then the policy's actions and the labels.

**The form** (`zkbot/policy/groups.py`, sim ABI 17).  A group order is one unit's order, the anchor's, with a radius and a set of roles: the sim gives it as well to every unit of the anchor's side of those roles within the radius of the anchor.  The radius is one of ten, 60 to 2,250 elmos, each 1.5 times the one before (`GROUP_RADII`); roles are the bot catalog's 19, numbered in sorted order, of which the 15 with a mobile type can be chosen (`choosable`).  Only move, fight and attack go as groups (`GROUP_KINDS`).  It was first a centre and a radius; an anchor makes it a choice of a unit the policy already has a token for, as a player clicks a unit to start a box.

**What the script sends** (`Seat::group_plans`, `sim/scripts/src/army.cpp`; scripts ABI 4).  Of the fighters' move, fight and attack orders it sends this step, those that are the same order to the letter (kind, target, place) go as one group order: the anchor is the member nearest their middle, the radius the smallest of the ten that reaches the farthest member by an elmo.  If none does, or the circle would take in any other own unit of those roles, they're split in two across their wider spread and each half is tried again; any left alone go as unit orders.  So the sim does just what the orders alone would do: those it could select are taken as every own unit with a role, a superset of what it selects; a member is inside the radius and anyone else outside it by an elmo, which float rounding can't undo; and the sim applies group orders before unit orders, while a fighter's other orders in a step all come after its plan's.  `SimEnv.group_orders` (on by default) turns it off.

**Recorded as the policy would give them** (data version 5).  The army's orders in `army_orders`, `script_record` and the `.zkr` files are those sent alone, and each group order as its anchor's order followed by a record of kind `GROUP_MARK` (100) for the same unit, with the radius as x and the roles as target; the other members have none, so their label is keep.  `.zkr` keeps orders exactly, so the format didn't change.  `policy_sim_env.policy_orders` reads a mark as the radius and roles of the order before it; `sim_orders` writes them the same way for the policy's orders, and `split_groups` makes the sim's group orders of them.

**The policy's heads** (`policy_latent.LatentActor.group`, `.roles`; `policy_model.sample_actions`).  A unit whose kind is move, fight or attack then chooses alone or one of the ten radii (11 ways, from its reading and an embedding of its kind; alone starts 2 nats ahead, as keep does), and if a radius, each choosable role in or out on its own (a Bernoulli each, from the same and an embedding of the radius).  The log-probability adds the group choice for those kinds and the roles' for a group; PPO's entropy bonus has two more heads, `group` (floor 0.3 x ln 4) and `roles` (mean over roles, floor 0.3 x ln 2).  Action tokens add an embedding of the radius and a projection of the roles.  A unit given another's group order in the same step keeps its own choice if it made one: group orders go first.  The actor's weights don't load into older checkpoints' shapes; cloning has to be run again on version 5 data.

**Checks.**
* 16 script games with variants, 2,000 decisions each, with group orders and without: no frame differed.  Of the army script's 17,748 orders, 5,224 (29%) went in 1,548 group orders, 3.4 units each, so the sim got 14,072 (79%).  The centre form grouped 48%: ten radii and an anchor at a member, not at the middle, need bigger circles that take in others more often.
* `test_group_orders_play_the_same_games_as_the_same_orders_unit_by_unit_and_are_recorded_as_given`: 600 decisions both ways, same frames, and at every decision what's recorded, expanded by `groups.expand`, tells every unit what the orders alone told it.
* `test_the_policys_group_orders_reach_the_units_they_select`: a policy's group order through the environment reaches exactly the units `expand` selects (at least 4).
* `test_what_decoding_makes_reads_back_as_the_same_choices`: random choices with groups, decoded, through the sim's orders and back, encode to the same kind, enemy, produce, group and roles.
* A small version 5 recording (35 games): no order the policy can't say, 117 group labels in 2,670 decisions read; 30 cloning steps run on it.  Full suite: 596 passed.

**Not done:** cloning or RL with the group heads; whether the policy uses them well.  How much more the script would group if a group could take units already doing that order wasn't measured: exactness keeps a unit that's already on its way, and so isn't re-ordered, from sharing its squad's circle.

### 11.16 The encoder retrained with air (2026-11-08)

At the user's request: `frame_vae.py train` as before (`--autoencoder --region-pool 32 --two-part --questions --muon-lr 0.008`, 25,000 steps, 50 minutes) on `frames/pool4`, collected with the sim's air and second factories, to `runs/encoder-air/final`.  The actor now reads it (`policy_latent.ENCODER`).

On the same 8,000 frames of `pool4`'s held-out maps, the old encoder (`encoder-questions`, which never saw aircraft) and the new:

| | old | new |
|---|---|---|
| reconstruction R2 | 0.926 | 0.924 |
| unit counts R2, own ground | 0.503 | 0.411 |
| unit counts R2, own / enemy air | -0.12 / -0.12 | -0.21 / -0.18 |
| where, any within 0.05 | 0.706 | 0.743 |
| where, top within 0.05 | 0.598 | 0.605 |
| what: type / side | 0.670 / 0.964 | 0.678 / 0.968 |
| what: hp R2 / vx R2 | 0.559 / 0.637 | 0.563 / 0.657 |
| aircraft asked about: found / type | 0.744 / 0.559 | 0.761 / 0.579 |
| aircraft: vx R2 / hp R2 | 0.586 / 0.469 | 0.610 / 0.431 |

Mixed: the questions, which ask what the latents say about a place or a unit and are nearer what the actor reads, are better on all but aircraft health; the count grid is worse.  The switch is because the actor has to be cloned again anyway for the group heads, and this encoder has seen the units it will meet.  **Not checked:** an actor trained on each.

### 11.17 A bigger encoder, 40 latent tokens, on themed scenarios and the C++ scripts' games (2026-11-08)

At the user's request: 40 latent tokens of 32 (1,280 numbers a frame, from 24), an encoder of 1.5 to 2 times the parameters, wider and deeper, and frames of what the C++ army script seldom makes, aircraft fighting above all.

**The data** (`tools/frame_data.py`, `frames/pool5`).  Three sources, 1M frames each, on the map pool with its held-out maps left out:
* `script`: the Python frame games (`frame_world.FrameGames`), as before.
* `scenario`: random scenarios, now 40% of them as before (`mixed`) and 60% themed (`FLAVOURS`): air battles 20% (one side all aircraft, the other aircraft and ground AA, small bases, 3,000 to 40,000 metal), air raids 10% (aircraft against a base that spends 30 to 70% of its building budget on AA turrets, the attackers' clusters placed most of the way to the enemy), fleets 10% (ships against ships, only on maps with water; on a map without, a mixed one), heavies 10% (units of 1,200 metal or more against a mixed army) and swarms 10% (units of 120 metal or less on both sides).  Which side gets which is by chance.  AA turrets are what shoot air and not ground (`_shoots_only_air`); the torpedo turret passes too, from how its weapon is rated.
* `army`: the C++ army and economy scripts' games with variants (`SimEnv(..., ["scripts"], variants=True)`), which the actor is cloned from and the encoder hadn't seen.  Game numbers are now spaced so that three sources fit in int32 (`SOURCES_APART`, `GAMES_APART`).

How often a frame holds aircraft (`seen_in`, own / enemy): script 23% / 23%, scenario 48% / 48%, army 0.2% / 0.2%.  Ships: 3%, 9%, 0%.  So the C++ script all but never flies by the time its games end, and what the encoder knows of air is from the scenarios.  The grouping change of 11.18 was built while the collection ran; any worker that loaded the scripts' library after it played the new grouping, so the `army` frames may hold some of each.  The change only tells some units their order again; it wasn't checked which workers had which.

**The encoder** (run `encoder-big/final`): as `encoder-air` (`--autoencoder --region-pool 32 --two-part --questions --muon-lr 0.008`, 25,000 steps), with `--queries 40 --d 144 --rounds 3`: 1,650,624 parameters to `encoder-air`'s 915,920 (1.8 times), 82 minutes on the RTX 4070.  `policy_latent.ENCODER` is now it; the actor's sequence has 44 tokens a frame where it had 28, and it has 4,096 more parameters (1,550,478) for its 40 slot embeddings.

On the same 8,000 frames of each collection's held-out maps (`pool4` as in 11.16; `pool5` taking each source's shards in turn, so the three are alike):

| | `pool4`: questions / air / big | `pool5`: questions / air / big |
|---|---|---|
| reconstruction R2 | 0.926 / 0.924 / **0.945** | 0.933 / 0.926 / **0.970** |
| unit counts R2, own ground | 0.503 / 0.411 / **0.726** | 0.307 / 0.228 / **0.722** |
| unit counts R2, own / enemy air | -0.12 / -0.21 / **-0.04**, -0.12 / -0.18 / **-0.01** | 0.13 / 0.15 / **0.50**, 0.11 / 0.22 / **0.62** |
| where, any within 0.05 | 0.706 / 0.743 / **0.865** | 0.624 / 0.672 / **0.781** |
| where, top within 0.05 | 0.598 / 0.605 / **0.740** | 0.486 / 0.493 / **0.601** |
| what: unit found / false | 0.790 / 0.798 / **0.900**, 0.077 / 0.076 / **0.033** | 0.778 / 0.793 / **0.890**, 0.100 / 0.098 / **0.043** |
| what: type / side | 0.670 / 0.678 / **0.812**, 0.964 / 0.968 / **0.983** | 0.650 / 0.666 / **0.818**, 0.974 / 0.977 / **0.988** |
| what: hp R2 / vx R2 | 0.559 / 0.563 / **0.747**, 0.637 / 0.657 / **0.807** | 0.592 / 0.589 / **0.764**, 0.547 / 0.567 / **0.740** |
| aircraft asked about: found / type | 0.744 / 0.761 / **0.885**, 0.559 / 0.579 / **0.759** | 0.712 / 0.736 / **0.871**, 0.450 / 0.474 / **0.722** |
| aircraft: vx R2 / hp R2 | 0.586 / 0.610 / **0.784**, 0.469 / 0.431 / **0.677** | 0.572 / 0.615 / **0.797**, 0.466 / 0.486 / **0.689** |

Better on every measure on both, by a lot: aircraft type 0.58 to 0.76, own ground counts 0.41 to 0.73.  **Not separated:** how much is the 40 tokens, the size, or the data; and what any of it does for an actor (the actor has to be cloned again in any case, for 11.18).

### 11.18 Actions one after another, as Pluto acts (2026-11-08)

At the user's request, after comparing the model with Pluto's (a StarCraft: Brood War bot that gives one action a step, a selection or a command, through a few heads chosen in turn): a decision is now a short list of actions, chosen one after another, where every unit chose its own order at once before.  The user's reason: Zero-K's unit AI is made to need as little micro as it can, so the per-unit form's one strength, cheap micro for every unit at once, buys little.  `docs/model.md` section 5 has the model.

**The script's grouping first** (`Seat::group_plans`).  Whether a few actions a decision could say what the army script says was measured first, on 35 version 5 games: actions a look (units told something, a group order counting once) had a median of 1, a 95th percentile of 8, a 99th of 28 and a maximum of 111, and at most 8 a decision would have cut 28% of the orders.  The big ones were pushes late in the game: in looks of more than 8 orders (22 on average), 82% of them went to one place, yet they made 1.8 group orders.  The script doesn't send a unit an order it already has (`_emit`: the same kind, within `REORDER`, 200 elmos), and those units, of the same roles and inside the circle, stopped it as units the group would wrongly take in.  Now such a unit doesn't stop it: the group order takes it in, and tells it its order again, to the letter of the group's.  So the games are no longer exactly those of the orders alone; the test that said so is now `test_group_orders_tell_each_unit_what_it_was_told_alone_and_again_what_it_was_already_told` (the same games up to the first such unit, and each taken-in unit's last order a move or fight of that kind within 200 elmos).  On 74 games recorded with it: 1.28 actions a look, median 0, 95th percentile 6, 99th 11, maximum 41.

| at most | looks with more | orders cut |
|---|---|---|
| 8 | 2.1% | 7.3% |
| 12 | 0.7% | 2.7% |
| 16 | 0.3% | 1.1% |

**The form** (`policy_latent.py`, `policy_model.py`, `zkbot/policy/actions.py`).  Up to 16 actions a decision (`actions.SLOTS`).  For each, a decoder (`plan`, two blocks) reads the frame's latent tokens and the actions chosen so far, and a pointer picks a unit, the anchor, or done: a unit that may be told something other than keep and that no action before picked; done starts 3 nats up (`DONE_START`).  The anchor reads the frame with a query of its own embedding and its slot (`reading`, the old per-unit `read_unit`), and the heads choose its order as each unit's did (kind, never keep; region, enemy, fine, produce, group, roles), with its masks (`per_action`).  The action becomes a vector (`action_vectors`: what each acting unit's part of the action tokens was), which the next slot reads; the action tokens read the same vectors.  A unit no action picks keeps.  An action's log-probability is its anchor's (or done's) plus its heads'.  `Decision`, `encode` and `decode` stay per unit: `actions.to_slots` and `to_decision` go between.  The actor has 1,546,382 trained parameters, 435,737 of them the decoder and anchor; it was 1,082,199.

**PPO** (`policy_train.py`).  A step keeps its actions as chosen (`Step.anchor` first, each field one entry an action).  Each action, done included, has its own clipped ratio, averaged within the decision; one advantage a decision, as before.  The entropy bonus has an `anchor` head (floor 0.3 x ln 4).  `actions_a_decision` is logged; `share_kind*` is now of actions, so keep is no longer in it.

**Cloning** (`policy_clone_train.py`).  The script's orders, as the policy's choices per unit, become actions in a random order, at most 16 (1.1% cut, `cut_share`), with done after.  They have no order of their own, so an action's anchor scores as any unit the script told something and no action before it picked (`evaluate_actions`' `labelled`: the log of the sum of their probabilities), and done once none is left; its heads score the one given.  So the actor is taught which units to give orders to, and not an order among them that isn't in the data.  The keep weight and its shift are gone: done stands for what's left, and a unit the script left alone is one no action picks.  Held-out measures: `anchor_right` (the likeliest anchor is a unit still to be told), `stops_early`, `done_right`, `kind_right` and `right_KIND` given the anchor, `nll_act`, `nll_done`.

**Checks.**  Full suite: 650 passed.  New or rewritten: a sampled decision scores the same when evaluated, for an untrained actor and for one that's never done; a script's actions scored as a set score at least as well as in their order, and the last of them the same; no unit is picked twice and every choice is legal for its anchor; the action tokens read actions as a set; segments in training score as their decisions did in play.  30 cloning steps ran on the 74 new games.

**Not done:** cloning long enough to compare with the per-unit form, and RL.  The decision rate is as before: cloning counts the decisions where the script looked, every 6th, and the policy decides at every one.

600 steps of cloning on CPU (8 stretches of 24 decisions a step, `--lr 1e-3`, on the 74 new games; held out: 2 games, 82 actions and 192 dones): the likeliest anchor was a unit still to be told 5% of the time at step 150 and 39% at 600, done where there was more to do fell from 95% to 61%, and done where the script was done stayed at 93%.  The kind is right 93% given the anchor, most of them factories producing.  So the set loss learns; how far it goes is for a real run.

### 11.19 A diversity bonus, after Pluto's (2026-11-08)

At the user's request: Pluto's reward has "a small bonus for building less-frequently seen units and tech", zero-sum.  Here (`policy_sim_env.SimEnv._diversity`, `rarity`; `policy_train.Config.diversity`):

* Each unit a side finishes, of a type a factory can make that moves (122 types), is worth `diversity` x its type's rarity.  Rarity is 0 for a type the policy's sides make at least as often as the average of those 122, and rises on a log scale the rarer it is: log(1 / (N x share)) / log N, at most 1.  The shares are counts of what the policy's sides finished, over every game of the batch, decayed by 0.999 a decision (a half-life of 700 decisions), with one of each type added so that nothing is rare from the start.
* Both sides earn it, and `reward` takes it zero-sum, as Pluto does: side 0's shaped reward is the change in the value balance plus its bonus less side 1's.  It's in the shaped part, so it fades with beta into the result (03-training.md 1.3) as the value balance does.  Against the script, the script's rare units count against the policy; that adds noise, not a bias, as the policy can't change them.
* Only what's finished counts, so ordering a rare unit and cancelling it earns nothing; the units a side starts a game with count as made already.  Buildings and tech aren't counted: in the sim the economy script builds them, so they're not the policy's to choose.  Pluto's bonus is for building; using a rare unit well earns what it earns through the value balance.
* `diversity_a_decision` is logged.

**How big.**  On 155 script games with variants (1,526 decisions on average), rarity counted over both sides, at 0.005 a side's bonus came to 0.0025 a game on average (0.019 at most), against a sum of 1.35 of the value balance's changes: nothing.  17 of the 54 types made had a bonus (amphibious AA 0.6, the plane constructor 0.63, the light scout plane 0.57, the cloaked AA 0.35, ...); constructors and raiders were the most made, 4 to 7% each.  The default is 0.05: a unit of a type at rarity 0.6 is then worth 0.03, about what losing a 300-metal unit costs at 10,000 metal on the field (the value balance is (a - b) / (a + b + 1000)).  On those games that's 0.025 a side a game on average, 1.9% of the value balance's changes, and 0.19 at most (14%); the plan's ceiling was 5% (1.3).  It's a guess, not a tuned figure.

**Checks.**  `test_finished_units_of_rare_types_earn_a_zero_sum_bonus_and_the_first_units_none`: a type made 10,000 times has rarity 0 and one never made more than 0.8; nothing at the first decision; over 300 decisions of 4 games the policy's sides earn some, the counts grow, and every seat's reward is the negative of its opponent's.

**Not done:** a PPO run with it, or any check that it changes what the policy makes.

### 11.20 `army-5`, and cloning 18 times faster: a decision every 3 s, and the age bias's backward (2026-11-09)

**`army-5`** (`/mnt/d/zk-learned/clone/army-5`, data version 5): the C++ army and economy scripts against themselves with variants, on the map pool's training maps, 12 workers, seed 5.  114,223 games, 401.5M decisions (both sides), 138 GB: 368 bytes a decision, not the 239 of 11.12's 64 games, as these run longer and hold more units.  1 h 56 min at 58,100 decisions a second.  Endings: 71,183 on a lead, 39,180 at the time limit, 3,860 by elimination.  `check` on 20 sides: no order missed, 107 of 704 orders group orders; labels keep 98.0%, fight 1.2%, produce 0.6%, move 0.2%.  1.1M units' current-order places were clamped to -25..125% of the map, about 0.005% of the unit records.

The first try was stopped at 5 minutes: each worker grew about 3 MB a second.  Not a leak: each made its own copy of each map's masks, site samples and part labels (`policy_opening.Ground`) and the C++ scripts' copy of them, as it first played the map, about 6 MB a map.  On the 298-map pool that's about 1.6 GB a worker, 20 GB for 12 on top of the shared 7.5 GB pool.  `policy_sim_env.share_maps` now makes them all before the workers fork (`Ground.warm`, and the scripts' data with every map added), so the workers share them: 12 level off at 5.4 GB together.  Games play the same either way (`test_shared_maps_are_used_and_play_as_each_env_making_its_own`).

**`clone-5`**, `/mnt/d/zk-learned/runs/clone-5`: the action form (11.18) with `encoder-big` (11.17), a decision every slot as before, 16 stretches of 48 a step, every 6th decision counted (where the script looked).  52 counted decisions a second.  Stopped at 11,500 of 20,000 steps, when the user chose a decision every 3 s instead.  On 96 held-out stretches (1,489 actions, 759 dones):

| Step | Loss | Anchor right | Kind right | Move right | Done right | Stops early | NLL of actions | NLL of done |
|---|---|---|---|---|---|---|---|---|
| 500 | 4.67 | 0.578 | 0.950 | 0.00 | 0.961 | 0.403 | 5.86 | 0.47 |
| 3,000 | 2.79 | 0.667 | 0.988 | 0.87 | 0.881 | 0.234 | 3.96 | 0.66 |
| 6,000 | 2.51 | 0.648 | 0.987 | 0.89 | 0.950 | 0.306 | 3.66 | 0.41 |
| 9,000 | 2.34 | 0.678 | 0.990 | 0.89 | 0.951 | 0.263 | 3.44 | 0.43 |
| 11,500 | 2.27 | 0.678 | 0.989 | 0.89 | 0.945 | 0.259 | 3.32 | 0.37 |

"Anchor right" is how often the likeliest anchor is a unit the script told and no action before picked; "stops early" how often done is likelier than any such unit while some are left.  Kinds are all but learned (fight and produce 1.00; move from 0 to 0.89 on 75 labels).  Choosing whom to tell is the hard part, and still improving when stopped.

**Speed.**  At 52 counted decisions a second, 20,000 steps would have taken 82 hours.  An `nvidia-smi` reading of 3% had suggested the workers held it up; the user saw the GPU at 90-95% in Task Manager, and that reading was a moment between steps.  What was done, in order, each measured on 600 of `army-5`'s files:

1. **A decision every 3 s** (`--decide-every 6`, the user's choice).  The script looks every 6 decisions, and only those count, but the actor read every frame: a stretch of 48 decisions and its 32 frames of history were 80 frames encoded for 8 counted decisions.  Now the actor reads only the frames it decides at, `policy_latent.window_frames(6)`: a window of 4 frames (8 s rounded to the rate, 9 s back), a reach of 6.  A stretch of 48 is 48 looks and 6 frames of history.  PPO must play at the same rate.  262 a second (5x).
2. **Decoding only what a sample needs** (`zkrecord.read(side=, until=)`, `policy_clone_data.read_side`).  A worker read the whole game, both sides, for 54 frames of one: 0.24 s of a sample's 0.32 s.  Records are deltas along the game, so it still reads from the start, but stops after the chunk holding the stretch's last frame and doesn't unpack the other side.  488 a second.
3. **The age bias's backward.**  A PyTorch profile of a step: 77% of the GPU's time was one kernel, `indexing_backward_kernel`, 250 ms a call, twice a step.  `TimeBlock.bias` indexed the age table (heads x window) once for every pair of tokens, about 2,400 x 2,400 a layer, and the backward of that indexing adds 5.6M gradients one at a time into a few dozen places.  `_age_bias` now indexes it a pair of frames at a time and spreads that over their tokens, which backward sums; the same values and gradients (`test_the_age_bias_by_frame_is_the_age_bias_by_token_and_learns_the_same`).  A GPU step took 1.07 s and takes 0.49 s.  716 a second.
4. **One encoder call and one action-token call a step**, over all the samples' frames joined (`pm.join`, which now keeps the true names the encoder reads), in place of 16 of each.  With samples made beforehand a step took 0.57 s; with the workers making them as it ran, 0.93 s.  Fewer torch threads changed nothing, so what was left was the main process taking in about 120 MB of samples a step while issuing a step's thousands of small kernels.  Joined, the latents differ from one sample at a time by at most 6e-4 (different padding) and the action tokens by 7e-7.  961 a second with 6 workers, 918 with 12.

Longer stretches (8 of 96 in place of 16 of 48) gained 10% on the GPU once history was 6 frames, not 32, and were left out for their more correlated batches.  Not done: sending samples through shared memory rather than pipes, or not sending the decision frames twice (`seen` and `heads` hold mostly the same frames now), worth up to 1.4x by the 0.57 s step.

**`clone-6`**, `/mnt/d/zk-learned/runs/clone-6`: as `clone-5` but `--decide-every 6`, 8 workers, 20,000 steps of 16 x 48 looks, 15.4M of `army-5`'s 67M looks.  860 counted decisions a second, about 5 hours; 1,550,374 trained parameters (the age tables are smaller).  Its checks hold 96 stretches of 48 looks: 8,395 actions, 4,573 dones.  At step 500: anchor right 0.625, kind right 0.951, done right 0.949, NLL of actions 5.26 (`clone-5` at 500: 0.578, 0.950, 0.961, 5.86, on other stretches, having read a sixth as many of the script's decisions).  Starting took 11.6 minutes, most of it reading 114,224 files' headers over `/mnt/d` (3.5 minutes for `clone-5`).

**Tests:** `tests/policy`, 317 passed.  New: the partial read reads the start of the whole (`test_one_side_as_far_as_a_frame_reads_as_the_start_of_the_whole`, over 8-frame chunks), the age bias by frame, the shared maps.

### 11.21 `army-6`, a 10M actor, flex attention, and a trainer that keeps the GPU fed (2026-11-09)

**`army-6`** (`/mnt/d/zk-learned/clone/army-6`, data version 6): as `army-5` (11.20), with the economy script's orders recorded after the army script's, both looking every 6 slots, and reclaim (`04-economy.md`).  83,820 games, 300.6M decisions, 105 GB; stopped by the user at 75% of the 400M asked.  Every game reads back whole (167,640 sides).  The 1,044-game `army-6-test` passed `check`: 0.35% of orders missed; build, assist and reclaim all there.

**The actor at 10M** (the user, 2026-11-09): `policy_latent.Size(d=256, heads=8, layers=4, plan_layers=4)`, 10,537,808 trained parameters; the old 1.9M is `SMALL`.  The critic is d 256, 4 layers, 8 heads, 3,221,769 parameters.  A reach of 12 decisions.  Every GPU script caps its memory (`set_per_process_memory_fraction`, `--gpu-memory` 0.85): past the card, the Windows driver spilled into 30 GB of system RAM instead of failing.  8 stretches of 48 a forward pass peak at 4.5 GB; 16 don't fit.

**The cloning step, the main process alone** on samples made beforehand, 8 stretches (`policy_clone_train.forward`, backward, optimizer):

| | ms a step | Peak GPU |
|---|---|---|
| Dense attention, after staging every host copy at the step's start and removing the syncs mid-pass | 230 | 4.8 GB |
| Flex attention, the age bias as a score_mod | 562 | 4.4 GB |
| Flex attention, the age bias frozen (no gradient) | 159 | 4.4 GB |
| Flex attention, the age bias as appended query and key dimensions (`TimeBlock.bias_dims`) | 157 | 4.5 GB |
| ... and `--compile default` on `sequence` | 155 | 4.4 GB |
| ... and `--compile reduce-overhead` (CUDA graphs) | 147 | 3.6 GB |

The dense path's biggest cost was the time transformer scoring every pair of a stretch's ~2,640 tokens, ~93% masked.  Flex attention skips the masked blocks, but a score_mod reading the age table gets its gradient a pair of tokens at a time, as atomic adds into ~136 places: 11.20's mistake again, inside the kernel.  As appended dimensions the bias is a dot product, and its gradient comes through dq, summed a frame at a time.  It matches the dense path to 1e-4 in outputs and gradients without TF32, whole and newest only (`test_flex_attention_reads_and_learns_as_the_bias_does`; with TF32 on, as `policy_train` sets, the two paths round differently by ~3e-3).  `--attention flex` is the default on CUDA; compiling adds little and is off.  Flex is compiled with automatic dynamic shapes, so a second shape (a held-out batch, a short game's stretch) doesn't pile up recompiles until it falls back to eager, which scores every pair: 8 to 7 to 8 stretches stayed at 157 ms.

**The trainer.**  The user saw the GPU at 100% for a second, then idle for one or two.  The trainer took a whole step's 64 stretches before its first forward pass, with 16 being made: the GPU and the workers waited on each other in turn.  Each pass's stretches are now taken as it needs them, with `max(2 x samples, 3 x workers)` reads in flight.  Then the workers were the limit, which was cut three ways:

* **Several stretches a read** (`--per-read` 4, `_samples`): a record is changes along the game, so a stretch is decoded from the game's start, a third of a worker's time.  A worker now cuts 4 stretches one after another from one decoding, and the next stretch's 12 frames of history are this one's end, featurized once.  The trainer draws stretches at random from 4 passes' worth made (`Prefetch(mixed=)`), so a pass isn't one game's.  Each stretch is what reading it alone gives, its own names dropped (`test_stretches_cut_from_one_read_are_those_read_one_at_a_time`).
* **`featurize._masks`** built each unit's reachable regions in a loop over units, a quarter of featurize; now all at once.  The golden fixtures are unchanged.
* **15 workers.**

| Trainer, `army-6-test` on `/mnt/d` | Decisions a second |
|---|---|
| As it was, 12 workers | 791 |
| A pass's stretches taken as needed | 1,145 |
| 4 stretches a read, `_masks` at once | 1,356 |
| History shared between a read's stretches | 1,397 |
| 15 workers | 1,462 |

The workers alone, from WSL's own disk: 1 makes 393 decisions a second, 4 make 1,431, 8 2,137, 15 2,392 (the CPU is a Ryzen 7 5800X3D, 8 cores, 16 threads).  `/mnt/d` is ~15% slower; one thread a worker changed nothing.  What's left is featurize, 1.7 ms a frame in ~1,500 small numpy calls, and the main process, which needs most of a core to issue a step's kernels and shares the cores with 15 workers.  A C++ featurize would be the real fix (numba did most of it: 11.22).

**`clone-7`**, `/mnt/d/zk-learned/runs/clone-7`: the baseline P3-P8, C1 and C4-C8 are measured against (`05-literature.md`), each run the same: `--decide-every 6 --samples 8 --accumulate 8 --lr 6e-4 --muon-lr 0.008 --workers 15 --steps 1000 --check-every 250` on `army-6`, a whole cosine schedule of 1,000 steps (the user: 1,000 is enough for a baseline), 3.1M looks.  Starting lists the 83,820 files in 194 s.
44 minutes in all, 1,326-1,427 decisions a second.  Its checks hold 96 held-out stretches: 14,277 actions, 4,512 dones, 142,144 units that could be told something.

| Step | Loss | Anchor right | Kind right | Done right | Stops early | NLL of actions | NLL of done | Ordered BCE (base 0.331) | Ordered right |
|---|---|---|---|---|---|---|---|---|---|
| 250 | 5.26 | 0.779 | 0.872 | 0.981 | 0.210 | 4.06 | 0.38 | 0.193 | 0.938 |
| 500 | 2.78 | 0.840 | 0.923 | 0.944 | 0.126 | 3.19 | 0.41 | 0.162 | 0.945 |
| 750 | 2.40 | 0.854 | 0.937 | 0.952 | 0.114 | 2.90 | 0.36 | 0.151 | 0.948 |
| 1,000 | 2.27 | 0.856 | 0.938 | 0.958 | 0.114 | 2.84 | 0.34 | 0.148 | 0.949 |

By kind at 1,000 (right / NLL, labels): produce 1.00 / 0.55 (3,877), fight 0.99 / 4.44 (4,181), move 0.92 / 3.64 (625), assist 0.92 / 0.61 (3,364), build 0.81 / 7.81 (1,762), reclaim 0.57 / 3.80 (466), attack 0 / 3.44 (2).  "Right" is the kind given the anchor; the NLL is the whole action's, its pointers and places too.  So the economy's kinds are learned, but a build's 7.8 nats are in what it builds and where (anchor, type, place, offset; not yet split by head), and reclaim's kind is right only 57% of the time.  The curve was still falling at the end (NLL of actions 2.90 to 2.84 over the last 250 steps, at a learning rate going to 0), so this is a baseline to compare against, not where cloning ends.  The checkpoint is `runs/clone-7/actor.pt`.

**Tests:** `tests/policy`, 323 passed at the 10M size.  New: flex attention against the dense bias (CUDA), a run's stretches against each read alone.

### 11.22 P3-P8, C1 and C4-C8 against `clone-7`, the cloned actor in closed loop, and featurize in numba (2026-11-09)

The user: implement P3-P8, C1 and C4-C8 (`05-literature.md`) in one go, then measure them against `clone-7` (11.21), each run with the baseline's command and a name of its own.

| | What it became | Measured |
|---|---|---|
| P3 DAgger | `policy_clone_eval.py --record DIR`: the cloned actor plays its side while the scripts only advise it, and its games are recorded as the scripts' are (`game-dagger-*.zkr`, which `sides()` and `check` read as any other).  libzkscripts ABI 7: a seat flagged `SEAT_ARMY_LABEL` gets the army script's orders as labels only, never sent and never as group orders; `SEAT_ECO_LABEL` the same for the economy script, where the policy runs its own economy.  `policy_clone_train.py --mix DIR --mix-share 0.5 --init RUN/actor.pt` fine-tunes on them mixed with `army-6` | the next step (chain2), on the better of `clone-7` and `clone-8` |
| P4 copying | `--eval-only --blind`: the held-out check with every action token zeroed; `--act-dropout p` | blind is worse on every measure (below), so the actor reads its past actions without leaning on them alone; `--act-dropout` left at 0 |
| P5 width | `--size small`: `policy_latent.SMALL`, 1.9M | `clone-7s` |
| P6 | `--finish-winners f`: the last f of the steps draw only the winning sides of decided games (51,651 of 164,220 sides); `--temperature` in play (`policy_model.TEMPERATURE`) | `clone-8` (f 0.1); the evals at 1.0 and 0.8 |
| P7 | `policy_train.py --init RUN/actor.pt` starts PPO's actor as a clone, at its size, and keeps a frozen copy for a k3 KL to it, weighted from `--kl-start` to `--kl-end`; `--clone-weight`, a cloning loss on what the scripts advise on the actor's own frames (`SimEnv.advise`); `--dual-clip`; `--critic-bins` and `--critic-span`, an HL-Gauss critic (a distribution over bins from -span to span); `--critic-warmup` iterations with the actor's learning rate at 0; `--entropy-floor 0` from a clone; `--fine-sim` (below) | a smoke run (below).  Not run: the user is retraining the encoder from scratch first, and PPO will start from the clone stage's last output |
| P8, C5 | z, 32 numbers a side (`policy_clone_data.Z_*`): whether the knobs are known, whether the counts are; the army script's 7 role weights (logs) and 5 knobs; its preset (rush, turtle, skirmish); and, in hindsight, its finished extractors, energy, factories, defences and army at minutes 3, 6 and 10 (log1p / 3).  `policy_clone_data.py zindex --data DIR` writes a dataset's (`z.npz`).  `--z` adds `z_in(z)` to every token of every frame, zero at first (`z_in` is made without drawing from the generator, so every other weight is as before); `--z-dropout 0.2`: the whole z unknown in that share of samples, the counts alone in as many again.  In play, `policy_clone_eval.py --z variant` gives the knobs and preset, the counts unknown | `clone-8`; the eval with `--z variant` |
| C1 | Not applicable: the sim and the scripts have no queue (an order replaces the last), so a queued head would have nothing to learn.  Actions a decision were measured in 11.20 | |
| C4 | `--economy-weight w`: BUILD, ASSIST and RECLAIM count w times in the loss, which is divided by the number of actions as before | `clone-8` (2.5) |
| C6 | `SimEnv.policy_economy_share`: whether the policy runs its own economy is drawn for each game at its start (the economy script then only advises); `--economy-start` and `--economy-end` in PPO | the evals run it in every game |
| C7 | `--income-weight`: a zero-sum term in the shaped reward, the change in (a - b) / (a + b + 5) of the two sides' metal income (`INCOME_CONSTANT` 5, as the value balance's) | PPO smoke only |
| C8 | `policy_clone_eval.py`: wins, draws and losses, and each side's metal income, finished extractors and value at minutes 3, 6 and 10 of each slot's first game | the evals below |

**P4**, `clone-7` at step 1,000 on its 96 held-out stretches:

| | NLL of actions | Anchor right | Kind right | Stops early | NLL fight | NLL move |
|---|---|---|---|---|---|---|
| Normal | 2.840 | 0.856 | 0.938 | 0.114 | 4.44 | 3.64 |
| Blind | 3.122 | 0.840 | 0.924 | 0.141 | 5.02 | 4.35 |

**The runs**, each 1,000 steps on `army-6` with `clone-7`'s command (11.21), held out at the end:

| | `clone-7` | `clone-8` | `clone-7s` |
|---|---|---|---|
| Flags beyond the baseline's | | `--z --z-dropout 0.2 --economy-weight 2.5 --finish-winners 0.1` | `--size small` |
| Parameters | 10,537,808 | 10,546,256 (`z_in` 8,448) | 1,928,792 |
| NLL of actions | **2.840** | 2.857 | 3.080 |
| Anchor right | 0.856 | **0.860** | 0.843 |
| Kind right | 0.938 | **0.942** | 0.925 |
| Done right | **0.958** | 0.945 | 0.954 |
| Stops early | 0.114 | **0.108** | 0.126 |
| Ordered BCE | **0.148** | 0.149 | 0.156 |
| NLL build / reclaim | 7.81 / 3.80 | **7.56 / 3.56** | 8.05 / 3.99 |
| NLL fight / move | **4.44** / 3.64 | 4.67 / **3.62** | 4.89 / 4.22 |
| Builds right / reclaims right | 0.809 / 0.567 | **0.829 / 0.588** | 0.776 / 0.509 |
| Decisions a second | 1,326 | 1,439 | 620 |

`clone-8` is even with the baseline overall: the economy's weight did what it was for (builds, reclaims, stopping early), and fights and done paid for it.  The check gives `clone-8` its z; told nothing of it, `clone-8` scores 2.886 (anchor right 0.856, kind right 0.939, done right 0.941), so the strategy is worth 0.03 nats here, and without it `clone-8` is 0.05 behind the baseline.  `clone-7s` is worse on every measure, by 0.24 nats on actions and most on moves: width matters here, as Unplugged found.  The gap widened through the run (0.17, 0.19, 0.23 and 0.24 nats at steps 250 to 1,000, the same schedule for both), and the small actor's anchor right stopped at 0.843 from step 750 while the 10M's still rose: the small one is saturating on 3.1M looks.  So its playing as well in closed loop (below) says the actor is limited by something else first there, not that width doesn't matter; as cloning and PPO feed it more data the small one will saturate (the user).  Its 620 decisions a second aren't comparable: the parity runs and tests of featurize's port shared the CPU with it.  Starting a run now takes 7 s, not 3 minutes: `sides()` keeps each file's side and frame counts in the dataset's `sides.npz`.

**The cloned actor in closed loop** (`policy_clone_eval.py`): the actor plays side 0, the army script side 1, 64 games on the pool's training maps (as the data was recorded), both army scripts drawing variants, 12 workers; each slot's first game counts.  The sim has to step a slot at a time with the actor deciding every 6th, as the data was recorded.  Stepping the sim at the actor's rate (`SimEnv(decide_every=6)`) instead changes what happens between decisions, and on a 4-game smoke run `clone-7` fell from 2.75 extractors at minute 3 to 1.0 at minute 10; stepping a slot at a time, it held 4.75, 6.0 and 6.0 (the script 5.5, 12.75 and 14.5).  `policy_train.py --fine-sim` does the same for PPO.

| Eval | Wins / draws / losses | Extractors at 3 / 6 / 10, actor (script) | Income at 10, actor (script) | Value at 10, actor (script) |
|---|---|---|---|---|
| `clone-7`, T 1.0 | 3% / 17% / 80% | 4.0 / 4.2 / 5.7 (5.0 / 9.3 / 17.3) | 24.6 (55.2) | 10,146 (15,073) |
| `clone-8`, T 1.0 | 2% / 19% / 80% | 4.1 / 3.8 / 5.1 (5.0 / 9.8 / 18.3) | 24.2 (56.6) | 9,856 (15,730) |
| `clone-7`, T 0.8 | 2% / 20% / 78% | 4.1 / 4.3 / 5.2 (5.1 / 9.2 / 15.4) | 24.8 (50.7) | 9,716 (14,071) |
| `clone-8`, T 0.8 | 3% / 19% / 78% | 4.0 / 4.5 / 5.6 (4.9 / 9.4 / 17.0) | 25.3 (55.4) | 10,263 (15,172) |
| `clone-8`, T 1.0, z from the variant | 5% / 19% / 77% | 4.0 / 4.7 / 6.2 (5.0 / 9.9 / 18.5) | 26.6 (59.4) | 11,041 (15,212) |
| `clone-7s`, T 1.0 | 5% / 16% / 80% | 3.9 / 3.8 / 5.0 (5.0 / 9.6 / 18.0) | 22.0 (57.3) | 10,107 (15,589) |

Each eval took 4.5-6 minutes for ~25,650 decisions; 41 to 46 of the 64 games reached minute 10.  No condition differs from another: 1 to 3 wins in 64, losses 77-80%, and the same economy.  The cloned actor keeps up with the script for the first 3 minutes, then stops expanding: by minute 10 it holds 5-6 extractors to the script's 15-18, and less than half its income.  Neither a lower temperature (P6), the economy's weight (C4), z from the variant (P8; the best economy of the six, within the noise) nor a fifth of the width (P5) moves it, though held out they differ by up to 0.24 nats.  So what holds the actor back in play is drift into states the script's games never show, which is what DAgger (P3) labels.

**PPO from a clone, smoke run** (16 games, 3 iterations of 512 decisions, `--init` from `clone-7`, the flags of the table's P7 row): at a learning rate of 1e-4 the actor drifted from the clone too far to keep; at 1e-5 the KL to the clone went from 0.003 to 0.023, PPO's own KL per iteration stayed at 0.005-0.008, and 10% of ratios clipped.  A token budget of 165,000 a pass ran out of the card's 85%; 40,000 fits.

**featurize's loops in numba** (`featurize_kernels.py`), a cheaper fix than 11.21's C++ featurize: selection and its sorts, the unit rows (instance, order memory, rays, DPS around), the regions' live columns and the masks, each one loop over units in place of numpy's many small calls.  Sites, features, summaries and the global row stay numpy.  The tensors are the same to the bit, not merely close: float32 arithmetic stays float32 (every constant a float32, since a Python float would widen numba's arithmetic to float64 where numpy's stays float32), a mean is numpy's own pairwise sum, and what numpy summed with bincount is summed in float64 in the same order.  hypot, atan2 and log1p came out equal on this CPU (no AVX-512, so numpy's float32 functions are libm's, as numba's are).  Checked on 47,000 frames of `army-6` (60 + 80 games, both sides, every 4th to 8th frame; a third of the second set tiled 4x with new uids, so 2,465 frames went past the caps and 1,068 had summaries): every field, dtype and shape equal, the critic's hidden tokens too, and the golden tests pass.  Without numba the same code runs in plain Python, equal and ~50x slower.

| | numpy | numba |
|---|---|---|
| featurize, a frame (mean 134 units) | 1.59 ms | 0.40 ms |
| a cloning worker's decision (decode, featurize, labels) | 2.85 ms | 1.54 ms |
| first call | | 13 s to compile, 0.4 s from numba's cache |

Both timed with a cloning run on the GPU beside them.  Decoding is now a worker's largest part (0.67 ms a decision), then labels (`actions.encode`, `to_slots`, ~0.2 ms).  numba's cache knows only the file it compiled, not the modules whose numbers it froze in, so the kernels hold rawframe's numbers as literals checked on import, and take the type stats' width from the array.  A recursive function loaded from that cache crashed (a segfault on the first call), so numpy's pairwise sum runs on a stack of its own.

**Tests:** `sim/tests` and `tests`, 668 passed (3.4 minutes).  New: z reads as nothing until it's learned, and then as the strategy; the scripts only advise a label seat, and the economy script only where the policy runs its economy; z's layout and dropout; the economy's weights; HL-Gauss; the kernels' mean is numpy's to the bit; a copied bundle with new arrays is packed anew.

### 11.23 A whole run in one command, z in PPO, and an unknown z as cloning showed it (2026-11-10)

The user: `clone-8` told its variant is the best of 11.22's evals (5 wins in 64, where the others had 1 to 3, and the best economy); write a script for a whole run, with the option of training the encoder first, then 10,000 steps of cloning, DAgger rounds and PPO.  No testing for now.

**`tools/policy_pipeline.py`** runs these stages in order, each with settings that were already used:

| Stage | What | Settings from |
|---|---|---|
| `frames` | with `--train-encoder`, if `--frames` (by default `ROOT/frames`) has no `manifest.json`: `frame_data.py collect`, 1M frames each of the frame games, scenarios and the C++ scripts' games | `pool5` (11.17) |
| `encoder` | with `--train-encoder`: `frame_vae.py train`, 25,000 steps, to `ROOT/encoder`; every later stage reads its `vae-last.pt` (otherwise `--encoder`, by default `policy_latent.ENCODER`) | `encoder-big` (11.17) |
| `zindex` | the data's `z.npz`, if it has none (`army-6` has one) | |
| `clone` | `policy_clone_train.py` on `--data` (`army-6`), 10,000 steps (one cosine schedule), held out and saved every 500 | `clone-8` (11.22): `--z --z-dropout 0.2 --economy-weight 2.5 --finish-winners 0.1`, lr 6e-4, Muon 0.008 |
| `record-N` | the last clone plays the army script with `--z variant`, the scripts advising it, until 256 of its games are kept (`--dagger-games`).  Its first 64 are its closed-loop eval, in `evals/record-N.json` | P3 |
| `zindex-N` | `z.npz` over every round's games | |
| `dagger-N` | 500 steps from the last clone, half of its samples from every round's games so far and half from `army-6`; lr 2e-4, Muon 0.003 | P3 |
| `eval` | the last clone, `--z variant`, 64 games: `evals/final.json` | C8 |
| `ppo` | `policy_train.py --init` the last clone, `--z variant`, 64 games, 4,096 decisions an iteration, 2M decisions (`--ppo-hours` caps it in time), the critic alone for its first 5 iterations | the smoke run (11.22): lr 1e-5 to 3e-6, KL 0.5 to 0.1, cloning loss 0.5, dual clip 3, HL-Gauss 51 bins, economy share 0.25 to 1, income 0.5, token budget 40,000, not concurrent |

`--dagger-rounds` is 2 by default.  `--clone-args`, `--dagger-args` and `--ppo-args` go at the end of those commands, so they override what's set.

**How it runs.**
* **One stage at a time.** Each stage is the tool's own command, in a process of its own, run one after another: no two load the map pool at once, and each hands back the GPU when it ends.
* **Logs and state.** Output goes to `ROOT/logs/STAGE.out`, and `ROOT/pipeline.json` lists the stages done with each one's time and exit code.
* **Carrying on.** The same command again carries on from the first stage not done.  `--redo STAGE` runs that stage and every one after it again.  `--dry-run` prints the commands.
* **Restarts.** The encoder and PPO resume from their own checkpoints.  A cloning stage starts over, setting its old `log.jsonl` aside.  A round recorded again first deletes its own games.
* **Stopping.** A `stop` file in `ROOT` halts it once the running stage ends.  One in `ROOT/encoder` or `ROOT/ppo` stops that training at its next report, and the pipeline with it, the stage not marked done.
* **DAgger aggregates.** Every round records into `ROOT/dagger` under a prefix of its own (`policy_clone_eval.py --record-prefix game-daggerN`), so each fine-tuning draws from all the rounds' games.

**Times, from earlier runs, not measured on this script.**
* **Cloning:** at `clone-7`'s rate (1,000 steps in 44 minutes) the 10,000 steps take about 7 hours.  The numba featurize may make that less; that isn't measured.
* **A DAgger round:** about 45 minutes.  An eval of 64 games took 5 minutes, so recording 256 games takes about 20, and 500 steps of fine-tuning about 22.
* **The encoder:** `encoder-big` trained in 82 minutes, after the collection.
* **PPO:** its rate in the fine sim isn't known yet.

**The encoder's retrain** is meant for the ids' new sizes (64 players, 8 allyteams).  That change to `featurize.MAX_PLAYERS` and `MAX_ALLY_TEAMS` isn't made here, so make it before a run with `--train-encoder`.  Frames are stored featurized, ids included, so `frames/pool5` can't serve after it; let the pipeline collect new frames.

**An unknown z.**
* **The bug.** Cloning with `--z` gives every sample a z, all zeros where it's dropped, and `z_in` adds its learnt bias to those as to any z.  But play with `--z none` and the held-out "z unknown" measure passed no z at all (None), which skipped `z_in` altogether.  That was a state the z-trained actor had never been shown.
* **The fix.** None now adds `z_in`'s bias, the same as zeros.  An actor trained without z has a zero bias, so it plays as before.
* **What it touches.** These were measured before the fix, so not as cloning meant them: `clone-8`'s z-unknown NLL (2.886 in 11.22) and its evals at T 1.0 and 0.8 without z.  The z-variant eval and every held-out measure with z are unaffected.  None of them was run again.

**z in PPO.** `policy_train.py --z variant` (it needs `--variants`) works like `policy_clone_eval.py --z variant`.
* **In play.** Each acting seat is told its side's knobs and preset, the counts unknown (`seat_z`).
* **In the update.** Each step keeps its z, and the update reads each segment's z; so does the frozen clone for the KL.  A segment is one game's, so its first step's z is all of its steps'.

**Also:** `frame_vae.py train --gpu-memory`, 0.85 by default, as the other trainers have.

**Checked:** the changed files compile, and `--dry-run` printed the stages' commands as set out above.  The flags were checked against each tool's own.  No tests were run and nothing was trained: the user left testing for now.
