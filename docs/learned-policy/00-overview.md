# Learned policy: overview and shared contract

A single network plays Zero-K through the existing bot client, trained by self-play in
a simplified simulator, following StarCraft's Pluto (without its per-opponent opening
bandit).  Status: stage 1 is built and Gate 1 was waived by the user on 2026-11-05 (below); Phase B's plumbing is built (below).  Nothing has been trained on the sim yet.

**2026-11-08: the rules bot was removed** at the user's request, with ML stages 1 to 4 (`fightnet`, the value net, the
observer, `tune.py`), `prun.py` and its reports.  `bots/zkbot` is now only the policy's host (`host.py`).  These documents
are a record, so their references to that code (`brain.py`, `military.py`, `fightnet.py`, `prun.py`, line numbers) are left
as they were: they point into the git tag `rules-bot-final`.  Superseded parts of the policy were removed the same day: the
first actor over every token and its numpy forward, the toy, the JEPA frame model and the raider-only fight script.  Current
state: `HANDOFF.md`.

* [01-simulator.md](01-simulator.md): the simplified environment
* [02-policy.md](02-policy.md): the network, its observations and actions
* [03-training.md](03-training.md): the training pipeline, evaluation and rollout

**This file wins.**  The three plans were written in parallel; where one disagrees with this
file, this file is right and the plan is out of date.

## Why

Rule changes stopped moving the win rate: 2/18 with and without the latest flags, on
TitanDuel 2.2 against CircuitAI Hard at 3x (`results/rev5-*`, `results/rev6-*`, HANDOFF
section 6e).  Pluto's approach: one policy network, pure self-play PPO-style actor-critic,
no scripted play, zero-sum shaped reward plus win/loss, CPU inference in play
(https://davechurchill.ca/starcraft/cog/results/2026/PLUTO_README.txt).  It won CoG 2026, but
its training compute and environment are not published, so we can't tell how much training
this takes.  The closest low-budget precedent is Gym-µRTS: 300M steps, about 60 hours, one GPU,
one small map (arXiv 2105.13807).

## Is this worth doing?

Each stage has a stop rule, and stage 2 is useful alone: a learned army controller on the
rules economy.  The largest risks, in order:

1. **The budget may be too small for the full game.**  The estimate is 0.5-2B decisions, which
   is 10-40 days at the planning rate, with no published figure to check it against.
2. **The policy may learn to exploit the sim.**  Stage 2's engine gate is the first real test.
3. **PPO may not cope with this action space.**  The first 1M decisions show whether heads collapse.

## Shared decisions

| Topic | Decision |
|---|---|
| Timing | Engine 30 frames/s.  Sim tick 5 frames (6 Hz).  Policy decision every 3 ticks: 15 frames, 2 Hz.  Live bot pins `block_n_frames` to 5 in policy mode. |
| Sim language | C++17 built with CMake in WSL, C ABI loaded with ctypes, 16 workers each owning whole games.  Rust isn't installed; pure Python is too slow. |
| Training | PyTorch 2.15 in WSL (`~/zkml`, RTX 4070 12 GB, 16 threads, 40 GB RAM cap), repo at `/mnt/c/Users/Thoma/Documents/sources/zk-ArenaAI`. |
| Budget | Two training passes per collected decision (`K=2`), **about 50M decisions/day** at 70% of the GPU, on 02's measured 1,700 samples/s.  100M/day would need one pass.  Measure K=1 and K=4 at T-M2, on the real model. |
| Network | v1 actor d=192, 4 layers, 2.71M parameters; separate critic 0.44M.  Sized from measured rates (02 section 3.5).  Pluto's 315M is about 116x larger. |
| Live play | Numpy forward in the bot, no PyTorch.  Targets: mean at most 35 ms per decision, p99 at most 60 ms (`policy.budget_ms`), and the game holds at least 2.7x when run at 3x.  Measured 30 ms for the encoder at 243 tokens; the whole forward is measured in shadow mode (P-M2).  If over: ONNX Runtime, then d=128.  A GPU server is a last resort. |
| Raw frame | Sim and live bot produce the same raw frame.  **Owner: `bots/zkbot/policy/rawframe.py`, fields in 02 section 2.2.**  It holds only what the live bot can really supply, so no facing, reload or acks.  The C++ header is generated from it, and a test compares the two. |
| Featurizer | One numpy module, `bots/zkbot/policy/featurize.py`, in both paths, with golden tests on recorded frames.  Masks are computed there too. |
| Fog | The policy sees what a player sees: visible units, radar blips, its own state.  A blip's type is known only if the unit is in LOS, or was seen and has stayed in radar contact since (`rts/ExternalAI/AICallback.cpp:593-597`).  The critic may see everything in training. |
| Units in the network | One token per unit: type stats from `catalog`, plus an exact name vocabulary with an unknown row and 30% dropout in training, plus instance state.  No per-name logic. |
| Terrain in the network | One token per `terrain.py` region, connected along gates.  Exact unit positions and 24 ray values keep fine detail. |
| Actions | The 16 kinds in 02 section 4.1, with KEEP as the default that emits nothing.  One output per controllable unit, at most 256 (the rest stay with rules at stage 2).  Commit lock of 3 s and "idle units can't KEEP" are masks.  Stage 2 enables kinds 0-9; factories and construction are stage 3. |
| Weapons | No projectile simulation.  Beam-type weapons hit at once.  Others schedule damage for the flight time, applied if the target is still within the blast then, so dodging still exists. |
| Movement | Spheres, no terraform.  A ground `move` walks straight when the way is clear, as Zero-K's `cmd_raw_move.lua` does, and otherwise follows gates and segment centres from a table made when the map is exported, with no fields and no pathfinding at run time (01 section 8.6). |
| Re-ordering | Its speed cost is **measured in the engine** (S-M4), not assumed, and then modelled.  Without it the policy would learn that re-ordering is free.  Measured: a plain move order stops a unit until its next slow update (15 frames), and a fight order costs nothing (01 section 8.3). |
| Reward | Zero-sum potential difference of own value only (03 section 1.3), blended into win/loss +1/-1 by `beta`.  Each kill already lowers the victim's value, so there's no separate kill term.  Gamma 1, GAE lambda 0.95. |
| Opponents | Current weights first, as Pluto does; a league only if the triggers in 03 section 4.5 fire.  CircuitAI exists only in the real engine, and is the measure of progress. |
| Rules baseline | Frozen flags `hitrun,mexpatrol,nearbuild,raidfirst,raidprobe,seeturrets,spearhead` (the "before" arm of rev5 and rev6).  Later rule changes don't enter comparisons.  At stage 2 both arms use these flags, and the policy arm adds `policy`. |
| First scope | TitanDuel 2.2, ground factories and static defence.  Air follows, since CircuitAI builds it. |
| Storage | Runs and checkpoints on `D:\zk-learned` (110 GB free).  **C: backs WSL's disk, even though `df` inside WSL shows far more free than C: has** (77 GB on 2026-11-05).  Raw frames stay in RAM for one iteration and are featurized per minibatch; nothing is spilled or archived, apart from a 1% sample kept for analysis. |

## Stages and gates

A stage starts only when the previous gate passed.

| Stage | What | Gate |
|---|---|---|
| 1. Sim fidelity | Combat-only sim; no training. | **Both:** (a) replaying 44,733 logged fights beats out-of-fold `fightnet` and the tracked value baseline (AUC 0.56 at 20 s), with game-block bootstrap CIs above 0 (01 section 5.1); (b) the scripted engine battery is within its thresholds (01 section 5.2). |
| 2. Army control | Self-play with scripted economy and reinforcements, then the network commands the army on the rules economy. | Real engine against CircuitAI Hard, 100 pairs, pre-registered: paired log X15 and log M15, +22% (03 section 6.3).  Win rate reported, not gated. |
| 3. Full game | Economy and construction in sim and policy. | 60% against anchors in sim by 1B decisions, then the live policy without rules beats the rules bot against CircuitAI Hard. A move from 10% to 20% wins needs about 215 pairs. |
| 4. Fine-tuning | Domain randomisation in sim, then continued training in the real engine. | Engine gate at its pre-registered effect; sim-to-engine gap within bound (03 section 6.6). |

**Gate 1 outcome (2026-11-05): waived by the user.**  1(a) failed: advance 0.591 against out-of-fold `fightnet` 0.665 and value 0.564 at 20 s.  1(b) is at 105/113 of its checks.
The user read this as probably unusual terrain and places tanks can't leave, said it's fine so far, and chose to skip the wider engine comparison and go on.
So the sim's fight model is unproven, not validated, and Gate 2 in the real engine is the first test of whether what it teaches transfers.

**Phase B outcome (2026-11-05): plumbing built, one budget missed.**  The toy met its win marks without learning focus fire unless forced to (03 section 10.2), so later gates should use several scripted opponents.  Numbers are in 01 section 8.7, 02 section 13 and 03 section 10.  Shadow mode ran live with no commands sent: forward mean 17 ms, p99 at most 37 ms, game at least 2.74x.  Numpy against torch: worst difference 7e-06 against the 1e-04 bound.  256 games of about 390 units: 0.34 ms per game decision against the 0.5 ms budget.  **The miss: training runs at about 360-400 decisions/s per pass on real frames, not the 1,700/s this plan assumed, so two passes give about 15M decisions a day, not 50M.**  The numbers and options are in 03 section 10.3.  It is the user's call before long runs start.

If gate 1(a) fails while 1(b) passes, or 1(a) passes only on the no-blip, no-memory subset,
stop and review.  The fight label is noisy and hides unseen enemies, so a failure there may
not mean the combat model is wrong; that review is a decision for the user, not a pass.

## Build order

Milestones keep the numbers in their own plan: `S-` for 01, `P-` for 02, `T-` for 03.

| Phase | Milestones | Ends with |
|---|---|---|
| A. Stage 1 | S-M0 data, S-M1 map, S-M2 combat core, S-M3 fight replay, S-M4 engine battery (includes re-order cost and raw move) | Gate 1.  Nothing else starts before this. |
| B. Plumbing | P-M0 schema and recordings, P-M1 featurizer, P-M2 net, export and shadow mode; T-M0 toy PPO; S-M5 binding, S-M6 raw frame and fog; T-M1 binding | Live shadow latency within budget; PPO solves the toy; a sim frame passes the golden test |
| C. Stage 2 | S-M7 orders and scripted baseline; T-M2 sanity (2a); T-M3 and P-M3 army training (2b, 2c); T-M4 and P-M4 live hand-off | Gate 2 |
| D. Stage 3 | S-M8 economy and build; P-M5 economy heads; T-M5 full game | Gate 3 |
| E. Stage 4 | T-M6 randomisation and engine training; S-M9 remaining mechanics, air | Gate 4 |

## Conflicts between the plans, resolved

| Conflict | Resolution |
|---|---|
| Frame schema: 01 had `heading`, `weapon_ready`, `para`, `ack`; 02 found the live bot can't supply them | 02's schema.  `ack` stays inside the sim for diagnostics. |
| Action list: 01 had 7 orders; 02 has 16 kinds | 02's kinds.  S-M7 implements kinds 0-9, including FIGHT, ATTACK_GROUND, MANUAL_FIRE, JUMP and CLOAK. |
| Factories: 03's stage 2c trained them; 02 masks them until stage 3 | Stage 3.  In 2c a scripted production schedule supplies reinforcements. |
| Live budget: 02 said 167 ms, 03 said 35 ms | 35 ms mean is the target, 60 ms p99 the guard, and 2.7x game speed the real test. |
| Live model: 03 proposed d=128 or a GPU server; 02 measured d=192 at 30 ms | d=192 numpy.  P-M2 measures the whole forward, since 30 ms covers the encoder only. |
| Token cap: 03 sized for 600; 02 caps at about 750 | 02's caps; 03's "up to 600 units" per decision is 256. |
| Buffer: 03 stored 64-d tokens (5 GB); 02's tokens are 140-d with N x N buckets and masks | Store raw frames (80 bytes per record, about 1 GB per iteration) and featurize per minibatch.  At 1.3 ms and K=2 that's about 2-3 threads at 50M/day. |
| Featurizer cost: 1.3 ms (02) vs 0.4 ms (03) | 1.3 ms: 03's version had no rays or pair buckets. |
| Module layout: 03's `policy_hook.py` vs 02's `policy/` | 02's layout.  Trajectory capture is `bots/zkbot/policy/trajectory.py`. |
| Live weights: 03's prun arm used `.pt`; 02's bot reads `.npz` | Export to `.npz` first; `ZKBOT_POLICY` names that file. |
| Live stage 2 gate: 02 said 8 games per arm | 8 games is a smoke test; the gate is 100 pairs. |
| Hashed name embedding vs exact vocabulary | Exact: 615 names in 512 buckets must collide, and the hash can't detect unknown names anyway. |
| Flow fields: 01 had an LRU of 64 and no budget for computing them | Superseded by 01 section 8.6: no fields at all.  Move orders use the straight-line check first, then gates and segment centres from a per-class next-hop table. |

## Measured so far (2026-11-04)

Throwaway benchmarks, not in the repo; a guide, not a promise.

| What | Result |
|---|---|
| C++ tick: spatial hash, targeting, blast damage, movement; 400 units, 16 busy threads | 370 µs.  Not a full sim. |
| Actor and critic training, bf16, 243 tokens (02) | 2,560 samples/s at 1.0M; **1,720 at 2.7M (v1)**; 1,500 at 4.8M |
| Plain transformer, no critic or region GRUs, bf16 and compile, 200 tokens (03) | 5,028 samples/s, but compile gave the real model no gain, so it isn't used (03 section 10.3) |
| Inference, v1, 243 tokens, GPU (02) | 7,000 decisions/s |
| Numpy on CPU, v1 encoder, 243 tokens (02) | 30 ms per decision |
| Real engine at 3x, 8 parallel games (03, rev6 timing) | About 1,000 games/day, mean game 658 game-seconds: 2.6-4.4M learnable decisions/day |
| Sim, planned, at 50M decisions/day | About 10,000 20-minute games/day; 11-19x the real engine in decisions |
| Entities seen at a game's busiest point (estimate from state logs, 18 games) | Median about 184, max about 460 |
| Army size (611 rules-bot runs, 10 s samples) | Median 24, p95 107, 3.5% of samples over 128, max 969 |
| Land regions on a map, 111 maps | Median 31, max 120 |
| Roster | 122 factory-built mobile types: 13 builders, 109 others, 81 non-builder ground.  242 Lua gadgets. |
| Fight logs | 46,389 snapshots in 184 games, 44,733 usable (without `terraunit`); median 15 entities |
| Paired log-ratio SD per game (`tuned-ab`, `rp-ab`) | M10 0.25-0.29, M15 0.32-0.51, X10 0.53-0.57, X15 0.51-0.69 |

## Sources

All in `C:\Users\Thoma\Documents\sources` (`~/Documents/sources`): `zk-ArenaAI` (this repo),
`ControllerAI`, `Zero-K`, `RecoilEngine` (CircuitAI under `AI/Skirmish/CircuitAI`).
