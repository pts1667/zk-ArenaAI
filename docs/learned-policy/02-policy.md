# Learned policy: observations, network, actions

**Reconciled with the other plans:** [00-overview.md](00-overview.md) lists what changed and wins any remaining disagreement.

Status: P-M0 to P-M2 built, behind flags `rawframe` and `policyshadow`, and no commands are sent yet (section 13).  Shared decisions are in
[00-overview.md](00-overview.md); this file follows them and lists objections separately.

Citations are repo-relative (`bots/zkbot/...`) unless prefixed `ControllerAI/`, `Zero-K/` or
`RecoilEngine/` (siblings under `C:\Users\Thoma\Documents\sources`).  Numbers marked
**measured** come from throwaway scripts in `C:\Users\Thoma\.claude\jobs\9ba24cc0\tmp\policy_bench\`
(numpy 2.3.5 in miniconda on Windows; PyTorch 2.15 cu132 in WSL on the RTX 4070, bf16 autocast).
Anything not checked is marked **unverified**.

## 0. Decisions for review

1. **Units are tokens with no name logic.**  Type stats come from catalog data, instance state from
   the live observation, and a 16-d hashed name embedding is learned.  Nothing in the featurizer or the
   net branches on a unit name.  Movement class comes from engine move data, not from name prefixes (1.4).
2. **Steady orders come from structure, not penalties.**  Orders are a KEEP default, the controller's own
   order memory is an input, and a 3 s commit lock is an action mask.  An idle unit cannot KEEP.
3. **Inference is numpy, not ONNX or PyTorch.**  Measured 30 ms per decision for the v1 encoder at 243
   tokens, against a budget of about 167 ms wall per decision at 3x.
4. **v1 actor: d=192, 4 layers, 2.7M parameters, plus a 0.44M critic.**  Measured training throughput
   does not support 100M decisions/day with two passes (3.5).  The target is 50M/day with two passes, or
   100M/day with one.  This is an objection to the contract.
5. **Stage 2 hand-off is one call in `military_step` plus one skip in its loop.**  The policy takes up to
   256 ground units.  Air, phantoms, Gremlins, the commander, factories and any units past the cap keep
   the rules.  With `ZKBOT_FLAGS` lacking `policy`, the path is unchanged.

## 1. What the live bot can give the policy

### 1.1 The observation

ControllerAI publishes, per unit, `defId`, `name`, `allyTeam`, `pos`, `vel`, `health`, `maxHealth`,
`experience`, `buildProgress`, `isBeingBuilt`, `isCloaked`, `isParalyzed`
(`ControllerAI/src/ControllerAISerialization.cpp:448-475`).  The three sections are `units` (own),
`enemies` (`GetEnemyUnits`, `inLos: true`) and `radarBlips` (`GetEnemyUnitsInRadarAndLos`, minus
those in LOS; `ControllerAI/src/ControllerAICommands.cpp:42-86`).  A blip carries `pos`, `vel`,
`inLos: false`, `isRadarBlip`, and `defId` only when the def is known
(`ControllerAISerialization.cpp:477-492`).  Every engine event is buffered per observation
(`ControllerAI/src/ControllerAIEvents.cpp:30-41`), including UNIT_IDLE, topic 7
(`RecoilEngine/rts/ExternalAI/Interface/AISEvents.h:38`).  The bot drops `vel`, `isCloaked` and
`isParalyzed` today: `make_unit` keeps health, max health, build progress, being-built and experience
only (`bots/zkbot/intel.py:28-53`).  Enemy velocity is kept separately in `enemy_vel`
(`bots/zkbot/blips.py:86-89`).

| Policy field | Live source | Sim must supply | Note |
|---|---|---|---|
| def_id | `units/enemies/radarBlips[].defId` | yes; -1 for blips not identified | blip rule is the engine's, see unverified |
| pos x,y,z | `pos` | yes | y minus ground height is altitude |
| vel x,z | `vel` (all three sections) | yes | elmos per frame; bot uses x30 (`blips.py:88`) |
| hp_frac | `health/maxHealth` | yes; NaN for blips | blips have no health |
| build_frac, being built | `buildProgress`, `isBeingBuilt` | yes | |
| cloaked, paralyzed | `isCloaked`, `isParalyzed` | yes | |
| in LOS | `inLos` | yes | |
| experience | `experience` | yes | |
| idle | UNIT_IDLE events since last decision | must emit topic 7 | `engagements.py:45` already reads topics |
| own order memory (kind, age, target point, target uid) | bot's `unit_order` dict, `(typ, x, z, target, frame)` (`brain.py:144`, `military.py:715-762`) | record the last command applied per unit, with frame | this is what a player remembers, not engine truth |
| blip top speed | `blip_tracks[uid]["smax"]` (`blips.py:91-120`) | same definition over the same 2 s window | |
| remembered enemy structure | `enemy_structures` (`threat.py:91-97`) | keep last-seen static enemy buildings | ghost buildings are visible to a player |
| facing | not in observation | omit | |
| reload, stockpile, energy | only `unit_details` / `unit_queue_state` queries | omit | a per-unit query at 2 Hz is too costly (unverified timing) |
| current queue | `unit_queue_state` query only | omit; use order memory | `ControllerAI/docs/api/queries.md` |
| pending projectiles, weapon fire | not observable | omit | do not expose them in sim either |
| hidden enemy units | never | critic only | |

### 1.2 Commands and timing

Commands queue and run at the next engine update (`ControllerAI/docs/api/commands.md`).  The bot's
`block_n_frames` is 6 in fights and 10 otherwise (`bots/zkbot/brain.py:179-184`,
`bots/zkbot/intel.py:240-267`).  Observations therefore come every 6 or 10 frames.  The military gate
is `frame - last_military_frame >= 15` (`brain.py:370`).  So the bot decides every 18 frames under
block 6, or 20 under block 10, not 15 (derived from these rates, not measured in a game).  Policy mode must set `block_n_frames` to 5 so that frames
divisible by 15 are observed (objection 2).

### 1.3 Radar blips

A blip has `vel` and `pos`, and a `defId` only if the unit is in LOS, or was seen and has stayed in radar contact
since (`RecoilEngine/rts/ExternalAI/AICallback.cpp:593-597`; assumes `springai::Unit::GetDef` reaches this call).  The featurizer sees the blip's observed top speed
(sustained over 2 s, as `track_blip` computes), its altitude, and a `TYPE_KNOWN` flag.  Type stats are
zero for an unknown blip.  The bot's guessed type (`blips.py:180`) is a rules output and is not
used.  Blip position accuracy is unverified.  The sim applies the same identity rule.

### 1.4 Movement class

The bot's own helper maps class from name prefixes (`bots/zkbot/hazards.py:19-32`: `spider`, `tank`,
`veh`, `hover`, `amph`, `ship`, `sub`), which the user rules out.  The dump lacks move data.  ControllerAI
serialises `moveData` (`ControllerAI/src/ControllerAISerialization.cpp:105-125`, attached at line
364): `maxSlope`, `depth`, `slopeMod`, `speedModClass`, `terrainClass`, `xSize`, `zSize`.
`tools/dump_defs.py` requests full defs, so the trimmed `bots/zkbot/data/units.json` (no `moveData`
key) can carry them.  The policy's movement class is then the engine's `speedModClass` and
`terrainClass`, mapped to the six `terrain.MOVE_CLASSES` (`bots/zkbot/terrain.py:109-116`).  Whether
`speedModClass` alone separates kbot, tank, spider and amph is unverified.  Lead: the same change
would let `hazards.py` drop its prefix table.

### 1.5 Type roles

Use `catalog.infer_role` and `catalog.tags`, not `units.role_of`, which reads hand tables and
`ROLE_OVERRIDES` by name (`bots/zkbot/units.py:105-118`).  `fightnet.py` already does this
(`fightnet.py:91-125`).

## 2. Raw frame and featurizer

### 2.1 Ownership

`bots/zkbot/policy/rawframe.py` owns the schema: field names, dtypes, enums, caps and the global
field list.  It is the only Python definition, because the live bot and the training data loader both
read it.  The C++ header in the sim (01) is generated from it, and a test compares the field
lists.  The live producer is `rawframe.from_brain(brain)`, a read-only function over the Brain state
listed in 1.1.  The featurizer (`bots/zkbot/policy/featurize.py`) is numpy and is the only code that
turns frames into tensors.  Training runs it in data workers, because it is cheap (2.6).

### 2.2 Raw frame fields

Per decision, one frame:

| Record | Fields | Note |
|---|---|---|
| UNIT (array) | `uid` i8 (pointer identity only), `player` i2 and `ally_team` i2 (the game's numbers, -1 unknown; `player` is the engine's team), `knowledge` u1 (0 in LOS, 1 radar, 2 remembered, 3 hidden, critic only), `def_id` i4 (-1 unknown), `pos` f4x3, `vel` f4x2, `hp_frac` f4 (NaN blip), `build_frac` f4, `flags` u2 (BEING_BUILT 1, CLOAKED 2, PARALYZED 4, IN_LOS 8, IDLE 16, TYPE_KNOWN 32), `experience` f4, `blip_top_speed` f4, `ord_kind` i1 (-1 none), `ord_age` f4 (frames), `ord_x,ord_z` f4, `ord_target` i8 (uid or -1) | order fields count only on the viewer's own units |
| SITE (array) | `site_id` i4, `x,z` f4, `income` f4, `status` u1 (free, ours, enemy, building), `region_id` i2 | from `map.spots` and `update_spot_status` (`intel.py:211`) |
| FEATURE (array) | `feat_id` i8, `x,z` f4, `metal` f4, `def_id` i4 | reclaimable wrecks in sight with at least 1 metal, with what's left of them: observation `features` (`ControllerAI/docs/api/state.md`), and in the sim the same by line of sight (01 section 8.10).  In frames since schema 3 |
| GLOBAL (vector) | named by `GLOBAL_FIELDS`: frame, metal and energy current/income/storage, commander hp, commander uid, base x,z, map w,h, frames since last enemy contact, count of unfinished own structures, count of idle own builders, the viewer's own `player` and `ally_team` | the last two aren't net inputs: they say who the frame belongs to |
| EVENTS | `idle_uids` (own units with UNIT_IDLE since the last decision) | |
| STATIC BUNDLE (once per game) | region records (centre, area, height, domain, tags, access per class, spots); gate records per edge (width per class, walking length, choke, ramp and shore tags); spot list; 16-elmo passability for kbot; ground heights | built by `terrain.TerrainMap` in both paths |

The static bundle must come from the same heights in both paths.  Water maps move the terrain after
load (`HANDOFF.md` 6e), so the live bundle must read heights after the water has been set.  Otherwise
the region graph differs between sim and live (unverified which maps are affected).

### 2.3 Featurizer outputs

| Tensor | Shape | Content |
|---|---|---|
| `unit_x` | U x 140 | type stats (below), instance state, order-memory block, 24 ray values, name id |
| `reg_x` | R x 40 | centre, area, height, domain, tags, access per class, metal spots and income, ramp and choke counts, value this decision of own alliance, other alliances in LOS and on radar |
| `site_x` | S x 12 | income, status one-hot, region centroid distance |
| `glob_x` | 40 | GLOBAL fields, normalised |
| `unit_ally`, `unit_player` | U each | ids as the viewer has them, each through a learned embedding: own is 0, others count from 1 skipping own, and an unknown id or one past the table shares the last row (16 alliances, 32 players).  Both seats of a 1v1 see the same numbers |
| `tok_kind` | N | what is known, not whose: 0 seen, 1 blip, 2 remembered, 3 summary, 4 region, 5 site, 6 global, 7 feature (stage 3), 8 hidden (critic), 9 memory state |
| `enemy_tokens` | E | the unit tokens that can be aimed at: any other alliance's, in LOS or on radar |
| `pair_bucket` | N x N int8 | unit-unit: log2 distance (12) x hostile, same alliance or same player (3), and in place of those, a unit and what it's ordered to shoot, each way (2); unit-region: same region or hop 1-6+ (8); region-region: hop 0-6+ (7); all else one bucket per token-kind pair |
| `feat_x` | F x 5 | a wreck or the like in sight: x, z, log1p metal / 8, distance to base, distance to army (both over map size).  F is at most 32.  Its pair buckets are 12 log2 distances to each unit, and one for each other group.  Region column 33 is log1p of all the metal in sight in that region / 10, so what the cap leaves out still shows |
| `kind_mask` | U x 16 | legal action kinds, see 4.1 and 4.4 |
| `ptr_*_mask` | U x E, U x U, U x R, U x S, U x T | legal pointer targets; T is the candidate type list |
| `route_bias` | U x R | walking cost from the unit's region to each region; class-specific, a per-game region-to-region table on the gate graph (3.3) |

Normalisation is fixed, not learned, so it does not drift with training: positions over map size
(to -1..1), velocities over the unit's own catalog speed (clipped to +-2), log1p for costs, hp and
dps over constants, log for times.  Stats are read from catalog data, not from the live game.

Order memory is what the bot last told a unit, on own units only, not the engine's command queue: the kind (16, one-hot), its age, the direction and distance to
its goal, whether it has a target, whether that target is in sight, and whether the commit lock holds.  For an order with a target, the goal is where the target is now, in LOS or
on radar; if it's in neither, those columns are zero.  The pair buckets mark which token it is, so a target's token can count who is shooting it.  A target out of sight reads
the same as a dead one and releases the lock.

Type stats (catalog, `bots/zkbot/catalog.py`): log cost, log hp, log dps ground and air, range, max
speed, aoe, flags (air, mobile, builder, factory, defence, unarmed), build speed and range, footprint,
the 13 `infer_role` roles one-hot, the 6 move classes one-hot, and the ability tags one-hot from
`abilities.json` (`catalog.tags`, 22 tags).  About 60 values.

Rays (spatial detail, see 3.2): for each unit, 8 sectors of 45 degrees, each giving the free distance
to the first impassable cell (to 400 elmos), enemy dps within 600 elmos in the sector, and own dps
within 600 elmos.  That is 24 values, computed with numpy on the passability grid.

Name id: a learned 16-d embedding over an exact vocabulary of catalog names plus an unknown row (about 615
names, against `fightnet.py`'s 512 hash buckets, which must collide), dropped to the unknown row 30% of the time
in training.  A checkpoint stores the vocabulary and the training catalog hash, and a mismatch logs a warning.

### 2.4 Caps and what happens over them

| Group | Cap | Over the cap |
|---|---|---|
| Own controllable (ground combat, see 5.1) | 256 | Sorted by cost, then distance to the army centre.  The rest keep the rules (stage 2) and join the pool below. |
| All units, of any player or alliance | 592, controllable first | In order: other alliances' units in LOS, then theirs on radar, then everything else (own builders, factories, static and air, allies, remembered structures).  Within each, by max(cost, 100) / (distance to our army + 200).  The rest are summarised. |
| Summary tokens | up to 12 | One per (whose, what's known of it, coarse role) with count, hp sum, dps sum, centroid.  The twelve largest are kept. |
| Regions | all | 31 median, 120 max across 111 maps (`HANDOFF.md` 6e) |
| Sites | 24 | Nearest to the base first |
| Features | 32 | Most metal for the distance from the army, metal / (distance + 200).  Ties by id.  Each region still counts all of its metal |

There is one pool, not one per side, so the same model reads a 1v1, a 5v5, a 3v3v3 or a free-for-all: who a unit belongs to is an id on the unit (2.3), and only the
viewer's own player's units get actions.  Any fight against up to 336 targets keeps them all, however far off they are, at 256 controllable.

Typical N is about 110-220 on recorded games.  Worst case is about 750 (the whole pool on a 120-region map).  Army size from 611 rules-bot runs
(`results/*/metrics-team0.csv`, 100k samples of 10 s, **measured**): median 24, p90 79, p95 107,
p99 243; 3.5% of samples exceed 128.  Per-game peak: median 44, p90 115, max 969 (`results/h127-circmax/game-003`).

The caps were first 128 controllable and 64 enemies, then 256 and 256 in separate pools, then this one pool.  The weights don't depend on them.
Cost with every enemy seen, on a 30-region map (**measured**, numpy, random weights, median of 20; one thread or sixteen made no real difference):

| Units a side | Tokens | Targets | Featurize + forward | Training, 256 games, concurrent, one pass |
|---|---|---|---|---|
| 40 | 135 | 40 | 10 ms | 584-644/s, 7.3 GB |
| 128 | 311 | 128 | 32-34 ms | not run |
| 192 | 439 | 192 | 58-59 ms | not run |
| 256 | 567 | 256 | 81 ms | 58-61/s, 8.0 GB |
| 400 | 649 | 336 | 99-101 ms | not run |

GPU memory is peak allocated, against a limit of 10.2 GB.  On recorded games the same pass takes 8-22 ms at 110-221 tokens.

An earlier run on a 76-region map gave 46 ms at 485 tokens against the 58-59 ms here at 439.  Part of that may be noise: in one later run the same decision took 44 ms at best and 50 ms at the median.  It isn't explained, so take the table above.

So the 60 ms `policy.budget_ms` (5.3) falls at about 192 a side, with everything in view, and five decisions over it in a row lower the caps for 30 s (5.3).  P-M2 has to measure how often that happens in real games.

### 2.5 Generalisation to unseen or rebalanced units

A new unit enters through its stats, its tags and its role.  The name embedding is unknown for it and
the net must not rely on it: 30% dropout trains this.  A rebalanced unit changes its stats and gets
the same treatment.  Behaviour learned on old stats may not transfer, so the catalog hash check (2.3)
and a retrain are the remedy.

### 2.6 Featurizer cost

Numpy pass with 170 units, 48 regions and 24 sites, including rays and pair buckets: 1.3 ms
(**measured**, synthetic arrays, `policy_bench/np_bench.py feat`).  At 100M decisions/day this is about
1.5 core-seconds per second, so training workers must featurize on the fly and not store tensors.
The raw frames need to stay in RAM only for the update's epochs (one or two, on-policy).

## 3. Network

### 3.1 Architecture (v1)

| Block | Size | Notes |
|---|---|---|
| Token inputs | unit MLP 140 to 192 to 192 (GELU); region 40 to 192; site 12 to 192; global 40 to 192 | plus token-kind embedding |
| Gate message layer | 1 layer over regions, gate features from the static bundle (width per class, length, choke, ramp, shore) | h_r += sum over neighbours n of W_gate(gate features) * h_n; this is the literal "connected along gates" |
| Encoder | 4 pre-LN blocks, d=192, 4 heads, FFN 768 | attention bias per layer and head = RBF(pair_bucket) @ W, computed once per decision for all layers |
| Recurrent memory | memory token from the previous global state; global GRU 192 to 384; per-region GRU 192 to 32 fed back into region tokens | 3.4 |
| Action heads | kind: 192 to 16 (masked); pointer queries: 5 linears, 192 to 192, scored against key sets; fine position: 9-way from (unit, region); route bias added to the region pointer logits | 4 |
| Critic | separate encoder, d=128, 2 blocks, over the privileged frame, plus the actor's GRU state; value head | 3.6 |

Measured actor parameters: 2.71M, including the GRU and heads.  The name embedding (8K) and the region
GRU (about 25K) are extra.  Critic: 0.44M.

### 3.2 Keeping spatial detail

Region tokens lose fine position, so fine detail comes in through three channels.  (a) Each unit token
carries its exact `pos` and `vel`, and the 24 ray values, which give free distance to walls and the
enemy and own dps in each direction.  Kiting and retreat room are visible this way.  (b) The pair
bucket gives attention a log-distance bias between every pair of units and an in-region or hop bias
between units and regions, so the net can learn "range-sized neighbours" without absolute coordinates.
(c) Fine move targets are a 9-way choice inside the chosen region (section 4.2), so the net decides
position relative to the enemy, not in absolute map coordinates.

Blast avoidance is only partly observable: the net sees enemy dps and aoe in each ray sector.
Pending shells are not observable live, so the sim does not expose them.

### 3.3 Region graph

Regions are tokens.  Adjacency enters in three places: the gate message layer (one step, using gate
width, length and tags), the hop-count attention bias (union of gates with width > 0 over all
classes; per-class reachability is applied to the masks only), and the walking-cost bias on the region
pointer (`route_bias`: a region-to-region walking-cost table per class, built once per game from the
gate graph; `TerrainMap.route`, `bots/zkbot/terrain.py:1352`, is the reference for one pair).  The bias
is a learned scale on walking seconds, so near regions are preferred until the policy learns otherwise.

### 3.4 Recurrent memory and fog

Fog memory lives in the network, as Pluto does it.  Two states persist: the global GRU (384) and the
per-region GRU (32 per region).  The memory token and region feedback let every decision read them.
Per-unit memory is not kept, because uids change and dead units disappear; a unit's intentions are in
its order-memory features.  Live, the state arrays are held by the policy object and reset at game start.
In training, a rollout stores the state at the start of each segment (length 32), so PPO unrolls each
segment from its stored state with no burn-in pass.  Burn-in would double the training cost (3.5).

Memory horizon is about the segment length in training (32 decisions, 16 s) plus what the GRU has
learned to keep.  Scouting knowledge older than that is only in the recurrent state, so the 20-minute
requirement is a risk to measure, not something the architecture guarantees.

### 3.5 Size and budget

Measured with `policy_bench/torch_bench2.py` (GPU, bf16 autocast, TF32 on; 170 unit + 48 region + 24 site
+ 1 global = 243 tokens; forward at batch 256, training at batch 128 with actor and critic, Adam):

| Config | Actor / critic params | Forward, decisions/s | Train, samples/s | Numpy CPU, ms per decision (encoder only) |
|---|---|---|---|---|
| d=128, L=3 | 1.02M / 0.44M | 9,700 | 2,560 | not run |
| **d=192, L=4 (v1)** | **2.71M / 0.44M** | **7,000** | **1,720** | **30** |
| d=256, L=4 | 4.80M / 0.44M | 5,500 | 1,500 | 37-44 |

Attention cost, batch 128, 243 tokens, forward and backward (**measured**,
`policy_bench/variant*_bench.py`): no bias 4.3 ms per layer (17 ms for four, extrapolated); a learned
per-pair gather 23 ms for one layer, because its backward is a scatter; a shared RBF matmul computed
once for all four layers 21 ms.  So the shared bias is kept, and the per-layer gather is not used.
The numpy and torch benchmarks used 116 unit inputs, not the 140 in 2.3; the difference is in the first
projection only.

Budget arithmetic (100M decisions/day = 1,160 decisions/s):

* Inference for rollouts at d=192: 1,160 / 7,000 = 17% of the GPU.
* Training, one pass: 1,160 / 1,720 = 67%.  Two passes: 135%, which does not fit.
* So v1 targets 50M decisions/day with two passes (training 67%, rollouts 8%) or 100M/day with one.

Pluto's 315M parameters are about 116 times the v1 actor.  Scaling the measured rate linearly in
parameters gives about 60 decisions/s on this GPU and about 3.5 s per decision on a CPU (extrapolated,
not measured).  Pluto's size is set by a 128x128 tile grid, 6-frame steps and a 4096-d GRU; this design
has 243 tokens and 2 Hz decisions.  The overview's 5M model at d=256 fits 50M decisions/day with two
passes (about 87% of this GPU, measured rates) but not 100M/day with two.  v1 uses d=192 for headroom.

### 3.6 Critic and privileged information

The critic sees the same frame plus hidden enemy units (`knowledge` 3) and enemy state the actor cannot
observe, and the actor's GRU state.  It is a separate network, so no privileged input can reach the
actor's trunk.  Training only; there is no critic at inference.  The value is zero-sum from the
reward (03).  An auxiliary head predicting the enemy's visible-value share is optional.

## 4. Action space

### 4.1 Kinds and their live commands

Every controllable unit has one kind per decision.  KEEP is the default and emits nothing.

| Kind | Target | Live command | Existing code | Mask rule (both paths) |
|---|---|---|---|---|
| 0 KEEP | none | none | n/a | illegal if the unit is IDLE (4.4) |
| 1 MOVE | region pointer + fine 9-way | `move` pos | `military.py:750` `_order` | mobile; region reachable for its class; not commit-locked |
| 2 FIGHT | region pointer + fine 9-way | `fight` pos | `order_fight` (`military.py:715`) | as MOVE, armed |
| 3 ATTACK_UNIT | enemy-token pointer (LOS only) | `attack` targetId | `order_attack` (`military.py:721`) | armed with a weapon that can hit the target's class (catalog `dps_air` / `dps_ground`); target in LOS |
| 4 ATTACK_GROUND | enemy or blip token pointer + fine 9-way at 200 elmos | `custom` cmd 20 | `order_attack_ground` (`military.py:729`) | armed; target in range x1.5 + 300 |
| 5 STOP | none | `stop` | new | not during a committed attack |
| 6 MANUAL_FIRE | enemy-token pointer | `custom` cmd 105 [x,y,z] | `beh_manual_fire` (`behaviours.py:218-260`) | tag `manualfire`; in catalog manual-weapon range; memory of last shot older than the weapon's reload (catalog) |
| 7 JUMP | region pointer + fine 9-way | `custom` cmd 38521, params [x,y,z] (unverified format) | new | tag `jump`; charged; region walkable or within catalog jump range, and one its class can stand on |
| 8 CLOAK_ON / 9 CLOAK_OFF | none | `custom` cmd 37382, state 1 / 0 (unverified; the Off/On state convention is in `Zero-K/LuaUI/Configs/customCmdTypes.lua:40`) | new | tag `cloak`, state from `isCloaked` |
| 10 BUILD | site pointer + type pointer | `build` defId, pos, facing | `construction.py`, `production.py:257` | builder; type in `buildOptions`; site free; unfinished cap (4.3) |
| 11 ASSIST | own unfinished unit pointer | `guard` targetId | `construction.py:300`, `production.py:372` | builder; target unfinished |
| 12 REPAIR | own damaged unit pointer | `repair` targetId | `construction.py:796` | builder; target damaged |
| 13 RECLAIM | feature pointer | `reclaim` featureId | `reclaim.py:100` | builder; feature within reach |
| 14 PRODUCE | type pointer over the factory roster | `build` on the factory uid | `production.py:257` | factory finished; roster from `catalog.roster`; queue below cap |
| 15 reserved | | | | always illegal |

Stage 2 enables kinds 0-9 only; kinds 10-14 are masked off for every unit.  Their weights exist but get
no gradient until stage 3.

### 4.2 Pointers and positions

Pointer heads are scaled dot products between a unit's query and keys of a candidate set, masked
per unit.  Key sets by kind: enemy and blip tokens for kinds 3, 4 and 6; own unit tokens for 11 and 12; region
tokens for 1, 2 and 7; site tokens for 10; type tokens for 10 and 14; feature tokens for 13 (stage 3).
Type keys come from the type MLP on the catalog stats, so a new roster unit is a new key, not a new
output.

Coarse position is the region pointer.  Fine position is 9 options: the centre of the region, or one
of eight compass points at half the region's equivalent radius, in a frame whose north points from our
army to the enemy centroid (or to the map centre when no enemy is known).  The point is snapped to the
nearest passable cell (`TerrainMap.area_of`, `terrain.py:1332`).  The frame makes the choice
rotation-equivariant.

### 4.3 Factories, construction, abilities

Factories (stage 3) pick a roster unit by a type pointer, so the head does not need to know the
roster's names.  The cap on queued units (memory) keeps the queue short.  Construction is the same
pointer scheme over sites from the static bundle.  Assist, repair and reclaim point at tokens.
Ability kinds are three: manual fire, jump, and cloak on or off.  A unit gets them through its tags
(`manualfire`, `jump`, `cloak`).  The other tags (shield, regen, emp, slow and so on) are input features
only in v1.

Jumping is on in training and in the bot's runtime (2026-11-07).  Charge is frame flag `JUMP_READY`
(schema 4), a column of the unit token (`FEATURE_VERSION` 5): the sim sets it for an own unit with a full
charge that isn't already busy jumping, and the live bot reads rules param `jumpReload`.  The region
pointer has its own mask, `Tensors.ptr_jump`, so that a jump can cross a wall or a gap MOVE can't, as long
as the region is in range.  Range comes from `bots/zkbot/data/jumprange.json`, generated with the catalog
and part of its hash.  Weights and frames from before this are invalid.

To stop too many unfinished buildings at once, BUILD is illegal when the number of unfinished own
structures reaches `policy.build_cap` (default 2 + builders / 3).  ASSIST is always legal for an idle
builder that has an unfinished target in reach, so the cheap move is to finish, not to start.

### 4.4 Masks and the two steadiness rules

Masks are computed by `featurize.py` from the frame and the catalog.  Live and sim use the same
function, so the mask cannot differ between them.

* **Commit lock.**  A unit whose last order is younger than `policy.commit_decisions` (default 6, which is
  3 s at 2 Hz) may only KEEP.  The lock ends early when the unit took damage in the last decision, its
  order target died or left LOS, or it is IDLE.  This removes the 1-2 s stop-and-start structurally.
* **Idle units cannot KEEP.**  An idle unit must choose MOVE, FIGHT, ATTACK, STOP (a no-op) or a job.
  This prevents idle constructors and idle army units that the net would otherwise leave.

A unit always has a legal MOVE, since its own region is a legal target, so no unit is left with no
legal kind.  KEEP is also the initial default: the kind logit for KEEP starts at +2, so the untrained
policy changes little.  Targets that match the current order within the `smoothorders` tolerance
(`bots/zkbot/smoothorders.py:37-62`) are not re-emitted, so a changed kind with the same destination
is a no-op on the wire.

### 4.5 Joint log-probability and variance

The policy factorises per unit: the joint log-probability of a decision is the sum over units of
log pi(kind_i) + log pi(target_i | kind_i) + log pi(fine_i | kind_i).  A KEEP unit contributes only the
KEEP term.  For PPO, the joint ratio is exp of a sum over up to 256 units, so a small change in each
unit pushes the joint far outside the clip.  The v1 loss therefore uses a per-unit ratio, clipped
separately, averaged over units in the decision (the MAPPO-style per-agent clip with a shared
advantage).  The entropy bonus is averaged per unit too.

Variance sources, and the controls:

| Source | Control |
|---|---|
| Credit spread over 256 units sharing one advantage | per-decision advantage normalised by batch; loss averaged over units, so the step does not grow with army size |
| Long horizon at 2 Hz | GAE lambda 0.95; value per decision |
| Dog-piling and herding (many units on one region or enemy) | monitored (section 6); shaping only if it appears |
| KEEP-dominated decisions | low-variance by design; most units contribute little gradient |

If herding appears in the logs, the next step is an autoregressive decoder over units in uid order,
where each choice updates the mask for the next (exact chain-rule log-probability, optional in v2).
v1 does not do it, because it runs one sequential pass per unit and the parallel heads are cheaper.

### 4.6 Order steadiness

Four things hold orders steady: KEEP is the default and emits nothing; the order-memory features
(kind, age, destination, target) tell the net how old its order is; the commit lock and idle rule are
masks; and the wire filter drops re-emitted identical orders.  The sim must also model the engine's
cost of a re-order, or the policy learns that re-ordering is free.  S-M4 measures that cost in the engine.

### 4.7 Mapping to ControllerAI

The command for each kind is in the table in 4.1.  Ground attack and manual fire use the custom IDs
already in the bot.  Two IDs are new to the bot, CMD_JUMP 38521 and WANT_CLOAK 37382
(`Zero-K/LuaRules/Configs/customcmds.lua:89-94`).  The emit filter is the one `_order` and
`order_settled` use, keyed on `unit_order`, so orders from the policy and from the rules share one
memory.  Commands go through `self.queue` and `client.send_commands` in the same frame, as now
(`bots/zkbot/main.py:106-109`).

## 5. Live integration

### 5.1 Hand-off in the bot

Hook points, all in `bots/zkbot/`:

| Where | Change | Rules when the flag is off |
|---|---|---|
| `brain.py:306-309` (events read) | accumulate topic 7 into `policy_idle` | unchanged; the set is unused |
| `brain.py:368`, `intel.py:240-267` (block rate) | in policy mode, `block_n_frames` is 5 | unchanged |
| `brain.py:370-374` (military gate) | unchanged call to `military_step` | unchanged |
| `military.py:173-199` (`military_step`) | after `plan_army`, `policy_units = self.policy_step(plan)` (returns set of uids it commanded, empty on failure); in the loop, `if uid in policy_units: continue` | `policy_units` is empty, so the loop is unchanged |
| `brain.py` `on_startup` | construct the policy object if `policy` is in `FLAGS` | none |

Policy-controlled units: army minus air, `gremlin_scouts`, cloaksnipe phantoms, and the commander
(not in the army anyway), in priority order, capped at 256.  Factories are not in the army.  The
policy's own commands use `_order`-like helpers, so `unit_order` is the single memory.

Flags (`bots/zkbot/params.py`): `policy` (stage 2 army), `policyshadow` (build and log, send no commands),
`policyview` (write the spectating log), `policyeco` (stage 3), `rawframe` (record raw frames for golden
tests).  Parameters through `P()`: `policy.max_units` (256), `policy.commit_decisions` (6),
`policy.temperature` (1.0), `policy.budget_ms` (60), `policy.build_cap`.

### 5.2 Inference runtime

| Option | Measured | Dependency | Verdict |
|---|---|---|---|
| numpy forward (fightnet-style, weights in `.npz`) | 30 ms per decision, encoder only, d=192, 243 tokens | numpy, already imported by `fightnet.py` | **Chosen** |
| ONNX Runtime | not measured | onnxruntime in the bot's Python; installed in miniconda, absent in WSL | Fallback if numpy is over budget; export of attention bias and dynamic token counts needs work |
| PyTorch CPU | 9-20 ms for 5.1M at 150/400 tokens (overview) | torch in the bot | Excluded by the contract |

Budget: one decision per 15 frames is 0.5 s of game time, or about 167 ms wall at 3x, but the engine
needs most of that, so the targets are a mean of at most 35 ms, p99 of at most 60 ms, and a game that holds at
least 2.7x (overview).  The 30 ms covers the encoder at 243 tokens only; heads, GRU and larger late-game token
counts come on top, and P-M2 measures them in shadow mode.  Eight games in parallel on 16 cores give each
bot about two cores, so the policy process should set `OMP_NUM_THREADS=2` (the thread effect was not
verified; the two timings above did not change with the setting).  The 60 ms `policy.budget_ms` is a
guard, not a target.

### 5.3 Recurrent state and failures

The state arrays (global 384, region 32 per region) live on the policy object.  They reset at game
start.  A skipped decision (fallback, below) does not advance them, so the state stays a function of
the decisions the net actually made.  A gap of more than three decisions resets the state.  Training
should include such gaps, at about 5% of decisions.

Failure behaviour, in order:

1. Missing or incompatible weights (catalog hash, feature version in `.npz`): policy off for the game,
   logged once.
2. Exception in featurize or forward: logged with traceback, `policy_units` empty for this decision, the
   rules run for everything, and the policy is retried next decision.  `main.py:104-109` already sends
   commands after a frame error, so a policy exception cannot stop the frame loop.
3. Over budget: five consecutive decisions over `policy.budget_ms` lower the caps to 128 controllable and a pool of 192 units for 30 s
   (about 300 tokens at most on a 76-region map, so under 35 ms, section 2.4; not measured at exactly these caps) and log it.  The weights don't depend on the caps, and units past them keep the rules, as past any cap.  If it
   is still over budget at the lower caps, the policy is off for 30 s.
4. Two exceptions in a row: policy off until the next game.

## 6. Testing and spectating

| Test | What it checks | Where |
|---|---|---|
| Golden frame | Featurizer output on 5 frames recorded from live games (`rawframe` flag), against reviewed `expected_*.npz`; tolerance 1e-6 | `tests/policy/test_golden.py` |
| Schema parity | Field list in `rawframe.py` equals the C++ header's; live `from_brain` on recorded observations equals the stored raw frame | `tests/policy/test_schema.py` |
| Mask rules | One hand-made frame per kind row in 4.1; every real unit has at least one legal kind; no pointer mask is all false for a legal kind | `tests/policy/test_masks.py` |
| Shape | Padding, N, cap overflow summaries, token-kind counts | `tests/policy/test_shapes.py` |
| Permutation equivariance | Permuting units (with their rows and pair buckets) permutes unit logits; permuting regions permutes region pointer logits | `tests/policy/test_equivariance.py` |
| Export parity | PyTorch fp32 vs numpy, 100 random frames, max abs difference below 1e-4 for logits and value, and across 10 recurrent steps | `tools/policy_parity.py` |
| Latency | ms per decision in shadow mode in a real game; from the `policy` log line | live, `policyshadow` |
| Behaviour metrics | Orders per unit per minute (the existing `count_order` "soon" counter, `smoothorders.py:72`); idle builders (`brain.py` `builder_watchdog`); unfinished structures; herding share | logs and `tools/policy_view.py` |

Spectating (the user watches games): `policyview` writes `policy-team<N>.jsonl` each decision: frame,
value, mean entropy, kind counts, and for each unit its uid, top kind with probability, target uid or
region and fine option.  `tools/policy_view.py` reads that file and the state log and writes one static
HTML page that replays the 2 Hz snapshots (unit dots by side, order arrows, a value curve, kind counts).
Refresh by re-running it.  A `policy` line every 60 s goes to the team log: frame, value, entropy, kinds,
units controlled, failures.  A game overlay would need a LuaUI widget, which is out of scope.

## 7. Sequencing

| Milestone | Deliverable | Acceptance | Size |
|---|---|---|---|
| M0 Schema and data | `rawframe.py`; `moveData` in the catalog dump; `rawframe` recorder; golden recordings from 3 live games | schema parity test; recorded frames replay | small |
| M1 Featurizer | `featurize.py`, masks, rays, terrain bundle; numpy type and region tables | golden, mask, shape and equivariance tests pass | medium |
| M2 Net and export | numpy forward, torch model, `.npz` export, parity tool; shadow mode in the real game (`policyshadow`) | parity below 1e-4; shadow latency under 60 ms; no commands sent | medium |
| M3 Army training (stage 2) | PPO for kinds 0-9 in the sim (01, 03) with fixed economies; needs the stage 1 gate passed first | learned policy beats the rules army in sim metrics, against a scripted opponent and in self-play | large |
| M4 Live army hand-off | `policy` flag and the hooks in 5.1; block 5 | 8-game smoke test at 3x, at least 2.7x held, then the 100-pair gate (overview gate 2; 03 section 6.3) | medium |
| M5 Economy heads (stage 3) | kinds 10-14, site and type pointers, unfinished cap, factories; `policyeco` | in sim, no idle builders and no unfinished pile-up beyond the cap; live A/B without rules economy beats rules (gate 3) | large |

Stage 2's army-only heads come first.  Economy kinds are masked off in M3 and M4, so their weights
do not train until M5.

## 8. Module layout

| Path | Role |
|---|---|
| `bots/zkbot/policy/__init__.py` | exports `PolicyMixin` |
| `bots/zkbot/policy/rawframe.py` | schema (owner), `from_brain` producer |
| `bots/zkbot/policy/featurize.py` | raw frame to tensors and masks; the one featurizer |
| `bots/zkbot/policy/tables.py` | type-stat table (catalog), name vocabulary, region tables from `terrain.py` |
| `bots/zkbot/policy/net.py` | numpy forward and recurrent state |
| `bots/zkbot/policy/actions.py` | kinds, decode, emit filter, mapping to commands |
| `bots/zkbot/policy/runtime.py` | `PolicyMixin.policy_step`, flags, fallbacks, logs |
| `bots/zkbot/data/policy.npz` | exported weights and metadata (catalog hash, feature version) |
| `tools/policy_train.py` | PyTorch PPO trainer in WSL (03) |
| `tools/policy_export.py`, `tools/policy_parity.py` | export and parity |
| `tools/policy_view.py` | spectating page |
| `bots/zkbot/policy/trajectory.py` | engine trajectory capture for stage 4 (03 section 5.2) |
| `tests/policy/` | tests in 6 |

## 9. Risks and how to detect a dead or saturated policy

| Risk | Detection | Response |
|---|---|---|
| Exploration: the random policy hovers in KEEP or attacks the nearest enemy | KEEP share above 0.98 for 1,000 decisions, or non-KEEP decisions below 1%; kind histogram logged per 10 min | prior logits (a non-imitation warm start, objection 6), or a curriculum in the sim |
| Saturation: logits collapse to one kind | max probability above 0.99 on more than 30% of unit-decisions; entropy near zero; explained variance of the critic below 0.05 | entropy bonus up; temperature up; check advantage scale |
| Herding and dog-piling | share of units sharing a region or target above 60% with at least 10 units | shaping or the autoregressive decoder (4.5) |
| Order churn | orders per unit per minute above 6 (the user's complaint is 30-60) | commit lock length; check the emit filter |
| Idle and unfinished pile-up | idle builders above 20% for 30 s; unfinished structures above the cap | mask caps (4.3) |
| Latency overrun | over budget five times in a row | numpy to ONNX, or d=128 |
| Feature drift from rebalance | catalog hash mismatch at load | retrain |
| Sim-to-live gap (combat model, unseen behaviours) | sim win rate high, live metal ratio flat | the fight-fidelity gate (01); live A/B |
| Memory horizon too short | a scout-then-attack probe in sim fails at delays above the segment length | longer segments, or stored-state burn-in at 2x cost |

## 10. Objections to the shared contract

Resolved in the overview: 1 (two passes, 50M/day), 2 (tick 5, block 5), 3 (cap 256, first 128), 4 (exact vocabulary), 6
(same `terrain.py` and heights in both paths).  Objection 5 stays open: training starts with no prior, and one is
added only if the exploration diagnostics in section 9 fire, with the user's decision, because it bends "no scripted play".

1. **Throughput (00, "5M network with two training passes is about 100M decisions/day").**  Measured
   training is 1,720 samples/s for the v1 actor and critic, and 1,500 for d=256 (3.5).  Two passes at
   100M/day need about 2,300 samples/s, which no measured size reaches.  With one pass, 100M/day fits
   (d=192 about 84% of the GPU with rollouts; d=256 about 98%).  With two passes, about 50M/day fits
   (d=192 about 75%).  Please pick one.  This plan targets 50M/day with two passes.
2. **Timing (00, "sim tick 6 frames; decision every 15 frames").**  15 is not a multiple of 6, so the two
   do not line up (the lowest common multiple is 30).  The live bot also decides every 18 or 20 frames
   under its current block rates (1.2).  Proposal: sim tick 5 frames (decision every 3 ticks); live
   `block_n_frames` 5 in policy mode.  01 and 03 must agree.
3. **"Every controllable unit gets its own output" (00).**  Units are unbounded (969 in one logged game).
   Proposal: a cap of 256 controllable units (first 128) with the rules for the rest (2.4).  This is an addition to
   the contract, not a change to its intent.
4. **Hashed name embedding (00).**  Followed as written.  The catalog has about 615 names, so an exact
   vocabulary with an unknown row is smaller and collision-free.  The hash is the one `fightnet.py` uses,
   so it was kept for consistency.  Lead's call.
5. **Pure self-play (00).**  Followed.  Exploration may still need a non-imitation prior, such as default
   logits favouring FIGHT toward the nearest enemy during the first updates.  Imitation is not proposed.
   This needs the lead's decision because it goes against "no scripted play".
6. **Region graph parity (00, "one featurizer" and terrain shared).**  If live heights and archive heights
   differ on water maps (`HANDOFF.md` 6e), the region graph differs and so does the policy's view.  Both
   paths must read heights the same way (2.2).

## 11. Needs from 01 (sim) and 03 (training)

From 01:

* Field names and enums exactly as in 2.2, with `player`, `ally_team`, `knowledge`, `def_id` (-1 for blips without identity),
  `flags` bits, and `ord_kind`, `ord_age`, `ord_x`, `ord_z`, `ord_target` written by the controller's own
  command log.  Commands take effect at the next tick after issue.
* UNIT_IDLE-equivalent `idle_uids` each decision (topic 7 semantics: a unit with no orders).
* Tick of 5 frames (6 Hz), decisions every 3 ticks.
* Re-order cost: measured in the engine at S-M4 and modelled in the sim.  This is the user's complaint.  The
  size and the cause aren't known yet, so nothing here assumes a value.
* Blips: `vel` and `pos` every tick; `def_id` = -1 until the unit has been in LOS of that team, if that is
  the engine rule (unverified).  Blip top speed on the same 2 s window as `blips.py`.
* Enemy structures stay as ghost memory until seen dead, with their last-seen frame.
* Hidden enemy units exposed only to the critic, as `knowledge` 3.
* No pending projectile or shell records to the policy.
* Static bundle built by the same `terrain.py` code on the same height arrays as live (2.2, objection 6).
* Critic-only records must not reach the actor's inputs.

From 03:

* PPO with per-unit ratio clipping, averaged over units (4.5), and advantage normalised per batch.
* Segments of 32 decisions with stored initial GRU state, no burn-in (3.4).
* Raw frames stored in compact form or regenerated, never full tensors; featurization in data workers
  (2.6).
* Pass count explicit in the config, so the 50M/100M budget is checkable.
* Entropy per unit and kind histograms logged per update (6 and 9).

## 12. Unverified or not measured

* Blip position accuracy, and that `springai::Unit::GetDef` reaches `CAICallback::GetUnitDef` (identity rule read at `AICallback.cpp:593-597`).
* Whether `GetFriendlyUnits` includes allied units (the plan assumes own only, which is true in 1v1).
* Games past 1v1.  The model and the frame carry per-unit ids, and a test runs three alliances and six players through the net, but nothing is trained on such
  frames.  The sim has two sides of one player each.  Live, the observation has `allyTeam` per unit and no per-unit `team`, so a hostile unit's player falls back to its
  alliance; `springai::Unit::GetTeam()` exists, so ControllerAI could send it, and `from_brain` reads `team` when it's there.
* Whether the observation's `radarBlips` includes static buildings.
* The `JUMP` (38521) and `WANT_CLOAK` (37382) parameter formats; the cloak state convention is inferred
  from `customCmdTypes.lua:40`.
* Whether `speedModClass` and `terrainClass` separate kbot, tank, spider and amph.
* Cost of `unit_queue_state` per unit (not measured); the plan avoids it.
* Numpy thread behaviour under the bot's environment (`OMP_NUM_THREADS` effect not verified).
* ONNX Runtime speed (not measured).
* Featurizer cost on real frames (1.3 ms is synthetic).
* Engine behaviour of ghost buildings after they leave LOS (assumed from the bot's `enemy_structures`
  memory, `threat.py:91-97`).
* Whether UNIT_IDLE fires for factories and static units, and whether it fires on a stop.
* Army-size statistics come from the rules bot; the policy's army will differ.
* Live and archive heights on water maps (objection 6).
* The 20-minute memory requirement, which is a risk to test, not a result.

## 13. As built (P-M0 to P-M2, 2026-11-05)

Code is in `bots/zkbot/policy/` (`rawframe`, `bundle`, `tables`, `featurize`, `net`, `actions`, `runtime`).  With no policy flag
set, the heavy modules aren't imported, and the bot's only new calls return at once.

**Measured live, whole forward with untrained weights, 3 games on TitanDuel at 3x:**

| Measure | Result | Budget |
|---|---|---|
| Forward, mean | 16-17 ms | 35 ms |
| Forward, p99 | 28-37 ms | 60 ms |
| Forward, max | 133 ms; over 60 ms in 0.2-0.3% of decisions | |
| Game speed | 2.74, 2.87, 2.89x; recording only 2.91, 2.90, 2.85x | at least 2.7x |
| Featurizing, real frames | 0.9-2.2 ms | 1.3 ms (synthetic estimate) |
| Tokens, controllable | Mean about 130, most 214; up to 87 controllable | |
| Numpy against torch | Worst 7e-06 over 100 frames of 10 recurrent steps | 1e-04 |
| Parameters | Actor 2,937,194; critic 477,505 | 2.71M plus extras; 0.44M |

**Changes from the plan above:**

* Dimensions: unit 126 (type 58, instance 18, order 23, rays 24, summary 3), region 33, site 9, global 16.
* Names are a separate `unit_name` input, not part of `unit_x`.
* Added flags DAMAGED and RESERVED; `ord_age` on a memory record is frames since seen.
* `experience` is in the frame but isn't an input: the sim has none.
* `ptr_enemy` has one mask per pointer kind (attack unit, attack ground, manual fire).  Only enemy and region pointers exist so far.
* Rays are per move class, from a compact bitmask at 16 elmos.
* `economy=False` is the default: metal, energy, unfinished structures, idle builders and site ownership all read as zero or
  free.  The sim has no economy at stage 2, and an input training never saw vary could only mislead the net in a real game.
* Altitude has a 40-elmo dead zone (01 section 8.7).
* `route_scale` starts at -3.0, about 0.05 nats per walking second.  It first started at 0.74 nats per second, under which a
  30-second walk cost 22 nats and no far region was ever sampled.

**Golden tests:** `tests/policy/test_golden.py` holds 5 live frames and 4 sim frames against reviewed output, at 1e-6, in 248 KB.
Rewrite them with `tools/policy_golden.py` only after a change you meant.

**Section 12 items now answered:** move class comes from `smClass`, depth and `maxSlope` (`tools/sim_data.py`, written to
`data/moveclass.json`), which separates kbot, tank, spider, amph, hover and ship.  Featurizer cost on real frames is above.  Ghost
buildings do persist in recordings: up to 69, over the token cap of 16, so the cap is in use.  Live and archive heights match on
TitanDuel, though not yet checked on a water map.  Still open: the JUMP and cloak parameter formats, which S-M7 and P-M3 reach.
