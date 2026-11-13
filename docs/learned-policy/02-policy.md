# Observations and actions

The contract between the game and the network: what a seat sees (the raw frame), what the network reads (the featurizer's
tensors), and what it can tell its units (actions, their masks and how they decode to orders).  The sim and the live bot share
all of it.  The network itself is in `docs/model.md`.

| Code | What |
|---|---|
| `bots/zkbot/policy/rawframe.py` | the raw frame's schema (the only definition) and the live reader `from_host` |
| `bots/zkbot/policy/bundle.py`, `tables.py` | a map's static bundle (regions, gates, spots, passability), and the type table (stats, names, rosters) |
| `bots/zkbot/policy/featurize.py`, `featurize_kernels.py` | raw frame to tensors and masks; the per-unit loops are numba kernels |
| `bots/zkbot/policy/actions.py`, `groups.py` | actions, encoding orders as choices and decoding choices to orders, group orders |

## The raw frame

One frame a decision, for one seat (`SCHEMA_VERSION` 5; version 4 is this without the transports' two flags, and still
reads).  It holds only what a player can know and the live bot can supply: no facing, weapon reload or shots in flight.

| Record | Fields | Notes |
|---|---|---|
| UNIT (array) | `uid`, `player`, `ally_team` (the game's numbers, -1 unknown; `player` is the engine's team), `knowledge` (0 in sight, 1 radar, 2 remembered, 3 hidden: the critic's only), `def_id` (-1 unknown), `x y z`, `vx vz`, `hp_frac` (NaN for a blip), `build_frac`, `experience`, `blip_top_speed`, `flags`, order memory (`ord_kind`, `ord_age` in frames, `ord_x`, `ord_z`, `ord_target`) | Flags: BEING_BUILT, CLOAKED, PARALYZED, IN_LOS, IDLE, TYPE_KNOWN, DAMAGED, RESERVED, JUMP_READY, PULLING, CARRIED (an air transport holds it), LOADED (a transport holding something).  Order memory is what the controller last told its own units, as a player remembers it, not the engine's queue.  For a PRODUCE, `ord_target` is the def id made.  On a remembered structure, `ord_age` is how long ago it was seen |
| SITE (array) | `site_id`, `x z`, `income`, `status` (free, ours, enemy, building), `region_id` | metal spots; the viewer's status from the extractors it knows of |
| FEATURE (array) | `feat_id`, `x z`, `metal` left, `def_id` | wrecks in sight with at least 1 metal |
| GLOBAL | frame, metal and energy (current, income, storage), commander health and uid, base x z, map size, frames since enemy contact, unfinished own structures, idle own builders, the viewer's `player` and `ally_team` | the last two say whose frame it is, and aren't inputs |
| Static bundle (once a game) | regions (centre, area, height, domain, tags, access per move class, spots), gates per edge, spots, passability, heights | built by the same `terrain.py` code from the same heights in both paths |

`sim/include/zksim/rawframe.hpp` is generated from `rawframe.py`, and a test compares them.

**Live** (`bots/zkbot/host.py`, `rawframe.from_host`), from ControllerAI's observation:

* Units, enemies in sight and radar blips each carry `defId`, `allyTeam`, `pos`, `vel`, health, build progress, cloaked and
  paralysed.  A blip carries `defId` only while its type is known.
* Idle units come from the UNIT_IDLE events (topic 7) since the last decision.
* Order memory is what the bot told its units.  A blip's top speed is taken over 2 s.  Remembered structures are kept by the
  host.
* ControllerAI doesn't send a unit's player yet, only its allyteam, so a hostile unit's player falls back to its allyteam.  That's
  exact in a 1v1.  `springai::Unit::GetTeam()` exists if it's wanted.
* The static bundle must be built from the heights after water is set.  This matched the archive on TitanDuel; it hasn't been
  checked on a water map.

## The featurizer

`featurize(frame, bundle, table)` gives fixed-normalisation numpy tensors.  It takes 0.4 ms a frame of about 134 units with
numba, giving the same tensors to the bit as the numpy it replaced.  Without numba it's plain Python, right but about 50x slower.

**Tokens, in this order:**

    controllable units | every other unit, most important first | summaries | regions | sites | features | global

| Group | Numbers each | Cap, and what happens past it |
|---|---|---|
| Units | 129: type stats, instance state, order memory, 24 terrain rays (free distance, enemy and own DPS in 8 sectors), summary counts | 1,280, controllable first (up to 512, by cost, then distance to the army).  After those come other alliances' units in sight, then on radar, then the rest, each by max(cost, 100) / (distance to our army + 200) |
| Summaries | as units | 12 tokens, one per (whose, what's known, coarse role), for what the unit cap left out |
| Regions | 34: 27 static, then what's there now (own value, seen and radar enemy value, DPS, wreck metal, ...) | all of them (120 at most on the maps seen) |
| Sites | 9 | 24, nearest the base first |
| Features | 5 | 512, by metal / (distance + 200); a region's wreck metal still counts every one |
| Global | 16 | |

* **Who owns a unit** is two ids numbered from the viewer: own is 0, and the others count from 1, skipping its own.  An unknown
  id, or one past the table, shares the last row.  The tables have 64 players and 8 allyteams (`MAX_PLAYERS`,
  `MAX_ALLY_TEAMS`).  Both seats of a 1v1 see the same numbers.  The token kind says what's known (seen, blip, memory, summary),
  not whose.
* **Names**: an exact vocabulary of catalog names plus an unknown row.  For the actor, 30% of the vocabulary reads as unknown
  in each game, so it learns to go by stats too.  The frozen encoder always gets the true names.
* **Normalisation is fixed**: positions over map size, velocities over the unit's catalog speed, log1p for costs, and constants
  for health and DPS.  Stats come from the catalog, not the live game.
* **The economy is on** (`economy=True`) everywhere the actor's or the encoder's frames are made: metal, energy, income,
  storage, unfinished structures, idle builders and site status.
* **For the critic only**: up to 256 enemy units the seat can't see (`hidden_*`), nearest the seat's army first.  It was 64
  until 2026-11-11 (the user): a late team game, free-for-all or chicken game has hundreds or thousands hidden.
* **`FEATURE_VERSION`** (10) is what a saved encoder or actor was trained to read.  Raise it with any change to the tensors'
  meaning.  Loading checks it, and `ENCODER_FEATURE_VERSIONS` lists what an encoder reads alike.  Frame shards store featurized
  tensors, ids included, so a change means collecting frames again.  Version 7 (2026-11-10) resized the id tables for
  free-for-all.  Version 8 (the same day, the user's numbers) raised the caps for long and many-sided games, from 592 unit
  tokens with 256 controllable and 32 features: an 80-minute 1v1 had 656 records in sight on a side and 609 of its own.  So
  every encoder, actor and frame pool from before is invalid, `full-1`'s too.  Version 9 (2026-11-11) is air transports:
  a unit's carried and loaded flags, three kinds, and aircraft among the controllable.  Nothing had been trained at 8.
  Version 10 (the user, 2026-11-13), found by the C++ port: the DPS sectors are floored, so sector k is the 45 degrees
  around k pi/4 from +z towards x, as ray k is (truncated, sector 0 had been 90 degrees wide, 4 and 5 22.5, and the -x
  side's one off from the rays); and the frame is over 80 minutes, the sim's limit (`GAME_FRAMES`), not 20, which ran to 4.
  On the golden frames only those columns changed, each unit's DPS kept to 6e-7.  `full-2`'s encoder and every actor
  trained on it read version 9.
* **Allies** are units like any other in the pool, told apart by their ids: the viewer's allyteam is 0 and its own player 0.
  Only the viewer's own units are controllable.
* **Golden tests** (`tests/policy/test_golden.py`): 5 live frames and 4 sim frames against reviewed output, at 1e-6.  Rewrite
  them with `tools/policy_golden.py` only after a change you meant.

## Actions

A decision is a list of up to 16 actions (`actions.SLOTS`), chosen one after another, then done.  Each action picks one of the
seat's controllable units, the anchor, and gives it an order.  A unit no action picks keeps what it's doing.  An action's
order may go as a group order to the units around its anchor.

**Controllable**: own mobile units with weapons, air transports, factories, silos' missiles, the Trinity, switches (Newton,
Sumo), constructors, and the commander once the opening is over (`featurize._select`, `table.acts_in_place`,
`table.constructor`, `table.transport`).

* **Aircraft** are controllable since `FEATURE_VERSION` 9 (2026-11-11).  Before it the policy had none of its own to
  command: armed aircraft were left out, and a transport is one.  It's a choice of this change, not the user's, and one
  line in `featurize_kernels.select` takes it back.

* **Factory plates** are factories in the type table (`specials.json`'s `plates`, from the dump's `child_of_factory`), though the
  engine's def says they aren't: a plate can be told what to PRODUCE and be helped, as in the sim.  Its type stats keep the
  def's flag.  FEATURE_VERSION wasn't raised for it (2026-11-10): no recorded game or frame had held a plate.
* **Caretakers** can't be told anything (they've nothing to build), and need nothing: the sim has an idle one help what's
  being built in its reach, as Zero-K's default widget does.

### Kinds

24 kinds (`rawframe.KIND_COUNT`).  The sim obeys the 20 in `policy_sim_env.KINDS`, keep among them.  The engine commands are
where each would go once the bot sends commands: none is sent yet, and those marked "unverified" haven't been tried in a game.

| Kind | Chooses | Engine command | Legal when |
|---|---|---|---|
| 0 KEEP | | none | never an action: it's what an unpicked unit does |
| 1 MOVE | region, fine | `move` | mobile, the region reachable for its class |
| 2 FIGHT | region, fine | `fight` | as MOVE, and armed |
| 3 ATTACK_UNIT | a target in sight | `attack` | armed with a weapon that can hit the target's class.  A silo's missile: any ground target in its range |
| 4 ATTACK_GROUND | a target in sight or on radar, fine at 200 elmos | custom 20 | armed, target within 1.5x range + 300.  Not in the sim's kinds |
| 5 STOP | | `stop` | not in the sim's kinds |
| 6 MANUAL_FIRE | a target | custom 105 | a manual weapon, in its range, reloaded.  The Trinity aims this way, within its manual range |
| 7 JUMP | region (its own mask), fine | custom 38521 (unverified) | can jump, charged, the region within jump range and standable |
| 8, 9 CLOAK_ON / OFF | | custom 37382, state 1 / 0 (unverified) | can cloak; the one it isn't |
| 10 BUILD | type, place, offset | `build` | a constructor or the commander, with an option it builds |
| 11 ASSIST | one of our units being built, or a factory | `guard` | a target of that kind in reach |
| 12 REPAIR | | `repair` | not in the sim |
| 13 RECLAIM | a wreck | `reclaim` | a constructor with a wreck in sight |
| 14 PRODUCE | type | `build` on the factory or silo | it makes something other than what it's making now |
| 16 THROW | region, fine | custom 105 (unverified) | a thrower, ready, with someone beside it to throw |
| 17, 18 PUSH / PULL | | custom 35666, state 1 / 0 | a switch; the one it isn't doing |
| 19 PLACE_BEACON | region, fine | custom 35170 (unverified) | a teleporter |
| 20 WAIT_AT_BEACON | a region with one of our beacons | custom 35171 (unverified) | walks, and a finished beacon of ours is reachable |
| 21 LOAD | a unit | `load units` (75) on it (unverified) | an air transport holding nothing, and a unit it can lift: one of ours, or another alliance's in sight that stands still.  Not one being built or already carried |
| 22 UNLOAD | region (JUMP's mask), fine | `unload unit` (81) at the place (unverified) | a loaded transport, and a region its load's move class can stand in |
| 23 DROP | | custom 35000, Zero-K's Drop Cargo (unverified) | a loaded transport, whatever else it's doing.  The button's 7 s of reload aren't in the frame, and a drop too soon does nothing |

What a transport can lift is the type table's (`lift`, `load_class`, `lift_size`, `load_size`, from `specials.json`): the
Charon lifts light and medium loads, the Hercules any, and a load too wide for one is refused (01-simulator.md,
"Transports").  The sim's own order numbers for the three differ from these (22 to 24), and `policy_sim_env` maps them both
ways.

**Commit lock**: a unit told something in the last 3 s (`COMMIT_DECISIONS` slots) can't be an anchor, unless it was damaged, its
target was lost, or it's idle.  This keeps orders steady.

### Where an action points

* **Region and fine**: a region pointer (each region's walking cost from the unit lowers its score by a learned amount), then 9
  options.  Option 0 is the region's centre, and 1-8 are compass points at half its equivalent radius, with north towards the
  enemy.  That makes the choice the same whichever way the map faces.
* **Targets**: one pointer per kind (attack unit, attack ground, manual fire, assist, load), over the tokens that kind may aim
  at.  With a transport of ours about, our own units it could carry are among those tokens.
* **BUILD**: a type (the same head as PRODUCE, over the constructor's options: 187 producible types, 127 from factories and silos
  and 60 built), then a place, then an offset.  The place is one of our unit tokens or a site.  The offset is on it, or one of 8
  compass points at 64, 128 or 256 elmos.  The sim builds at the nearest spot the building's footprint fits, on ground level
  enough for it (`SimEnv.place_build`; 01-simulator.md, "Where a building can go"), and an extractor at the free spot
  nearest the point.  On the economy script's builds, that comes out p50 19 elmos from the script's
  place, with 99% within 150.
* **RECLAIM**: a pointer over the wreck tokens.
* **Group orders** (MOVE, FIGHT and ATTACK_UNIT only): alone, or one of 10 radii from 60 to 2,250 elmos (each 1.5x the one
  before), then each of 15 roles in or out.  The sim gives the order as well to every unit of the anchor's team of those roles
  within the radius of the anchor, and to no ally's.

### Encoding and decoding

* `actions.decode` turns choices into orders, for the sim (`policy_sim_env.sim_orders`) or the engine.  `actions.to_slots` and
  `to_decision` convert between the list of actions and what each unit was told.
* `actions.encode` is the inverse.  It turns the scripts' orders into the policy's choices (the cloning labels), taking the
  nearest legal region, fine option or place, and counting what can't be said: 0.35% of orders in `army-6`'s check.  A group
  order is recorded as its anchor's order followed by a `GROUP_MARK` record holding the radius and roles.

## Deployment: C++ inference

The bot is to run with nothing to install (the user, 2026-11-12): its inference path in C++, the nets in onnxruntime, and no
Python, numpy or numba.  onnxruntime builds exist for Windows (x64, ARM64), Linux (x64, ARM64, glibc 2.28 or newer) and macOS
on Apple Silicon only (1.31.1).  The type table, the featurizer and the actions are done (2026-11-13); training doesn't call
them yet.

### libzkpolicy

`sim/policy/` (C ABI in `zkpolicy.h`, about 2,300 lines), bit for bit with the Python, which stays the reference:

* **What's in it**:
  * the type table, read from a flat file (below);
  * `featurize.py` and `featurize_kernels.py`: a raw frame (`rawframe.hpp`'s records), a map bundle's arrays and the table to
    every field of `fz.Tensors`, the masks and the critic's hidden units among them;
  * `actions.py` and `groups.py`: `decode`, `encode`, `fine_point`, `world_positions`, `to_slots`, `to_decision`, and the
    groups' `bucket` and `expand`.

  Not in it: the bundle builder (a bundle's arrays are handed in), the nets, the decision loop, the engine side.
* **The type table is a flat file Python writes**, `bots/zkbot/data/policy_table.bin` (444 KB): `tables.TypeTable`'s arrays
  as the kernels read them (`fz._table_pack`), the roles, the def ids' rows and the names, behind a header with the table's
  hash.  Not the catalog's JSON with a vendored reader: the table is made by rules (`tables.py`, `catalog.py`, `units.py`:
  classes, roles, specials, what each plant makes) that C++ would have to repeat and keep in step, where a flat file has one
  maker and can't drift.  C++ only checks its header and sizes.  `~/zkml/bin/python sim/py/zkpolicy.py table` rewrites it,
  and `test_zkpolicy.py` fails while it's stale.
* **How it's the same to the bit**:
  * numpy's types followed op by op: NEP 50 keeps a Python float with a float32 in float32, and takes a float32 with a
    float64 or an index to float64.  Its float32 sums are done pairwise as numpy does them (blocks of up to 128, halves split
    at multiples of 8);
  * glibc's libm (`log1pf`, `hypotf`, `atan2f`, `log1p`, `hypot`, `sin`, `cos`), which numpy and numba call on this machine
    (AVX2, so no SVML), built with `-O3 -ffp-contract=off` (no fused multiply-adds);
  * numpy's orders (NaN last in a sort, lexsorts stable, `argmin` taking the first NaN), x86's float to int (INT_MIN when out
    of range) and Python's modulo;
  * two kernels are quicker and give the same: a DPS sector by the sign of a cross product with each boundary, back to
    `atan2f` within 1e-4 of one (no difference in 10^8 pairs, half of them near boundaries; made for the truncated sectors,
    which differ from the floored only on the -x side), and a radius test that rules out by squared distance before `hypotf`.
* **Checked** by `tools/policy_parity.py` on 49,451 frames of `army-7` (corpus and results in `/mnt/d/zk-learned/parity/`):
  * 311 sides: 80 1v1s, 80 team games, 80 free-for-alls and 71 chicken games;
  * each side's frames spread over its game, plus its fullest (the caps of 1,280 units, 512 controllable, 512 wrecks and 256
    hidden all reached) and those with a transport loaded and with plates;
  * every frame featurized as training does, every 4th with the defaults, and every 4th with all 24 kinds on a copy whose own
    units are partly made Djinns, beacons and switches, which the scripts never build: 74,254 featurizations, every field
    to the bit;
  * the scripts' 157,226 orders encoded and decoded back, and 3.2M orders of random legal decisions decoded and encoded
    again, of every kind the masks allow (THROW, PUSH, PULL, PLACE_BEACON, WAIT_AT_BEACON, LOAD, UNLOAD and DROP among
    them), with slots both ways, and 145,074 group expansions.

  No difference, in 76 s on 8 processes; and none again after version 10 and the actions' fixes below (`check-fv10.json`).
  `tests/policy/test_zkpolicy.py` does the same on the golden frames.
* **What the port found in the Python, fixed on both sides** (the user, 2026-11-13): the DPS sectors and the frame's scale
  (version 10, above); a THROW's reach read by `t.unit_name`, which a training game's dropped names
  (`policy_train._drop_names`) had made unknown, so a Lobster threw at its own feet, now by `true_name` (the binding passes
  the real names); `fz.fine_point` in float32 (numpy's scalars) where encode's `fine_points` are float64, now float64 both; and
  encode keeping fields of a unit's earlier order when a later one replaced it, now the last order that can be put, whole.
  The math library stays as it is: the bits match on glibc's libm on x86-64, so the bot on Windows will need the same.
* **Speed** (`policy_parity.py speed`: 15,106 frames of 96 sides, featurized as training does, in WSL on the 16-thread CPU
  while a training job used 5 to 8 of its threads), per frame:

  | Unit tokens (frames) | numba | libzkpolicy |
  |---|---|---|
  | 0-127 (5,565) | 0.45 ms | 0.24 ms |
  | 128-383 (5,194) | 0.90 | 0.51 |
  | 384-767 (2,871) | 3.0 | 1.7 |
  | 768-1,280 (1,476) | 11.1 | 5.6 |
  | all | 2.13 | 1.13 |

  `featurize_many` in batches of 64 frames, a frame: 1.32 ms on 1 thread (1.03 with the buffers kept from batch to
  batch, which saves the pages' first touch), 0.24 on 8 and 0.19 on 16, which is 11x numba in one process.
* **Building**: `cmake -S sim -B ~/zksim/build && cmake --build ~/zksim/build -j8` makes `libzkpolicy.so` beside the sim's
  libraries.  `sim/py/zkpolicy.py` looks for it at `ZKPOLICY_LIB`, else beside `ZKSIM_LIB`, else in `~/zksim/build`.
* **Calling it** (`import zkpolicy as zp`, from `sim/py`):
  * `zp.featurize(frame, bundle, table, ...)` is `fz.featurize`'s drop-in: the same `fz.Tensors`, whose arrays are views of
    one buffer C++ wrote.
  * `zp.featurize_many([(frame, bundle), ...], table, threads=8, ..., buffers=None)` does many frames on C++ threads with the
    GIL let go, each into a buffer of its own or the caller's (`zp.bounds` gives their sizes).  The threads are made for
    the call, so a process can still fork after it.
  * `zp.decode`, `zp.encode` (of `act.Order`s, or of `zp.ORDER_DTYPE` records), `zp.fine_point`, `zp.world_positions`,
    `zp.to_slots`, `zp.to_decision`, `zp.bucket` and `zp.expand` are actions.py's and groups.py's.
  * A table or bundle is copied into the library once and kept on the Python object, remade if its arrays are replaced.
  * From C or C++: `zkp_table_new` on the file's bytes and `zkp_bundle_new` on a bundle's arrays, then `zkp_bounds` and
    `zkp_featurize` into the caller's buffers, which hands back each field's offset and shape.  `zkp_decode` and
    `zkp_encode` take a `zkp_tensors` of those arrays.
* **Switching training over** (not done):
  * `policy_train`'s `_worker_chunk` and `_worker_batch`, `policy_clone_train._samples`, `policy_clone_data` and
    `frame_data` call `zp.featurize_many` or `zp.featurize`, with a worker's buffers kept.  `Player._features` applies
    `_drop_names` after, as now;
  * the encoding of the scripts' orders calls `zp.encode`, and rollouts and the clone eval `zp.decode`;
  * the bundles' handles (`zp._bundle`) are made before workers fork, and `libzkpolicy` is built in `~/zksim/build`;
  * encoding the scripts' orders from their records, without `se.policy_orders`' Python loop, needs a map from the sim's
    catalog ids to the table's rows;
  * C++ threads could take the place of the featurizing processes, with `pm.collate` still in Python.
* **Still to do**:
  * the bundle builder (`terrain.py`, `bundle.py`), checked on every pool map;
  * the nets and the decision loop, and the engine side (below);
  * a libm of the bot's own: the bits hold against glibc on x86-64 without AVX-512.  Windows' CRT, or numpy with AVX-512's
    SVML, may differ in the last bit of `log1pf`, `atan2f` and the like, so the bot and training's library should share one.

### The rest

* **The nets in ONNX**: the encoder, the time transformer and one action step, each exported on its own.  The decision loop
  (sampling, masks, up to 16 actions until done, as `policy_model.sample_actions`) is C++, checked against torch on recorded
  decisions.  Attention exports by the `scaled_dot_product_attention` path with a mask, not flex attention: ONNX's
  FlexAttention is an experimental op (`ai.onnx.preview`), and a decision's window is at most 4 frames of 44 tokens.
* **The engine side**: first a C++ client of ControllerAI in place of `main.py`, `client.py` and `host.py`.  In the end, a
  native Skirmish AI the engine loads, as it loads CircuitAI, so the bot is one folder in Zero-K's AI directory: its
  library, onnxruntime's and the model files.  The opening and the economy script go with it.
* **Size**: of the 4,900 lines of Python that were on the bot's path, the featurizer (1,100) and the actions and groups
  (350) are ported, and the unit tables (550) are the flat file now.  Left: `terrain.py` 1,400, the raw frame schema 300
  (already generated as `sim/include/zksim/rawframe.hpp`), the engine glue 650.
* **Order**: the featurizer and the actions (done), the bundle, the nets and the decision loop, then the engine side.

## Not yet checked live

* The JUMP, cloak, throw and beacon command formats.
* Transports: `from_host` sets neither CARRIED nor LOADED (whether ControllerAI's observation says what a transport holds
  isn't known), and the LOAD, UNLOAD and DROP command formats are untried.  Where a carried unit is reported to be
  matters too: UNLOAD's mask finds a transport's load as the carried unit at its place.
* Aircraft as the policy's own: nothing here rearms a bomber or tells a plane where to land.
* Whether ControllerAI's `radarBlips` include static buildings.
* Whether UNIT_IDLE fires for factories, and on a stop.
* How accurate a blip's position is.
* A thrower's reload isn't visible live, so it always counts as ready.  `PULLING` is never set live, since the bot would have to
  remember what it told a switch.
* Live and archive heights on a water map.
* Caretakers: the AI gets no widgets, so the bot has to give its own the patrol order Zero-K's default widget gives a player's.
