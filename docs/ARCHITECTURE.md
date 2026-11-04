# zkbot architecture and redesign plan

Status: proposal plus the first steps (marked **done**).  The research
behind it is summarised in section 7; the full report (with sources) was
produced by a research subagent and is quoted where it matters.

## 1. Where the bot is now

* ~9,500 lines of Python in 20 modules, all composed into one `Brain` class
  by 18 mixins.  Every mixin reads and writes `self`: **186 distinct
  attributes** are assigned on the brain, **95** of them created lazily via
  `getattr(self, "x", default)`.  Any module can depend on any other's state,
  which is why a change in raiding can break construction.
* Decisions are long if-chains.  `military_step` was one 500-line function
  (context computation, then a 20-branch per-unit chain).
* Unit knowledge is by name: `units.FACTORY_UNITS`, `ROLE_OVERRIDES` and
  ~200 string literals (`"cloaksnipe"`, `"cloakaa"`, `"factorycloak"` ...).
  Phantoms, Gremlin spotters and scouting only exist for the cloak factory.
* ~1,300 numeric literals, 34 module constants and 47 `ZKBOT_FLAGS` checks.
  Flags are good A/B switches but they are never removed, so dead branches
  accumulate.
* Tested almost only on TitanDuel 2.2, Cloakbot factory, vs CircuitAI:Hard,
  4 games per batch, sequentially (~4 games/hour).

What works and should be kept: the frame loop and transport; the threat map;
blip identification; raid squads v2 and raider micro; slow-ball engagement;
overwhelm attack; Phantom/Gremlin logic; burrowed mexes; extraction-based
resign; emergency turrets (opt-in); the metrics CSV and logs.

## 2. Goals

1. Robust to **maps** (land/sea/size), **starting factories** (all ten), and
   **game changes** (balance patches, new units): unit facts come from the
   game's own data, not from names in code.
2. **Separable subsystems** with explicit inputs/outputs, so economy,
   production, army and intel can be developed and read independently.
3. **General decision making**: few, well-named decision points scored from
   continuous inputs, instead of special cases.
4. **Measured progress**: fast parallel games, paired A/B with real
   statistics, a fixed test matrix, and room for light learning/tuning.

## 3. Target architecture

```
Layer 0  adapter      client.py, main.py — websocket, observations, query cache,
                      command buffer (dedupe / rate limit / who-issued-it)
Layer 1  knowledge    catalog.py (stats + abilities + descriptions → roles,
                      rosters, manual weapons), mapdata.py (+ pathability,
                      regions), params registry (all tunables, typed defaults)
Layer 2  world model  units/enemies/ghosts (intel.py, blips.py), threat maps
                      (threat.py), economy state, event log — read-mostly
Layer 3  managers     Strategy (posture, budget split, role-mix targets, resign)
                      Economy+Construction (what/where/who to build)
                      Production (role mix → factory queues via catalog)
                      Army (squads, objectives, engage/retreat → ArmyPlan)
                      Intel/Scouting
                      plug-ins: air, snipers, burrow, raids
Layer 4  behaviours   per-unit priority lists (behaviours.py) using the plan
```

Per frame: observation → adapter → world model → managers (each at its own
cadence) write *intents* (`ArmyPlan`, build queue, production targets) →
behaviours turn intents into unit orders → command buffer → engine.

Rules that make it hold together:

* **Managers own their state** in a small object (`self.army_state`,
  `self.eco_state`, ...) instead of loose attributes on the brain; other
  managers read it, only the owner writes it.  Migrate one subsystem at a
  time; the mixins can stay as the mechanism while state moves into objects.
* **Unit names appear only** in `catalog` overrides and opening scripts.  A
  code review rule; `tools/catalog_check.py` shows where data and hand labels
  disagree.
* **Every tunable lives in the params registry** with its default, a range
  and a one-line rationale; flags become registry overrides
  (`ZKBOT_PARAMS="raid.offense_share=0.4"`), which is also what a tuner
  needs.
* **Decision style per layer**: utility scoring (considerations × response
  curves, multiplied so a single veto kills an option) for "pick one of N"
  (targets, objectives, placement, what to build); small state machines with
  hysteresis/commit timers for squad modes (gather → advance → engage →
  retreat); ordered behaviour lists for micro.  No GOAP/HTN: too costly to
  debug for what they would add here.

## 4. Unit knowledge: stats are not enough (user review)

Some units' job is defined by an ability, not by their stats: the Dante's
manual-fire napalm salvo, the Reef's stockpiled disarm missiles, the
Scorpion's EMP lightning, Aspis/Felon shields, Iris area cloak, Sumo/Black
Hole gravity, Dominatrix capture, Crab's armoured curl.  So the catalog
combines three sources:

1. **Stats** (`data/units.json`, `data/weapons.json`): cost, HP, speed,
   range, DPS, AoE → `catalog.stat_role`.
2. **The game's description** (`data/abilities.json` → `desc_roles`), e.g.
   "Assault/Riot Strider", "Raider/Riot Corvette", "Cloaked Scout/Anti-Heavy".
   Dual roles are kept (`catalog.roles(name)`); stats break the tie for the
   primary role.
3. **Abilities** from the unit Lua definitions and scripts
   (`tools/extract_abilities.py`): `manualfire` (with the D-gun's range,
   damage, AoE, reload, effects and `manualfire_desc`), `cloak`,
   `area_cloak`, `shield`, `jump`, `emp`, `disarm`, `slow`, `fire`,
   `gravity`, `capture`, `torpedo`, `stockpile`, `kamikaze`, `transport`,
   `jammer`, `radar`, `drones`, `morph`, `regen`, plus script markers
   (`armor` curl, `floating`, `takeoff`, `radar_fire_toggle`, ...).

Primary roles inferred this way agree with the hand labels for 100/108 units
that have one; most of the rest are cases where the game's description
disagrees with the hand table (Sumo, Crab and Venom are "Riot" in-game).

**Done:** catalog with stats + description + abilities; it fills roles
missing from the hand tables and picks factory units for roles a factory
table lacks (e.g. Shield AA/artillery); a generic `manual_fire` behaviour
fires any army unit's D-gun weapon (Dante napalm, Scorpion EMP, Bantha EMP
missile, Detriment blaster, Reef disarm) at the best cluster/value target in
range (opt-out flag `nomanualfire`).
**Next:** use abilities in threat estimates (enemy Dante salvo radius,
disarm/EMP range when choosing engagements), shield units as a group anchor,
cloak units for raids, jump units ignoring cliffs in pathing.

## 5. Migration plan (each step A/B-tested, see section 6)

| Phase | Step | Status |
|---|---|---|
| P0 infra | `tools/prun.py` parallel games, code snapshots, paired arms, memory guard, global slots; `tools/abstat.py` | **done** |
| P0 infra | `ZKBOT_FACTORY` override for factory generalisation tests | **done** |
| P1 knowledge | `catalog.py` + `extract_abilities.py` + `catalog_check.py` | **done** |
| P1 knowledge | replace name checks (Phantom, Gremlin spotter, scouting, air) by catalog roles/tags | next |
| P1 knowledge | `params.py`: exact-token `FLAGS` set for every module (fixes `"emergency" in "noemergency"`), `P()` registry with ranges and `ZKBOT_PARAMS` overrides; resign/raid micro/blip/ball/spotter constants registered; config logged at game start | **done** |
| P1 knowledge | register the remaining inline constants (≈1,300 numeric literals) as they are touched | ongoing |
| P2 structure | army: `ArmyPlan` composed of `Posture`, `CommanderFocus`, `SearchPlan`, `SlowBallPlan`, `RaidPlan`, `MainGroup` (one planner method each, `armyplan.py`) + ordered behaviour list (`behaviours.py`); `military_step` 500 → 25 lines | **done** |
| P2 structure | raid squads, responders, pincers → a raid manager owning its state | |
| P2 structure | economy split by domain: `economy.py` = task board (176 lines, was 1,176); `build_resources.py` (mex/energy/storage/pylons), `build_facilities.py` (factories/strider/air pads/caretakers), `build_support.py` (radar/silo/seismic), `build_defense.py` (turrets) | **done** |
| P2 structure | economy: separate placement (where) from task generation (what); builder assignment (who) as a scored market | |
| P2 structure | state objects per manager instead of loose brain attributes | |
| P3 generality | factory gauntlet (all ten factories) and map suite as standing tests | gauntlet ready |
| P3 generality | openings per factory/map class as data; composition = role mix → catalog | |
| P4 learning | engagement logging (`engagements.py`: snapshot of both sides of every fight + losses 20 s later → `engagements-team<N>.jsonl`, ~50–100 per game) | **done** |
| P4 learning | `tools/combat_model.py`: fits logistic regression / gradient boosting on the logged fights (CV grouped by game) and compares with the current power-ratio rule; `--save` writes `data/combat_model.json` | tool ready; needs ~100+ games of logs |
| P4 learning | use the fitted model for engage/retreat decisions (A/B against the ratio rules) | |
| P4 learning | SPSA/CLOP tuning of registry parameters in groups of 3–6 | |
| P4 learning | Thompson sampling over a few plan families per (map class, factory, opponent) | |

## 6. Evaluation

* `python tools/prun.py <tag> --games N --arm new= --arm base= --code base=<dir>`
  plays both arms on the same map and side for every game index (paired), in
  parallel.  Each run uses a snapshot of `bots/zkbot`, so editing code during a
  batch is safe, and `--code` can point an arm at an older version (e.g.
  `git archive <commit> bots/zkbot | tar -x -C results/_base_<commit>`).
* Throughput: a headless engine commits ~3.3 GB at the start and up to
  ~4.7 GB in long games, a bot ~0.5 GB, so ~8–9 games fit in 48 GB (the
  runner waits for 9 GB of free commit before each start).
* **CPU affinity:** the engine pins its main thread to one fixed logical core
  (0x4000 on this machine) and ignores `SetCoreAffinity`, so until h119 every
  parallel game shared that one core (8 games at ~1–2x, and a game capped
  at 1x fell to ~0.3x next to uncapped ones).  `zk-startup.ps1
  -EngineAffinity <mask>` resets the engine's process affinity when the game
  starts, which resets all its threads; prun passes all cores.  8 parallel
  games now reach ~12 game-minutes in the first wall-minute (was 3).
* **CircuitAI's random starting factory dominates results.**  Pooled over
  h100–h119 (TitanDuel, Hard): vs shield 15/15, cloak 13/16, but hover 3/12
  and amph 3/9.  A 4-game batch mostly measures which factories CircuitAI
  rolled.  `--enemy-factories cloak,shield,veh,tank,hover,amph,spider,jump`
  pins CircuitAI's factory per game index (via its `disabledunits` AI option,
  `zk-startup.ps1 -CircuitOptions`), two consecutive indices per factory (one
  per side), so paired arms face the same matchup; `abstat.py` prints results
  per enemy factory (inferred from the log for older batches).  Note that
  disabling the other factories also stops CircuitAI's later second factory.
* **Game speed changes CircuitAI's strength (confirmed, h120).**  Same code,
  same 16 pinned matchups, run side by side with the affinity fix: at a
  steady 1.00x we won 1/16, uncapped (~10x) 9/16; paired, only the fast arm
  won 8 games, the slow arm 0 (sign test p = 0.01).  CircuitAI's metal
  income is the difference: its cumulative metal at 5/10/15 min was
  4.5k/13.5k/26.7k at 1x vs 4.0k/10.9k/18.7k at max speed, while ours was
  4.9k/13.3k/22.6k vs 5.1k/15.4k/26.2k (ours is already equal at 5 min,
  before much fighting).  Our bot is frame-synchronous (the engine waits for
  it; no query timeouts), so the change is CircuitAI's.  Source
  (RecoilEngine/AI/Skirmish/CircuitAI, util/Scheduler.cpp): all its
  decision logic is frame-scheduled, but threat/influence/enemy maps
  (AllyTeam::Update, every 7 frames, skipped while the previous job runs)
  and every path and builder cost-map query (PathFinder::Run*, 4 pather
  threads) run on worker threads and are applied on whichever frame they
  finish.  An idle builder gets no task until its cost-map query is ready
  (BuilderManager::DefaultMakeTask returns nullptr), raid/attack squads move
  only when their path arrives.  At 10x a job of fixed wall time spans 10x
  as many game frames: staler threat maps and longer idle builders and
  squads.  Human
  games run at 1x, so **all historical win rates (70–100% at 10–16x) were
  against a weakened CircuitAI; at 1x the bot loses ~95%**.  Evaluate at
  `--speed 3`, shown equivalent to 1x in h121 (0/16; CircuitAI's metal
  14.1k/27.0k at 10/15 min vs 13.5k/26.7k at 1x) and prun's default; confirm
  key results at `--speed 1`, and compare arms only under the same speed.
* Statistics (`tools/abstat.py`): win rate with Wilson intervals; for paired
  arms, the discordant pairs and an exact sign-test p-value.  A difference of
  one or two games in 4-game batches is noise: p stays near 1.
* Recommended going forward (from the research): test candidates against the
  previous accepted version (self-play, win rate near 50% where tests are most
  sensitive) and keep CircuitAI Hard/Brutal as a gauntlet; use a sequential
  test (SPRT) on paired games.  Expected games to a decision (α = β = 0.05):
  0.50→0.60 ≈ 130, 0.50→0.65 ≈ 57, 0.50→0.70 ≈ 31.  Cap tests at ~400 games
  and reject changes that don't show an effect unless they simplify code.
* Fixed matrix: maps (small/large, land, water) × factories, same for every
  candidate, one map held out to catch overfitting.

## 7. Research summary

* Strong competitive bots (UAlbertaBot, SAIDA, PurpleWave, Steamhammer) are
  rule-based managers around a shared world model; robustness came from clean
  separation and solid fundamentals, not learning.
* CherryPi's blackboard + intents and Ares-SC2's ordered "maneuver"
  behaviours are the patterns adopted above; CircuitAI (the Zero-K/BAR bot)
  is data-driven: roles and responses in JSON/AngelScript, one engine for
  several games.
* Engage/retreat in top bots uses cheap combat estimation (FAP-style
  simulation or Lanchester models) with commit timers to stop flip-flopping;
  such models can be fitted from logged fights.
* Tuning noisy constants: SPSA/CLOP (chess engines), racing (irace); micro
  constants on scripted scenarios.
* Strategy selection across games: bandits/epsilon-greedy over a few
  openings, pooled across maps; learned build switching needed millions of
  games — not an option here.
* Pitfalls: overfitting to one opponent/map, combat-model inaccuracy
  (terrain, shields, AoE, cloak in Zero-K), decision flapping, accepting
  noise from many small A/B tests, unit-name special cases creeping back.
