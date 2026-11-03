# HANDOFF — zk-ArenaAI / zkbot

Date: 2026-11-02 (updated; the previous handoff was 2026-11-01)  
Working directory: `C:/Users/Thoma/Documents/sources/zk-ArenaAI`

## 1. Goal and current status

Build a from-scratch Zero-K AI (`bots/zkbot`) that can beat CircuitAI at every
difficulty, starting with **CircuitAI:Normal**, then **CircuitAI:Hard**, then
**CircuitAI:Brutal**. Primary test map: **TitanDuel 2.2**. The user (an
experienced Zero-K player) also plays the bot and reviews it; their gameplay
guidance is a design requirement, see section 6b.

Current status:

- **CircuitAI:Normal**: consistently beaten (n28, n29).
- **Caveat (2026-11-03, h120): all win rates below were measured at 10–16x
  game speed, where CircuitAI is much weaker.** At 1x (human speed) the
  current bot wins ~1/16 against Hard (section 3). Retuning at 1x is the
  main open task.
- **CircuitAI:Hard** (at high game speed): beaten consistently. Committed build: `h93` 3/4,
  `h94` **4/4** (metal ratio vs CircuitAI 1.39 / 1.46 / 1.73 at 5 / 10 / 15
  min); 14/18 since the human-review changes (`h91`–`h94`). Before this
  session: 0 wins.
- Against a human (the user, tank factory): lost in ~7 min (`results/human1`,
  review -> section 6b.4) and again in 6.6 min (`results/human2`: 14 Kodachis
  killed all 36 Glaives, 42 builder flees, never more than 4 mexes; review ->
  section 6b.9).
- Weakest matchup historically: CircuitAI amphibious (`amph*`); improved by
  the role overrides but still worth watching.

Second layer (section 6b.7: raid squads v2, slow-ball engagement, overwhelm
attack, Phantoms + Gremlin spotters): `h100` **4/4** vs the same code with
the layer switched off (`h101`) 2/4.

Radar-blip identification + raid early warning (section 6b.8): `h102`
**4/4** (metal ratio 1.37 / 1.62 / 2.06) vs the same code with it off
(`h103`) 2/4 (1.36 / 1.39 / 1.40).

Next targets: stage-2 slow-ball play if amph balls keep winning, another human game, then CircuitAI:Brutal and other
maps (section 9).

## 2. Environment / references

- Zero-K portable:
  `C:/Users/Thoma/Downloads/zero-k-portable`
- Engine: `2026.07.04` (headless-capable)
- Zero-K: `v1.14.10.3`
- ControllerAI: `0.3`
- Python: `3.13.5`
- RecoilEngine source:
  `C:/Users/Thoma/Documents/sources/RecoilEngine`
- CircuitAI source:
  `C:/Users/Thoma/Documents/sources/RecoilEngine/AI/Skirmish/CircuitAI`
- Zero-K source:
  `C:\Users\Thoma\Documents\sources\Zero-K`
- zkscraper:
  `C:/Users/Thoma/Documents/sources/zkscraper`
- ControllerAI docs: `results/SKILL.md` (also `GET /SKILL.md` from a live team)
- Test maps list: `maps.md`

Useful game facts:

- TitanDuel 2.2 is 5120×5120 with 32 metal spots, income 2.0 each.
- Start is 350M / 350E.
- SW start seen at roughly `(307, 311)` in some games, `(4813, 4813)` in
  others depending on side. Do not assume a fixed corner.
- `staticradar` has 2100 radar radius and 800 LOS.
- Overdrive: metal gain is roughly `0.25 * sqrt(energy)`.
- Hidden storage is 10000; visible storage is `raw - 10000` (`effective_storage`).
- Commander is `dyntrainer_support_base` (12 buildpower, 232 build range,
  support chassis).
- Default factory: `factorycloak`; `factoryhover` if water fraction > 0.25 or
  start is below water.

## 3. Running tests

Headless at max speed; do not use windowed mode unless the user asks.

**Preferred (from 2026-11-03): `tools/prun.py`** — parallel games, paired
A/B arms, code snapshots (safe to edit `bots/zkbot` while a batch runs),
never kills other engines (a human game can run alongside).

```bash
python tools/prun.py h120 --games 4 --arm new= --arm off=noresign      # paired A/B, 8 games in parallel
python tools/prun.py h121 --games 4 --arm new= --arm base=     --code base=results/_base_afae691/bots/zkbot                        # vs an older version
#   (git archive afae691 bots/zkbot | tar -x -C results/_base_afae691)
python tools/prun.py g1 --games 1 --arm shield= --arm-env shield:ZKBOT_FACTORY=factoryshield ...
python tools/abstat.py results/h120-new results/h120-off                # win rates, CI, sign test
```

Each game gets its own slot (ControllerAI port 3100+4n, host port 8500+4n,
write dir `results/_slots/slotN`, file-locked so concurrent batches don't
collide). An engine commits 3.3–4.7 GB, so ~8–9 games fit in 48 GB; the
runner waits for 9 GB free commit before each start. prun passes
`-EngineAffinity` (all cores): the engine otherwise pins its main thread to
one fixed core, so all parallel games shared it (h114–h119 ran like that, at
~1–2x). With it, 8 parallel games run ~10x early on. A killed batch keeps
its results.csv (written after every game) and continues with `--resume`.
Output per arm:
`results/<tag>-<arm>/game-NNN` + `results.csv` (report.py/abstat.py read it).
`ZKBOT_PARAMS="resign.low_share=0.25"` overrides registered parameters;
`ZKBOT_FACTORY=factoryX` forces the starting factory.
`--enemy-factories cloak,shield,...` pins CircuitAI's starting factory per
game (CircuitAI option `disabledunits`, new `zk-startup.ps1 -CircuitOptions`).
**Matchups matter more than anything else:** pooled h100–h119, vs shield
15/15, cloak 13/16, hover 3/12, amph 3/9. Use pinned, stratified batches.
**Game speed matters most:** CircuitAI is much stronger at 1x (human
speed). h120, same code and pinned matchups side by side: 1/16 at 1x vs 9/16
uncapped (~10x), paired 0 vs 8, p = 0.01. CircuitAI's cumulative metal at
15 min: 26.7k at 1x vs 18.7k at max speed (ours 22.6k vs 26.2k, equal at
5 min). Our bot is frame-synchronous, so the change is CircuitAI's (threaded
work finishes later in game time at high speed). Historical 70–100% win
rates were at 10–16x against a weakened CircuitAI. Evaluate with
`--speed 1`; compare arms only at the same speed.

Offline checks (seconds, no engine): `python -m pytest tests -q` (catalog
roles/abilities, factory fallbacks, flag/param semantics, brain composition).

**Older:** `tools/run.sh` + `tools/report.py` (sequential).

```bash
bash tools/run.sh h93 Hard 8 "TitanDuel 2.2" 22   # tag diff games map wall-minutes
python tools/report.py results/h93                # per-game WIN/LOSS, metal, mex/builders/army
ZKBOT_FLAGS=staticmix bash tools/run.sh ...        # A/B an opt-out flag
ZKBOT_PY=pypy3 bash tools/run.sh ...               # other interpreter (not installed; not worth it)
```

`run.sh` kills stray engines first (so never run it while a human game is
open), uses `-AlternateSides` for >1 game and blocks until done. The CSV
`outcome` column is not ours-vs-theirs; trust `report.py`, which finds our
team from `script.txt`. A 22-min wall cap is a "timeout" (often a win we
could not finish). Games now take ~1.5–3 min wall each.

Human game: `.\zk-startup.ps1 ControllerAI Player -Map 'TitanDuel 2.2'
-NoDownload -OutDir 'results/humanN' -Client 'python bots\zkbot\main.py {team}'`.

Bot log lines worth grepping: `profile wall=` (engine/brain/query wall-time
split each minute), `spend` / `lost` / `lost_branch` / `dmg_from` /
`enemy_seen` ledgers (every ~3 min), `raid squad`, `army mode`, `burrow`.

Older manual command:

```bash
cd "C:/Users/Thoma/Documents/sources/zk-ArenaAI"
taskkill //IM spring-headless.exe //F >/dev/null 2>&1 || true
taskkill //IM spring.exe //F >/dev/null 2>&1 || true
sleep 1
rm -rf results/hNN
nohup timeout 520 powershell -NoProfile -ExecutionPolicy Bypass \
  -Command ".\zk-startup.ps1 ControllerAI CircuitAI:Hard -Headless \
  -Map 'TitanDuel 2.2' -NoDownload -TimeoutMinutes 6 \
  -OutDir 'results/hNN' -Client 'python bots\zkbot\main.py {team}'" \
  >/tmp/hNN.out 2>&1 &
```

Then wait 120–170 seconds and inspect:

```bash
grep "^\[.*t=" results/hNN/game-001/zkbot-team0.log | tail -5
cat results/hNN/summary.txt
cat results/hNN/results.csv
```

Logs are in `results/<tag>/game-001/zkbot-team0.log`. Headless runs about
3–9× faster than wall time depending on load and combat.

For multi-game checks use `-Games N -AlternateSides` and compare `summary.txt`.
Judge over several games, not one.

Pitfalls:

- `set_settings` at game start can make the engine exit; the bot only sends
  observation-rate changes after frame 2.
- Stray `spring-headless.exe` / `spring.exe` processes hold ports
  `3017`/`3018`. Kill them before starting.
- If a diagnostic client stops calling `finish_frame`, the sync match stalls.
- Shell `timeout` can leave the game running; check result logs instead of
  blindly re-running.
- Use the `write` tool for large files; shell heredocs on this repo can hit
  `ENAMETOOLONG` / unterminated-heredoc errors.
- `taskkill` sometimes reports failure for a child process even after the game
  ended; check `results.csv` and `tasklist`.

## 4. Repository / bot layout

`bots/zkbot/`:

- `main.py` — process entry, connects to ControllerAI, drives `Brain`.
- `client.py` — fast `websockets.sync` + `orjson` client. Important for
  throughput; do not regress to `websocket-client` or JSON.
- `brain.py` — orchestration, startup, frame loop, state; MRO currently:
  `ThreatMixin, IntelMixin, EconomyMixin, ResourceBuildMixin,
  FacilityBuildMixin, SupportBuildMixin, DefenseBuildMixin,
  ConstructionMixin, ProductionMixin, StrategyMixin, TacticsMixin,
  DefenseMixin, ScoutingMixin, AirMixin, BurrowMixin, RaidMixin,
  PhantomMixin, BlipMixin, RaiderMicroMixin, ResignMixin,
  UnitBehaviourMixin, ArmyPlannerMixin, EngagementLogMixin, MilitaryMixin`.
  Design and migration plan: `docs/ARCHITECTURE.md`.
- `params.py` — `FLAGS` (exact tokens of `ZKBOT_FLAGS`) and the `P()`
  parameter registry (`ZKBOT_PARAMS` overrides).
- `catalog.py` — unit roles/rosters/abilities/manual weapons from data
  (`data/abilities.json`, built by `tools/extract_abilities.py`).
- `models.py` — `UnitInfo`, `Task`.
- `units.py` — static `units.json` helpers and factory-role tables.
- `mapdata.py` — map spots, heightmap, water, terrain helpers.
- `intel.py` — observations, enemies/radar, commander tracking, threat,
  spot status, observation-rate control.
- `economy.py` — the build task board and `economy_step`.
- `build_resources.py`, `build_facilities.py`, `build_support.py`,
  `build_defense.py` — build task generators by domain.
- `construction.py` — placement, builder assignment, fallback/reclaim,
  commander confinement.
- `production.py` — factory queues, unit-role selection, Gremlin production.
- `military.py` — army mode, home guard, `military_step` (plan → behaviours),
  responders, attack/hunt/raid target choice, kiting, order helpers,
  commander safety.
- `armyplan.py` — `ArmyPlan` and its parts (Posture, CommanderFocus,
  SearchPlan, SlowBallPlan, RaidPlan, MainGroup), one planner method each.
- `behaviours.py` — per-unit priority list `UNIT_BEHAVIOURS` (bomb, retreat,
  air, aa_escort, home_guard, raider_micro, respond, focus_commander,
  search, raider_squad, phantom, manual_fire, kite, ball_gather,
  outranged, local_target, regroup, defend, attack, rally).
- `engagements.py` — passive fight logging (training data) and enemy-death
  inference from EVENT_ENEMY_DESTROYED.
- `strategy.py` — sector clustering, terrain scoring, sector threat/strength,
  expansion/defense/attack/rally decisions.
- `tactics.py` — composition counters, retreat decisions, local target
  selection, raid group target selection.
- `defense.py` — defence intelligence and spending model, damage/death
  ledgers (dedup by `(uid, name)`), `desired_aa_count`.
- `scouting.py` — Gremlin forward observers.
- `air.py` — air factory play: Swift scouting with boost, Owl caps, bomber /
  combat-air budget.
- `burrow.py` — terraform contested extractor spots ~55 elmos down
  (`CMD_TERRAFORM_INTERNAL=39801`, `CMD_LEVEL=39736`) before building.
- `raiding.py` — persistent raid squads (join / merge / combine for a
  prize / route via fresh mexes and builders), slow-ball detection and
  commit/gather decision.
- `snipers.py` — Phantom quota, fire-at-radar off, volley targeting,
  evade/fire/approach/hold orders, Gremlin spotter ring points.
- `blips.py` — radar blip identification (speed + enemy factories) and
  raid early warning from blip/enemy trajectories.
- `raidmicro.py` — raider-vs-raider micro: no chasing, evade/detour
  around enemy raider groups, intercept at the predicted target, pincer,
  minimum offensive raider squad.
- `resign.py` — resign (self-destruct everything) when extraction income
  collapses.
- `threat.py` — weapon data, `combat_power`, `weapon_range`, `local_balance`.
- `data/units.json` — 615 unit definitions extracted from the live game;
  `data/weapons.json` — weapon stats (`tools/extract_weapons.py`).
- `units.ROLE_OVERRIDES` — roles for units that `units.json` leaves as
  generic `combat` (Buoy, Felon, Sumo, Crab, ...).

`tools/`: `prun.py` (parallel paired batches), `abstat.py` (A/B stats,
per enemy factory), `run.sh` (old sequential runner), `report.py` (batch
report), `demostats.py` (demo reader), `extract_weapons.py`,
`extract_abilities.py`, `catalog_check.py`.

Zero-K source of truth for command IDs: the rapid pool in the portable
install (`packages/*.sdp` → `pool/xx/<md5>.gz`, e.g.
`luarules/configs/customcmds.lua`: `DONT_FIRE_AT_RADAR = 38372`, param 1 =
don't fire at radar).

`results/` is gitignored. `results/SKILL.md` is a local copy of ControllerAI
docs, not part of the bot package.

## 5. Existing architecture and completed behaviours

### Commander / opening
- Commander `dyntrainer_support_base`.
- Factory plop at frame 0 (`factorycloak`, or hover on water).
- Commander is now **confined to home economy/defence**:
  - only `energy`, `fusion`, `storage`, `radar`, `defense` tasks,
  - task limit 700 elmo from base,
  - if farther than 900, it returns home,
  - commander retreats if an enemy is within ~900,
  - never wanders after distant wreckage.
- This fixed a verified Hard loss where the commander died at t≈487 at
  `(2951, 2207)` far from base. After the fix, `h55` survived to ~2110s.

### Economy
- Mex expansion, energy ring placements, storage, pylon/overdrive linking.
- Energy task completion integrity: only buildings first seen after task
  creation can complete energy/fusion; 50-elmo match radius for packed rings.
- Reclaiming wrecks, factory assist, caretakers.
- Sector-aware mex priority: threatened/enemy/frontier sectors penalized;
  terrain defense and income bonuses.
- Late-game second/third/fourth factories; air factory under conditions.
- Builders are limited and must survive; builder count target is roughly
  3–6 depending on army/time/income.

### Military
- Global army modes: `gather`, `defend`, `attack`.
- Home guard keeps roughly 25% (2–10) of army at home in attack mode.
- Raider squads share one group target and stage before attacking; no
  one-at-a-time raider trickle.
- Counter-aware target selection and retreat:
  - cheap raiders may trade themselves;
  - expensive units retreat along safe paths;
  - no retreat if the local fight is clearly won.
- Enemy commander tracking, out-of-LOS attack-ground at last known position.
- Focus fire, army group anchors, frontier rally points.
- Air branch exists (`factoryplane`/`factorygunship`) for hunting/scouting.

### Sectors
- `strategy.py` clusters metal spots with union-find; radius
  `max(420, min(680, diag * 0.085))`.
- Adds base/enemy anchors when no nearby cluster.
- Scores terrain height/variance/slope/water.
- Tracks ownership, threat, our/enemy strength, buildings, frontier.
- Used for expansion scoring, defense anchors, attack target, rally point.
- State updates about every 20 frames.

### Performance
- Adaptive `block_n_frames`: normally 6, 2 in combat with 1800-frame
  hysteresis.
- Frame throttles for economy/military/spot/summary queries.
- Fast client transport is critical.

## 6. Latest session work — requested AA / defence / radar / Gremlin changes

### 6.1 New `defense.py`

Tracks:

- Real aircraft:
  - `_is_real_aircraft()` excludes commander drones (`drone*`) and
    `planelightscout` (the `staticradar` morph target), plus fake helper units.
  - `note_air_contact()` sets `air_seen_frame`; `aircraft_seen()` is historical.
  - AA decisions use this flag, not just current contacts.
- Damage events:
  - `read_battle_events()` consumes ControllerAI observation events:
    - topic `9` = `EVENT_UNIT_DAMAGED`
    - topic `10` = `EVENT_UNIT_DESTROYED`
  - Tracks per-sector attack pressure for units/mexes/turrets.
  - Tracks recent attacker max weapon range, globally and per sector.
  - Tracks enemy artillery from:
    - current/remembered scouted `arty`/`heavyarty`,
    - unattributed non-paralyzer turret damage,
    - known artillery damage.
  - **Paralyzer/disarm damage is excluded** from artillery inference.
  - Unknown turret damage is treated as possible artillery; steep downward
    impact direction strengthens that classification.
- Spending model:
  - New units are recorded as `defense` (static defences) or `mobile`
    (mobile combat/builders) in `_defense_spend_records`.
  - `desired_defense_share()`:
    - base ~0.07,
    - + attack pressure (capped +0.18),
    - + low-HP/retreating strain (capped +0.20),
    - + artillery estimate (capped +0.16),
    - + home threat (+0.05),
    - clamped to 0.04–0.52.
  - `defense_budget_ok()` compares recent/pending defense metal with desired
    share; urgent tasks get a larger allowance.
  - `defense_step()` logs every ~600 frames:
    ```text
    defense turrets=... spend=... desired=... attack=... strain=... arty=... air=... gremlins=...
    ```
- Turret selection:
  - `defense_unit_for(pos, aa=False)` chooses the cheapest turret whose range
    is at least the recent attacker range near `pos`.
  - Default (no attack-range evidence) is 500, i.e. Picket rather than Lotus.
  - Candidate order: Lotus 460, Gauss 560 (dominated by Picket), Picket 610,
    Stinger 620, Sunlance 750, anti-heavy 1200.
  - AA candidates: AA laser 1000, flak 1000, far AA 1800, heavy AA 2400;
    only if aircraft has been seen.
  - Anti-heavy is limited to one base-line turret and is **not** built on
    forward mexes or front tasks; if only anti-heavy can outrange an attacker,
    the mex is left to the mobile army.

### 6.2 AA gating

- AA is only built if a real aircraft has been seen.
- `tactics.counter_role()` only requests AA when historical aircraft seen is
  true (plus real composition air).
- `production.py` only requests AA after aircraft seen.
- `economy.ensure_defense_tasks()` only creates an AA turret after
  `aircraft_seen()`.
- No commander drone or radar-morph scout triggers AA.
- **Exception:** Gremlin (`cloakaa`) is queued as a dedicated artillery scout
  even without aircraft; it is not counted as normal AA and is controlled by
  `scouting.py`.

### 6.3 Turret spending / placement changes in `economy.py`

- `ensure_mex_defense_tasks()`:
  - starts at t=150s / metal income ≥8,
  - uses dynamic turret choice,
  - cap 2–8 from desired defense share,
  - skips far mexes before 360s,
  - skips anti-heavy,
  - logs `mexdef <Name> at ... req=... urgent=...`.
- `ensure_front_defense_tasks()`:
  - after 900s and army ≥20,
  - dynamic type,
  - cap from desired share,
  - skips anti-heavy,
  - logs `frontdef ...`.
- `ensure_defense_tasks()`:
  - ignores remote threat positions more than 1700 elmo from base,
  - fortifies owned threatened sectors only if <2200 elmo from base,
  - defense cap derived from desired share, with early caps of 1/2/4,
  - dynamic turret type,
  - one anti-heavy maximum,
  - logs `defense turret ... req=... urgent=... share=.../desired`.
- `defense_pending_cost()` ignores unassigned defense tasks older than 1800
  frames so a dead builder’s abandoned far task cannot block a replacement
  near home.
- `HANDOFF` note: despite these gates, Hard builders still die and the base
  collapses; static defence is selected correctly but not necessarily
  completed.

### 6.4 Radar coverage

- `ensure_radar_tasks()`:
  - target count `3 + min(3, len(arty) // 2)`,
  - anchors: base, rally point, front owned metal, artillery group center,
    attack sector,
  - uses ~1900-elmo coverage spacing for the 2100 radar radius,
  - avoids anchors with enemy within 500,
  - avoids sending a builder more than 1900 from base unless the army is
    within 700 of that anchor,
  - radar priority raised to 1760 when we own artillery.
- Artillery can attack radar-only blips:
  - `tactics.best_local_target()` includes radar contacts for `arty` /
    `heavyarty`.
  - `military.py` issues `ATTACK-ground` (custom `cmdId 20`) at the blip
    position instead of an invalid target id.
  - Artillery local reach uses actual weapon range × 0.95, not the old 520
    elmo cutoff.
- Tests showed radar counts up to 4 in Normal (`n29`), but Hard often loses
  radar with builders.

### 6.5 Gremlin forward observers — new `scouting.py`

- Only produced by `factorycloak`.
- Requires our own artillery to exist (`artillery_units()`), then targets:
  - 1 Gremlin if <4 own artillery,
  - 2 if ≥4.
- `next_unit_for_factory()` queues `cloakaa` and increments
  `gremlin_pending`; `DefenseMixin.note_new_units()` detects the new
  `cloakaa` and calls `note_new_gremlin()`.
- Excluded from normal `self.army`:
  `read_units()` subtracts `gremlin_scouts`.
- Holds fire permanently:
  ```python
  {"type": "custom", "unitId": uid, "cmdId": 45, "params": [0.0]}
  ```
  (`CMD_FIRE_STATE=45`, `FIRESTATE_HOLDFIRE=0`).
- Keeps ~420 elmo from enemies (`GREMLIN_DECLOAK_SAFE`) to stay cloaked.
- Moves toward enemy economy/attack sector with move orders, not fight.
- Withdraws when a dangerous contact gets close; logs at most every 300
  frames per Gremlin.
- First scout order logs:
  ```text
  gremlin N scouting (x, z) with fire-at-will off
  ```

### 6.6 Tactics / production changes

- `tactics.py` ignores drone/radar-morph aircraft in enemy `air` composition;
  they count as `other`.
- `counter_role()` uses a balanced quota (`raid/skirm/riot/assault/arty`,
  typically 24/24/20/20/8%) and only adds AA when aircraft has been seen.
- `production.py` no longer uses `any(u.is_air ...)` for AA decisions.
- Gremlin production is checked before emergency AA/combat production.
- Air factory roles still hunt/scout; not a current focus.

## 6b. Session 2026-11-02 — Hard beaten

### 6b.1 User guidance (design requirements)

No commander morph. Second factory at ~40–45 metal/s or 20–25 min (usually
plane, else gunship); no 3 factories by 20 min. Swifts scout (1 s boost
through AA); Owls are mid/late radar/LOS for artillery. Burrow contested mexes
~50–60 elmos. Early game: many Glaives and aggressive raiding; gradual riot
transition; skirmishers only against slow kitable units; LLT + occasional
Picket vs early tanks; commander expands and fights small raider groups; don't
engage units that outrange you; raid squads retarget off defended spots and
merge when close / for a valuable target. Phantoms (mid game): cloaked,
fragile, stay cloaked when firing, 2 volleys kill an unupgraded commander,
good vs slow heavies; need **LOS** (not just radar) for mobile targets, so
pair them with Gremlin spotters (ceil(phantoms/2), at least 1); turn
fire-at-radar **off** for Phantoms.

### 6b.2 Economy / build fixes

- Energy target follows metal income (construction drains E 1:1); energy
  priority rises when metal floats because energy is short.
- Caretakers placed on a ring within build range of the factory (not via
  `closest_build_site`, which drifted out of range); dynamic priority.
- Second factory rule per 6b.1; also when metal floats for ~90 s.
- Lost factories are rebuilt at rotating anchors (two fixed anchors were both
  blocked → 30 min of capped metal).
- Burrowing of contested mexes (`burrow.py`).
- Enemy base guess ignores spawn boxes near our own base (side-key mismatch
  made "enemy base" = our base in about half the swapped games); a roaming
  commander only stands in for the base until a factory is seen.

### 6b.3 Military

- Reactor (singularity/fusion) blast avoidance for targets and attack points.
- AA quota from recent air contacts (`desired_aa_count`) and AA escorts.
- `hunt_target`: with overwhelming power and no attack target, go for any
  known ground enemy, else sweep unvisited points (army used to sit at rally
  for 25 min at 20x power).
- Dynamic retreat threshold (`tactics.retreat_threshold`): base 30%/45%
  scaled by local balance, depth in enemy territory, unit cost, global army
  balance. A flat lower threshold (`lowretreat`, removed) lost 5/7.

### 6b.4 After the human review (all default on; opt-out flags)

| Change | Flag to disable |
|---|---|
| Phase-based unit mix (≈60% raiders early → mid mix by 15 min); riots scale with enemy raiders seen; skirms only after enemy riots/assaults seen; no early arty | `staticmix` |
| Emergency production builds raiders, riots only vs 4+ raiders; dead cons rebuilt first (≥3 builders) | `staticmix` |
| Commander takes mexes up to ~1600 elmos early (+mex score), fights raider groups it outpowers | `staticcomm` |
| All raiders raid in up to 3 squads; targets judged by defending metal value (riots/turrets ×1.5) vs half the squad's value; too-defended targets blacklisted 80 s and retargeted | `oldraid` |
| Units outranged by a stronger enemy step out of range unless locally ≥1.5x | `nooutrange` |
| Default turret without attacker evidence: LLT, every 3rd a Picket | `oldturret` |
| Outranged step-back only vs stronger (≥1.3x) riot/assault/heavy units | `nooutrange` |
| Static AA count scales with recent air contacts and air damage (max 6), placed at air-damage hotspots; more mobile AA under air damage | `oldaa` |

### 6b.5 Speed

- `block_n_frames` 6 in combat, 10 when calm (was 2 most of the game);
  `ZKBOT_BLOCK=fight,calm` overrides.
- `client.query` caches `elevation_at` (300 frames), `can_build_at` /
  `closest_build_site` (90 frames); cache cleared when a build is rejected.
- Builder idle checks once per second per builder.
- Profile: the engine is ~77–82% of wall time, bot compute 10–16%, so PyPy
  was not pursued (it also lacks `orjson`). Whole games now take 77–176 s wall.

### 6b.6 Other opt-in flags kept for experiments

`focus`, `sticky`, `buildfirst`, `defregroup`, `mexdefprio` (each tested
worse or neutral), `burrowall`, `noburrow`, `nokite`, `noregroup`,
`oldretreat`, `staticretreat`, `nohold`, `noresponse`, `nohunt`;
`ZKBOT_SECOND` pins the second factory.

### 6b.7 Second layer (all default on; opt-out flags)

| Change | Flag to disable |
|---|---|
| Raid squads v2 (`raiding.py`): persistent squads, raiders join the nearest squad within 1400, squads within 700 merge (≤1.5x cap), cap 4/6/8 by game time, max 3 squads | `oldsquads` |
| Combine squads for a prize (commander worth 1500, else nearest factory 1000) that no single squad may take: gather 750 out, strike when all gathered or after 30 s | `nocombine` |
| Route raid squads via a fresh mex (< 2 min) or a mobile con when the detour ≤ min(1.25x, +600) | `noroute` |
| Slow ball: densest cluster of slow (< 60) enemy combat units, ≥ 800 metal, containing a ≥ 600-cost heavy, ≥ 50% of nearby enemy army value; raiders join the main group; gather out of range and commit when our metal value ≥ 1.0x (0.7x once committed); mix 40% raid / 35% skirm | `noball` (`anyball` drops the heavy requirement) |
| Overwhelm: army attacks when ≥ 40 combat units, ≥ 4x the enemy's power and no near threat | `nooverwhelm` |
| Phantoms (`snipers.py`): cloak factory, ≥ 10 min, ≥ 25 income, after ≥ 3 enemy heavies or a ball; fire-at-radar off (38372 [1]); volleys on LOS targets (commander first with ≥ 2), keep enemies outside 335; Gremlin spotters ceil(n/2) circle the target at 330 | `nophantom` |

The broad slow-ball rule (any slow cluster) went 8/12 vs baseline 10/12, so
it was narrowed to balls with a heavy. Phantoms are rare in practice (only
built after 3+ enemy heavies); they fired in h98 but are not yet proven.

### 6b.8 Radar blips (all default on; opt-out flags)

Blip fields (logged with `ZKBOT_FLAGS=blipdump`): `allyTeam`, `inLos`,
`isRadarBlip`, `pos`, `vel` (exact, elmos/frame). `defId` / `name` only when
the engine already knows the type from earlier LOS; never-seen blips used to
be `def0` contacts that counted for nothing.

| Change | Flag to disable |
|---|---|
| Identify unknown moving blips: sustained top speed (min over 2 s, max over life) ≤ candidate's speed; candidates = mobile build options of the enemy's known/inferred factories, air vs ground by height above ground; weighted by times seen, speed closeness, cheapness; adopt the top type of the dominant role (≥ 45%). Checked on LOS entry (`blip check ... acc a/b role c/b` in the log) | `noblipid` |
| Raid early warning: seen/guessed raiders and ground blips > 85 elmos/s whose straight track passes within 450 of our mex or constructor in 4/8/12 s; constructors there step aside (eta ≤ 10 s, local power < 1.2x), responders go to the warned asset | `noblipwarn` |
| Guessed (never-seen) units count in `seen_enemy_roles` for the unit-mix counters | `noblipcount` |

Accuracy (h102, logged sample): 9/9, 8/9, 431/600 exact (487/600 role) in a
long Cloakbot game, 31/60 vs Spider + Tank. Main miss: slow blips guessed as
Flea (`spiderscout`) once many Fleas were seen; the seen-count prior swamps
the speed evidence when the blip moves slowly. Possible fix: cap the prior,
or require a higher sustained speed before trusting a fast-unit guess.

### 6b.9 Raider micro after the second human review (default on)

User review: a fleeing raider shoots its chasers, so never chase raiders in
their range; Glaives are faster than Kodachis (which outrange them), so path
around them on offence and intercept at their estimated target on defence;
pincer Kodachis with several raid groups (close range favours Glaives);
always keep at least one raid squad on offence after ~2 min (the user never
saw our raiders hit their mexes).

| Change | Flag to disable |
|---|---|
| Enemy raider groups (cluster 450, seen in last 5 s, power/range/velocity). A raider near a group it cannot beat locally (1.3x) or that outranges it moves away (fire at will) toward its destination; a stronger raider never follows a group moving away | `oldmicro` |
| Offence: waypoint around the first enemy raider group within range+350 of the path | `oldmicro` |
| Defence: hold 120 short of the asset the group is heading for (early-warning trajectory, else our nearest mex/con); collapse once the group is within 380 of the asset and local power ≥ 0.9x | `oldmicro` |
| Pincer: groups of ≥ 2 near our base/assets; ≥ 4 raiders with ≥ 1.4x power split by side, stage at ±70° at range+260, strike when in place or after 15 s; abort < 3 raiders or < 0.9x | `nopincer` |
| Offensive reserve: from 2 min, max(squad size, 35%) raiders keep raid squads even in defend mode | `nooffense` |

Results: h104+h106 6/8 vs `oldmicro` h105+h107 7/8 (noise), metal ratio at
15 min 1.75 / 2.02 vs 1.47 / 1.30. CircuitAI rarely plays tank raiders, so
the Kodachi case needs the next human game. Pincers rarely trigger vs
CircuitAI.

### 6b.10 Emergency turrets (opt-in: `emergency`)

User idea: when a builder flees, check whether its refuge has enough
turrets / nearby raiders to beat the raid; if not, build an emergency turret
at high priority. Implemented in `construction.emergency_turret_check`
(called from `retreat_builder`): raider-led threats ≤ 900 power only, refuge
near base (1800) or one of our extractors, turret at the refuge pulled 100
toward base, priority 1900, max 2 builders, one pending, ≥ 15 s apart,
bypasses the defence budget.

- First version (turret toward the danger, any threat, anywhere): h108 1/4,
  builder flees 232–980 per game (builders walked back into the raid to
  build it; 19–25 emergency turrets per game).
- Narrowed version: h110 3/4 + h111 2/4 = 5/8 vs off (h109 4/4 + h112 3/4)
  7/8, lower metal ratio, so it is **off by default**. CircuitAI does not
  raid like a human; worth enabling for the next human game.

### 6b.11 Resign (default on; `noresign`)

User rule: resign when metal extraction income (2-min average, reclaim and
overdrive excluded) is below 30% of its peak 2-min average for 1–2 min.
Extraction = summed `income` of the spots holding our finished extractors.
There is no resign command; Zero-K disables its "self-D everything ⇒ resign"
blocker in 1v1, so the bot self-destructs every unit and then stops issuing
orders (game ends normally, scored as a loss). Guards: not before 6 min,
peak ≥ 4 m/s, low for 90 s, and not while our army power ≥ 1.5x the
enemy estimate (h113 g2 resigned at 67 min with 5x the enemy's army power
because the map's extractors were gone; the guard was added after that).
`ZKBOT_RESIGN="share,frames,min_s,min_peak"` overrides for testing (`r1`:
forced resign at 2:22 ended the game as a CircuitAI win).

## 6c. Session 2026-11-03 — redesign groundwork

Plan and rationale: **`docs/ARCHITECTURE.md`** (current-state audit, target
layers, migration table, evaluation method, research summary).

* `tools/prun.py`, `tools/abstat.py` — parallel paired testing (section 3).
* `catalog.py` — unit knowledge from data: stat roles (`stat_role`), the
  game's description roles (`desc_roles`, dual roles kept in `roles()`),
  ability tags and manual-fire weapons from `data/abilities.json`
  (`tools/extract_abilities.py` reads the Zero-K unit Lua + scripts:
  manualfire/D-gun stats, cloak, area_cloak, shield, jump, emp, disarm, slow,
  fire, gravity, capture, torpedo, stockpile, kamikaze, drones, morph, ...).
  `units.role_of` uses it for ground units the hand tables call "combat";
  `units.factory_unit` uses it for roles a factory table lacks (opt-out
  `noinfer`). `tools/catalog_check.py`: 100/108 agree with hand roles.
* `armyplan.py` — `ArmyPlan` composed of `Posture`, `CommanderFocus`,
  `SearchPlan`, `SlowBallPlan`, `RaidPlan`, `MainGroup` (+ responders), each
  from one planner method; `behaviours.py` — the per-unit priority list
  (`UNIT_BEHAVIOURS`), one small method per branch. `military_step` is now
  ~25 lines (was ~500); logic unchanged.
* New behaviour `manual_fire` (opt-out `nomanualfire`): any army unit with
  a manual-fire weapon (Dante napalm, Scorpion, Bantha, Detriment, Reef)
  fires it at the best cluster / most valuable target in range.
* `params.py` — `FLAGS` exact-token set used by every module (the old
  string `"emergency" in "noemergency"` was True), `P()` parameter registry
  with ranges + `ZKBOT_PARAMS` overrides; config logged at game start.
* `ZKBOT_FACTORY` override for factory generalisation tests.
* `economy.py` split by domain: it keeps the task board (`economy_step`,
  `add_task`, `finish_task`, ...); task generators moved unchanged to
  `build_resources.py`, `build_facilities.py`, `build_support.py`,
  `build_defense.py` (mixins placed right after EconomyMixin in the MRO).
* `engagements.py` (passive, opt-out `noenglog`): every fight is logged as
  one JSON line in `engagements-team<N>.jsonl` (both sides' cost, HP,
  power, DPS, range, speed, roles, turrets, height; losses 20 s later) as
  training data for a combat-outcome model. ControllerAI forwards
  EVENT_ENEMY_DESTROYED (topic 18) **without a unit id**, so deaths are
  matched to enemies that vanished (not into radar) in the same
  observation. This also fixes the metrics column `enemy_killed_value`,
  which was always 0. `python tools/combat_model.py` fits/compares outcome
  models on all logged fights (needs ~100+ games before it means anything).

## 7. Test evidence from latest changes

### Hard, session 2026-11-02 (TitanDuel, alternating sides)

| Run | Result | Notes |
|---|---|---|
| h72–h78 | 11/19 | energy fix, caretaker ring |
| h79–h85 | 6/31 | h79 defend-regroup + mexdef priority hurt → now opt-in |
| h86 | 3/6 | 2 timeouts (factory never rebuilt; army idle at rally) |
| h87 | 5/6 | hunt target, factory anchors, enemy-base guess |
| h88 | 3/6 | block 6/10 + query cache; 2 amph losses |
| h89 | 2/7 | `lowretreat` (dropped) |
| h91 | 6/6 | role overrides, dynamic retreat, review changes |
| h92 | 1/4 | lost 3 with 1.4–3x metal: broad `outranged` step-back vs Buoys, heavy bombers vs 1 AA turret |
| h93 | 3/4 | `outranged` limited to stronger riots/assaults; static AA scales with air threat |
| h94 | 4/4 | same build (committed `ae3c24c`) |
| h95 | 2/4 | + raid squads v2, broad slow ball |
| h96 | 4/4 | + overwhelm attack |
| h97 | 2/4 | Reaper-heavy tank game 25 kills / 405 deaths |
| h98 | 2/4 | + Phantoms (fired in 2 games) |
| h99 | 3/4 | baseline: `noball,oldsquads,nooverwhelm,nophantom` |
| h100 | 4/4 | slow ball narrowed to balls with a heavy (committed) |
| h101 | 2/4 | baseline flags again (2 losses, one amph) |
| h102 | 4/4 | + radar-blip identification and raid early warning (committed) |
| h103 | 2/4 | same with `noblipid,noblipwarn,noblipcount` |
| human2 | loss | vs the user (tank), 6.6 min |
| h104 | 3/4 | raider micro (section 6b.9); loss = amph ball |
| h105 | 4/4 | `oldmicro` |
| h106 | 3/4 | micro + pincer fix (committed); loss vs vehicles |
| h107 | 3/4 | `oldmicro` |
| h108 | 1/4 | emergency turrets v1 (dropped) |
| h109 | 4/4 | `noemergency` |
| h110 | 3/4 | emergency turrets narrowed |
| h111 | 2/4 | emergency turrets narrowed |
| h112 | 3/4 | `noemergency` → emergency made opt-in |
| h113 | 2/4 | resign on: resigned in both losses (g2 at 5x army power → army guard added) |
| h114 | 4/8 | resign army guard; parallel prun, CPU-starved (~1–2x) |
| h115 | 1/4 vs 2/4 | refactor (new) vs afae691 (base), unpinned |
| h116 | 3/16 vs 6/16 | refactor vs base, pinned enemy factories, starved ~1–2x |
| h117 | 6/8 | base afae691, 2 parallel (~5–12x) |
| h118 | 4/16 vs 2/16 | refactor vs base (pooled with h116: 7/32 vs 8/32, no regression) |
| h119 | max 5/16 | speed test, invalid: all games shared one core (affinity bug) |
| h120 | **1/16 at 1x vs 9/16 at ~10x** | same code and matchups; paired 0 vs 8, p = 0.01 |

Previous session's evidence:

### Normal wins

- `n28`: win, 117s wall, ~1090s sim. Logs:
  - `gremlins=2`
  - `radar=2`
  - `defense turrets=8`
  - real aircraft seen at ~t1070; `counter aa` appeared only then.
- `n29`: win, 115s wall, ~1030s sim. Logs:
  - `radar=4`
  - `defense turret turretlaser ... req=185`
  - `mexdef turretmissile ... req=0`
  - `mexdef turretmissile ... req=530`
  - `gremlin 18323 scouting (2019, 2020) with fire-at-will off`
  - no AA before real aircraft.

### Hard losses

- `h58`: loss ~1320s. Turret choices included `req=530` Picket and
  `req=610` Picket; several builders died en route to defense tasks.
- `h59`: loss ~1340s. No demo captured; base still collapsed; turret count
  fell to 0 while desired defense share was high.
- `h60`: loss ~1220s. Inferred artillery estimate rose to ~9 from
  unattributed turret damage; anti-heavy was chosen for extractors before the
  anti-heavy skip was added.
- `h61`: loss ~1240s. After the anti-heavy skip, anti-heavy only appeared on
  the base line (`defense turret turretantiheavy ... req=980`), while mex/front
  tasks used Lotus/Picket. No AA was built (no real aircraft). Builders and
  army still collapsed.

## 8. Known issues / traps to investigate next

New (2026-11-02):

- Editing bot files while a batch runs changes the code for every game that
  starts afterwards; finish or stop batches before editing.
- `run.sh` kills all spring processes at start, including a human game.
- Long games (40–90 min) still show 20–50 builder deaths and `retreating` as
  the largest loss branch; Hard losses are now mostly these grinds.
- `report.py` marks games with an empty demo as LOSS "(no demo)": those are
  wall-limit timeouts, check `results.csv`.

From the previous session (several now addressed):

1. **Hard army attrition**: current Hard games survive 20+ minutes but lose
   the army and then the base. Investigate:
   - army grouping / focus fire / target selection against Hard’s skirm/assault
     ball;
   - builder survival and repair;
   - whether home guard / defense anchor pulls too many units away;
   - economy rebuilding after mex loss.
2. **Builder survival**: many defense/radar tasks fail because the only
   builder dies en route. Threat anchors are now capped near base, but forward
   mex/front tasks still cost builders.
3. **Turret budget vs actual completion**: `defense spend` is often low while
   `desired` is high; turrets are selected correctly but not completed. This
   is not merely a budget bug.
4. **Anti-heavy limit**: one base anti-heavy is allowed; forward mex/front
   tasks skip anti-heavy. Re-check if this is too restrictive versus enemy
   artillery.
5. **Artillery inference**: all unattributed non-paralyzer turret damage is
   treated as possible artillery. Watch for false positives (cloaked raiders,
   shield/disarm damage where `paralyzer` is not set).
6. **AA no-build case**: in Normal `n28`, real aircraft was seen late but no
   static AA completed before game end. This is not a gating failure; verify
   the mobile AA queue if this matters.
7. **Gremlin utility**: Gremlins survive but often just withdraw; measure
   whether they actually provide LOS for artillery or whether a different
   forward position/path is needed.
8. **`read_battle_events` and stride**: called after the `stride` early
   return in `brain.on_frame`. Default stride is 1, so normal tests capture
   events. If `ZKBOT_STRIDE > 1`, events on skipped frames are lost. Fix before
   relying on stride.
9. **Radar coverage**: radar target count/anchors are implemented, but Hard
   frequently has only 1–2 radars due builder deaths. Verify coverage near
   artillery targets, not just tower count.
10. **`set_settings` / stray processes**: see test pitfalls above.

## 9. Suggested next work

Current priority order (2026-11-02):

Items 1, 1b (stage 1) and 2 are implemented (section 6b.7), item 3 too
(section 6b.8; not yet: blip-based slow-push and artillery warnings); the
descriptions are kept as design notes.

1. **Raid-squad merging** (agreed with the user): form squads by position;
   merge squads within ~600–800 elmos; combine squads for a valuable target
   (commander, factory, mex cluster) that no single squad may take but their
   combined value may; gather outside the defence range, then strike together.
   **Routing:** if a squad's path to its target passes near one of our
   builders or a recently built mex (detour ≤ ~25% or ~600 elmos), go
   through it first, so a raid squad is likely near new expansions when an
   enemy raid arrives; clear the waypoint once the squad centre is within ~350.
1b. **Slow-ball engagement** (user: large raider + Ronin groups beat Buoy +
   Grizzly; a Grizzly needs ~30 raiders mixed with Ronins to die before it
   retreats): detect enemy armies dominated by slow units (< ~60 speed);
   then raiders mostly join the main group (only extras raid), Ronins too;
   gather ~650 out of range and commit only when group metal value ≥ ~1.0×
   the ball's (Ronins weighted up); chase a retreating ball; mix ≈ 40% raid /
   35% skirm, fewer riots. The `combat_power` sqrt(dps·hp) score overrates
   small units here (1.3× power ⇒ only ~13 Glaives vs a Grizzly).
   **If that is not better, stage 2 (user):** skirmishers, artillery and
   Phantoms kite the ball first and focus its riot units; raiders wait out
   of range and go in only when they counter what is left and most of the
   ball's riot value is dead/heavily damaged. Sumo and Dante (riot/hybrid
   riot heavies) are never a raider target.
2. **Phantoms** (cloak factory, from ~10 min once enemy riots/slow assaults
   have been seen; count scaled by those and by metal income):
   fire-at-radar off (`custom cmdId 38372, params [1]`), near max range (700)
   behind the army, keep enemies outside 155 (decloak), synchronized volleys
   on one target, commander first when ≥2 Phantoms reach it, then slow heavies.
   Gremlin spotters: `ceil(phantoms/2)`, at least one; in front, outside
   decloak radius, constantly moving between ring points; hold fire.
3. **Radar-blip raider early warning**: estimate blip speed from successive
   positions (first log raw blip fields to see whether `defId` / `vel` are
   present); ground blips > ~85 elmos/s are raiders (Glaive 115, Kodachi 106);
   group co-moving blips, extrapolate 10–15 s; if the path passes within ~400
   of our mex/builder, raise an early threat there: nearest raid squad /
   responders intercept ahead of it, builders step aside, commander responds
   if inside its radius and the raid is small.
   **Generalize to blip identification:** candidates = build options
   (`units.json` `buildOptions`) of the enemy's known factories / inferred
   faction whose top `speed` ≥ the blip's max sustained speed (2-s windows,
   radar error jitters positions), ranked by closeness and by types already
   seen from that enemy. Feed weighted candidates into the composition
   counters (riots/skirms/anti-heavy before contact), the threat/local-balance
   estimates, slow-push warnings and artillery detection (e.g. tankheavyarty
   38). Log guess vs. real type when a blip enters LOS to measure accuracy.
4. Play the user again (tank opener) and take the next review.
5. CircuitAI:Brutal; then other maps (`maps.md`).
6. Builder survival in long games (20–50 builder deaths), friendly fire
   (`?own` damage up to 40k).

Older list (2026-11-01, mostly done):

1. Reproduce a Hard loss with current code (`h62+`) and analyze army
   composition/casualties using zkscraper or log summaries.
2. Focus on **army control and builder survival**, not more turret tuning:
   - keep a coherent ball, focus fire, retreat damaged expensive units;
   - protect builders with escorts or a builder-safe rear rally;
   - rebuild mexes/energy after attacks.
3. Add repair/rebuild logic for damaged turrets and mexes; use strain data.
4. Evaluate whether `desired_defense_share` should cap lower (e.g. 0.40) to
   leave metal for mobile units; H60/H61 showed high desired shares while the
   army disappeared.
5. Once Hard is stable, run multi-game Normal/Hard checks:
   `-Games 3 -AlternateSides`, compare `summary.txt`.
6. Only after Hard is consistently beaten, move to **CircuitAI:Brutal**.
   Hard vs Brutal differs mainly in `economy.json` (`buildpower`, `mex_max`,
   `build_delay`).
7. Test other maps from `maps.md` once TitanDuel is stable.

## 10. Git state

The 2026-11-02 commit contains all bot changes (incl. new `air.py`,
`burrow.py`, `threat.py`, `data/weapons.json`), `tools/` (`run.sh`,
`report.py`, `demostats.py`, `extract_weapons.py`), the `zk-startup.ps1`
taskkill fix (a failed kill used to abort multi-game runs) and this file.
`results/` stays untracked; never commit demos or `results/SKILL.md`.
Branch `zkbot-hard`: `ae3c24c` (Hard beaten), the section 6b.7 layer,
then radar blips (6b.8), raider micro (6b.9), emergency turrets (opt-in)
and resign (6b.10–11); `main` is fast-forwarded
to `zkbot-hard` and pushed. Then the 2026-11-03 redesign groundwork
(section 6c: catalog/abilities, params, behaviours/army plan, economy split,
engagement log, prun/abstat, engine affinity fix) on `zkbot-hard`.
