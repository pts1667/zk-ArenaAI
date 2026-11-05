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

- **Direction (2026-11-04):** rule changes stopped moving the win rate (2/18 vs 2/18
  in the last A/B, section 6e), so the user asked for a Pluto-style self-play policy
  trained in a simplified simulator.  The implementation plan is on branch
  `learned-policy-plan`, starting at `docs/learned-policy/00-overview.md`; the flags below remain the rules bot's baseline, none adopted.
  Stage 1 is built, and **Gate 1 did not pass** (section 6f); stage 2 awaits the user's decision.
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

Headless at 3x (prun's default `--speed 3`, see below); do not use windowed
mode unless the user asks.

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
rates were at 10–16x against a weakened CircuitAI. **3x is equivalent to
1x** (h121: 0/16, CircuitAI metal 14.1k/27.0k at 10/15 min vs 13.5k/26.7k
at 1x), so prun defaults to `--speed 3` (16 games, 4 parallel, ~45 min).
Use `--speed 1` only to confirm key results; compare arms only at the same
speed.

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
- `terrain.py` — levels, ramps, regions and the gate graph (section 6e; imported
  only by `hazards.py`, and only when flag `avoidlava` meets harmful liquid).
- `hazards.py` — keeps builders off lava and void using `terrain.py` (section 6e).
- `raidprobe.py` — raiders-only opening and probing raid squads (section 6e).
- `nearbuild.py`, `mexpatrol.py` — sized responses to small raids (section 6e).
- `smoothorders.py`, `buildfocus.py`, `reclaim.py`, `pylongrid.py`, `attackline.py`, `heavycounter.py` — from
  the spec3 review: order rate, constructors, overdrive grid, army order (section 6e).
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

## 6d. Session 2026-11-03 (later) — loss attribution and ML stages 1–4

See **`docs/ML.md`** for the plan, environment and commands.

* **Metrics**: `tools/abstat.py` prints per-arm geometric means and paired
  sign tests of `M10/M15` (our/CircuitAI metal produced, from the demo) and
  `X10/X15` (metal killed / lost): nearly every game is lost, so win counts
  cannot separate arms. Log lines carry game time (`[hh:mm:ss gMM:SS]`).
* **Death log** `deaths-team<N>.jsonl` (`engagements.py`): killer, weapon,
  seconds under fire, damage by attacker, branch, depth, local balance.
  Weapon def ids are resolved at runtime (`learn_weapon_defs` queries each
  unit def once: weapon mounts + death/self-d explosions): `~name` inferred
  hidden attacker, `own:name` friendly fire / own explosions,
  `boom:<explosion>`, `env:<source>`. h124 (48 games) re-labelled: unknown
  killers 9% → 2.5%; death explosions 3.2% of all metal lost (fusion
  `atomic_blast` ~0.8k/game in our base; enemy pylon `estor_buildingex`
  kills raiding Glaives); friendly fire lands 2% of killing blows (Rocko
  and Reaver into our own Glaives); 10 fusions died without any damage
  event (unfinished frames decaying? new `built` field will tell).
* Where metal is lost (h124 base, by killer role): first 10 min raiders
  25%, turrets 18%, riots 18%; 10–20 min assaults 19%, skirmishers 18%,
  turrets 16%. Victims 10–20 min: Glaive 26%, Reaver 18%, Rocko 12%.
* **ControllerAI** forwards ENEMY_DAMAGED (topic 17, ~15k events/game) with
  the enemy id: now used for exact enemy-loss labels. It drops the
  `weaponDefID` of UNIT_DESTROYED and the ids of ENEMY_DESTROYED (one-line
  fixes in `ControllerAISerialization.cpp` if ever needed; not done).
* **Stage 1** `fightnet.py`: entity-level fight logger (`fights-team<N>`,
  every unit near a fight incl. turrets, air, remembered structures,
  blips; losses after 10/20 s from damage events), featuriser shared with
  `tools/train_fightnet.py` (set transformer, WSL + torch.compile), numpy
  inference of exported `data/fightnet.npz` (parity 6e-8), decision hooks
  behind flag **`fightnet`** (retreat threshold, attack target, hold/meet
  an attack, chase). Not trained yet (needs the d1 batch).
* **Stage 2** `statelog.py` (`state-team<N>`, every 10 s) +
  `tools/train_value.py` (temporal transformer; targets: winner, metal
  ratio 5 min ahead).
* **Stage 3** `bots/observer/main.py` + `prun.py --observe`: CircuitAI vs
  CircuitAI, each with an allied passive ControllerAI observer logging
  CircuitAI's full force and every unit it starts; `tools/observe_dataset.py`
  spend-mix comparison and behaviour-cloning examples.
* **Stage 4** `tools/tune.py`: SPSA over `P()` parameters with paired
  +/- prun games. The fight constants are now parameters (same defaults):
  `army.attack_ratio/hold_ratio/min_units/target_ratio/outranged_hold/
  chase_ratio`, `retreat.hp_cheap/hp`, plus `def.base_share/max_share/
  mex_cap/mex_scan` and `fightnet.*` thresholds.

### 6d.1 First ML results (d1: 37 games + 8 partial vs Hard at 3x, 3/37 won; o2: 8 observer games)

Both batches were stopped early by Claude Code under system memory
pressure (8 + 2 parallel games); resume with `--resume`, fewer jobs.

* Stage 1 (12.5k fight snapshots): predicting "fight won" (their loss >
  ours in 20 s), the bot's power-ratio rule has AUC 0.49–0.58 over four
  game splits, i.e. about chance; the set transformer 0.54–0.63. Better in
  3 of 4 splits but not yet reliable: needs ~100+ games. A preliminary
  model is exported to `data/fightnet.npz` (15 epochs); flag `fightnet`
  is untested.
* Stage 2 (7.2k steps): win prediction impossible yet (3 wins); metal
  ratio 5 min ahead: MAE 0.17 vs 0.233 for the median (26% better), then
  overfits. Not exported.
* Stage 3, spend mix by phase (CircuitAI: metal of units started; ours:
  net value added, which flatters categories that survive):

  | phase | CircuitAI | ours |
  |---|---|---|
  | 3–7 min | energy 19, **defence 16**, skirm 14, mex 10 | energy 18, assault 15, raid 14, riot 13, **defence 7** |
  | 7–12 min | **defence 26**, energy 17, assault 10 | energy 24, riot 15, raid 12, **defence 5** |
  | 12–20 min | **energy 22, defence 18**, assault 11 | riot 22, skirm 15, assault 14, **energy 12, defence 3** |

  h124's higher `def.base_share` was neutral; whether it actually raised
  realised defence spending to CircuitAI's level is unchecked (h124 had no
  state logs). Next: an arm whose realised spend follows CircuitAI's mix.

### 6d.2 Full d1, label fix, CircuitAI-macro arms (h127 pending)

* d1 resumed at 6 parallel and finished: 64 games, 7 won; metal ratio
  1.14 / 1.01 / 0.91 at 5 / 10 / 15 min.
* **Label bug:** `terraunit` (the engine's terraform helper, listed at
  100,000 metal) was in 17% of fight snapshots, inflating our side's value
  and logging fake losses (117 snapshots > 20k). Excluded now
  (`fightnet.NOT_UNITS`; also from the state log and spend counter);
  `train_fightnet.py` drops old snapshots that contain it. The stage 1
  numbers in 6d.1 were measured on contaminated labels.
* Stage 1 on full d1, clean: AUC net **0.59–0.62** vs power-ratio
  **0.50–0.55** on 3 of 3 splits. Exported (`data/fightnet.npz`, all
  games, 20 epochs). Predictions regress to the mean (10/50/90% of
  predicted exchange 0.36/0.69/1.2), so the `fightnet.*` thresholds were
  moved to those quantiles (attack 1.0, hold 0.6, chase 0.7, retreat
  stays ≥1.2, leaves early ≤0.5).
* Stage 2 on full d1: metal ratio 5 min ahead MAE 0.15–0.17 vs median
  0.25–0.28 (~40% better). Win AUC unmeasurable (1–2 won games per split).
  Exported (`data/valuenet.pt`, 10 epochs; offline use only).
* Stage 3: `state-team<N>.jsonl` now logs `spent` (metal committed per
  category, dead units included), so our realised mix is comparable with
  CircuitAI's. CircuitAI's turret metal (9 games, <20 min): Lotus
  (turretlaser) 28%, Stinger (turretheavylaser) 24%, Picket
  (turretmissile) 15%, Warden (turretriot) 12%, Gauss 10%; half its
  energy metal is one Singularity. Its turrets come from a per-cluster
  "porcupine" whose value cap scales with income (MilitaryManager::
  MakeDefence): our turret caps are fixed counts (base perimeter ~4
  anchors, 8 extractor turrets, 8 pending), which is why h124's higher
  share was neutral. New arms: `circdef` (that porcupine) and `lessriot`
  (see docs/ML.md). Smoke: `circdef` built 17–22 turrets, realised turret
  share 4–10% in a low-income game; most of our riots come from the
  reactive rules, not the cycle `lessriot` changes.
* **h127** (16 games/arm): base 0/16, `fightnet` 1/16, `circdef` 0/16,
  `circdef,lessriot`+`def.cluster_sec=40` 1/16. Nothing significant:
  `fightnet` M15 0.66 vs 0.81 (paired x1.24 for base, p=0.12: if
  anything worse economy), `circdef` X10 0.47 vs 0.39 (p=0.45). But the
  realised spend shows the CircuitAI arms did not get CircuitAI's mix:
  turrets 5–11% (CircuitAI 16–27%), riots unchanged. 25–55% of `circdef`
  jobs went stale unassigned: **build power**, not turret rules, is the
  limit. CircuitAI fields 4 / 8 / 11 / 13–15 mobile constructors at
  4 / 8 / 10 / 16+ min (o2), we ~5–7 throughout (our 1.25 BP per metal/s
  formula is CircuitAI's config value, but not what it actually does).
* **Raider overproduction:** `counter_role` keeps standing *head-count*
  shares (late raid 0.24). Glaives are cheap and die first, so the quota
  refills them without end: ~150 dead Glaives (10k metal) per game in
  h127-base, raiders ~20% of all metal (CircuitAI 5–7%).
* New arms: `morebp` (~0.6 mobile constructors per metal/s, max 15,
  `eco.cons_per_income`) and `spendmix` (role quotas measured on metal
  started in the last 4 min, CircuitAI's late army mix as target). Smoke
  `spendmix`: still 30% raiders late, because that enemy was skirmisher-
  heavy and the counter rule (raiders beat skirmishers) adds raid weight.
* **h128**: base 1/16, `morebp` 0/16, `morebp,circdef` 0/16. More
  constructors match CircuitAI's spend on builders (24% / 18% of metal at
  0–3 / 3–7 min) and with `circdef` lift turrets to 11–13% from minute 3,
  but the thinner early army trades far worse: X10 0.24 / 0.29 vs 0.39
  (13/16 pairs favour base, p=0.02 each), M15 0.65 / 0.75 vs 0.84.
  Rejected: CircuitAI's macro does not transfer piecewise; its bigger
  build power works because its army holds (its early raiders and
  turrets), ours does not.
* **h129**: base 3/16, `spendmix` 2/16; M15 0.81 vs 0.86, X10 0.35 vs
  0.43, all n.s. Raiders fell to 11% of late spend but air rose to 26%
  (second factory, outside `counter_role`). Not adopted.
* Stage 3 summary: four piecewise imitations of CircuitAI's spend mix
  (turrets, fewer riots, constructors, spend-based quotas) moved the
  realised mix toward CircuitAI's without improving results; `morebp`
  clearly hurt. The macro gap is a consequence of losing fights (X ~0.4),
  not its cause, so the next lever is fighting (stage 1/4), not the mix.
* Stage 4 started: `tools/tune.py army1` over `army.attack_ratio,
  hold_ratio, chase_ratio, target_ratio, outranged_hold, min_units,
  retreat.hp, hp_cheap`, 20 iterations x 8 paired games, 6 parallel
  (state: `results/tune-army1/state.json`; `--resume` continues).
* Stage 4 result, not adopted: `army1` finished 20 iterations (159 pairs) at
  `army.attack_ratio=1.376,army.hold_ratio=1.024,army.chase_ratio=1.12,
  army.target_ratio=1.674,army.outranged_hold=1.595,army.min_units=7.058,
  retreat.hp=0.3838,retreat.hp_cheap=0.2136`.  Plus-minus scores swung between
  -0.43 and +0.54 with no trend.  A/B against defaults (`results/tuned-ab-*`, 20
  games per arm, TitanDuel, Hard, 3x, varied enemy factories): 1/20 base, 0/20
  tuned; M15 0.83 against 0.84; X15 0.44 for both; X10 0.52 base against 0.43
  tuned (paired 15/20, p=0.04, one of four tests, gone by minute 15).  Defaults
  stay.  8 game pairs per step is too noisy for this kind of tuning.
* o2 finished: 24 observer games; CircuitAI's mix is stable (table above
  within 1–3 points). `results/o2/macro.jsonl` holds the 9k windows.

## 6e. Session 2026-11-04 — terrain regions (bot uses them only behind `avoidlava`)

`bots/zkbot/terrain.py` builds a walkable-distance map graph from height and
slope arrays.  No bot module imports it, so behaviour is unchanged.  (prun
snapshots `bots/zkbot` per batch and `tools/tune.py` runs one batch per
iteration, so an edit to an imported module would leak into a running tune.)

- **Passability:** the engine's speed formulas for kbot, tank, spider, amph,
  hover and ship, with limits from `gamedata/movedefs.lua`.  Reachability uses
  the full 16-elmo grid; regions grow on a 32-elmo grid (64 on huge maps).
- **Levels and ramps** (the user's definition): a level is a large flat area; a
  ramp is a walkable slope joining two levels at least 40 elmos apart.  A slope
  with no level at its top, or a sheer cliff, is not a ramp.  Flatness tolerates
  map noise (slope < 0.02); ground that still climbs 40 elmos at that limit is
  re-split at 0.006.  The strict limit alone lost TitanDuel's west plateau.
- **Chokes, by shape:** open ground is wherever a disk fits; the passable rest is
  narrow.  A narrow part touching open ground in two separate places joins them,
  and the narrowest point of the widest path through it is a choke, unless a way
  around within 1000 elmos is wider than the choke.  So a compound's gates all
  count (Castel Gods' centre has four), corners never do, and a gap beside a
  rock doesn't.  Land chokes use kbot passability; straits use ship passability.
- **Regions:** seeded from metal clusters, start positions, large levels and
  open ground, grown by walking cost.  Land and open sea never share a region;
  borders prefer chokes and ramps; small metal-less regions join their
  neighbours.  A choke with no seed behind it can sit inside one region.
- **Graph:** gates per movement class and per connected area, with width and
  walking length.  A gate on a choke is tagged `choke`, as is a narrow border
  with a long way around.  Crossings use per-class full-grid labels: a cell under
  a work-grid wall takes the region it is walkably joined to.  (Labelling it by
  nearest region made false gates through sealed walls with walkable flanks.)
- **Lava and void:** any water damage makes the plane lava, or void from the
  engine's ground limit up (1000 per slow update, 2000 per second in mapinfo).
  No class may enter and neither makes sea regions, since the engine would path
  units through lava and they would burn.  Void also blocks the 20 elmos above
  the plane: The Hole and Hexian Arena destroy anything below y = 20 by gadget.
  Lava doesn't, as Carabanossi has 18% of its map on a real shelf at 10-20 elmos.
  Hover may cross lava only where Zero-K's `map_lava` runs (`hover_safe`, live
  from the `lavaLevel` game rules param), which cancels their damage; a map's
  own gadget doesn't (Banana Republic).  `map_lava` also counts as lava when the
  engine reports no damage.  `water_damage` is the engine's and the bridge's
  value, per slow update: half the mapinfo figure.
- **Water:** the engine's water plane is hardcoded at 0.  Zero-K's `waterlevel`
  and `waterpreset` shift the terrain (`GG.Terraform_RaiseWater`), and the map
  can also generate its terrain in-game, so use live heights.  `water_level`
  is a plain shift for anything else.

`tools/terrain_map.py` runs it offline from map archives and draws
`results/terrain/*.png`; `tests/test_terrain.py` has the synthetic maps.

Survey of all 111 maps in the portable install (no failures), by
`python tools/terrain_map.py --all --check-graph`: build 2.0 s
median, 4.4 s worst, offline with a tune running; land regions 31 median, 120
worst; ramps 10 median, 89 worst, 28 maps with none; chokes 9 median, 96 worst, 32
maps with none.  Metal spots
match except duck, DunePatrolRedux, Living Lands, Scorched Crossing and StormSiege
(the tool's reader only approximates mex_spot_finder.lua; a game uses its own
list).  Every spot pair the full grid connects also has a route on the graph.
10 maps have harmful liquid: 4 void (Hexian Arena, The Hole, Violence, ZeroWars
Crossfire), 6 lava (Banana Republic, Carabanossi, LimeQuat, Scorched Crossing,
Violet Rampart, Zed Remake).

**Raider opening, flags `raidfirst` and `raidprobe` (`bots/zkbot/raidprobe.py`), not
adopted:** from the user's spectated game (`results/spec1`): a Reaver by minute 1-2
and Glaives parked at minute 3.  Causes: the fixed opening mix queued the first riot
at about 65 s in the median game whatever the enemy had, Fleas counted as raiders
there, and a squad at an empty metal spot waited for the 20 s retarget.
`raidfirst`: raiders and constructors only until 5:00, riots and assaults phasing in
by 10:00; ends early on 6+ real raiders outnumbering ours or 3+ riots and assaults
seen.  `raidprobe`: an empty target means move on to the next unchecked point toward
the enemy spawn; rally 55% of the way there; a turret counts against a target only
if its range reaches it, and routes bend around coverage; squads accept defence up
to their own metal value (was half) and keep raiding in attack mode.  **Existing bug,
still there without the flag:** `raid_defense_value` ignores turrets in view (they
aren't `is_dangerous`, and are marked seen so memory skips them too).

A/B `results/rp-ab-*`, 12 games per arm, TitanDuel, Hard, 3x, varied factories:

| | base | raidfirst | both |
|---|---|---|---|
| Wins | 1 | 0 | 1 |
| First riot, median | 65 s | 350 s | 330 s |
| Median game length (metrics) | 1735 s | 1270 s | 1340 s |
| M10 / M15 | 0.99 / 0.88 | 0.93 / 0.79 | 0.99 / 0.88 |
| X10 / X15 | 0.50 / 0.42 | 0.34 / 0.31 | 0.39 / 0.40 |
| Glaive deaths by 7 min, per game | 19 | 32 | 34 |

The opening now looks as asked, but the raiders die without earning it, and the bot
loses sooner.  Only 16% of the early Glaive deaths in `both` were to turrets; enemy
commanders, riots and raiders kill more.

**Hit-and-run raids, flag `hitrun` (with `raidprobe`), and `seeturrets`, untested
beyond one short headless game and the user's spectated `results/spec2`:** the A/B
above showed Glaives reaching the enemy economy (91% of early deaths in the enemy
half) and dying at a local enemy force of 2.6x.  `hitrun`: move orders on the way in,
then "fight" armed units at the target, else attack constructors first, then the
nearest unarmed target; a squad takes a target only if worth twice its defence value;
a riot within 520 elmos, or seen there in the last 10 s, rules a target out whatever
the squad's size; a squad breaks off when the defence within 650 elmos passes that
limit; a finished target means a new order that tick; squads may be larger (6/8/10
by phase, was 4/6/8).  Raiders below 50% in a squad of 6 or more go back along its
route, then wander to safe metal spots, until 95% or 60 s.  `seeturrets` counts
turrets in view within the old radius (`raidprobe` uses their real range instead).

**From spectating `results/spec2`, flags `spearhead`, `nearbuild`, `mexpatrol`
(`nearbuild.py`, `mexpatrol.py`), no A/B yet:** at 1:02 a raid warning for one Flea put
the army in defend mode and recalled every Glaive.  `spearhead`: the first 3 raiders
never defend, skip the route via our expansions, and aren't replaced.  `nearbuild`: no
defend mode when local units, the commander, patrols and any combat unit finishing
within 5 s match the threat; while there's a threat, a factory building a combat unit
is on high priority (`CMD_PRIORITY` 34220).  `mexpatrol`: 22% of raiders, at least 2
(production ensures them from 1:00), only walk our extractors and answer threats.
Under `raidprobe`, all raiders now keep raiding in attack mode: they were idling at
the rally point once it began.

**From the review of `results/spec3` (first spectated win, g24:26, one game), nine
opt-in flags, no A/B yet.  Tests in `tests/test_spec3_review.py`.**

- `smoothorders` (`smoothorders.py`): units visibly stopped when re-ordered.  Repeats
  were only part of it (every 3 s even unchanged).  Attribution by behaviour showed
  most orders were destinations jumping 1000+ elmos between plans (patrol, raid
  squads, main attack, home guard).  So: destination may drift by 25% of the distance
  left, slow refresh, and a plan's order stands 5 s against other plans unless the
  unit has arrived; reactions aren't held back, and a squad breaking off is released.
  The `army mode` log line now has `orders=` and `order_sources=`, with or without the
  flag; `soon` counts orders within 3 s of that unit's previous one.
- `unstick` (`buildfocus.py`): the constructor idle all game was guarding a finished
  solar.  "Assist" is a guard order, which never ends, so its queue was never empty
  and the idle check never fired.  Guards are tracked and released.  A trap nearby:
  `cleanup_tasks` moves `created_frame` forward while a nanoframe exists, so energy
  tasks are normally closed by the idle check as "occupied", not as "done".
- `finishfirst` (`buildfocus.py`): with under 150 metal banked and at least
  1 + income/15 buildings unfinished (2 to 8), constructors help the nearest one instead
  of starting another, unless the job has priority 1600+.
- `fewercons` (`production.py`): at least 1 constructor before 2:00, 2 before 4:00, then
  3; 1 per 5 metal/s above 12, instead of 1.25x that.  A floor of 2 trapped one raided
  game on 3 extractors for 10 minutes (`results/rev3-new/game-002`).
- `reclaimcons` (`reclaim.py`): at 700+ safe wreck metal, 2 constructors (1 if under 3);
  at 150+, any above `desired_builder_count`; at most half.  Stays on a wreck while it
  exists, and prefers piles of them.  Safe means our side of the map, with no threat
  within 250 and no enemy within 500.
- `pylongrid` (`pylongrid.py`): units link when `dist < rangeA + rangeB`
  (`quadField:GetIntersections`).  Pylon when `count * 200 / gain <= 300 s`, where gain
  comes from `sqrt(energy/mexes) * 0.25` per extractor income.  One at a time, chains
  from the grid, priority 1400 plus the seconds it beats that payback by (at a fixed
  1100 they went stale unserved).  Replaces `ensure_pylon_tasks`, which built at most
  one, near a reactor.  The `grid extractors` log line shows joined and apart.
- `fastgather` (`attackline.py`): gathered means 50% within 1100 (was 60% within 700);
  the group meets 30% of the way to the target (at most 600 elmos) and only units 900+
  from there walk; a regroup lasts 20 s, then 40 s of attacking.  Raid stage radius
  1400 (was 900), squads act from 2 raiders, and combined squads meet between them.
- `assaultfirst` (`attackline.py`): within 1800 of the target, or 1200 of an enemy,
  followers hold behind the front (assault, heavy; riot if there are under 2): riots
  90 elmos, others 170, artillery 80% of range.  The hold point moves only after the
  front has moved 300 elmos, so this doesn't bring the stop-start back.
- `mexllt` (`build_defense.py`): from 2:30 a spot without a ground turret within 400
  gets a Lotus 110 elmos to the enemy side first, and unstarted extractor jobs wait for
  it too.  2 waiting at most; after 90 s with no constructor on it the extractor goes
  ahead, and the spot doesn't ask again for 3 min.  Under 4 extractors, income comes
  first.  Only jobs queued before 2:30 are converted: redoing later ones had turrets
  and extractors dropping each other in turn.

Measured on TitanDuel, 14 game minutes against CircuitAI Hard, not win rates (no game
reached an end; 0 wins, all timeouts).  Orders within 3 s of the unit's last order:
without these flags 4391 and 6469 (`results/rev3-old`), with them 430 and 466
(`results/rev3-new`) and 421-652 after the later fixes (`results/rev4`, 4 games).  All
orders: 8326 and 11529 against 1923-3064.  Armies differ in size, so those aren't exact
ratios.  Attack target flips remain the largest source (`attack:jump`).  Expansion looks
slower: 5-6 extractors at 4:00 and income 19-30 at 8:00 (rev4), against 8-10 and 26-34
(rev3-old), in too few games to tell which flag does it; `fewercons` and `mexllt` are
the suspects.  Not yet seen working: grid is still mostly apart (joined 2-4, apart 2-10
at 14:00) before the priority change, which has only had unit tests.

`results/spec4` (all these flags, spectated, enemy on spiders): lost at g21:45, one game,
so no verdict on the set; spec3's win was against shields.  Metal lost 25.2k, killed
8.5k; Crabe and Recluse did most of it.  Constructors stayed at 4 from 6:00 and income
at 17-22, with 17 extractor assignments and 27 extractor jobs gone stale, so constructors
now limit expansion (`fewercons` follows income, income needs extractors).  One extractor
on the grid nearly all game: 4 pylon jobs, 2 assigned.  Mexllt: 20 asked, 8 assigned, 13
gave up.  Largest order sources are new: `ball_gather:switch` 403 (move and fight flip at
the edge of the ball's reach) and `defend:switch` 359.  `attack_behind` barely ran.

**After spec4 (flags `stickytarget`, `armorcounter`; A/B `results/rev5-*`):** `fewercons`
adds a constructor per 3 safe unassigned extractor, turret or pylon jobs (at most 4,
from 2:00), since income alone trapped it.  `smoothorders` leaves a unit shooting a live
target for 2 s, and `ball_gather` moves on to 500 past the ball's reach once moving.
`stickytarget` (`attackline.py`) keeps the main group's target 20 s unless blacklisted.
`armorcounter` (`heavycounter.py`, reviewer's advice: Slings, Phantoms, massed Knights):
an armored heavy is slow, 800+ metal, a 100+ blast and the `armor` tag, which today means
only the Crab; Grizzly's blast is 14, and the raider-Ronin answer still applies to it.
Counted for 5 min after last seen.  Mix: assault .44, arty .30, raid .12, skirm .10, riot
.04.  Phantoms 1 more than Crabs, at most income/15, without the 10:00 and income 25
gates.  Raiders don't join and treat it like a riot.  The group commits only when
role-weighted metal, with raiders at 0, outweighs the ball; assaults then ignore being
outranged, and units with more range fire from 92% of theirs instead of waiting.

A/B before and after (TitanDuel, 3x, all factories): `results/rev5-*` 1/10 and 0/10,
`results/rev6-*` 1/8 and 2/8, so 2/18 each; no difference shown.  `--enemy-factories` steps
every 2 games, so rev5's 10 games only reached the first 5 entries and no Crab; rev6 covers
spider, amph and jump.  Orders within 3 s per minute: about 375 down to 60.  Crabs got their
first kill at 17:00-25:00 in 3 of 4 after-arm spider games, all lost, costing 3.4k-13.9k
metal (before arm: 2 of 4 games, 24:00 and 29:00, 3.2k and 3.3k, both lost).  Two of those three queued no Phantom (`production_counter`
bug, then fixed under `armorcounter`; that fix has unit tests only).  Not adopted.

**Factory on lava and void, flag `safefac` (`MapData.favours_hover`):** without it
`choose_factory` picks `factoryhover` when `water_fraction` > 0.25, which counts
lava and void as water (Hexian Arena, Violence, ZeroWars Crossfire, Carabanossi,
Zed Remake).  With it, any harmful liquid means `factorycloak`, including on Zed
Remake where hover is safe.  Not adopted yet: only a 4-game smoke test
(`results/safefac-smoke-*`; 1/2 wins against 0/2, too few to mean anything).
In `deaths-team*.jsonl`, `weapon_def` -5 is water damage.

**Builders and lava, flag `avoidlava` (`bots/zkbot/hazards.py`):** on maps with
harmful liquid only.  Skips metal spots no mobile builder can walk to, won't assign
a builder a job or a wreck it can't walk to, and refuses sites in or against the
liquid.  Not adopted yet.  Smoke test `results/avoidlava-smoke-*`, 4 games per arm
on Carabanossi and Violence, both arms with `safefac`, while the tune was running:

| | `safefac` | `safefac,avoidlava` |
|---|---|---|
| Wins | 1/4 | 0/4 (sign test p=1.00) |
| Constructors lost to lava, Carabanossi | 115 | 11 |
| Constructors lost to lava, Violence | 53 | 0 |
| All units lost to lava, Carabanossi | 450 (35% of metal lost) | 628 (46%) |
| All units lost to lava, Violence | 56 (6%) | 0 |

Violence's shore deaths were constructors walking to off-land spots, though each
died nearest a main-land spot: 118 mex assignments went off the main land without
the flag and none with it.  16 other jobs (radar, storage, defense) still end up
off it with the flag, with no lava deaths; `can_walk` checks the task position
before placement moves it, which may be why.  Carabanossi's army still burns, 478 of 543 lava deaths
in one game coming from units in a retreat branch: retreats go straight across
lava.  That is the next thing to fix, and these games don't show the flag helps
win.  **Open:** one flagged Carabanossi game logged 106 idle-builder warnings
(others 0-15), builders parked near (2010,3376) for 15+ minutes while tasks were
open.  Cause unknown, and a single game, so it may not be the flag.  `site_bad`
looks unlikely: the comparable unflagged game dropped more unplaceable energy tasks
(47 against 25) with only 5 warnings.  The
"N of M free spots out of reach" line also logs 40-59 times a game, as the free
list changes.

Still to check, in a live game:
1. **Possible existing bug:** six maps store flat placeholder heights (Alphabet
   Siege Dry, Fickle Rivers, LimeQuat Republic, Rainbow Comet, Random Plateaus,
   THISspace), and most of them generate terrain in-game.  The bot reads
   `heightmap` from the snapshot taken at AI load.  If that predates generation,
   `map.water_fraction` and `is_water` are wrong there (Fickle Rivers would read
   as all water) and factory choice follows.  Unverified.

Checked live on Carabanossi and Violence: `height_map` and `slope_map` give the same
mean height, regions, chokes, ramps, walls and share below 0 as the map files.
Queries took 0.24-0.31 s and the build 1.0-2.0 s at game start, with 6 tuning games
running alongside.

Not modelled, because the bridge does not expose them: typemap speed changes
(mapinfo sets a zero speed on Hexian Arena, Scorched Crossing, The Hole and Zed
Remake), blocking features.  Also not modelled: lava that rises in a game, and
Scorched Crossing's burn line 20 elmos above its lava (0.3% of that map).  The
flat-slope, level-size and choke limits were set by eye on a few maps.

First wiring to try, behind a flag and A/B tested: `path_threat` along real
routes instead of straight lines.  Testing it needs several maps; prun defaults
to TitanDuel only.

## 6f. Session 2026-11-04 (later) — learned policy, stage 1: Gate 1 not passed

Stage 1 of `docs/learned-policy/` is built: a C++17 combat and movement simulator with a ctypes binding, unit and map
data from an in-engine dump, a scripted engine battery, and the fight-replay gate.  The bot is unchanged.  **Stage 2
waits for the user's decision.**  All numbers and deviations: `docs/learned-policy/01-simulator.md` section 8.

- **Gate 1(a), replay of 44,733 logged fights: fail.**  20 s AUC: sim 0.591, out-of-fold fightnet 0.665, tracked value
  0.564.  Sim minus fightnet -0.074 [-0.092, -0.053]; sim minus value +0.027 [+0.011, +0.045].  At 10 s it ties
  fightnet.  Sim fights run far closer to the death than real ones (0.86 to 0.91 predicted against 0.39 to 0.46 measured
  in the top bin).  Don't tune the sim against these snapshots.
- **Gate 1(b), engine battery: 108/113.**  Misses: 4 blocked-move walk times and 1 detour length, from route
  choice; speed along engine paths matches.
- **Finding for the rules bot, not yet used:** a plain move order stops a unit until its next slow update (15 frames;
  a tank keeps 0.65 of its distance when re-ordered every second, 0.26 at every 0.5 s), while fight orders cost
  nothing.  This is the likely cause of the stop-start the user saw.  `smoothorders` isn't in the frozen baseline, and
  its docstring gives a different cause.  Without it the bot repeats unchanged orders after 90 frames if moved under
  150 elmos (`military.py:750-760`).  Any change goes behind a flag and an A/B.
- **Budget:** 423 µs per 400-unit tick with shared fight goals on Titan Duel against a 370 µs plan; 17.7 ms when each
  unit paths to its own enemy, because of run-time Dijkstra.  The pathing change below removed that.  Profile with
  `cmake -S sim -B ~/zksim/build-prof -DZKSIM_PROFILE=ON`, then `tools/sim_profile.py`.
- **Pathing (2026-11-05):** no pathfinding at run time.  `sim/py/zknav.py` splits per-class terrain segments to a limit (4 parts, 90% seen)
  and builds gates and a next-hop table when a map is exported.  Units walk straight, or along the route via gates and centres.  57 MB of
  fields became 1.3 MB on Titan Duel, and 1.2 GB became 8.7 MB on Carabanossi.  122 to 247 µs per 400-unit tick, against 370 planned.  A unit making
  no progress for 3 s aims straight at its next gate or goal, for longer each time.  Battery 105/113.  Same-segment arrival 98 to 100%; other-segment
  91 to 100% on both maps.  About 8% of long tank moves on Titan Duel still never arrive, so watch this in self-play.  Gate 1(a) rerun unchanged, and still
  awaits the user's decision (`01-simulator.md` 8.6).

Setup, in WSL from the repo: `cmake -S sim -B ~/zksim/build && cmake --build ~/zksim/build`; the library is
`~/zksim/build/libzksim.so` (`ZKSIM_LIB` overrides).  `~/zkml/bin/python sim/tests/test_sim.py` runs the tests.
`python tools/sim_data.py {catalog,maps,splits}` rebuilds the data; maps are gitignored.  `tools/sim_engine.py` uses its
own write dir and port.  To stop a stuck run, stop only the process whose command line contains `results\simtools`.

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
| h121 | 3x 0/16, 2x 1/13 (stopped) | speed calibration: 3x matches 1x (CircuitAI metal @15m 27.0k vs 26.7k); our metal ratio 1.09/0.93/0.84 at 5/10/15 min, losses 2–4x kills |
| h122 | base 2/16, `staticmix` 1/16, `regenretreat` 2/16, `nooffense` 1/16 | `staticmix` worse economy (paired M10 x1.31 p=0.02, M15 x1.42 p=0.04): early raiders buy map control; others neutral |
| h124 | base 2/16, tur 3/16, tur2 2/16 | more static defence (`def.base_share` 0.14/0.22, mex turret cap/scan up): neutral (base vs tur X10 x0.81 p=0.45, M15 x0.94 p=0.79); not adopted |
| h127 | base 0/16, `fightnet` 1/16, `circdef` 0/16, circmax 1/16 | none significant; `fightnet` M15 0.66 vs 0.81 (p=0.12); `circdef` turrets only 5–11% of spend (jobs stale: too few constructors) |
| h128 | base 1/16, `morebp` 0/16, `morebp,circdef` 0/16 | more constructors: X10 0.24/0.29 vs 0.39 (p=0.02): rejected |
| h129 | base 3/16, `spendmix` 2/16 | n.s.; raiders 11% of late spend but air 26%: not adopted |

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
