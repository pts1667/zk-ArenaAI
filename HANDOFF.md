# HANDOFF — zk-ArenaAI / zkbot

Date: 2026-11-01  
Working directory: `C:/Users/Thoma/Documents/sources/zk-ArenaAI`

## 1. Goal and current status

Build a from-scratch Zero-K AI (`bots/zkbot`) that can beat CircuitAI at every
difficulty, starting with **CircuitAI:Normal**, then **CircuitAI:Hard**, then
**CircuitAI:Brutal**. Primary test map: **TitanDuel 2.2**.

Current status:

- **CircuitAI:Normal**: consistently beaten. Recent wins: `n28` (wall 117s,
  sim ~1090s), `n29` (wall 115s, sim ~1030s).
- **CircuitAI:Hard**: **not beaten yet**. Recent losses:
  - `h58`: game ~1320s, wall 186s
  - `h59`: game ~1340s, wall 149s (no demo captured)
  - `h60`: game ~1220s, wall 124s
  - `h61`: game ~1240s, wall 112s
- The latest Hard failure mode remains mid/late-game attrition: the army
  shrinks under CircuitAI’s ball, builders die, extractors go to zero, and the
  base collapses. Turret and air/artillery decision logic is now much more
  deliberate, but the economy/army cannot yet hold against Hard.

Do not claim Hard is solved. The next session should diagnose the Hard
attrition, not merely tune turret counts.

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

Typical command:

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
  `IntelMixin, EconomyMixin, ConstructionMixin, ProductionMixin,
  StrategyMixin, TacticsMixin, DefenseMixin, ScoutingMixin, MilitaryMixin`.
- `models.py` — `UnitInfo`, `Task`.
- `units.py` — static `units.json` helpers and factory-role tables.
- `mapdata.py` — map spots, heightmap, water, terrain helpers.
- `intel.py` — observations, enemies/radar, commander tracking, threat,
  spot status, observation-rate control.
- `economy.py` — task creation, mex/energy/defense/radar tasks, assignment.
- `construction.py` — placement, builder assignment, fallback/reclaim,
  commander confinement.
- `production.py` — factory queues, unit-role selection, Gremlin production.
- `military.py` — army modes, home guard, raider squads, focus fire, retreats,
  commander safety, direct orders.
- `strategy.py` — sector clustering, terrain scoring, sector threat/strength,
  expansion/defense/attack/rally decisions.
- `tactics.py` — composition counters, retreat decisions, local target
  selection, raid group target selection.
- `defense.py` — **new** defence intelligence and spending model.
- `scouting.py` — **new** Gremlin forward observers.
- `data/units.json` — 615 unit definitions extracted from the live game.

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

## 7. Test evidence from latest changes

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

Priority order:

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

## 10. Current git state before handoff commit

Modified/untracked bot files:

- Modified: `bots/zkbot/brain.py`, `construction.py`, `economy.py`,
  `intel.py`, `military.py`, `production.py`
- New: `bots/zkbot/defense.py`, `scouting.py`, `strategy.py`, `tactics.py`
- Existing commit before this work: `4913701 Add zkbot AI with modular brain
  and fast websocket transport`

The handoff commit should include all bot changes plus this file. `results/`
must remain untracked. Do not include local demos or `results/SKILL.md` in the
commit.
