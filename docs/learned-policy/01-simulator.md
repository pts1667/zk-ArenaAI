# 01. Simplified Zero-K simulator: implementation plan

Status: stage 1 built; **Gate 1 not passed, awaiting a decision** (results in section 8, which overrides older text
here).  Stage 2 hasn't started.  Engine and game facts are cited `path:line` against
`C:\Users\Thoma\Documents\sources\RecoilEngine` (RE), `...\Zero-K` (ZK) and this repo (zk-ArenaAI).
Numbers marked **measured** come from a throwaway Python pass over the repo's data while planning
(method stated where used). Everything else is from reading source, or is marked unverified.

**Reconciled with the other plans:** [00-overview.md](00-overview.md) lists what changed and wins any remaining disagreement.

Shared decisions are taken from `00-overview.md`. Where this plan disagrees, it is listed under
"Objections to the shared contract" at the end.

## 0. Decisions for review

1. **Tick 5 frames, not 6.** The contract's 15-frame decision cannot land on a 6-frame tick (2.5 ticks).
   Proposed: 5-frame tick (6 Hz), decision every 3 ticks. Cost: about 20% more sim time per game.
2. **The stage 1 gate compares against a model trained on all games.** The exported `fightnet.npz` is
   trained on every game (HANDOFF 6d.2). The fair comparator is fightnet retrained on the same train
   split, with grouped cross-validation over the 184 games.
3. **Add a value-ratio baseline to the gate.** On the logged snapshots, cost x hp of tracked units
   alone already gets AUC 0.56 at 20 s (**measured**, section 5.1). fightnet's 0.59-0.62 is a small
   margin above that. The gate needs bootstrap CIs by game and a baseline row.
4. **C++17 sim, C ABI, ctypes binding, one thread per game.** No pybind11 build dependency; 16
   worker threads each own whole games; the batch step is one ctypes call with preallocated numpy
   buffers (section 4.4).
5. **Authoritative unit data is an in-engine LuaRules dump, not the lupa extraction.** The lupa route
   reads raw unit files and misses gamedata post-processing (armordefs.lua rewrites `damage` tables).
   lupa stays as an offline cross-check (section 3).

## 1. Fidelity triage

Rule: an error is acceptable when the policy cannot profit from it. "Exploit risk" names the most
likely way RL would find the error. Tags: **E** exact in the sim (formula copied from source),
**A** approximate (stated method), **O** omitted (with the stage it returns in).

### 1.1 Movement and pathing

| Mechanic | Tag | How in the sim | Source | Exploit risk |
|---|---|---|---|---|
| Top speed | E | `speed/30` per frame (`UnitDef.cpp:442-443`, `MoveType.cpp:72`). ZK kbot "speed" 115.5 gives 3.85 elmo/frame. | `Zero-K/units/cloakraid.lua` | Low |
| Acceleration / braking | E | Per-frame speed change bounded by `maxAcc` / `maxDec` (`UnitDef.cpp:455-459`, `IPathController.cpp:6-14`). Integrated per frame in closed form over the tick. | `acceleration`, `brakeRate` in unit files | Low |
| Turn rate | A | `turnRate` is heading units out of 65536 per frame (`SpringMath.h:16-17`, `GroundMoveType.cpp:518`). cloakraid 3000 = 16.5 deg/frame. Turn is sub-stepped every 2 frames if the calibration in 5.2 shows more than 10% error on turn-around time. | `turnRate` | Medium: kiting at tick scale |
| Slope and depth speed per class | E | Copy `CMoveMath::GroundSpeedMod` (`GroundMoveMath.cpp:12-27`) and the `MoveMath.cpp:81-104` class switch; `terrain.py:speed_field` already reproduces it in Python and is the oracle for C++ unit tests. | `terrain.py:152` | Low |
| Terrain type multipliers (`tt.kbotSpeed`) | O | `terrain.py` does not expose them; TitanDuel's typemap is unverified. | `MoveMath.cpp:81-104`, `terrain.py` docstring | Low if the map has none; check on TitanDuel |
| Speed sampled at the centre square only | E | Same as engine: `GroundMoveType.cpp:1300-1305`. | | Low |
| Raw move | A | ZK replaces a ground `CMD_MOVE` with `CMD_RAW_MOVE` (`cmd_raw_move.lua:555-560, 759-763`): the unit walks straight if `IsPathFree` holds in 16-elmo steps, re-checked every 500 elmos of travel when blocked (`:170, :180, :474-485`). Sim: the same line test on the passability grid, flow field otherwise. FIGHT and ATTACK use the path branch; to be checked in S-M4. | `cmd_raw_move.lua` | **High**: without it paths and arrival times are wrong on open ground |
| Re-order cost | A | Each new order on a moving unit costs speed, which the user saw as stop-start every 1-2 s. **Measured** in the S-M4 battery (8.3): a plain move stops the unit until its next slow update, while fight orders cost nothing. The sim reproduces this. | S-M4 battery | **High**: if free, the policy re-orders constantly and loses speed live |
| Pathfinding | A | Engine path is QTPFS for ZK (`modrules.lua:19-22`; the engine default is HAPFS, `ModInfo.cpp:130`; the mapping was not fully traced). Replaced by a Dijkstra flow field per (move class, goal region) on the 16-elmo grid using `speed_field` costs, precomputed per map and shared read-only (4.5). Region gates (`terrain.py:Gate`, `Edge.best_gate`) give the policy's region tokens and a coarse route length for long moves. | `terrain.py:289-300` | **High**: corridor squeezes and route cheating. Detect with 5.3. |
| Move classes | E | Six classes from `terrain.py:MOVE_CLASSES` (kbot, tank, spider, amph, hover, ship), which match `Zero-K/gamedata/movedefs.lua`. | `movedefs.lua` | Low |
| Blocking by structures and features | A | Structures are blocked cells; wrecks and rocks are ignored (`terrain.py` docstring: features not modelled). | `terrain.py` docstring | Medium on maps with features; check |
| Unit collision | A | Spheres: radius = `collisionVolumeScales.x / 2` from the unit file (cloakraid: 18 gives 9 elmo). Overlap is resolved by separation, because ZK sets `allowUnitCollisionOverlap = false` (`modrules.lua:34`). Pushing enemies is on (`modrules.lua:31`). | `modrules.lua:31-34` | Medium: clumping and body-blocking |
| Crush | O | `crushStrength` (`movedefs.lua`) is omitted. Stage 2 only if army tests show body-blocking. | `movedefs.lua` | Low |
| Collision damage | O | `allowUnitCollisionDamage = true` (`modrules.lua:33`) is not simulated. Magnitude unknown; measure in scripted engine tests (5.4). | `modrules.lua:33` | Medium for ramming |
| Hover, ship | A | Hover uses `HoverSpeedMod`; ships are stage 4 (TitanDuel has 1.9% water). | `MoveMath.cpp` | Low for stage 1 |

### 1.2 Targeting, weapons, damage

| Mechanic | Tag | How in the sim | Source | Exploit risk |
|---|---|---|---|---|
| Reload, burst, projectile count | E | `reloadTime` starts at the first round; burst and projectiles per shot. | `WeaponDef.cpp:79, 82-83` | Low |
| Auto-target order | A | Engine sorts by `TargetWeight` (script) with bad-target categories last, then takes the first valid (`Weapon.cpp:228-232, 720-776`). Sim: sort by category preference, then distance; ZK's `unit_target_priority` gadget (480 lines) overrides this and is not read in full. | `Weapon.cpp:720-776` | **High**: focus fire is the thing a policy will learn |
| Overkill avoidance | A | `unit_overkill_prevention` (calls `SetUnitTarget`) retargets when pending shots already kill. Sim: count pending scheduled damage at impact. Without this the sim gives too much overkill. | `unit_overkill_prevention.lua` | Medium |
| Target leading | A | Engine predicts with `dist / projectileSpeed` and no target velocity (`Weapon.cpp:311-316`, with the engine's own TODO). Sim does the same. | `Weapon.cpp:311` | Low |
| Turret turn | A | Instant turret rotation, except non-turret weapons get a cone (`tolerance`, `WeaponDef.cpp:112`). `unit_turn` and `unit_turn_without_interia` are approximated as the same. | `WeaponDef.cpp:112` | Low |
| Range | E | `range` from weapon defs; chase when out of range for attack orders (see 4.7). | `weapons.json` | Low |
| Accuracy and spray | O | Assume hit (user direction). Units with nonzero `sprayAngle` will hit more often in the sim. | `WeaponDef.cpp:133` | Medium |
| Projectile flight | A | Per contract: impact at `ceil(dist / weaponVelocity)` ticks (`Cannon.cpp:100-134` predicts flight time; `MissileLauncher.cpp:65`). Damage applied at impact if the target is still inside the blast. | `Cannon.cpp:100-134` | Medium: dodge exploit at the tick scale |
| Hitscan | E | Rifle (`Rifle.cpp:23-37`) and BeamLaser (`BeamLaser.cpp:264`) hit at once. ZK units use no Rifle. LightningCannon not read (unverified; treat as instant). | | Low |
| Beam damage over time | A | BeamLaser with `beamtime`: damage per frame x frames, applied per tick. | `WeaponDef.cpp:141` | Low |
| Blast falloff | E | Copy `rts/Game/GameHelper.cpp:146-150`: factor `(R + 0.001 - d) / (R + 0.001 - d*edge)`. The sim uses centre distance minus the unit radius, not the collision-volume surface (`GameHelper.cpp:132`), which the sphere model cannot reproduce exactly. | `GameHelper.cpp:107-150` | Low |
| Blast delay | A | Engine queues damage by `dist / explosionSpeed` (`GameHelper.cpp:~160-170`). Sim uses the flight-time rule above. | `GameHelper.cpp:~160-170` | Low |
| Damage types, armour classes | E for ground | Classes are `ELSE`, `PLANES`, `SHIELD`, `CHICKEN` (`armordefs.lua`). Ground units take the `default` value. Only `planes` and `shield` differ. Post-processing sets `default` to the max over classes (`armordefs.lua` weapon loop). | `Zero-K/gamedata/armordefs.lua` | Low |
| Shields (`Shield` weapon type, 3 factory units) | A | Absorb damage up to a stored amount with regen. Projectile interception not modelled. | `Zero-K/units/*.lua` | Medium for shield units |
| EMP | A | Engine: `paralyzeDamage` accumulates; stun while it exceeds max health; decays at `maxHealth x (15/30) x empDeclineRate` per slow update (`Unit.cpp:992-1009`). ZK uses `paralyzeOnMaxHealth = true` (`modrules.lua:92-93`). Sim copies this, with `empDeclineRate` read from the engine dump. | `Unit.cpp:992-1009` | Medium |
| Slow (`timeslow`) | A | Multiplies speed and reload for a duration; ZK applies it via rules params read by unit scripts (`unit_timeslow.lua`). Duration and factor from the unit file. | `unit_timeslow.lua` | Low |
| Disarm | A | Disables weapons for the duration (`disarmed` rules param, read by `unit_mex_overdrive.lua`). | `disarm_defs.lua` | Low |
| Death explosion | A | `explodeAs` weapon fired at death (`UnitDef.cpp:649`); self-destruct via `selfDestructAs` (`UnitDef.cpp:650`). | `UnitDef.cpp:649-650` | Low |
| Jump | O (stage 2 approx) | Seven factory units have `jump`. Timed parabolic move with no stun. Needed before factoryjump units are fielded in army play. | `jump_defs.lua`, `unit_jumpjets.lua` (794 lines) | **High** if omitted and used by the policy |
| Experience | O | ZK sets `experienceMult = 0`, `powerScale = healthScale = reloadScale = 0` (`modrules.lua:132-137`). Engine XP effects are read at `ModInfo.cpp:346-348` and guarded in `Unit.cpp:1453-1460`. `unit_xp.lua` only records XP. | `modrules.lua:132-137` | None |

### 1.3 Sensing, fog, stealth

| Mechanic | Tag | How in the sim | Source | Exploit risk |
|---|---|---|---|---|
| Ground LOS | A | Radius = `sightDistance` (`UnitDef.cpp:480`). Engine uses a raycast with terrain heights for LOS and radar (`LosHandler.cpp:92`, `LosMap.cpp:489-505`). Sim: raycast on the 16-elmo grid, cached per (cell, height bucket). Mip levels (`modrules.lua:97-110`) are ignored. | `LosHandler.cpp:92` | **High** for ridge-line hiding; check with 5.3 |
| Air LOS | A | Circle at 1.5 x LOS (`UnitDef.cpp:481`). | `UnitDef.cpp:481` | Low |
| Radar | A | Circle at `radarRadius`, same raycast as LOS (`LosHandler.cpp:92`). Blips carry no type (`fightnet.py` FLAG_BLIP). | `LosHandler.cpp:92` | **High** if blips leak type |
| Radar position wobble | A | `unit_radar_wobble` (`SetUnitPosErrorParams`). Stage 2. | `unit_radar_wobble.lua` | Low |
| Cloak | A | Cloaked units are hidden unless within `minCloakDistance` (`UnitDef.cpp:501`) or in LOS when `decloakRequiresLineOfSight` (`modrules.lua:101`). Energy cost (`cloakCost`, `UnitDef.cpp:497`) is in the economy stage. | `UnitDef.cpp:497-501` | Medium |
| Global LOS | config | `globallos` modoption (`game_globallos.lua`) makes LOS global for allies. Off for matchmaking, to be confirmed. | `game_globallos.lua` | If on, fog is wrong |
| Sphere sensor | A | `unit_spherical_los` (`SetUnitSensorRadius`). | `unit_spherical_los.lua` | Low |

### 1.4 Economy, production, construction

| Mechanic | Tag | How in the sim | Source | Exploit risk |
|---|---|---|---|---|
| Mex income | A | `extractsMetal` per spot from the unit def (`UnitDef.cpp:305`). The base value is not in `staticmex.lua` (unverified where ZK sets it). ControllerAI serialises `extractsResource` (`ControllerAISerialization.cpp`). | `UnitDef.cpp:305` | Low once read from the dump |
| Generators | E | `metalMake`, `energyMake` (`UnitDef.cpp:320-322`). | `UnitDef.cpp:320` | Low |
| Overdrive | A (stage 3) | `energyToExtraM(E) = sqrt(E) x 0.25` (`unit_mex_overdrive.lua:265-267`); allocation by `OptimizeOverDrive` (`:899`). Payback and refund (`:50-55`) are approximate. | `unit_mex_overdrive.lua` | **High**: allocation optimum is exactly what a policy would exploit |
| Pylon grid | A (stage 3) | `bots/zkbot/pylongrid.py` link rule (`pylongrid.py:1-10`). | `pylongrid.py` | Medium |
| Team income multipliers | E | `allyTeamIncomeMult` from the `team_N_econ` modoption (`unit_econ_mult.lua:71`). | `unit_econ_mult.lua:71` | Low |
| Build step | E | `step = min(amount / buildTime, 1 - progress)`; resources drawn per step; stall if short (`Unit.cpp:1995-2025`). Build speed = `workerTime` (`UnitDef.cpp:362`). | `Unit.cpp:1995-2025` | Low |
| Construction decay | E (off) | Disabled in ZK (`modrules.lua`, construction block). | `modrules.lua` construction | None |
| Nanoframe rate limit | O | `unit_nanoframe_rate_limit.lua` (54 lines, not read). Stage 3. | | Unknown |
| Factory queue | A | `CFactory::StartBuild` / `UpdateBuild` / `FinishBuild` (`Factory.cpp:173, 209, 255`): queue, resource wait, exit placement approximated. | `Factory.cpp:173-255` | Low |
| Metal in wrecks | A | Wreck feature from `corpse` (`UnitDef.cpp:292`), metal value from the feature def, not verified. Sim: wreck value = fraction of cost. | `Unit.cpp:153, 452` | Medium: reclaim farming |
| Reclaim | O | Stage 3. `modrules.lua` reclaim block (lines 62-74) sets `featureEnergyCostFactor` and `reclaimMethod`. | `modrules.lua:62-74` | Medium |
| Assist and repair | A | Same build-step formula; repair via `repairSpeed` (`UnitDef.cpp:366`). | `UnitDef.cpp:366` | Low |
| Resurrect, capture | O | Not on the stage 1-3 roster. Capture (`capture_defs.lua`) is stage 4. | | Low |
| Commander, morph | O (stage 3) | `unit_morph` (1266 lines) and `unit_commander_upgrade` (755 lines) are not read in full. | | Medium at game start |
| Stockpile | O (stage 3) | `unit_stockpile.lua`. | | Low |

### 1.5 Terrain hazards

| Mechanic | Tag | How in the sim | Source | Exploit risk |
|---|---|---|---|---|
| Deep water, cliffs, slopes | E | `terrain.py` impassability and speed. | `terrain.py:152-175` | Low |
| Lava and void | A | No class enters (`terrain.py` docstring); lava damage is not simulated. `map_lava.lua` (918 lines) is not read; the kill rule is taken from `terrain.py`. | `terrain.py` docstring | Low if no entry |
| Water planes | E | Engine plane at 0; ZK `waterlevel` moves terrain (`terrain.py` docstring). | `terrain.py` docstring | Low |

## 2. Zero-K Lua gadgets

Count: `Zero-K/LuaRules/Gadgets` holds 247 entries. That includes 3 directories (`CAI`, `Include`,
`Shaders`) and 2 text files, so there are **242 `.lua` gadgets**. The overview's 247 should read 242
(see objections). **Measured**: I read headers and grepped engine API calls per gadget. I did not
read every file in full. Files marked "not read" need reading before the stage that uses them.

### 2.1 Changes balance enough to model

| Gadget | Effect (from calls and header) | Handling | Stage |
|---|---|---|---|
| `unit_mex_overdrive` (2090 lines) | Energy to metal via `sqrt(E) x 0.25`, grid optimisation, payback | Copy the formula and the optimiser's objective, approximate payback | 3 |
| `unit_energy_drain` | Continuous energy use via `UseUnitResource` | Model per unit | 3 |
| `unit_econ_mult` | Team income multipliers and auto-handicap | `allyTeamIncomeMult` from modoption; handicap ignored | 3 |
| `unit_regeneration` | Idle regen from `customParams.idle_regen` (`unit_regeneration.lua:26`). The header says "air units", but the gadget applies to any unit with the param. | Model; check which ground units carry the param | 2 |
| `unit_repair_speed` | Changes costs for repair (`SetUnitCosts`) | Approximate as repair rate | 3 |
| `unit_timeslow` | Slow via rules param and `SetUnitTarget` | Approximate, see 1.2 | 2 |
| `unit_paralysis_damage` | EMP, `SetUnitHealth` | As 1.2 | 2 |
| `unit_stealth`, `unit_cloak_shield`, `unit_decloak_damaged` | Cloak, LOS mask, decloak on damage | As 1.3 | 2 |
| `unit_is_on_fire`, `unit_ward_fire` | Burn damage over time | Approximate as DoT | 2 |
| `unit_invincibility`, `unit_untargetable` | Flags | Model | 2 |
| `unit_overkill_prevention` | Retargets to avoid overkill (`SetUnitTarget`) | Model, see 1.2 | 1 |
| `unit_target_on_the_move`, `unit_target_lookahead`, `unit_proximity_targetting` | Lead and proximity targeting | Approximate with the engine's leading rule | 1 |
| `unit_target_priority` (480 lines) | Overrides target order ("the engine seems to be based on random numbers") | Not read in full. Stage 1 uses the category rule; read before stage 2 | 1 |
| `weapon_reaim_time` | Delay after aim changes (`SetUnitWeaponState`) | Model as reaim delay | 1 |
| `weapon_single_hit` | One hit per unit per projectile | Model | 1 |
| `weapon_area_damage` | Area damage behaviour (`SetProjectileCollision`) | Covered by blast rule | 1 |
| `weapon_shield_merge` | Shield projectile merge | Approximate with shields | 2 |
| `weapon_impulse`, `weapon_noexplode_speed_damage` | Push and speed-based damage | Impulse applied to separation; the speed rule is not read | 2 |
| `weapon_projectile_retarget`, `weapon_projectile_radar_homing`, `weapon_projectile_chain`, `weapon_projectile_gravity`, `weapon_projectile_hold` | Projectile behaviour | Missiles approximated as flight time; the rest not read | 2 |
| `unit_spherical_los`, `unit_radar_wobble` | Sensors | As 1.3 | 2 |
| `unit_stockpile` | Stockpile | Model | 3 |
| `unit_jumpjets`, `unit_jumpjet_pilot`, `unit_jumpjet_pilot_2`, `unit_impulsejumpjets`, `unit_fall_damage` | Jump and fall damage | Approximate, see 1.2 | 2 |
| `unit_shield_charge`, `unit_shield_link` | Shield units | Approximate | 2 |
| `unit_morph`, `unit_commander_upgrade` | Morph and level-ups | Not read in full | 3 |
| `unit_explosion_spawner`, `unit_self_destruct`, `unit_instant_self_destruct`, `unit_noselfd` | Death and self-destruct effects | Model | 1 |
| `unit_attributes`, `unit_attributes_generic` | Per-unit attribute overrides (`SetUnitRulesParam`) | Not read; check the unit files for overrides | 1 |
| `map_lava` | Lava handling | Terrain rule, see 1.5 | 1 |
| `unit_rezz_hp`, `unit_nanoframe_death` | Resurrect HP, nanoframe death | Stage 3 | 3 |
| `game_globallos` | Global LOS modoption | Fog switch | 1 |
| `cmd_factory_stop_production`, `cmd_retreat` | Stop production, retreat command | Map to order kinds (4.7) | 2 |

### 2.2 Not relevant to ground-factory play, or out of scope

| Gadget | Reason |
|---|---|
| `unit_xp` | No stat effect in ZK (1.2). Visual only. |
| `unit_bounty` | Only `SetUnitRulesParam`; no effect found. |
| `game_perks`, `game_rogue_progression` | Not read beyond the header ("Perks", "Unlock system"). Verify before stage 2 that matchmaking games have no stat changes. |
| `unit_terraform`, `weapon_terraform_explosions` | Terraform disabled by user direction. |
| `unit_capture`, `capture_defs` | Stage 4. |
| `unit_teleporter`, `api_teleport_helper` | One unit type. Stage 4. |
| `unit_grey_goo`, `unit_global_zombie_resurrect`, `api_zombie_helper`, `chicken_control` | Chicken and zombie modes. |
| `game_sudden_death`, `game_point_control`, `game_share_mode`, `game_support_powers`, `game_support_mana_handler`, `game_resign`, `unit_planetwars`, `mission_*` | Game modes, campaigns, the bot's own resign rule. |
| `unit_antinuke_handler`, `unit_missilesilo`, `unit_missilewarn`, `unit_bait_prevention` | No nukes on the factory roster. |
| `unit_air_plants`, `unit_aircraft_crash`, `unit_bomber_*`, `unit_fighter_pullup`, `unit_gunship_*`, `unit_control_gunship_*`, `unit_zenithskylinecheck`, `unit_vertical_swarm`, `unit_carrier_drones` | Air, stage 5 (after the ground stages). |
| `ai_*`, `CAI`, `cmd_*` except the two in 2.1, `dbg_*`, `lups_*`, `gfx_*`, `feature_fx`, `wake_fx`, `unit_glow`, `unit_tint`, `unit_icongenerator`, `unit_model_rescale`, `unit_grass_cutter`, `unit_retexture_custom_param`, `cus_gl4`, `api_*` (not listed above), `start_*` (scenario setup only), `mex_placement`, `mex_spot_finder` (the map dump supplies spots), `endgame_graphs`, `awards`, `mod_stats`, `typemap_options` | Debug, visual, UI, plumbing or setup. About 150 of the 242 fall here by name and header. |

## 3. Data pipeline

### 3.1 What exists and what is missing

`bots/zkbot/data/units.json` (615 entries; 122 factory-built mobile types, of which 13 are builders
and 109 are not, **measured**) holds cost, HP, speed, `losRadius`, `radarRadius`, `maxWeaponRange`,
`buildSpeed`, footprint-free build data, and role. `weapons.json` (276 units) holds per-unit DPS
(ground and air), `aoe`, `range`, and per-weapon `{name, dps, range, air, ground, aoe, paralyzer,
type}`. `abilities.json` holds tags. `ControllerAISerialization.cpp` adds `armorType`, `turnRate`,
`radius`, `mass`, `waterline`, `maxWaterDepth`, `wreckName`, `extractsResource`, `resourceMake`,
`storage`, `buildDistance`, `reclaimSpeed`, `resurrectSpeed`. The dump is available live through
`unit_def_by_id`.

Missing for the sim (not in any current JSON):

| Field | Needed by | Where it is in source |
|---|---|---|
| `acceleration` (`maxAcc`), `brakeRate` (`maxDec`) | movement | unit `.lua`, `UnitDef.cpp:455-459` |
| `movementClass` and move-def values (`footprint`, `maxSlope`, `maxWaterDepth`, `slopeMod`, `depth` coefficients, `crushStrength`) | pathing and speed | `movedefs.lua` |
| `collisionVolumeScales`, `collisionVolumeType` | collision radius | unit `.lua` |
| `reloadtime`, `burst`, `projectiles`, `weaponVelocity`, `beamtime`, `sprayAngle`, `tolerance`, `turret`, `accuracy`, `movingAccuracy`, `dynDamage*`, `edgeEffectiveness`, `paralyzeTime`, `impulseFactor`, `stockpile` | weapons | `WeaponDef.cpp:71-176` tag names |
| damage per armour class (`damage.planes`, `damage.shield`, `damage.default` after post-processing) | damage | `armordefs.lua` |
| `cloakCost`, `cloakCostMoving`, `minCloakDistance`, `decloakDistance` | cloak | `UnitDef.cpp:497-501` |
| `idle_regen`, `timeslow` and other `customParams` | effects | unit `.lua` |
| `extractsMetal` per mex | income | `UnitDef.cpp:305` (value from the dump) |
| `empDeclineRate` | EMP | engine `globalUnitParams` (`Unit.cpp:998`) |

### 3.2 Extraction

Primary source: a LuaRules gadget, `tools/sim_dump.lua`, run once inside a game or headless engine.
It writes `UnitDefs`, `WeaponDefs`, move defs and armour defs as JSON, after post-processing.
This is the authoritative source for the sim, because `armordefs.lua` and `unitdefs_post.lua` change
values that the raw files do not show. This needs one engine run; it is not started by this plan.

Cross-check: extend `tools/extract_weapons.py` (lupa, raw unit files) to read the new fields, and
diff against the dump. Report any field that differs. Fields that differ are post-processed (expected
for `damage.default`, `customparams.stats_damage`).

Output: `sim/data/units.v2.json`, `sim/data/weapons.v2.json`, `sim/data/movedefs.json`,
`sim/data/armordefs.json`. Keep `bots/zkbot/data/*.json` as they are; the bot still uses them.

Existing tools reused: `tools/extract_weapons.py` (PRELUDE, `lower`, `num`), `tools/extract_abilities.py`
(tags, roles), `tools/dump_defs.py` (ControllerAI dump), `tools/catalog_check.py` (role comparison).

### 3.3 Map data

`tools/terrain_map.py` reads `.smf` from the map archive and produces a height map (8-elmo squares),
the slope map (16-elmo cells, `terrain.py:slope_from_corners`), metal spots `(x, z, income)`, and the
region graph (`terrain.TerrainMap`, with `Gate`s per class). Output per map: one `.npz` with `height`,
`slope`, `water_damage`, `spots`, `start_positions`, plus the pickled region graph for the policy's
tokens. TitanDuel 2.2 (`maps.json`, **measured**): 10 x 10 map units, height -105 to 360, 32 metal
spots, 6 start positions, water share 1.9%.

Existing gap: `tools/terrain_map.py --check-graph` compares region routes with the 16-elmo grid. The
sim's flow fields are the same grid, so the same check applies to sim paths (5.2).

## 4. Architecture

### 4.1 Layout

```
sim/
  CMakeLists.txt              C++17, Release -O3, -ffp-contract=off, no -ffast-math
  include/zksim/
    types.hpp                 ids, fixed enums, catalog index
    world.hpp                 SoA state, handles with generation
    units.hpp, movement.hpp, combat.hpp, economy.hpp, fog.hpp
    rawframe.hpp              generated from bots/zkbot/policy/rawframe.py
    capi.h                    C ABI
  src/                        one .cpp per header
  tests/                      doctest (single header, no dependency); unit and golden tests
  data/                       generated JSON and map .npz (maps gitignored; JSON committed)
  py/zksim.py                 ctypes wrapper, buffer management
tools/sim_dump.lua            in-engine dump (LuaRules)
tools/sim_fightlog_replay.py  section 5 harness
```

Build in WSL: sources at `/mnt/c/Users/Thoma/Documents/sources/zk-ArenaAI/sim`, build directory at
`~/zksim/build` (not on /mnt/c, for speed). Compilers: `g++`, `clang`, `cmake` (overview).

### 4.2 Data structures

State is struct-of-arrays per entity class, with fixed capacity per game (512 units, 128 structures,
64 shots in flight per tick slot). Units are addressed by `uint32` handle = slot index plus a 16-bit
generation. Hot arrays: position (`x, z, h`), velocity (`vx, vz`), heading, HP, paralysis, cooldown
per weapon slot (max 4 weapons per unit type), order kind and target handle, team, type index.
Type data is read-only and shared across games (`const` tables built once per process).

Scheduled damage: a ring buffer of tick slots, sized to the longest flight time plus margin. Each
slot is a vector of `(shot_id, source handle, target handle or blast centre, weapon index, damage
array)`. This follows the engine's `waitingDamages` ring (`GameHelper.cpp:~160-170`).

### 4.3 Tick order (5-frame tick; decisions every 3 ticks)

| Step | Work | Cadence |
|---|---|---|
| 1 | Apply policy actions in ascending handle order; ack codes (4.7) | decision ticks |
| 2 | Economy: extraction, generation, overdrive (stage 3), storage clamp | every tick |
| 3 | Builders and factories: build steps, queues | every tick |
| 4 | Targeting: re-evaluate weapons with no valid target, and every 3 ticks for the rest | 3 ticks |
| 5 | Firing: create shots; hitscan damage now; projectile shots scheduled | per weapon |
| 6 | Resolve scheduled damage due this tick; blast falloff; overkill bookkeeping | every tick |
| 7 | Movement: velocity and heading update (sub-stepped if 5.2 says so), flow-field lookup, collision separation | every tick |
| 8 | Deaths: HP <= 0, death explosions queued, wreck created, metal value recorded | every tick |
| 9 | Status timers: paralysis decay, slow and disarm timers | every tick |
| 10 | Fog: LOS and radar per team | decision ticks, plus cache refresh |
| 11 | Raw frame emission | decision ticks |

Note the engine's slow update is 15 frames (`GlobalConstants.h:60`) and the decision cadence is 15
frames, so decisions and the engine's own slow-update logic line up.

### 4.4 Batch API, threads, binding

Choice: C ABI in `capi.h`, loaded with `ctypes`, numpy buffers passed by pointer.

Alternatives considered:

| Option | Verdict |
|---|---|
| pybind11 | Works. Adds a build-time dependency in WSL and per-object overhead for every frame record. Rejected for now. |
| Python multiprocessing, one sim per process | 16 copies of maps and higher IPC. Rejected. |
| One sim with intra-game threading | Games are small (about 400 units). Threads per game would spend most time on synchronisation. Rejected. |
| Rust or other | Not installed (overview). |

API (sketch):

```c
zk_create(const zk_config*, zk_handle** out);            // once per worker slot
zk_reset(zk_handle*, const zk_scenario*, uint64 seed);   // scenario = map id + start setup
zk_step_batch(zk_handle** games, int n,
              const zk_action* acts, const int* n_acts,  // flat, per game
              zk_frame* frames_out,                       // preallocated, decision ticks only
              zk_ack* acks_out);
zk_snapshot(zk_handle*, void* buf, size_t cap);           // debug and replay
```

Threads: a pool of 16 workers, each owning a fixed slice of games (no work stealing). A batch of
`n` games is split across workers; the call returns when all have stepped. One `step_batch` call
steps every game by one decision interval (3 ticks). Results are written to disjoint buffers, so
no locking.

Determinism: each game has its own RNG (xoshiro256**, seeded by `(seed, game index)`), no
iteration over hash maps, all sums in handle order, compiler flags as in 4.1. Test: the same
seed set gives identical frame hashes with 1 worker and with 16 workers (tests/determinism).

### 4.5 Spatial index and budget-critical paths

Uniform grid, cell 128 elmo, rebuilt each tick by counting sort (no per-unit insert). Range queries
use cells within `range / 128` rings. Max range in the roster bounds the ring count (ZK long-range
units about 820 elmo, so about 7 rings).

Flow fields: one float grid per (move class, region centre), precomputed once per map and shared
read-only across games. 320 x 320 cells gives about 400 kB per field, so 31 regions x 6 classes is about
75 MB per map (120 regions: about 290 MB). The policy's fine 9-way offset steers locally from the region's
field, so offsets add no fields. Structures block cells dynamically and are handled by local avoidance, not by
recomputing fields; S-M2 measures whether that's good enough. Per-goal Dijkstra at run time isn't budgeted in
6.1, and at about 100k cells it would cost far more than a tick (estimate).

### 4.6 Raw frame (shared with the live bot)

The sim and the live bot each emit the same raw frame. The featurizer (02) reads only this.

Reconciled: `bots/zkbot/policy/rawframe.py` owns the schema, with the fields in 02 section 2.2.  It holds
only what the live bot can supply, so `heading`, `weapon_ready`, `para` and `ack` are not in the frame; the sim
keeps them as internal state, and `ack` stays available for diagnostics.  `sim/include/zksim/rawframe.hpp` is
generated from that file, and a test compares the field lists.  The live producer is `rawframe.from_brain` (02).

What the sim must emit each decision, beyond positions and health:

- `side`: 0 own, 1 enemy in LOS, 2 radar blip, 3 remembered structure, 4 hidden enemy (critic only).  Side 4
  must never reach the actor's inputs.
- Blip identity: `def_id` is -1 unless the unit is in LOS, or was seen and has stayed in radar contact since
  (`rts/ExternalAI/AICallback.cpp:593-597`, LOS_PREVLOS with LOS_CONTRADAR).
- `blip_top_speed` over the same 2 s window as `bots/zkbot/blips.py`.
- Order memory (`ord_kind`, `ord_age`, `ord_x`, `ord_z`, `ord_target`), written from the controller's command
  log, as a player remembers it, not read from unit state.
- `idle_uids`: units with no orders, matching topic 7 (UNIT_IDLE).
- No pending shots: live play can't see them.

The static bundle (regions, gates, spots, passability) comes from the same `terrain.py` code and the same height
arrays as live.  Region dynamic values are computed by the featurizer from the unit records, not emitted.

<!-- Superseded proposal, kept for the reasoning on sizes and record kinds. -->
First proposal, superseded by 02 section 2.2 (dtype names are numpy-style):

| Record | Fields |
|---|---|
| Header | `schema_version u2` (=1), `game_id i4`, `frame i4`, `time_s f4`, `own_team u1`, `n_units i4`, `n_regions i2`, `n_sites i2` |
| `unit` (one per visible entity; own units always) | `side u1` (0 own, 1 enemy, 2 neutral), `flags u2` (BLIP, MEMORY, UNBUILT, STUNNED, CLOAKED, IN_LOS, IN_RADAR, FACTORY, STATIC), `type i4` (catalog index, -1 unknown or blip), `uid i4` (live engine id or sim handle), `x f4, z f4, h f4`, `heading f4` (radians), `vx f4, vz f4` (own units, or visible enemies), `hp f4` (fraction), `build f4` (fraction), `para f4` (paralysis / max HP), `weapon_ready f4` (own: weapon 0 cooldown fraction), `order u1` (order kind, 4.7), `target i4`, `ack u1` (last order result, 4.7), `region i2` (-1 if none) |
| `region` | `rid i2`, `own i2`, `enemy_visible i2`, `metal_spots i2`, `owned_spots i2` |
| `site` | `sid i2`, `x f4, z f4`, `kind u1`, `valid u1` |
| `team` | `team u1`, `metal f4`, `energy f4`, `metal_income f4`, `energy_income f4`, `metal_storage f4`, `energy_storage f4`, `spent_metal f4` |

Limits: `n_units` up to 512 own plus visible enemies; the featurizer's cap (roughly 460 entities at
the busiest point, overview) fixes the array size. Blip records carry `type = -1` and no HP.

### 4.7 Fog and action application

Fog per side, computed at decision ticks: own units always visible to own side. Enemy entity visible
if (a) inside any own unit's LOS (raycast), or (b) inside any own radar circle, which gives a blip, or
(c) a structure seen once stays as MEMORY with its last state. Air LOS is a plain circle.

LOS cost: a raycast per unit per decision is too expensive (estimate, not measured: about 3,800 cells
per unit at 560 elmo and 16-elmo cells). Plan: cache per (cell, quantised height of eye) with
invalidation when a unit moves across a cell, and report the measured cost in M6.

Actions: 02 section 4.1 is the list, one slot per own controllable unit (at most 128) and factory each
decision.  Sim behaviour for each kind, with the ack the sim records for diagnostics:

| Kind | Sim behaviour | Ack when it cannot be honoured |
|---|---|---|
| KEEP | Current order continues | n/a |
| MOVE(region, fine) | Straight line if the way is clear (`cmd_raw_move.lua`), otherwise flow field | `UNREACHABLE` if no path (keeps current order) |
| FIGHT(region, fine) | As MOVE, but stops to engage what comes in range | `UNREACHABLE` |
| ATTACK_UNIT(enemy) | Chase until in range; fire | `TARGET_LOST` if the enemy is not visible or dead |
| ATTACK_GROUND(point) | Move into range, fire at the point | `UNREACHABLE` |
| STOP | Stop and fire at will | n/a |
| MANUAL_FIRE(enemy) | Fires the manual weapon, on its own reload | `TARGET_LOST` |
| JUMP(region), CLOAK on/off | Per 1.2 and 1.3; needed by S-M7 for units with the tags | `UNREACHABLE` |
| BUILD, ASSIST, REPAIR, RECLAIM, PRODUCE (stage 3) | Per 1.4 | `NOT_BUILDABLE`, `INSUFFICIENT` (stalls, not an error), `ROSTER_MISS` |

Rules: actions are applied in handle order at decision ticks; an action on a unit that died this
tick acks `TARGET_LOST`. Units that finish their order revert to idle, which fires at the nearest
enemy in range and does not move (engine idle behaviour, to be confirmed in 5.4). The `ack` field
is for diagnostics only. The policy learns of failures from masks and from its own order-memory
features, since the live bot has no equivalent feedback.

## 5. Validation

### 5.1 Stage 1 gate: replay of logged fights

Data: `results/*/game-*/fights-team0.jsonl`. **Measured**: 184 games, 184 team-0 files, 46,389 raw
snapshots. 1,656 contain `terraunit` and are dropped by `train_fightnet.py:load`, leaving **44,733**
usable snapshots (the overview's 46,389 should be 44,733 for the gate). Median 15 entities per
snapshot (max 96, the logger's cap). Measured share of snapshots containing: static structures 84.7%,
unbuilt entities 52.1%, remembered enemy structures 43.8%, radar blips 29.6%, retreat-flagged units
14.0%. Snapshots with both sides present: all of them (the logger needs contact). The label
`their loss > our loss at 20 s` is 27.3% positive; 11.1% of snapshots have zero loss at 20 s.

Entity layout: `[side, name, dx, dz, dh, hp_frac, flags]` (`fightnet.py`, `fight_entities`). Flags:
1 static, 2 air, 4 retreat, 8 blip, 16 unbuilt, 32 memory.

Baselines measured on the same snapshots (AUC for the label, **measured**, computed in this session
from the logs with the same `log((their+50)/(our+50))` score as `train_fightnet.py`):

| Score | AUC at 10 s | AUC at 20 s |
|---|---|---|
| Value ratio, tracked units only (cost x hp, no blips or memory) | 0.51 | 0.56 |
| Value ratio, all entities | 0.45 | 0.50 |
| fightnet (HANDOFF 6d.2, its own split) | not stated | 0.59-0.62 |
| Power ratio (`train_fightnet.py`, HANDOFF) | not stated | 0.50-0.55 |

A few snapshots gave non-finite scores in the baseline; I did not investigate (indicative only).

The gate, as proposed:

1. **Comparator.** Retrain fightnet with `tools/train_fightnet.py` in grouped 5-fold CV over the 184
   games (same arguments; the split logic is `random.shuffle` over sorted game directories, seed 0).
   Score fightnet's out-of-fold predictions. The exported `fightnet.npz` is not used as the comparator.
2. **Sim prediction.** For each snapshot, eight seeds, 20 s horizon. Our and their losses are
   counted on the same tracked units as the logger (non-blip, non-memory), with cost x fraction, like
   `_fl_collect`. Score is `log((their+50)/(our+50))`, the same as fightnet.
3. **Metric.** AUC against `their > our` at 20 s (primary) and 10 s. Also the calibration of
   predicted fraction-lost against measured, and sign agreement.
4. **Uncertainty.** Block bootstrap by game (184 blocks, 1,000 resamples). Snapshots of one game are
   strongly correlated (2 s spacing, same fight). Report the CI for the difference
   `AUC_sim - AUC_fightnet_oof` and for `AUC_sim - AUC_value_tracked`.
5. **Pass.** The sim's AUC exceeds the out-of-fold fightnet AUC, and the 95% CI of the difference is
   above 0. Also reported: the sim must beat the tracked value baseline with a CI above 0, or the
   sim adds nothing over counting value.
6. **Subsets.** Report separately: (a) snapshots with no retreat flag (sim has no retreat, 86%); (b)
   snapshots with no blip or memory entity (where the loss label is most complete).
7. **Default behaviour.** Fixed before any results are looked at:
   - Mobile units with no order: hold position, fire at the best target in range.
   - Primary variant ("advance"): mobile units with no target in range move at full speed toward
     the nearest enemy until in range.
   - Second variant ("hold"): no movement, fire only. Reported, not used for pass or fail.
   - Snapshot units start at the logged positions, with zero velocity, facing the nearest enemy,
     HP from `hp_frac`, paralysis zero, no cooldown. Unknown types (none in the logs) would use
     the catalog default.
   - Static entities and unbuilt entities: unbuilt counts as no fire and no HP scaling until built
     (treated as HP-proportional build fraction, not built).

Is the gate meaningful: the tracked-value baseline (0.56 at 20 s) sits within 0.03-0.06 of fightnet.
A sim that only reproduces value counts would pass a weak gate. Hence the baseline row and the
requirement to beat it with a CI. The gate could be too hard if the sim's combat is correct but the
label is dominated by unobserved enemies (blips and memory account for much of the loss ambiguity);
that is why subset (b) is reported.

Flaws in the existing protocol, to fix in the gate:

- Comparator is in-sample (see objection O2).
- Count (46,389) includes the 1,656 `terraunit` snapshots (O3).
- Label counts only tracked units, so the enemy's unseen losses are never counted. Their losses are
  undercounted exactly where the enemy is partly unseen.
- The AUC has no CI; with about 27 validation games the margin of 0.03-0.06 is inside the noise.
- The label is loss in 20 s, not win or loss; it is a proxy. This is acceptable for stage 1 only.

### 5.2 Movement, economy and build checks

Unit tests (doctest, in `sim/tests`):

| Test | Oracle | Pass |
|---|---|---|
| Speed field per class and cell | `terrain.speed_field` on 1,000 random height and slope cells | exact within float error |
| Top speed reached | `ceil(speed/30 / acc)` frames from the formula in `GroundMoveType::ChangeSpeed` | exact |
| Turn-around | 180 deg at `turnRate` 3000 = 11 frames (formula) | within one frame |
| Blast falloff | `GameHelper.cpp:146-150` formula, computed by hand for edge 0, 0.5, 1 | exact |
| Overdrive per mex | `energyToExtraM` and the optimiser on a fixed grid, from the Lua | exact (stage 3) |
| Build step | `Unit.cpp:1995-2025` for a partial resource pull | exact |
| EMP decay | `Unit.cpp:992-1009` | exact |
| Flow field vs terrain route | route length on the 16-elmo grid vs `--check-graph` | within 5% |

Engine comparison (needs an engine run; not started): a headless scripted battery in a LuaRules gadget
(`tools/sim_scenarios.lua`, not written): spawn fixed unit sets at fixed positions, issue orders at
fixed frames, log per-unit position, heading, HP, weapon cooldown each frame to CSV. Run the same
scenario in the sim. Compare:

| Scenario | Metric | Threshold to pass |
|---|---|---|
| Straight-line move, cloakraid | arrival frame | within 2 frames |
| Turn-around | heading vs frame | within 1 frame |
| Two-unit duel, no overkill | time to kill | within 10% |
| Blast at a unit cluster | damage per unit | within 10% |
| Pathing around a wall | path length | within 5% |
| Terrain slope speed | speed per cell | within 2% |
| Re-order every 0.5, 1, 2, 4 s, per move class, on MOVE and FIGHT | mean speed against no re-orders | sim within 10% of the engine; the measured table sets the penalty |
| Clear and blocked moves | path taken, raw or pathfinder; arrival frame | within 5% |

Economy and build checks, using logs already produced by the bot (`state-team<N>.jsonl`, which the
HANDOFF says logs `spent` and income): compare realised income per minute vs the sim's
extraction-and-generation prediction for the same mexes and pylons. This needs the mex positions and
the pylon grid from the logs. Check build timing on one factory unit: the engine's `buildTime` vs
the sim at equal resources.

### 5.3 Exploit detection (also used for stage gates)

- Tick sensitivity: run the same policy in the sim at 3, 5 and 6-frame ticks. Win rate or metal
  exchange changing by more than the CI means the policy is exploiting tick granularity.
- Path sensitivity: swap the flow field for a straight-line cost in a test; large change means
  the policy is learning corridor squeezes.
- Blip leakage: count frames where the policy's observation contains enemy types that the engine
  would not show (type of a BLIP record must be -1). Test: assert in the featurizer golden test.
- Sim-vs-engine divergence: at each checkpoint, the army-only A/B in the real engine (stage 2 gate)
  is compared with the sim's win rate on the same policy. A gap beyond the CI is a red flag.

### 5.4 Unit and integration tests

Determinism (1 vs 16 workers, same frame hashes); snapshot round-trip; raw frame golden test
against a recorded frame (shared with 02); action ack tests (each row of 4.7); `zk_step_batch`
throughput test.

## 6. Sequencing

| Milestone | Deliverable | Acceptance test | Size |
|---|---|---|---|
| M0 Data | `sim/data/*.json` from the engine dump; lupa cross-check report | All 122 mobile roster units complete; dump vs lupa differ only in post-processed fields | medium |
| M1 Map | TitanDuel `.npz` (height, slope, spots, region graph, region fields) and a synthetic flat map for T-M2 | `speed_field` oracle tests pass; route check passes | small |
| M2 Combat core | Movement, targeting, weapons (hitscan, scheduled), blast, death, wrecks, value accounting | 5.2 unit tests; determinism test; 400-unit tick within budget (6) | large |
| M3 Stage 1 gate | Replay harness, fightnet CV retrain, baselines, report with CIs | Section 5.1 pass criteria, reported honestly even if they fail | medium |
| M4 Engine comparison | Scenario gadget and sim runner | Section 5.2 thresholds | medium |
| M5 Binding | C ABI, ctypes wrapper, 16-worker batch | 256-game batch steps within budget; determinism | medium |
| M6 Raw frame and fog | Emission, LOS cache, schema file | Featurizer golden test (with 02); fog cost measured | medium |
| M7 Army-only self-play | Kinds 0-9 (4.7) and acks; re-order penalty from M4; scripted attack-nearest baseline; scripted production schedule for 2c (03 section 3) | Stage 2 prerequisite: policy plays a full army-only game | medium |
| M8 Economy and build | Mex, generators, overdrive, build, factories | 5.2 economy and build checks | large |
| M9 Remaining mechanics | Jump, morph, stockpile, capture, air (stage 5) | Per mechanic tests | large |

The combat-only sim (M0-M3) is useful alone: it can score any two armies on a map, which is the fight
evaluator the bot already needs (fightnet) and can be used to pick attack timing in the rules bot.

### 6.1 Per-tick budget (5-frame tick, 400 units, 16 busy threads)

The overview's 370 µs is for the combat loop at 400 units with 16 threads (**measured** by the overview
author, throwaway code). Plan targets below are estimates, not measurements:

| Subsystem | Target µs / tick | Note |
|---|---|---|
| Spatial grid rebuild and queries | 25 | counting sort |
| Targeting (every 3 ticks) | 60 | amortised |
| Firing and scheduling | 30 | |
| Scheduled damage and blast | 40 | |
| Movement, flow lookup, separation | 150 | largest term |
| Deaths, wrecks, status timers | 20 | |
| Combat total | **325** | within the measured 370 |
| Economy and build (stage 3) | 50 | |
| Fog (decision ticks, amortised) | 40 | estimate; cache dependent |
| Raw frame emission (decision ticks, amortised) | 20 | |
| Total stage 3 | **435** | leaves headroom under 370 + 50 + 40 + 20 |

Game arithmetic (from the budget, not measured): 20 minutes at 6 Hz (5-frame tick) is 7,200 ticks.
At 435 µs that is 3.1 s per game per thread. With 16 threads the sim alone is about 5 games per
second, which is about 450,000 games a day as an upper bound, before policy inference. The overview's
live-play budget is a separate constraint.

## 7. Risks

| Risk | Where RL would exploit it | Detection | Mitigation |
|---|---|---|---|
| Focus fire ordering | Target choice is the most learnable thing; the engine's `TargetWeight` and ZK's priority gadget are not reproduced exactly | Compare target-choice logs in scripted battery (5.2); A/B in the real engine | Read `unit_target_priority.lua` before stage 2 |
| Dodging projectiles at tick scale | Move out of blast in the last tick | Tick-sensitivity test (5.3) | Use 5-frame ticks and sub-stepping for projectiles, or account for dodge probability |
| Corridor and route cheats | Flow-field paths differ from QTPFS | Route length test; straight-line variant | Keep the flow field on the same grid as `terrain.py` |
| Blip leakage | Enemy composition inferred from radar blips in the sim but not in the engine | Featurizer golden test; blip records carry no type | Enforce in the schema |
| Ridge hiding | Terrain LOS approximated | LOS test against engine's visible set on a scripted map | Raycast on the 16-elmo grid |
| Overdrive allocation | Exact optimum a policy can learn; the sim's optimiser may disagree | Income test against state logs | Copy the optimiser and test it against the Lua |
| Jump and teleport omitted | Policy plans around units that behave differently in the engine | Per-unit calibration errors | Stage 2 approximation before factoryjump units are fielded |
| Reward accounting | Wrecks or units that die out of view counted differently from fightnet's `track` rule | Compare reward breakdown with fightnet's tracked rule (03) | Share one accounting function |
| Non-determinism | Noise looks like skill | Determinism test (4.4) | Per-game RNG, no shared state |
| Fog cost | LOS cache gets expensive with 400 units | Measured in M6 | Coarser LOS update cadence if needed |

## 8. Stage 1 results (2026-11-04)

Built: S-M0 to S-M4, in `sim/`, `tools/sim_data.py`, `tools/sim_engine.py`, `tools/simgame/`,
`tools/sim_battery.py` and `tools/sim_fight_gate.py`.  **Gate 1 did not pass** (rerun after the pathing change in 8.6: unchanged).  Under "Stages and gates" in the
overview that stops the work for a decision by the user; stage 2 has not started.  Sections 0 to 7 above are
the plan as written, and this section overrides them where they differ.

Run again with:

```
python tools/sim_battery.py engine                                # Windows, about 70 s
wsl: ~/zkml/bin/python tools/sim_battery.py compare               # results/simbattery/
wsl: ~/zkml/bin/python tools/sim_fight_gate.py --tag gate         # results/simgate/; about 2 min of replay, fightnet 45 s per fold if not saved
wsl: ~/zkml/bin/python sim/tests/test_sim.py
```

### 8.1 Gate 1(a): replay of logged fights, fail

44,733 snapshots, 184 games, 8 seeds, no unknown unit names.  95% intervals from 1000 resamples of whole games.

| Score | AUC at 10 s | AUC at 20 s (primary) |
|---|---|---|
| Sim, advance | 0.631 [0.621, 0.642] | **0.591** [0.580, 0.604] |
| Sim, hold | 0.617 [0.607, 0.627] | 0.589 [0.580, 0.599] |
| fightnet, out of fold | 0.644 [0.632, 0.657] | **0.665** [0.654, 0.677] |
| Value ratio, tracked | 0.506 [0.494, 0.516] | 0.564 [0.552, 0.576] |

| Difference | 10 s | 20 s |
|---|---|---|
| advance - fightnet | -0.013 [-0.030, +0.005] | **-0.074 [-0.092, -0.053]** |
| advance - value | +0.125 [+0.110, +0.141] | **+0.027 [+0.011, +0.045]** |

The pass needs both 20 s differences above 0.  The sim beats counting value and loses to fightnet.  At 10 s it
ties fightnet.  Subsets at 20 s, advance / fightnet / value:

| Subset | Share | AUCs | advance - fightnet |
|---|---|---|---|
| No retreat flag | 86% | 0.595 / 0.662 / 0.564 | -0.068 [-0.088, -0.046] |
| No blip or memory | 40.8% | 0.598 / 0.656 / 0.498 | -0.058 [-0.075, -0.042] |
| Both | 35.6% | 0.598 / 0.657 / 0.494 | -0.059 [-0.077, -0.041] |

So removing retreats, blips and memory doesn't close the gap.

What the misses look like (advance, 20 s):

- Mean loss per snapshot: predicted 358 ours, 302 theirs; measured 240 and 136.
- In snapshots predicted to lose 0.4 to 1.0 of the tracked value, the sim predicts 0.855 of ours and measures 0.464
  (n = 23,366), and 0.908 of theirs against 0.393 (n = 28,013).  **Sim fights go much closer to the death than real
  ones do.**
- Sign agreement on who loses more is 0.585 (hold 0.420).  Spearman on loss size is 0.593 ours and 0.592 theirs.

Possible causes, none tested: real units retreat, disengage or get ordered away, while replayed units keep
fighting under a fixed behaviour; reinforcements enter real fights and not replayed ones; enemy losses out of view
aren't in the label; targeting or damage is wrong in ways the duels don't show.  The 20 s score falling below the
10 s score fits a replay whose units stop resembling what real units did, but doesn't prove it.

fightnet's out-of-fold 0.665 is above the 0.59 to 0.62 the plan assumed, which came from single
splits.  It is also a statistical model trained on the same kind of labels, so it learns how the
bot's fights tend to go, including retreats.  The replay has to get there from unit stats alone.

The full gate ran once, after the battery fixes.  A 2,000-snapshot smoke test had been seen earlier; every sim
change after it came from engine evidence.  **Don't tune the sim against these snapshots**: they would stop being a
test.

Data notes: 1,316 of 812,071 logged entities have negative health (lowest -5.5; the cause isn't traced).  They put
570 tracked entities in 550 snapshots and made 3 value scores non-finite.  Clamping changes no table above at three
decimals, and `sim_fight_gate.py` now clamps at load.  The published numbers come from the run before the clamp.

### 8.2 Gate 1(b): engine battery, 108 of 113

165 scenarios on Titan Duel 2.2; duels are repeated 5 times.

| Check | Result | Note |
|---|---|---|
| Straight line | 4/4 | |
| Turn-around | 4/4 | |
| Duel time to kill | 8/8 | see deviation 6 |
| Blast | 9/9 | |
| Slope speed | 8/8 | |
| Re-order, per class and rate | 64/64 | |
| Path length around a wall | 7/8 | route/tank/1: engine 3738, sim 2980, a long detour |
| Blocked-move walk frames | 4/8 | route/kbot/1 225.7 vs 252.8; route/tank/1 1314 vs 1015; route/spider/0 708 vs 760; route/spider/1 572 vs 697 |

Sim speed along the engine's own paths matches in every slope bucket, so the walk-time misses come from which route
is taken, not from the speed formula.  Unexplained: in 2 of 5 repeats of the cloakassault against vehassault duel, the
Knight does no damage to the Ravager for 700 frames.

### 8.3 Measured engine facts

Plain move orders cost speed, and fight orders don't.  **A CMD_MOVE stops the unit until its next slow update**
(`UNIT_SLOWUPDATE_RATE = 15`, `rts/Sim/Misc/GlobalConstants.h`); braking starts about 2 frames after the order and
walking restarts at the slow update (absolute frame %15 == 4 in 106 of 106 samples, in a small game).  FIGHT orders
kept 1.00 of the distance at every rate, even with a shifting goal.  This settles the open question in row "Re-order
cost" of 1.1.  `bots/zkbot/smoothorders.py`'s docstring says a unit slows "while it waits for a path"; the measurement
says otherwise.

Share of the no-reorder distance kept, ORDER_FRAME = 5:

| Class | 0.5 s | 1 s | 2 s | 4 s |
|---|---|---|---|---|
| kbot | 0.77 | 0.89 | 0.95 | 0.98 |
| tank | 0.26 | 0.65 | 0.82 | 0.93 |
| spider | 0.34 | 0.68 | 0.84 | 0.94 |
| hover | 0.10 | 0.55 | 0.77 | 0.91 |

The numbers depend on the order's phase against the slow update (another phase gave tank 0.01/0.52/0.76/0.90, kbot
0.51/0.76/0.88/0.95).  A goal shifted by 60 elmos behaves like the same goal.  The sim reproduces the class differences with one
delay rule and each class's acceleration.

Other corrections to the plan:

- Speed on a slope is directional (`CMoveMath::GetPosSpeedMod`, `GroundMoveMath.cpp:48`), and descending isn't
  slowed.
- Spray scatters sideways only.
- Burning ground and burn damage matter in duels (`weapon_area_damage.lua`, `unit_is_on_fire.lua`), so both are
  in the sim, along with script burst weapons (`script_reload`, `script_burst`).
- Units stop up to about 50 elmos short of a move goal.
- Level 1 trainer commanders have only the first weapon listed: heavy machine gun, or light particle beam for support.

### 8.4 Budget: missed for now

Single thread, 400 units, 200 ticks, mixed armies:

| Scenario | Flat map | Titan Duel |
|---|---|---|
| Hold | 57 µs | 70 µs |
| Fight orders, one shared goal per side | 151 µs | **423 µs** |
| Advance, each unit to its nearest enemy | 194 µs | **17,721 µs** |

`test_budget_400_units` in `test_sim.py` reports about 1250 µs: advance, with units spread over the whole map.  Plan: 325
to 370 µs.  No optimisation has been done.  These are single-thread numbers, and the plan's figure was with 16 busy threads,
so a loaded machine may be slower.

**Where it goes (2026-11-05):** building distance fields at run time.  `tools/sim_profile.py` on a `-DZKSIM_PROFILE=ON` build,
200 ticks (its armies differ slightly from the table above, and the timers add some cost of their own):

| Scenario | Total µs/tick | Dijkstra | Everything else | Dijkstra with a cache of 256 fields |
|---|---|---|---|---|
| Fight, one goal per side | 481 | 260 | 221 | 261 |
| Fight, 9 group goals | 128,637 | 128,317 | 320 | 727 (977 total) |
| Fight, every unit its own goal | 327,416 | 327,031 | 385 | 4,179 (4,461 total) |
| Advance, two armies | 9,693 | 9,411 | 282 | 6,032 (6,309 total) |
| Advance, whole map | 1,343 | 1,208 | 135 | 1,102 (1,230 total) |

One build is about 4.6 ms: Dijkstra with a binary heap over a window of up to 257 x 257 cells.  With 9 group goals and 4 move classes, 36 fields compete
for the 16 in `FIELD_CACHE`, so the cache thrashes: 28 builds per tick.  A larger cache gives identical runs, since results depend
only on map, class and goal.

The rest is within budget.  In the first scenario: movement frames 43, targeting 38, `line_free` 35, separation 20,
lookahead 17, all other phases under 7 each.

So self-play doesn't need faster combat code; it needs goals from a bounded set with fields built ahead of time, as 4.5 planned
(region centres per class, shared read-only between threads).  Replay's per-unit goals can stay slow.  Section 8.6 has what
was built, and these numbers are from before it.

### 8.5 Deviations from section 5, all chosen before seeing the full gate

1. Units start with a random weapon phase, not "no cooldown", since a logged fight is already under way.
2. The score uses mean losses over the seeds.
3. The label uses the logged `out` as it stands.
4. The battery lane is a levelled patch of Titan Duel (x 64-2064, z 576-1088, height 175.5), so slope doesn't enter
   the speed and order checks.
5. Battery metrics start when the unit leaves the start, and turning uses the half-turn rate, to remove slow-update phase.
6. Duel limit is max(10%, one 5-frame tick): a sim death is only seen at a tick's end, so short duels can't be closer.
7. Orders go on the tick grid, at frame 5.
8. Fire damage, sideways scatter and script bursts were added after the battery showed them missing.

### 8.6 Pathing: segment centres and a next-hop table (2026-11-05)

This replaces the per-goal fields that 8.4 profiled.  A run does no pathfinding: `tools/sim_data.py` works out routes when it
exports a map (`sim/py/zknav.py`), and they load with it.

**Segments.**  Start from a connected piece of a `terrain.py` region per move class (`fine_label`), with pockets under 64 cells
joined to a neighbour.  Split each one until its centres see 90% of it, or into 4 parts at most.  Leftover tiles go to the nearest
centre that sees them, even another segment's, and otherwise to the nearest in their own.  Parts under 16 cells join a neighbour.
Titan Duel: 24 regions, segments kbot 30, tank 53, spider 25, amph 30, hover 53.  Carabanossi: 76 regions, 113, 182, 81, 113, 182.

**Routes.**  Each pair of touching segments gets a gate on the far side, chosen so both centres see it where possible.  A next-hop
table gives the gate to take from any segment towards any other.  Rules in `World::waypoint`:

1. Line to the goal clear: walk straight.
2. Otherwise look up to 4 segments along the route and aim at the farthest gate or centre in sight.
3. Nothing in sight: aim at the segment's own centre.  Within 64 elmos of it, aim at the gate or goal anyway, since a turning circle can
   keep a unit from getting closer.
4. Stuck: a unit that covers under 20% of its top-speed distance in 3 s aims straight at the next gate or the goal, alternating, for 3 s, doubling
   with each attempt up to 24 s, and sliding along what blocks it.  Attempts reset when it enters a new segment.

**Memory**, in the C++ map:

| Map | One float per tile per field (before) | Label grid | Routes |
|---|---|---|---|
| Titan Duel | 57 MB, plus up to 68 MB per thread | 1.2 MB | 0.05 MB |
| Carabanossi | 1.2 GB, plus up to 68 MB per thread | 8.3 MB | 0.43 MB |

The grid is one short per tile per class, so a unit knows its segment.

**Speed**, µs per tick (`tools/sim_profile.py`, 400 units, 200 ticks).  There are no builds, so this is also the cold-start cost.

| Scenario | Original | Per-segment fields, total / long run | Now |
|---|---|---|---|
| Fight, one goal per side | 481 | 695 / 243 | 219 |
| Fight, 9 group goals | 128,637 | 2,106 / 270 | 247 |
| Fight, every unit its own goal | 327,416 | 2,797 / 263 | 234 |
| Advance, two armies | 9,693 | 1,039 / 273 | 246 |
| Advance, whole map | 1,343 | 636 / 132 | 122 |

The fight replay takes 32 s, down from 98 s originally.  With the timers on, `move.waypoint` is about 30% of a tick, and targeting 18%.

**Arrival** (`tools/sim_segment_check.py`, 300 reachable goals per class, within 4x the straight walk), same / other segment:

| Class | Titan Duel, now | Titan Duel, fields | Carabanossi, now | Carabanossi, fields |
|---|---|---|---|---|
| kbot | 99.3% / 96.0% | 99.3% / 100% | 98.7% / 91.3% | 96.0% / 93.3% |
| tank | 97.7% / 91.3% | 98.7% / 98.7% | 98.0% / 91.0% | 97.7% / 96.3% |
| spider | 100% / 100% | 100% / 100% | 100% / 98.3% | 98.3% / 99.0% |
| hover | 98.0% / 93.0% | 98.7% / 97.7% | 98.7% / 93.0% | 94.7% / 93.7% |

**The remaining misses are mostly stuck units, not slow routes.**  Of 200 Titan Duel tank goals across segments, 183 arrived on time, 1 late, and 16
never arrived within 16x: 10 stopped, 5 wandered, 1 circled.  For kbot, 197 arrived on time and 3 stopped.  That's with the rule above; before it,
tank was 182 on time with 18 failing, and kbot 194 with 5.  Tank and hover are the weak classes.  A policy could learn around this, so it needs watching in self-play.

**Engine battery: 105/113.**  Misses: route/tank/1 (never arrives; length 1626 against 3738), route/spider/0 (1228 against 1317, 811 against
708 frames), route/spider/1 (675 against 572 frames), route/hover/0 (1362 against 953, 274 against 195 frames), route/hover/1 (950 against 902).

**Gate 1(a) rerun (`--tag gate-stuck`): unchanged, still fail.**  20 s: advance 0.591, fightnet 0.665, value 0.564.  Advance minus
fightnet -0.075 [-0.092, -0.055]; minus value +0.027 [+0.010, +0.044].  The limit was chosen before this run, and the score hasn't moved
through three pathing designs, so pathing isn't what this gate responds to.

**Limits swept**, Titan Duel (`--max-parts`, `--seen-share`, `--height-band`, `--slope-band`).  Figures for tank, measured before the stuck rule; after the
visible-leftover and turning-circle fixes, except the first row:

| Limit | Segments, kbot / tank | Battery | Tank arrival, same / other |
|---|---|---|---|
| One centre per segment, before the fixes | 24 / 34 | 103 | 84.0% / 82.0% |
| 2 parts, 80% | 26 / 41 | 104 | 95.7% / 90.3% |
| **4 parts, 90% (chosen)** | 30 / 53 | 105 | 97.7% / 90.3% |
| 8 parts, 95% | 34 / 57 | 104 | 99.0% / 93.0% |
| 16 parts, 98% | 43 / 73 | 104 | 99.7% / 89.3% |
| Bands 80 / 0.2, 32 parts, 95% | 11 / 32 | 99 | 98.3% / 74.3% |

An exact split (no limit) did not finish within 500 s for 2 maps and 3 settings, so it was dropped.  Loose bands give the fewest segments,
but the worst routes and arrival, so they stay off.  Differences of 1-3 points are within this test's noise at 300 goals.

**Tried and replaced:** straight-only inside a segment (14-25% of kbot, tank and hover goals never arrived, battery 104); per-segment
whole-map fields plus a cached goal field (better arrival, battery 106, but 1.2 GB on Carabanossi and run-time builds).

## Objections to the shared contract

All seven are accepted in the overview.

- **O1. Tick and decision cadence.** 15-frame decisions on a 6-frame tick give 2.5 ticks. Proposed: 5-frame
  tick (6 Hz). Cost: roughly +20% sim time. Alternative: keep 6 frames and move decisions to 12 frames.
  This breaks the 2 Hz rule and is not proposed.
- **O2. Stage 1 comparator is in-sample.** The overview's "fightnet AUC 0.59-0.62" was measured on
  validation splits of models trained on about 85% of games. `fightnet.npz` is trained on all games
  (HANDOFF 6d.2). The gate must compare against a retrained, out-of-fold fightnet.
- **O3. Snapshot count.** 46,389 includes 1,656 `terraunit` snapshots. Usable: 44,733 (measured). Clarify
  the gate's denominator.
- **O4. Gate needs a value baseline.** The tracked-value ratio already reaches AUC 0.56 at 20 s
  (measured). The gate must beat it with a CI.
- **O5. Default behaviour on replay is not specified in the overview.** Section 5.1 proposes one, fixed in
  advance.
- **O6. Roster and gadget counts.** "122 factory-built mobile types" is right when builders are included
  (13 builders, 109 others, measured). The action slots in 02 depend on which set is meant. "247 gadgets"
  is 242 `.lua` files.
- **O7. Raw frame ownership.** The overview says one featurizer and one raw frame, but does not name the
  owner of the schema. Proposed here: schema owned by 01, featurizer by 02.

## Needed from 02 and 03

From 02-policy.md:

- Confirm the record and field names in 4.6 (`unit`, `region`, `site`, `team`, header), or send changes.
- Confirm the ack codes in 4.7 are enough for action masking. Confirm that BLIP records have `type = -1`.
- Confirm the unit-token fields the featurizer reads from `unit` (for example `vx, vz` on enemies).
- Confirm the decision tick is every 3 five-frame ticks.

From 03-training.md:

- Batch API signature (4.4) and the reset scenario format (map id, start setup, seed).
- Train/validation split for the fightnet CV (section 5.1 step 1). The same game lists must be used for
  stage 1 and for the stage 2 train/validation split.
- The reward accounting must use the same tracked-unit rule as fightnet (non-blip, non-memory), or the
  reward and the stage 1 gate measure different things.
- Tick assumption: 5 frames if O1 is accepted; 6 if not.

## Unverified claims

- LightningCannon behaviour (treated as instant): not read.
- Mex base metal rate: `extractsMetal` is read from the unit dump; the ZK source that sets it for `staticmex`
  was not found in `units/staticmex.lua` or `gamedata/unitdefs_post.lua` (searched, not found).
- `unit_target_priority.lua`, `unit_morph.lua`, `unit_commander_upgrade.lua`, `map_lava.lua`,
  `unit_jumpjets.lua`, `game_perks.lua`, `unit_nanoframe_rate_limit.lua` read by header or call list only.
- `empDeclineRate` default and ZK value: not traced.
- QTPFS vs HAPFS mapping for ZK: `modrules.lua:19-22` selects QTPFS; the engine's use of that flag was not
  traced to the path manager.
- `GameHelper.cpp` line numbers for delayed damage are approximate (`~160-170`).
- Engine idle-unit behaviour (fire at nearest, no movement): assumed, to be confirmed in 5.2.
- Turn sub-stepping need: a hypothesis to test; no calibration has been done.
- Every number in section 6.1 is a target, not a measurement.
- The value-ratio AUC baseline (section 5.1) was computed in this session on the logs (measured); a few
  snapshots gave non-finite scores that I did not trace.
- LOS cost estimate (section 4.7): arithmetic, not measured.
- This list and the plan above predate the build; section 8 records what the build measured.
