# 01. Simplified Zero-K simulator: implementation plan

Status: stage 1 built, Gate 1 waived by the user; S-M5 and S-M6 built (section 8.7).  Results in section 8 override older
text here.  S-M7 hasn't started; S-M8 is built (8.9, 8.10).  Engine and game facts are cited `path:line` against
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
| Overkill avoidance | A | `unit_overkill_prevention` holds a shot when incoming damage already kills. Sim: the same rule per weapon from each script's `BlockShot` line, without using the reload (8.12). | `unit_overkill_prevention.lua`, unit scripts | Medium |
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
| Shields (`Shield` weapon type: Aspis, Convict, Thug, Felon, Funnelweb, Nebula, Aegis, Odin's temporary one) | A | A sphere that stops hostile shots along their path, for what the weapon does to shields.  Charging costs energy and they share charge.  **Checked in a game, 8.11**. | `Zero-K/units/*.lua`, `CheckShieldCollisions`, `unit_shield_charge.lua`, `unit_shield_link.lua` | Medium for shield units |
| EMP | A | Engine: `paralyzeDamage` accumulates; stun while it exceeds max health; decays at `maxHealth x (15/30) x empDeclineRate` per slow update (`Unit.cpp:992-1009`). ZK uses `paralyzeOnMaxHealth = true` (`modrules.lua:92-93`). Sim copies this, with `empDeclineRate` read from the engine dump. | `Unit.cpp:992-1009` | Medium |
| Slow (`timeslow`) | A | Multiplies speed and reload for a duration; ZK applies it via rules params read by unit scripts (`unit_timeslow.lua`). Duration and factor from the unit file. | `unit_timeslow.lua` | Low |
| Disarm | A | Disables weapons and their reload, not walking, with its own level and timer. Built, see 8.8. | `disarm_defs.lua`, `unit_boolean_disable.lua` | Low |
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
| Cloak | A | Own or lent by an area cloak.  Hidden from sight (`LosHandler.cpp`) and from radar (`unit_stealth.lua`), so it can't be aimed at, only caught in a blast.  Costs energy twice a second, more moving; shown by an enemy within `decloakDistance`, damage, firing or building, then blocked 45 or 90 frames (own cloak) or 90 or 240 (lent).  **Checked in a game, 8.10**. | `unit_decloak_damaged.lua`, `unit_stealth.lua`, `unit_cloak_shield.lua` | Medium |
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
| Metal in wrecks | A | 40% of a unit's metal as a wreck, 20% as a heap, by the feature defs of `featuredefs_post.lua`.  Which by recent damage over health, as the scripts' `Killed` say.  A wreck blown up leaves a heap, a heap nothing.  **Checked in a game, 8.10**; wrecks aren't thrown by blasts. | `featuredefs_post.lua`, `unit_nanoframe_death.lua` | Medium: reclaim farming |
| Reclaim | A | Any builder but a factory, at its reclaim speed in metal a second, adding up, paid as income and lost past storage.  **Checked in a game, 8.10**.  Nothing orders it in training yet. | `modrules.lua:62-74`, `Feature.cpp` | Medium |
| Assist and repair | A | Same build-step formula; repair via `repairSpeed` (`UnitDef.cpp:366`). | `UnitDef.cpp:366` | Low |
| Resurrect, capture | O | Not on the stage 1-3 roster. Capture (`capture_defs.lua`) is stage 4. For now a capture weapon does nothing and its unit is out of armies (8.8). | | Low |
| Commander, morph | O (stage 3) | `unit_morph` (1266 lines) and `unit_commander_upgrade` (755 lines) are not read in full. | | Medium at game start |
| Stockpile | A | Both resources as it goes, all of a step or none; always on, as the auto-stockpile widget leaves it.  **Checked in a game, 8.10**. | `unit_stockpile.lua` | Low |

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
| `weapon_shield_merge` | Shield projectile merge; what a shot takes from a shield | Modelled, merge itself off by default (8.11) | 2 |
| `weapon_impulse`, `weapon_noexplode_speed_damage` | Push and speed-based damage | Impulse applied to separation; the speed rule is not read | 2 |
| `weapon_projectile_retarget`, `weapon_projectile_radar_homing`, `weapon_projectile_chain`, `weapon_projectile_gravity`, `weapon_projectile_hold` | Projectile behaviour | Missiles approximated as flight time; the rest not read | 2 |
| `unit_spherical_los`, `unit_radar_wobble` | Sensors | As 1.3 | 2 |
| `unit_stockpile` | Stockpile | Model | 3 |
| `unit_jumpjets`, `unit_jumpjet_pilot`, `unit_jumpjet_pilot_2`, `unit_impulsejumpjets`, `unit_fall_damage` | Jump and fall damage | Approximate, see 1.2 | 2 |
| `unit_shield_charge`, `unit_shield_link` | Shield units | Modelled, without priorities (8.11) | 2 |
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
| `unit_teleporter`, `api_teleport_helper` | Modelled since ABI 15: section 8.17. |
| `unit_grey_goo`, `unit_global_zombie_resurrect`, `api_zombie_helper`, `chicken_control` | Chicken and zombie modes. |
| `game_sudden_death`, `game_point_control`, `game_share_mode`, `game_support_powers`, `game_support_mana_handler`, `game_resign`, `unit_planetwars`, `mission_*` | Game modes, campaigns, the bot's own resign rule. |
| `unit_antinuke_handler`, `unit_missilewarn` | No nukes on the factory roster. `unit_missilesilo` is modelled since ABI 15: section 8.17. |
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

- `player`, `ally_team`: whose it is, in the game's numbers.  For now each of the sim's two sides is one player and one alliance.
- `knowledge`: 0 in LOS, 1 radar blip, 2 remembered structure, 3 hidden enemy (critic only).  Hidden records
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

Actions: 02 section 4.1 is the list, one slot per own controllable unit (at most 256) and factory each
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
overview that stopped the work for a decision by the user, who waived it on 2026-11-05 and skipped the wider engine comparison, so Phase B went ahead.  Sections 0 to 7 above are
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

### 8.7 Batch, raw frames and fog (S-M5, S-M6; 2026-11-05)

`zksim.Batch` steps games across threads and leaves both sides' frames in preallocated arrays (`sim/src/views.cpp`, ABI 2).
`sim/include/zksim/rawframe.hpp` is generated from `bots/zkbot/policy/rawframe.py`, which owns the schema.

| Measure | Result | Budget |
|---|---|---|
| 256 games, about 390 units each, 16 threads | 0.339 ms per game decision, frames for both sides included | 0.5 ms; 1.0 allowed |
| The same games as armies thin out, 240 decisions | 0.194 ms average | |
| Fog, one thread, starting at 400 units | 339 us per game decision for two sides, 5.7% of a tick's cost | Estimate was 40 us per tick, or 120 us per decision: 2.8x over, and it doesn't matter yet |
| Thread count | Same frame hashes at 1, 5 and 16 threads | Determinism |

Fog is one height-map ray per pair, within each unit's sight radius, with radar as a plain circle.  **A live frame recreated in
the sim** (`tools/policy_golden.py sim`, also in `sim/tests/test_sim.py`), over 5 frames and 329 units:

* Type, position, build, free-space rays: exactly the same features.  Health is within 1%.
* Ground altitude matched only after a change: live `y` is a unit's middle, up to 35 elmos up by type, and the sim's is the ground.
  The feature now has a 40-elmo dead zone, so it means "flying or not" in both.  Air units differed then, since the sim flew all
  of them at 150 elmos; since 8.18 each flies at its own height, and lands.
* **Sight: the sim saw 18 of the 23 enemies the live game saw, and none it didn't.**  Its fog is a little tighter.  Five frames is
  too few to tune against, so it stays.

Live against archive bundle on TitanDuel (objection 6 of 02): the same regions once renumbered, because the spot list comes in a
different order.  Heights and passability are identical and walking distances differ by at most 0.23 elmos.  The net has no
region ids, so numbering doesn't reach it.

Bug found by the recreate check: `Batch.reset` passed record-array columns to C, which read past each value because the columns
have gaps.  `zksim._floats` and `_ints` now refuse such arrays and `reset` copies them.

S-M7 is not started, so of the fighting kinds the sim obeys only KEEP, MOVE, FIGHT, ATTACK_UNIT and MANUAL_FIRE (8.8).  Construction came later (8.9).  `tools/policy_sim_env.py`'s start of two
mirrored armies from far-apart spots is now only for checks and benchmarks.

### 8.8 Special weapons (2026-11-05)

The exporter read weapon fields without knowing the `customParams` that give them their meaning, so a scan of all 152 custom
parameters on the 748 weapons, against what `tools/sim_data.py` reads and which game code reads them, found more than capture and
disarm.  Fixed, each from the game's source:

| What | Was | Now | Source |
|---|---|---|---|
| Capture (`is_capture`) | Dominatrix did 660 DPS of real damage | No damage.  Out of armies until capture is built | `unit_capture.lua` |
| Disarm | Treated as stun: stopped walking, no separate timer | Own level and timer.  Stops firing and reload, not walking.  Reads `disarmdamagemult`, so `shipscout_missile`, which has `disarmdamageonly = 0`, also does damage | `unit_boolean_disable.lua`, `disarm_defs.lua`, `unit_attributes.lua:86-90` |
| Disarm targeting | None | Up to 3 weapons share a target, others prefer one still armed; a disarmed unit is a worse target for everyone else | `unit_target_priority.lua:285-299` |
| Reef missile, and any manual or stockpile weapon | Never fired | `ZK_ORDER_MANUAL_FIRE`, numbered as the raw frame's kind 6.  Walks to range, fires, order done.  Stockpile by unit `stockpiletime`, stopped by stun and disarm, free until the sim has an economy | `CommandAI.cpp` `WeaponFired`, `Weapon.cpp:1023`, `unit_stockpile.lua` |
| Starburst missiles | Launcher had to face the target; flew straight | Launcher needn't face; climbs for `uptime`, then flies | `StarburstProjectile.cpp` |
| Heat rays (`dynDamageExp`): Scorcher, Desolator, Dante | Full damage at any range | Falls to nothing at range, muzzle to hit | `DamageArray.cpp:82-110` |
| Fake guns that do damage: Grizzly, Buoy, Bulkhead | Fired as well as the real gun | Never fire: the script's `AimWeapon` returns false | `amphfloater.lua:227-241` |
| Slow-only beams: Cyclops, commanders, missile | No slow: factor defaulted to 0 | Factor 1, as the game | `timeslow_defs.lua` |
| `timeslow_damage` (Sunlance), `timeslow_overslow_frames` | Ignored | Read | `timeslow_defs.lua`, `unit_timeslow.lua:61` |
| Stun, disarm and slow level | By max HP | By current HP, so hurt units fall sooner | `unit_paralysis_damage.lua`, `unit_boolean_disable.lua` |
| `overstun_time` | Ignored | Stun can run past the weapon's time, near the threshold | `unit_paralysis_damage.lua` |
| Doomed-target bookkeeping | Counted stun and disarm damage as health damage | Only what takes health | |
| Felon's gun (`shield_drain`) | Free | Takes charge from its own shield, and can't fire without it | `shieldfelon.lua` |
| Ronin, Phantom (`reload_move_penalty`) | Ignored | Slower while the main weapon reloads | `cloakskirm.lua` |

Checked and needing nothing: `extra_paratime` equals `emp_paratime` on all 5 playable weapons that carry it, and
`shieldstartingpower` equals `shieldPower` on Convict and Thug.  (This said "all 3" when written; it doesn't hold for all shields: 8.11.)

**Armies** (`policy_sim_env.army_types`) drew from every def: chickens, test units, and AA that can't hit the ground.  Now 40 types:
factory-built, can hit the ground, nothing flagged `ZK_W_UNMODELLED` (capture, spawned units, pushing or pulling, throwing).

**Re-order boundary.**  New battery phases showed a move order up to 2 frames before a slow update costs the engine a full cycle, and
in the sim it cost nothing.  A policy re-ordering each decision would have kept those units at full speed.  It now waits for the next
update.  The phases just above that are still nearly free in the sim (0.83 and 0.94 of full speed for a raider re-ordered every decision;
the median is 0.43).  The battery doesn't measure those phases, so watch for it in self-play.

**Engine battery, 194 scenarios: 119 of 138.**  The 25 new checks (`status/*` and 4 duels), engine against sim:

| Check | Engine | Sim |
|---|---|---|
| Disarmed unit walks (elmos; unhindered, engine / sim: 578 / 588) | 578 | 587 |
| Disarm lasts (frames, of 600) | 180 | 175 |
| Damage dealt in 600 frames, free / while a Racketeer disarms it | 2037 / 819 | 2090 / 861 |
| Reef missile: lands (frame), disarmed for (frames) | 100, 295 | 95, 295 |
| Scorcher kills a Glaive at 225 elmos (frames) | 97 | 105 |
| Grizzly against Ravager; Buoy against Knight (frames) | 202; 913 | 185; 910 |
| Ronin walking and firing (elmos) | 650 | 642 |

The 19 misses: 9 are the routing misses of 8.6, unchanged.  10 are new:

- **First Racketeer missile lands 35 frames late in the engine** (2 checks), about 1.2 s.  Later shots keep the 10 s rhythm.  Cause unknown.
- **Sitting on the stun threshold** (`status/emp-walk`, 2 checks): levels agree (peak 1.05 against 1.04), but a turret that holds a unit
  at 1.00-1.05 flips it over and under; it walks 300 elmos in the engine against 172.
- **Bantha's manual salvo** (3 checks): 12 missiles fly further and land later in the engine, and differ between runs.  10,000 metal,
  in no army.  Left alone.
- **Scorcher at 90 elmos**, 31.8 against 40 frames (1 check).  Close to the tick's size.
- **Kbot re-order at 0.5 s** (2 checks), 0.03 against 0.09 of full speed: both nearly stopped, but the check is a share.

**Known gaps**, all named so they aren't mistaken for modelled:

- Reef's drones: no carrier logic.  It has its missile only, and ships are out of armies.
- The policy can't see disarm: the raw frame has a paralysed flag and no disarmed one.  It sees the target stop firing, not the status.
- Gauss rifles (`single_hit`: Dagger, Gauss turret) hit only their target, not everything along the line.
- Mines, capture, impulse, throwing: out of armies via the flag.  Bulkhead's deploy-to-fire script isn't modelled, and it stays in armies.
- Stockpiling costs no metal, and a paralyser hit doesn't also extend a partial disarm.
- Air-only (`reammoseconds`, `shots_per_refuel`, thermite), water (`stays_underwater`), terraform and visual parameters are for later
  stages or have no effect on a fight.

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

### 8.9 Economy and the scripted opening (2026-11-06)

Part of S-M8, as far as an opening needs: `sim/src/economy.cpp`, ABI 7.  A world runs none of it unless `zk_world_economy` turns it on, so the gate's replays are unaffected.

| What | How | Source |
|---|---|---|
| Start | 350 metal and energy, storage 500 (the commander's own) | `start_setup.lua`, unit dump |
| Income | Innate 2 and 2.  Commander 4 and 6 while it lives.  Extractor: its spot's income, as the game reports it.  Solar: 2 energy.  Nothing while stunned; an extractor nothing while disarmed | `start_setup.lua`, `dynamic_comm_defs.lua` "econ", `unit_mex_overdrive.lua` |
| Overdrive | Below | `unit_mex_overdrive.lua` |
| Build step | Progress per second is build speed / build time, paid as it goes from both resources, stalling on whichever runs out.  First come, first served, in handle order.  Health rises with progress | 1.4 |
| First factory | A side's first ploppable factory is free and finishes at once.  The game ties this to the commander; the sim to the side | `start_unit_setup.lua:61-69` |
| Orders | BUILD (type at x, z), ASSIST (a unit; a factory means what it's building), PRODUCE (a type the factory repeats), numbered as the raw frame's kinds 10, 11 and 14 | |
| New units | Come out of the factory's front, or the nearest side their class can walk from.  Unfinished, they neither walk, shoot nor see | |

The commander's income wasn't in the unit dump, and with only the innate 2 and 2 the sim fell well behind real games.  **Logged games settled it**: over 188 games, income less 2 per finished extractor and solar is 6 and 8 in the first four minutes, in most samples.

Against those logs (rules bot, which also builds constructors):

| Time | Sim opening, metal / energy income | Logged median |
|---|---|---|
| 30 s | 8.8 / 10.1 | about 8-10 / 8-10 |
| 60 s | 11.3 / 12.2 | 12.0 / 10.0 |
| 120 s | 15.1 / 14.8 | 14.8 / 14.9 |
| 180 s | 16.9 / 16.5 | 18.2 / 17.9 (past here the rules bot keeps expanding; the opening stops) |

Build times, stalls and assisting follow the source and have unit tests, but **no engine battery covers them yet**.

**Regeneration** (`unit_regeneration.lua`) is in every world, not only one with an economy: a unit gains `idle_regen` health a second once `idleTime` frames have passed since it was last damaged, unless stunned or disarmed.  The sim adds it evenly where the game adds half of it every 15 frames.  This is the only source of healing there is, since `autoHeal` is 0 on every def once Zero-K's post-processing has moved it.  A real game agrees: hurt to half, Glaive waits 5 s then gains 20 a second, Kodachi 10 s then 5, Scorcher and Solar 60 s then 5.  The gate moved a little and still fails: advance at 20 s 0.593 against 0.591, at 10 s 0.638 against 0.631, and predicted loss per snapshot fell from 358 to 323 ours against 240 measured.

Not in: repair, Duck's faster healing under water (`amph_regen`), and `neededlink` (a heavy turret that won't fire off a powered grid).  Reclaim, wrecks, upkeep, cloak, stockpile cost and a shot solar came later (8.10).

**Overdrive** runs once a second, per side, as `unit_mex_overdrive.lua` does for an ally team of one player:

1. Energy to spend: income less what builders spent in the last second, if that's positive, times how full storage is.
2. Grids: finished, unstunned, undisarmed units with a pylon range share one if closer than the sum of their ranges (extractor 50, solar 100, wind 63, pylon 500).  They're worked out afresh each time.
3. An extractor's share goes by its income squared, over all grids.  A grid can't take more than its own generators make; the commander and innate income are in no grid, so they feed overdrive only up to what grids can carry.  What a full grid leaves goes to the others, and what no grid can take stays in storage.
4. An extractor earns its income times `1 + 0.25 sqrt(energy)` until the next update.

An extractor with one solar beside it takes that solar's 2 and earns 35% more.  In the opening that's about 0.8-0.9 metal per second from 120 s on, well under the 3-4 that 4-6 pairs could carry at full storage, since the commander and factory spend most of what's spare.

The income the sim reports is the game's: metal with its overdrive, energy after what overdrive drew.  `AddTeamResource` books the net as income (`Team.cpp`), so **the game's energy income falls when overdrive runs**.

`python tools/sim_engine.py overdrive` builds layouts in a real game and logs what the gadget makes of them.  **The sim matches it**:

| Layout (commander's 8 energy as well) | Game: drain, extra metal, team energy income | Sim |
|---|---|---|
| Extractor and solar 100 apart, storage full | 2.00, 0.709, 8.00 | 2.00, 0.709, 8.00 |
| 160 apart, no link | 0, 0, 10.00 | 0, 0, 10.00 |
| One extractor, 5 solars, full | 10.00, 1.585, 8.00 | 10.00, 1.585, 8.00 |
| The same, half full | 9.00, 1.503, 9.00 | 9.11, 1.513, 8.89 |
| Two grids, carrying 2 and 10, half full | 10.00, 2.126, 10.00 | 10.13, 2.138, 9.87 |
| Two extractors on a pylon, 10 solars, full | 20.00, 3.170, 8.00 | 20.00, 3.170, 8.00 |
| The same, half full | 14.00, 2.652, 14.00 | 14.26, 2.677, 13.74 |

The half-full gaps are the check, not the sim: the game's energy is held at exactly 250, and the sim's has one tick of income on it.  For this table the sim's spots earn the 2.0047 a real extractor earned here.  The game's list says 2.0, and that 0.2% is otherwise left alone.  The policy isn't told about grids, and its token for a unit has no grid number.  That matters once it builds (S-M7).

**Map facts.**  `python tools/sim_engine.py facts MAP...` starts a game on each map and writes what Zero-K published to `sim/data/map_facts.json`: start boxes (`start_boxes.lua`) and metal spots (`mex_spot_finder.lua`).  A game is asked rather than the files read because the choice between Zero-K's config, the map's and a fallback is the game's, and a map's file is Lua that may build its list in code.

* **That found a bug in the export.**  `tools/terrain_map.py` read 22 of Faultline's 44 spots: its layout file makes half of them in code.  Red Comet's spots give 2.3 metal, not the 2.0 the export assumed, and FrostyCove's 1.97-2.12.  `load_map` now prefers the facts file, and the five training maps are exported again.  All earlier sim training had this wrong.
* **A map may draw its boxes when it loads.**  Faultline's come out differently from one launch to the next, 3 layouts in 6 launches, so `facts` loads each map several times and keeps every layout seen, and a game draws one.  All earlier training saw a single draw.
* It also lists what's reclaimable on the map from the start, with at least 1 metal.  Those are placed in every game: Agalta 12 features of 120 metal, Faultline 18 of 720, FrostyCove 138 of 2,760, and none on Red Comet or TitanDuel.
* Start boxes are polygons, and the two on each training map are far from where the old rule (the two farthest spots) put commanders: on Agalta, mid-west and mid-east, not two corners.  FrostyCove's start points are corners of their own polygons.
* A game takes two of a map's boxes at random, as Zero-K's `allshuffle` does.  A start is the box's start point half the time, which is where logged bot games start (TitanDuel 4813, 4809).  Otherwise it's anywhere in the box on level ground, as a player could choose, snapped to 32 elmos so that plans can be kept.  The script's target is the enemy's actual start; the opening plans towards the enemy box's start point, which is all a player knows then.

**The opening** (`tools/policy_opening.py`), the same on both sides whoever controls the fighting: the factory first, about 160 elmos towards the enemy, which starts repeating its raider at once.  Then an extractor and a solar at each of the 6 nearest spots within 1,300 elmos and nearer us than them, 4 to 6 by map from the start points, and after that the commander assists the factory.  Commander and factory are drawn per side.  Finished by a median of 152 s (p90 194 s), later than the older starts' 124 s because there's more to build.

* A step over 2 minutes is given up.
* Eight factories: amph, cloak, hover, jump, shield, spider, tank, veh.  Tank was missing at first because `tools/sim_data.py` flagged a weapon as unmodelled if it had `area_damage_is_impulse` at all, and on 12 fire weapons the value is 0.  It now reads the value.  This also makes Dante, `missileslow` and `jumparty` eligible for the older armies' pool, if they pass its other filters.
* "Raider" is the bot's role data, so jump opens with Pyros at 220 metal and spider with Venoms at 190, against Glaives at 65.

### 8.10 Wrecks, reclaim, upkeep, cloak, stockpile, a shot solar (the rest of S-M8; 2026-11-06)

`sim/src/wrecks.cpp`, `sim/src/cloak.cpp`, more of `sim/src/economy.cpp`; ABI 8.

| Mechanic | The rule | Simplified |
|---|---|---|
| Wreck or heap | Finished unit: wreck (40% of its metal) if recent damage is at most half its health, else heap (20%).  Recent damage gains each hit and fades by a tenth **a frame** | A few scripts' `Killed` return other levels, ignored.  Flying units leave theirs where they die |
| Frame | Over 5% built: a heap in proportion, with health in proportion.  Under: nothing.  Over 80%: it explodes as well | |
| Blast on a wreck | Default damage to its health.  At zero a wreck becomes a heap, keeping the share already reclaimed; a heap is gone | **Blasts don't throw it.**  A Singularity threw a Glaive's heap 2,519 elmos |
| Reclaim | At the builder's reclaim speed, in metal a second, adding up across builders; no energy; paid as income, the rest lost past storage.  Either side may take any wreck | Costed in metal alone: a feature with energy in it takes as long as its metal says |
| Upkeep | Once a second, in energy: radars, jammers, an area cloak.  Unpaid, a radar stops until paid.  Not while stunned, disarmed or unfinished | First come, first served, not by priority |
| Cloak | See the table in section 1.3.  Commanders can't, without a module | Area cloak has its radius at once and covers friends at once |
| Stockpile | Metal and energy together as it goes, all of a step or none | Always on, without end: the stock isn't capped |
| Solar | A hit from a real weapon shuts it from that moment: no energy, no capacity on its grid.  19 frames on, armoured to a third.  120 frames after the last hit it opens: armour off at once, energy 38 frames after that | |

Only the solar closes: `energysolar` alone has `auto_close_time`.  Until now the sim gave it its armour always, which was wrong.

**Features are in the frame**, schema 3: id, place, what metal's left, def.  What a side sees is what's in the line of sight of its finished units, by the same raycast as units (02 section 2.3).

`python tools/sim_engine.py mechanics` plays these out in a real game and logs them.  It corrected two readings of the source, and sent us back to a third:

* **Radar doesn't see cloaked units.**  `LosHandler.cpp` alone would have them on radar, but `unit_stealth.lua` makes a unit radar stealthy while it's cloaked.
* **Recent damage fades every frame**, in `CUnit::Update`, not every slow update.  Three 100-damage hits on a 230-health Glaive leave a heap 10 frames apart and a wreck 30 frames apart.
* **A blasted wreck keeps what was reclaimed of it** (`CFeature::CreateWreck`).  This one is from the source: in the game's check the wreck survived its blast, so the heap's share wasn't seen.

| Check | Game | Sim |
|---|---|---|
| Glaive destroyed outright / big hit | wreck 26 / heap 13 | same |
| Half-built Glaive, 3% built | heap 6.5 metal, 115 health / nothing | same |
| Wreck in a blast, then its heap | heap / gone | same |
| Wreck loses, income gains, one builder / two | 5.0 / 10.0 a second | same |
| Energy a second: radar / unfinished radar | 0.80 / 0 | same |
| Jammer with its cloak on | 16.5 | same |
| Cloaked Scythe still / moving | 0.20 / 1.0 | same |
| Stockpile, each resource | 4.01 a second | 4.00 |
| Unpaid radar | radius 0, and back when paid | same |
| Solar's first hit to energy off / armour | 1 / 19 frames | within a tick / 19, by tick |
| Solar's last hit to armour off / energy back | 121 / 159 frames | 120 / 158, by tick |
| Damage to it, open / shut | 10.5 / 3.5 | same |
| Starts cloaked | frame 1 | at once |
| Enemy near, then gone, to cloaked again | 43 frames | 30-60, by slow update |
| Hit, to cloaked again | 90 frames, and shown only at the next slow update | 90, shown at once |
| Lent cloak: covered | 26 frames | up to 15 |
| Lent cloak: fired, to cloaked again | 240-255 frames | 240-255 |
| Sniper firing | stays cloaked | same |

Nothing issued RECLAIM in training at first: the opening's commander assists its factory for good, and the policy had only the fighters, with kind 13 masked until stage 3.  Since 2026-11-09 the economy script reclaims and the policy may (`04-economy.md`, "Reclaim").

**What it did to the numbers** (256 games, side 0's wins/draws/losses; the earlier figures are from before this section):

| | Before | After |
|---|---|---|
| Doing nothing against the script (so the script wins 83, then 82) | 2/15/83, median 680 s | 1/16/82, median 718 s |
| Script against script | 43/20/37, median 534 s; 167 eliminated, 38 on the lead, 51 at the limit | 36/27/36, median 564 s; 152, 34, 70 |
| Mirror factories draw / mixed | 44% / 13% | 47% / 16% |
| Seat steps a second | about 44,000 | 40,058 |
| Replay gate 1(a), advance at 20 s against fightnet | 0.593 against 0.665, CI of the difference [-0.091, -0.052], FAIL | 0.604 against 0.665, [-0.080, -0.042], FAIL |
| Advance at 10 s against fightnet | 0.638 | 0.642 against 0.644, CI [-0.019, +0.015] |

`results/simgate/m8.json`.  Income is the same at 60 s and a little lower later: metal and energy 15.9 / 15.8 at 180 s against 16.9 / 16.5, which takes it further under the logged 18.2 / 17.9.  The draw rate's rise is about two standard errors and hasn't been explained: Faultline's other layouts, solars being softer until shut, cloak and the new blast rules all changed at once.  The gate wasn't a target of any of this.

### 8.11 Shields (2026-11-06)

`sim/src/shields.cpp`, ABI 9.  Before this a shield was checked where a shot landed, so it covered what was inside it and let a shot pass through on its way to something beyond.  Beams cost almost nothing.

**What a shield stops.**  A shot stops at the first hostile sphere its path enters that can pay for it, and explodes there, so its splash reaches what's near the surface.  A shooter inside fires out freely, and a shield of the shooter's own side isn't in the way.  The spheres are 3D, on the ground at the unit's position.

**A shell's path.**  A parabola from the same two solutions as `Cannon.cpp`, with the higher one for the three units with `highTrajectory` 1 (Lobster, Tremor, Badger).  It can fly over a small bubble.  What's fired from or at the air, and missiles that climb first, count only where they land.

**What it costs.**  `damages[shield]` is what the weapon does to shields, and it doesn't fall with range: `weapon_shield_merge.lua` charges `shield_damage`, and the check is strict, so a shield with no more than that doesn't stop it.

For a beam `armordefs.lua` gives `shield_damage` per beam frame, 35 for the Stinger against 850 per shot, and `stats_shield_damage` for the whole shot.  The catalog holds the whole shot, and the sim, which has one hit where a beam has many, stops the share of it the shield can pay for.  The rest carries on to the next shield or the target.  103 beam weapons differ between the two figures, and no other type does.

**Charging.**  Only when active: a stunned or unfinished shield doesn't charge, which it used to.  Slow scales both charge and energy.  It takes `shield_drain` energy a second (Aspis 12, Funnelweb 48, Aegis and Nebula 9, others free), and it's all of a step or none of it.

**Sharing.**  As `unit_shield_link.lua`: spheres that touch, ranked by radius so that a shield doesn't start with a lower rank, a random pick every 2 frames from its links and one more choice that does nothing, 1% of the difference, with room left for what the lower one charges by itself.  Links are refreshed every 30 frames.

**Against a game** (`python tools/sim_engine.py mechanics`, `shield_*` lines; 900 is a Convict and 3600 an Aspis, both charged to that at the start; sim figures averaged over 16 seeds):

| Case | Game: target lost, shield left | Sim |
|---|---|---|
| Stinger, 10 s, Convict in the way | 1347, 14 | 1551, 11 |
| Same, Convict off the line | 2481, 900 | 2550, 900 |
| Minotaur shells, 10 s, Convict in the way | 640, 34-39 | 960, 35 |
| Stardust at something inside, near the near edge, 1 s | 109-128, 769-858 | 225, 706 |
| Same, deep inside | 0, 770 | 0, 769 |
| Sling, 20 s, Aspis half way | 0, 3540; every shell stops at x 46-53 of its 350-radius edge at x 50 | 0, 3600 |
| Sling, target inside an Aspis | 0, 3600; shells stop at x 463-477 | 0, 3600 |
| Sling, Convict half way, target beyond | 140, 900 (a second run: 140, 764) | 420, 900 |

| Charging and sharing | Game | Sim |
|---|---|---|
| Aspis from 0, 5 s | 250; energy +8/s full, -4/s charging | 250, 12/s |
| Aspis with no energy, 5 s | 100 stays 100 | same (unit test) |
| Two Aspis touching, 3000 and 0, at 2, 5, 10, 20 s | 2487/713, 2067/1433, 2059/1941, 2503/2497 | 2418/782, 2100/1400, 2067/1933, 2504/2496 |
| Aspis and Convict, 3000 and 0, at 2, 5, 10 s | 2748/374, 2445/860, 2668/900 | 2702/420, 2448/857, 2672/900 |
| Two Aspis out of reach, 5 s | 3250/250 | same |

The differences, as far as they've been followed up:

* **Sling against a Convict.**  The game's first shell hits the target and later ones mostly miss it, as `accuracy` also scatters up and down.  A shell that falls short can come down low enough to meet the bubble: one was at height 212 at x 316, where the full path is over 300.  The sim's scatter is sideways only (1.2).  Its shells land on target and overfly the bubble.
* **Minotaur.**  One more shell lands in the sim's window, as it fires at once and the game turns first.  The split on the shield is the same: three stopped.
* **Stinger.**  13% less gets through in the game, since it charges frame by frame and a shot is 827 there against 850.
* **Stardust.**  The sim's 2D hit point is a few elmos from a target just inside the surface.  The game hits higher up, and varies a lot between runs.

An early probe had 140 of damage inside a full Aspis.  It was a shell left in the air by the case before, which started inside the new sphere and so wasn't stopped.  The cases now wait for those.

**Left out:**

* A stun doesn't take charge to 0 (`unit_shield_charge.lua` does).
* Everything starts full.  In the game Aspis, Nebula, Aegis and Funnelweb start empty, and Felon at 1600 of 1800.
* Funnelweb's `shield_recharge_delay`.
* The temporary shield's falling charge rate and death at 0, `shield_rate_charge` and `shield_die_on_zero_charge`: 620 after 5 s in the game, 1000 here.  It's dropped by Odin, a bomber, so not in the armies.
* Charge priorities when energy is short: each shield pays in turn.
* Shells' scatter up and down, as above.

The raiders and commanders in the present games have no shields, so this matters only for games with other shield units.

**Replay gate 1(a):** unchanged, FAIL.  Advance at 20 s 0.604 against fightnet 0.665, CI of the difference [-0.080, -0.042], and at 10 s 0.642 against 0.644, CI [-0.019, +0.015], the same as 8.10 to three places.  The sim's prediction changed in 3.8% of snapshots for advance and 1.1% for hold.  `results/simgate/shields.json`.

### 8.12 Unit AI (2026-11-06)

`sim/src/unit_ai.cpp`, ABI 10.  Units now act on their own between orders, as in the game.  `ZK_BEHAVIOUR_ORDERS` is a game's behaviour; `ZK_BEHAVIOUR_SCRIPTED` does only what it's told and sees everything, for tests and the battery that measure one mechanic.

**What's in:**

* **Attack-move.**  At move state 1 (the default) a unit looks within its range + 100 elmos of the closest point on its line, chases the closest enemy, and returns to the line when that's over.  `MobileCAI`'s leash is 500 elmos plus range, from the point it left.  Roam widens the search and hold position stops it moving.
* **Idle.**  Attacks within range + 200 elmos, gives up after 150 frames, and walks home if it was idle before.
* **Fire states.**  Hold fire doesn't shoot, return fire shoots what shot it, fire at will is the default.
* **Tactical AI**, ported from `unit_tactical_ai.lua`, with its tables from the game: raiders run from riots when idle, skirmishers kite what would reach them, and step out of range of turrets and other skirmishers while they reload.
* **Overkill prevention** from each script's `BlockShot` line, per shot for weapons whose script fires the salvo, and without using the reload.
* **Bait prevention** at each unit's default level.  The gadget's default is off for AIs, and a human's widget or the AI interface turns it on, so the sim has it on.
* **Radar.**  An unidentified blip is aimed with an error.  Some weapons take `radar_wobble` and dot penalties from their defs.  By the user's rule, a weapon that only hits by touching (`ZK_W_IMPACT_ONLY`, such as Lance) can't hit a radar dot unless it's a structure already seen.  That's stricter than the game, where a Lance hit a mobile dot with 1 of 4 shots and an unseen structure with 2 of 4.
* **Stardust.**  Its script's heat is modelled, slowing reload with continued fire, and its 9 degrees of scatter.  `SCATTER_DAMAGE` in `tools/sim_data.py` takes 20% off its damage because the sim's scatter is sideways only.

**Against the game** (`python tools/sim_engine.py mechanics`, the `unitAI` probes in `zz_sim_mechanics.lua`; times in seconds):

| Probe | Game | Sim |
|---|---|---|
| Fight, target at range + 60 | chases, returns, at the goal by 20-24 s | same |
| Fight, target at range + 160; hold position, same | ignored | ignored |
| Hold position, target in range | kills, stays on the line | same |
| Leash against a fleeing scout | gives up about 4 s in, at x about 738 | same |
| Idle, target at range + 100 | closes to 158-168, kills | same |
| Idle at range + 300; idle on hold position | doesn't move | doesn't move |
| Idle, home again after a target | by 6 s | by 6 s |
| Hold fire and return fire, not shot at | 0 damage | 0 damage |
| Return fire, shot at | gun 1702 hp, enemy lost 690 | gun 1692, enemy lost 920 |
| Raider flees a riot | at about 400 elmos | about 410 |
| Skirmisher kiting a mobile unit, 24 s | holds about 438, 4 shots | about 411, 6 shots |
| Ronin stepping back from a Lotus | 185-262, about a 4 s period | 192-253, same period |
| Ronin against a Moderator | goes in to about 516 | steps at 192-261: **differs** |
| Ronin against a Ronin | 280-327 | 261: **differs** |
| Ogres at a Glaive | 1 shot | 1 shot (the test requires it) |
| Ogres at a Knight, shots | 11 | 13, measured before the last overkill fix |
| Bait prevention on, Lance at a solar | 0 damage | 0 damage |
| Bait prevention on, Lance at a Glaive | 2600 | 3000 |
| Lance at radar mobile / unseen structure / seen structure, 72 s | 2952 / 6000 / 12000 | 0 / 0 / 12000, by the rule above |
| Sling at a radar dot | not measured: terrain blocked the radar in the probe (`los_state` 0) | 701 damage |

**Stardust at 30 s, game against sim before the 0.8:**

| | 150 | 250 | 350 | 400 |
|---|---|---|---|---|
| Knight | 6512 / 6693 | 5994 / 6434 | 5395 / 5974 | 4247 / 5659 |
| Glaive | 6166 / 6526 | 4730 / 6043 | 3739 / 5399 | 2691 / 4988 |

The heat agrees, 0.967-0.994 against 0.972.  The mean of game over sim is 0.815, and 0.8 is a mean, so a Stardust is now a little weak up close and strong at range.

**Left out:**

* The swarm logic.  A game's raiders and skirmishers jink up to 136 elmos sideways and some close in; the sim's walk straight, so they arrive sooner (z 1729 against 1501 at 15 s in one probe).
* Ward fire, `fightOnlyOverride`, jammers, and the amphibians' sea table: they use the land one.
* Disarm in overkill prevention (`shieldarty`, `shipscout`), its repair trend, and scripts with computed damage (Merlin, bombers) or several `BlockShot` branches, where the first line is taken.
* Builders on attack-move fight like any mobile unit, with no reclaim on the way.  Air uses the same logic.
* Elevation scatter, except the Stardust's 0.8.
* **The policy can't set fire or move states yet**: both stay at the default.

**What it did to the numbers:**

| | Before (8.11) | After |
|---|---|---|
| Doing nothing against the script, side 0's wins/draws/losses | 1/16/82, median 718 s | 11/22/67, median 611 s |
| Script against script | 36/27/36, median 564 s; 152 eliminated, 34 on the lead, 70 at the limit | 46/8/46, median 491 s; 235, 0, 21 |
| Mirror factories draw | 47% | 26% (133 games, of a separate 1024) |
| Seat steps a second | 40,058 | 43,059 (a clean machine; the 40,058 may not have been) |
| Replay gate 1(a): advance at 20 s against fightnet | 0.604 against 0.665, [-0.080, -0.042], FAIL | 0.603 against 0.665, [-0.081, -0.043], FAIL |
| New `orders` arm at 20 s | | 0.613 against 0.665, [-0.067, -0.037] |
| New `orders` arm at 10 s | | 0.653 against 0.644, [-0.005, +0.023] |

`results/simgate/unit_ai.json`.  The gate's primary arm stays `advance`, so the gate still fails.  `orders` has the best agreement of the three on the gate's measure at both horizons, and it's the behaviour training plays, but not on every measure: at 20 s its sign agreement is 0.484 against advance's 0.560, and its rank correlations are 0.547 and 0.560 against 0.599 and 0.596.  Its mean losses are 143 and 156, against 240 and 136 logged and advance's 307 and 319.

"Doing nothing" is no longer nothing: idle raiders defend and attack what comes close, and that alone wins 11% against the script.

**The factory drawn now decides most games.**  1024 games of script against script: the side with the better factory by overall score won 91% of decided games between different factories.  Overall score, wins plus half draws:

| Spider | Tank | Amph | Shield | Jump | Cloak | Hover | Veh |
|---|---|---|---|---|---|---|---|
| 0.92 | 0.74 | 0.67 | 0.48 | 0.47 | 0.29 | 0.24 | 0.21 |

Only raiders get built in these games, so this is each factory's cheapest raider-role unit, not the factory: Venom for Spider, Kodachi for Tank.  **I haven't checked this against the engine.**  A factory that can't win has nothing for a policy to learn from, and it caps the win rate of a test: see `03-training.md` 10.6.

### 8.13 Bombs (2026-11-06)

ABI 11.  Limpet, Imp, Skuttle and Snitch now go off by themselves.  `kamikaze_distance` comes from each def's `kamikazeDist`, which is the Lua `kamikazeDistance` plus 25, so any ground unit with `canKamikaze` gets this (`subscout` too) and no unit is named in the code.  `update_kamikaze` in `sim/src/unit_ai.cpp` follows `CUnit::SlowUpdateKamikaze`, every 15 frames.

**When one goes off:**

* **Told to attack:** when the target is within reach.  An order at what it can't see lapses after four updates.
* **On fire at will:** when any enemy on the ground it can see is within reach, whichever way that enemy is going.
* **Told to self-destruct** (`ZK_ORDER_SELF_DESTRUCT`, `World.self_destruct`): at once.  It does nothing for other units.
* **Never:** while stunned, on attacking the ground, or at hold fire without an attack order.  Hold fire is what they all default to.

They have no weapon, so they never pick a target or chase on their own, whatever their fire state.  Going off adds twice the unit's maximum health to its recent damage, as `ForcedKillUnit` does, so it leaves a heap.

**What the game corrected.**  I first wrote it to go off only for an enemy coming closer, reading the engine's `AllowUnitKamikaze` call.  In the game a Limpet went off for a Knight standing still and for one walking away: the engine only passes that test to gadgets, Zero-K has none for it, and with none the answer is yes.

**Against the game** (`python tools/sim_engine.py mechanics`, `bombs()` in `zz_sim_mechanics.lua`):

| Case | Game | Sim |
|---|---|---|
| Limpet told to attack at 200: dies at frame, distance | 26, 116 | 35, 96 |
| Imp, same | 25, 109 | 35, 84 |
| Skuttle, same | 104, 45 | 110, 42 |
| Snitch, same | 40, 67 | 35, 99 |
| Told to attack at 400, unseen: walks (Limpet, Imp, Skuttle, Snitch), none goes off | 73, 75, 35, 66 | 109, 116, 44, 102 |
| Limpet at hold fire, Knight walks past | stays | stays |
| Limpet on fire at will, Knight approaching: goes off at distance | 139 | 126 |
| Same, Knight standing at 100, or walking away from 100 | goes off | goes off |
| Limpet on a fight order passing 60 from a Knight: hold fire / fire at will | walks on / goes off | same |
| Attacking the ground | stands there, stays | no order path sets it off, not played |
| Stunned, told to attack or self-destruct | stays | stays (test) |
| What's left | heap: 32, 25, 110, 34 metal | same |

**Blasts:**

| Blast, target at distance | Game | Sim |
|---|---|---|
| Limpet at 60: Knight's health lost, friend's | 105, 0 | 107, 0 |
| Limpet at 60: Knight's slow | 0.44 | not read: the state has no slow field, a test measures its walking |
| Imp at 60: Knight's stun, friend's | 0.91, 0 | 0.95, 0 |
| Snitch at 60: Knight's health lost, friend's | 994, 1024 | 1026, 1026 |
| Snitch at 80 | 896 | 931 |
| Skuttle at 80 | 2638 | 3386 |
| Skuttle told to attack, at about 43 | 5921 | 7055 |

**The sim's blasts run a bit strong.**  At 80 elmos, Snitch's and Skuttle's numbers both fit the game measuring 7 elmos farther from the blast than the sim does.  It's probably height, which the sim doesn't have.  That's 4% for Snitch, with its wide blast, but 19-28% for Skuttle, whose damage falls steeply over 90 elmos.  This applies to every blast and is left as it is.

**Two fixes along the way:**

* A death blast now carries its side, so a Limpet's no-friendly-fire blast spares friends.  Before, it hurt them.
* A death blast now comes before the remains, as in `ForcedKillUnit`.  Before, it could damage its own heap.

**Left out:**

* Burrowing, jumping and Blastwing were here, and are now in 8.14.
* A self-destruct waits for the next slow update in the game, up to 15 frames, except Imp's, which has `instantselfd`.  Here it's at once.  An order given while stunned is dropped.  By the engine's code it would finish once the stun ends, for those without `instantselfd`; the probe's 4 s didn't reach that.
* Bombs set off by fliers: the engine measures in 3D, and they're mostly too high up.
* Bombs in replays' `advance` and `hold` arms, which give no orders.

**The policy can't use bombs yet**, and they aren't in training's armies:

* `ptr_enemy` allows an attack only for what can hit, which a bomb can't.
* It can't set fire states, or self-destruct.
* So a policy's bomb can walk, and go off only when killed.
* The fight script's FIGHT orders leave them walking too, at hold fire.  That includes the frame model's scenarios.

**What it did to the numbers:** nothing in the present games, which have no bombs.  Script against script is still 46/8/46, and doing nothing against the script 11/22/67, game for game.  42,520 seat steps a second against 43,059, which is noise.  The replay gate wasn't rerun.

### 8.14 Jumping, digging in and Blastwing (2026-11-06)

ABI 12, `sim/src/jump.cpp`, `sim/tests/test_jump.py`.  Checked in a game: `python tools/sim_engine.py mechanics`, `jumps()`, `burrows()` and `blastwing()` in `zz_sim_mechanics.lua`.

**Jumping**, for every mobile unit with `canjump` (`World.jump`, `ZK_ORDER_JUMP`): it walks into range, goes when charged, flies an arc whose length sets the time, and reloads after landing.  Height exists only in the air: it counts for weapon reach, blasts and a bomb's reach.

**Digging in**, for everything with `idle_cloak` (Imp, Snitch, Flea, Blastwing; the Wolverine mine by the same rule).

**Blastwing:** told to attack, it goes off at a reach that grows with speed, up to 160, and on fire at will only within its kamikaze distance.  Its bomb keeps its speed.

| Case | Game | Sim |
|---|---|---|
| 300-elmo jump, landed at frame: Pyro, Jack, Constable, Jugglenaut, Skuttle | 76, 126, 84, 93, 65 | 80, 125, 85, 90, 65 |
| Ready again: Pyro, Jugglenaut | 376, 542 | 385, 540 |
| 700 away: leaves at frame, from | 106, 310 | 115, 335 |
| Jugglenaut's landing: at 60, at 120, friend at 100, itself | 710, 356, 514, 0 | 751, 398, 515, 0 |
| Skuttle landing by a Knight: dead at, damage | 72, 7294 | 75, 7187 |
| Snitch blast, Jack on the ground / at the top of its arc | 1238-1258 / 0 | 1193 / 159 |
| Jack's spike in the air | no damage | no damage |
| Imp, Snitch, Flea: cloaked again after a 300-elmo walk, frame | 115, 125, 90 | 95, 95, 80 |
| Blastwing back on the ground after 300 | 85 | 80 |
| Blastwing attacking at 200: gone at frame, from, damage | 39-41, 112-119, 265-288 | 30, 119, 236 |
| Same at 400 | 59-60, 167-175, 262-342 | 55, 128, 223 |
| Same at 800, out of sight | order lapses | order lapses |
| Passing a Knight at 60; at 120 or more | goes off; doesn't | same |
| Blastwing going off where it sits: frames from death to damage, target, friend | 1, 211-214, 221-224 | 5, 225, 225 |

**What the game corrected:**

* A jumper can be out of reach of what's under it, so `in_range` counts height.
* A Blastwing on fire at will passed a Knight at 120 without going off: only an ordered target extends its weapon's range.  Told to attack, it's looked at every tick.
* A Pyro told to move, fight or attack in the air stood where it landed for the 10 s watched.  The sim keeps it standing until its next order, and ends a move order at landing, where a game's lapsed some time within 12 s.
* A fix for every flier: they now slow down to stop at a goal, rather than overshooting.
* A burrowing unit's decloak distance is its def's, not the area-cloak one.
* Jugglenaut's and Detriment's landing blasts were being fired as ordinary guns.  They now go off only on landing.

**Left out:**

* Units landing at one place spreading out; bouncing off a structure.
* Landing in water too deep to walk in, which the game allows: the sim refuses that order.
* Turning during flight: it faces its goal at take-off.  Jump modules on commanders.  Detriment's weapons being blocked in its wind-up.
* A flight's order stays until the stun ends in the game; here landing ends it.
* Shots at a jumper in the air use only its map position, so they hit more easily than in a game.

**Not checked:** the case of a Knight walking up to a sitting Blastwing played only on rough ground (game: gone at 156-159 from 66, sim: 160 from 79); it didn't move in the flat-ground runs.  Script games, speed and the replay gate weren't rerun after this.

Pyro's flamer did about half its game damage here: fixed in 8.16.  The policy couldn't jump here: it can now (`02-policy.md` 4.3).

### 8.15 Floating structures (2026-11-06)

ABI 13, for the frame model's maps with water (`03-training.md` 11).  Catalog entries gain `min_water_depth` and `max_water_depth` from the defs, and flag `ZK_U_FLOATS` from `floatOnWater`.  A floating structure now reports its height from the water's surface, as ships and aircraft do, where before it read the sea bed.  A structure that doesn't float, such as an Amphibious factory, still stands on the bed.

Nothing else changed.  Script against script is 46/8/46, median 491 s, and doing nothing against the script 11/22/67, both as before.  Replay gate still fails: `results/simgate/jump.json`, 20 s AUC value_tracked 0.564, advance 0.604, hold 0.596, orders 0.611, fightnet 0.665.  Speed measured 39,600 seat steps a second against 42,500 before ABI 12's and 13's changes, about 7% slower or noise; not looked into.

### 8.16 Shots that go through (2026-11-07)

ABI 14.  A weapon with `noExplode` no longer stops at what it hits (`World::pierce` in `sim/src/combat.cpp`, `sim/tests/test_pierce.py`).  Flag `ZK_W_PIERCES` comes from the defs, for all except thermite bombs, whose burn is modelled apart.

In the engine such a shot is tested once a frame and can hit whatever it's inside on each, so hits on a target come to about 1 + chord / speed.  The sim does the same in one go: it finds every enemy on the line from the shooter through the aim point to the end of the range, and for each one applies the shot's blast with its power multiplied by 1 + 2 * sqrt(r^2 - offset^2 - height^2) / speed.  With `ZK_W_ONE_HIT_EACH`, from `single_hit` and `single_hit_multi`, that multiplier stays 1.  Shields stop it as before, and it never hurts its shooter.

**Against a game** (`zz_sim_mechanics.lua` modes `pierce` and `pierce_slow`), damage in the sim over the game's, standing targets, mean over 2-5 distances:

| Shooter | Glaive | Knight | Hermit | Minotaur | Lotus |
|---|---|---|---|---|---|
| Pyro (was about 0.4) | 1.11 | 1.11 | 1.08 | 1.03 | 0.91 |
| Dante | 0.99 | 1.03 | 0.98 | 1.01 | 1.00 |
| Gauss turret | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Dagger | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 |
| Ultimatum, one shot | 0.93 | 1.13 | 0.99 | 0.84 | 1.00 |

Single Pyro distances range from 0.71 to 1.69.

**Differences left:**

* **A walking target:** a Pyro against a Glaive crossing its line does 236 where the game does 430.  Not looked into.
* **Dagger:** 5 shots against the game's 6 in 15 s, which is the whole of its 0.83.  Not looked into.
* **Friends on the line:** in the game the shooter holds fire, and in the sim it fires and passes through them unhurt.
* **First shot:** the Ultimatum's first hit lands within 45 frames in the game, and later than that in the sim, so its row compares 90 frames against 45, one shot each.
* `assaultcruiser`, `bomberassault` and `striderdetriment` also have the flag and weren't compared.

**What it did to the factories,** in 1024 script games, overall score against 8.12's:

| | Spider | Jump | Tank | Amph | Shield | Hover | Veh | Cloak |
|---|---|---|---|---|---|---|---|---|
| 8.12 | 0.92 | 0.47 | 0.74 | 0.67 | 0.48 | 0.24 | 0.21 | 0.29 |
| Now | 0.91 | 0.74 | 0.70 | 0.57 | 0.45 | 0.30 | 0.22 | 0.16 |

The better factory won 93% of decided games between different ones, up from 91%.  So the fix, right as it is for a Pyro alone, makes the spread in `03-training.md` 10.6 a little wider, with Cloak now lowest.  **Still not checked against the engine.**  8.12's row is from ABI 10, and bombs, jumping and floating structures came in since, so not all of this change need be this fix.  The user expects a spread like this: the script only builds raiders, and some factories have more versatile ones.  So it's no sign of a sim fault, and it's a limit of the script as a test (`03-training.md` 10.6).

**Regression:** script against script 45/8/47 on 256 games, median 496 s, against 46/8/46 and 491.  Speed 39,350 seat steps a second against 39,600.  Replay gate still fails, unchanged: `results/simgate/pierce.json`, 20 s AUC value_tracked 0.564, advance 0.604, hold 0.596, orders 0.612, fightnet 0.665.

### 8.17 Grids, pushing, throwing, silos and teleporting (ABI 15, 2026-11-07)

`sim/src/abilities.cpp`, `sim/tests/test_abilities.py`.  The game side is `tools/sim_engine.py abilities`, on ground levelled for the purpose with `Spring.LevelHeightMap`.  Levelling mattered: on the first run's hills, slopes moved units far more than the guns did, and that run's "bursts" were all slopes.

**Grids.**  What has `neededlink` doesn't fire or build unless its grid makes enough.  The sim works it out from generators in pylon reach and calls the result `low_power`, which isn't in the unit token.  In the game the `lowpower` rule param agrees for no generator, a solar, one fusion (all short of a Cerberus's 50) and two fusions, a Singularity (enough).

**Not checked:** a Cerberus in the harness never fired, with or without power, even told to attack a seen Knight 1,000 away with 1,850 range (`grid_ordered_*`: 0 shots).  So the rule param is confirmed and a powered Cerberus's actual fire isn't, in this harness.  The sim's Cerberus fires when powered.

**Pushing** (`push_from`, `step_skid`; Newton and Sumo).  Elmos from the Newton, in the game and in the sim on flat ground, at the same frames.  `CMD.ONOFF` is refused by `cmd_onoff.lua`, and the game's order is `CMD_PUSH_PULL`.

| Case | Game | Sim |
|---|---|---|
| Glaive facing away, frames 0-150 | 300, 321, 349, 375, 402, 425 | 300, 315, 339, 363, 385, 409 |
| Knight facing away | 300, 306, 313, 320, 327, 334 | 300, 305, 313, 319, 326, 333 |
| Ravager facing away | 300, 302, 306, 308, 312, 315 | 300, 302, 306, 308, 311, 314 |
| Sumo facing away | 300, 301, 302, 304, 305, 306 | 300, 301, 303, 305, 306, 307 |
| Knight facing the gun or side on, 150 frames | 300 or 301 | 301 |
| Knight walking at it from 400, 150 frames | 409 | 392 |
| Glaive facing the gun, frames 0-75 by 15 | 300, 341, 470, 548, 555 | 300, 309, 351, 373, 399, 410 |
| Glaive side on | 300, 323, 454, 610, 662 | as facing the gun |
| Pull, Knight facing away, 150 frames | 399 | 400 expected from a test; 367 facing the gun, not measured in the game |
| Pull, Glaive from 400 | 381, 189, 43 at frame 45, then back and forth, 956 lost | 387, 380, 363, 307, 200, 87, 28 at frame 105, hurt in time |

So a unit facing the way it's pushed goes by as much as each hit (a Glaive at about an elmo a frame), and anything else skids against 0.35 of friction a frame: more than a hit gives a Knight, which then stays put and is held from walking.  The difference between pushing and pulling is the gun standing higher than its target (`GUN_STANDS_HIGHER`): a hit on a light unit lifts it off the ground only when pulling, and in the air nothing slows it.  A Sumo at a Glaive facing away drifts it at 1.6 elmos a frame in the game and 0.83 in the sim, which hasn't been looked into.

**Known differences.**  A Glaive under fire isn't thrown as far, 410-450 against 555-660, nor pulled in as fast, because a beam misses what's moving fast between ticks.  Slopes are left out, and on a hill they count for more than the push.  So are units hitting each other or a wall.

**Throwing** (`throw_around`, `update_throwers`).  500 elmos, on flat ground: up by 317, down at frame 147, going across from about frame 18 at 3.97 a frame, and landing 511 away.  The sim takes 148 frames, 317 and 510, with a plain parabola, so it's higher early on (79 in the game by frame 15, 115 in the sim) and within 20 from frame 45 on.  Across, it's within 5 elmos all the way.  Who goes: what's near of either side, not air, not buildings, not the Lobster.  With nobody to throw, the order waits.  Out of range, it walks at 1.8.  Reload 540 frames (541 measured).  **Known difference:** a thrower stunned just after firing stops the throw going across, as in the game, but there the flight's also shorter (120 frames, 155 up), and the sim keeps it whole.

**Silos and missiles** (`free_pad`, `launch_missile`, `destroy_missiles_of`).  In the game: one missile at a time, about 1,800 frames each, on 4 pads at (+-16, +-16), and no more after that.  The sim makes one by each 1,800 frames too.  Losing the silo takes them with it.  A missile with a seen enemy in reach fires by itself, by frame 190 at 1,000 elmos for a tactical (3,502 lost), so nothing has to order it.  In the game a fired missile stays a valid unit off the map for about 17 s; in the sim it's gone at once.

**Teleporting** (`update_teleporters`).  In the game: a beacon takes 270 frames from any distance and the Djinn doesn't move for it, then deployed after 36 (30 here).  A Knight comes over in 52 frames, a Glaive in 15 after it and a Sumo in 143 after that, so mass over 5, longest waiting first, coming out 55-73 from the Djinn with no order left.  In the sim: a beacon at 270, deployed at 35, and over at 65, 85 and 230, coming out 52-80 away.  Walking undeploys it.  Nothing shoots a beacon by itself, and it goes with its Djinn.

**Orders.**  `ZK_ORDER_PLACE_BEACON`, `ZK_ORDER_WAIT_AT_BEACON` and `ZK_ORDER_ON_OFF` are the sim's, as is `ZK_ORDER_MANUAL_FIRE` at the ground.  Frames have them from scripts (`FrameFight`, `BaseScript`), and the policy has kinds for them all since 2026-11-07 (`02-policy.md` 4.3; `policy_sim_env.sim_orders`), with frame flag `PULLING` (`UF_PULLING`) saying which a switch is doing, to its own side only.

**Still unmodelled:** capture, spawned units and area impulse, on `bomberassault`, `dronecarry`, `hoverminer`, `jumpblackhole`, `veharty`, `vehcapture`, `zenith`.  They can be in frames, and stand there without their weapon doing what it does.


### 8.18 Aircraft (ABI 16, 2026-11-08)

`sim/src/air.cpp`, `sim/tests/test_air.py`.  The game side is `tools/sim_engine.py air`: 15 aircraft flown north and turned back, 6 bombers sent once at a tough target 1,500 away with a plane plant 400 behind them, and the gunships set on a still target.  Before this every aircraft flew like a gunship at a fixed 150 elmos.

**Height.**  An aircraft is made on the ground, takes off when it has somewhere to be, and flies at its def's `wantedHeight`, climbing and landing at `verticalSpeed` (3 elmos a frame for planes, as measured).  Below 10 elmos it's on the ground: ground blasts reach it, and it can't fire.  Heights count for weapon reach (`in_range`), the frame's `y` and the state's `altitude`.  A plane takes 20 frames from the ground before it moves off, where the game took 8-12 for bombers and scouts and 28-34 for fighters; a gunship climbs to 0.9 of its height first (the game: 28-40 frames).

**Planes** (`ZK_U_PLANE`, everything that flies but doesn't hover) can't stop in the air.  They turn at a fixed rate, slowing to `turn_speed` when more than 0.5 rad off, and turn away from a goal that's close and behind (`StrafeAirMoveType.cpp`'s near-goal test).  One attacking flies at its target and over it, and comes round.  A move is over 100 elmos from its goal.

| Frames to turn half round | Game | Sim |
|---|---|---|
| planefighter | 72 | 70 |
| bomberprec | 54 | 55 |
| bomberheavy | 90 | 90 |
| planescout | 46 | 45 |
| bomberassault | 176 | 170 |

**Landing** (`ZK_U_LANDS`, from `unit_air_plants.lua`): what a plane plant makes lands when it has nothing to do, a gunship plant's hovers.  A unit that both plants make lands, as the plane plant has it.  A plane at the end of a move stops across and comes down, as in the game.

**Bombers** (`ammo`, `reammo_frames`, `pad_count`; `unit_bomber_command.lua`, `unit_refuel_pad_handler.lua`).  A salvo uses a load; with none left an attack order is over and the bomber makes for the nearest pad with room (a plane plant has 1, an airpad 4, a carrier 2), settles within 80 of it, rearms for `reammoseconds` at 10 energy a second, repairs at the pad's 2.5 build power, and sits there idle.  A fight order is kept.  Out of ammo or on the ground, it gives itself no targets.  Raven: one hit of 800, on the pad 265 frames later, rearmed in 240 (the game: 240).  Phoenix 540 (543), Likho 750 (753), Odin 900 (900).

**Bombs** (`ZK_W_BOMB`, AircraftBomb) go only on what the bomber's sent at, when where it would land (from the plane's height, speed and the bomb's own gravity) is within the salvo's spread of the target, steered by at most an elmo a frame of fall.  A fake bomb (`fake_weapon`) does no damage and marks the target.  Two bombers' real weapons are their scripts':
* **Thunderbird**: where its fake bomb would drop, `bomberdisarm.lua` fires 80 bolts of its lightning, one a frame straight down, and reloads at the end (`ZK_W_DROPPED`).  It flies straight on through the run.  A tank under the run is disarmed.
* **Odin**: its fake bomb is always blocked, and its thermite (`weapon_noexplode_speed_damage.lua`) falls and burns where it lands for 900 frames, ramping from 0 to 1,000 dps.  Here it's a dropped bomb leaving a burning area of 40 elmos, ramping from 50 to 950 dps for the same total.  On a still target: 14,927 against the game's 14,842.  Odin still isn't on any menu: its shield cluster spawns units.

**Strafing gunships** (`ZK_U_STRAFES`, `airStrafe`) circle what they shoot at at 0.8 of `combatrange`, or of range, at `min(top, sqrt(0.37 accel r))`.  A Krow doesn't strafe.

| Circle, elmos | Game | Sim |
|---|---|---|
| gunshipraid | 199 | 197 |
| gunshipskirm | 292 | 293 |
| gunshipassault | 124 | 132 |
| gunshipheavyskirm | 500 | 508 |

**The strider hub** is a factory here (`isfakefactory`); in the game it's a nanotower that builds striders beside it.  Like a Cerberus it needs a grid of 50 energy, so on its own it builds nothing.

**Known differences.**
* A Krow attacking holds 306 elmos away, the game 381.  It overshoots where it can first hit.
* Ground weapons can aim at aircraft, as in the game (fixed wing is only a bad target for most).  What hits a plane is whatever reaches its height.
* Transports don't carry.  Drones (Funnelweb's, the Reef's) aren't made.
* No crashing, no fighter pull-up, no `unit_vertical_swarm`.
* After rearming, the game's bomber flies about 115 elmos and lands again; here it stays on the pad.

### 8.19 Group orders (ABI 17, 2026-11-08)

`World::order_group` (`sim/src/world.cpp`), `zk_group_rec` (`capi.h`), `sim/tests/test_group_orders.py`.  One order to many units, as a player gives one to a box of selected units.  It names an anchor unit: the anchor, and every other unit of its side that is alive, mobile and finished, has a role in the order's bit mask, and stands within its radius of the anchor (in float, `dx*dx + dz*dz <= r*r`), are given the order as a unit order would give it.  The anchor is told whatever its own role; a dead anchor selects nothing.  Roles are each catalog type's, set once per batch (`zk_batch_set_roles`; `policy_sim_env` sets those of `zkbot.policy.groups.roles`, the bot catalog's in sorted order); a type with none is never selected.  `zk_batch_step` takes a game's group orders beside its unit orders and applies them first, each in turn, so a later group order replaces an earlier one for a unit both select, and any unit order replaces a group's.  Nothing about a unit records that its order came in a group.
