# The simulator

`sim/` is a simplified Zero-K in C++17, built with CMake in WSL and loaded from Python with ctypes.  It plays many games at once,
emits the same raw frame the live bot reads (02-policy.md), and runs the scripted players the policy is cloned from and trained
against.  Engine and game facts below were read from `../RecoilEngine` and `../Zero-K`, and most were measured in a real game
(see "Validation").

The rule for what to model: an error is acceptable if the policy can't profit from it.

## Building and the libraries

    cmake -S sim -B ~/zksim/build && cmake --build ~/zksim/build -j8

This builds three libraries in `~/zksim/build` (`ZKSIM_LIB` overrides where they're looked for).  Rebuild after any change under
`sim/`.

| Library | What | Python side |
|---|---|---|
| `libzksim.so` | the sim (`sim/src`, `sim/include/zksim`, C ABI in `capi.h`, ABI 18) | `sim/py/zksim.py` (`Batch`), `sim/py/zknav.py` (routes) |
| `libzkscripts.so` | the army and economy scripts (`sim/scripts`, `zkscripts.h`, ABI 8) | `tools/policy_scripts_cpp.py` |
| `libzkrecord.so` | the game recorder (`sim/record`, `zkrecord.h`), needs `libzstd-dev` | `sim/py/zkrecord.py` |

`sim/include/zksim/rawframe.hpp` is generated from `bots/zkbot/policy/rawframe.py` (`python tools/policy_schema.py`), and a test
compares the two.  The game around the sim (starts, scripts, rewards, the policy's orders) is `tools/policy_sim_env.py`'s `SimEnv`.

## The batch

* A batch holds many games and steps them a slot at a time on a thread pool of its own.  The caller's thread works too, and
  games are handed out costliest first, by the last step's time.  Any thread count plays the same games bit for bit.
* Several batches can step at once in one process.  A forked process with a live batch starts its own threads
  (`test_batch_threads.py`).  `zk_batch_step_begin` / `zk_batch_step_wait` let the caller work while the sim steps, and
  `zk_batch_set_threads` changes the pool.
* Each game has its own random generator, seeded by the seed and the game's index.  Sums run in handle order.
* Fog goes out from each side's own units.  The spatial grid is split by side, and a tick allocates nothing.

Speed, `tools/sim_step_bench.py`, 128 games on 8 threads: 142,943 game-steps a second at 1 minute into games, 57,889 at 8 and
33,213 at 16.  With both sides scripted, `SimEnv.step` runs 92,443 seat decisions a second at 8 minutes (2.8 ms a step).
Movement is the largest phase.

## Data

* **Units and weapons** come from an in-engine dump after Zero-K's post-processing (`tools/simgame/` gadgets, then
  `python tools/sim_data.py`), because `armordefs.lua` and `unitdefs_post.lua` change values the raw unit files don't show.
  `sim_data.py` also flags what isn't modelled (`ZK_W_UNMODELLED`: capture, spawned units, area impulse) and reads each
  `customParams` that changes a weapon's meaning.
* **Maps** are exported by `tools/sim_data.py`, with their routes worked out at export (see "Movement").  `sim/data/maps` is
  gitignored: after a pull, run `python tools/sim_data.py maps`.
* **Map facts** come from a real game (`python tools/sim_engine.py facts MAP...`, into `sim/data/map_facts.json`): start boxes,
  metal spots and reclaimable features.  A game is asked because a map's layout file is Lua that may build its lists in code
  (Faultline makes half its spots that way).  A map may also draw its boxes differently on each load, so `facts` loads each map
  several times and keeps every layout it sees.
* **The five training maps** (`TRAINING_MAPS`): Agalta, Faultline, FrostyCove, Red Comet and TitanDuel.
* **The map pool** (`/mnt/d/zk-learned/maps`, `tools/map_gen.py`): 98 real maps and 200 made-up ones (relief, water and
  symmetry drawn per map, with starts made reachable).  21 are held out (`held_out_maps`): TitanDuel and a tenth by the hash of
  the name.  The pool loads in 37 s to about 7.5 GB, so load it once before forking workers, and never in two jobs at once.
  `policy_sim_env.share_maps` builds each map's derived data before the fork too, so workers share it.

## What's modelled

Tags in this section: what's left out is listed under "Known gaps", and what was checked against a game is under "Validation".

### Movement

* Top speed, acceleration and braking as the engine integrates them.  Turn rate, and slope and depth speed per move class from
  `CMoveMath`.  Speed on a slope is directional: going downhill isn't slowed.
* Six move classes (kbot, tank, spider, amph, hover, ship), from each unit's `smClass`, depth and `maxSlope`.
* Units are spheres, kept apart by separation, and may push enemies.  A unit stops up to about 50 elmos short of a move goal.
* **Re-ordering costs what it does in the engine.**  A plain move order stops the unit until its next slow update (every 15
  frames).  That includes an order given up to 2 frames before the update.  A fight order costs nothing.
* **Routes are made at export, and nothing is pathfound at run time.**  Each `terrain.py` region is cut per move class into
  segments of up to 4 parts, whose centres see 90% of them.  A next-hop table gives the gate to take from any segment towards
  any other.  A unit:
  1. walks straight if the line to its goal is clear, as Zero-K's `cmd_raw_move.lua` does;
  2. otherwise aims at the farthest gate or centre in sight, up to 4 segments ahead;
  3. otherwise aims at its segment's centre;
  4. if stuck (under 20% of its top speed's distance in 3 s), aims straight at the next gate or the goal, alternating, for 3 s,
     doubling up to 24 s.
* Arrival within 4x the straight walk, Titan Duel, across segments: kbot 96%, tank 91%, spider 100%, hover 93%.  Most misses
  are units stuck, not slow routes.  Tank and hover are the weak classes.

### Weapons and damage

* Reload, burst and projectiles per shot, including script bursts.  A target is chosen by category, then distance.  Shots lead
  their target as the engine does (no target velocity).  Spray scatters sideways only.
* No projectiles.  Beams hit at once.  Other shots' damage lands after their flight time, if the target is still within the
  blast.  Blast falloff is the engine's formula, measured from the centre less the unit's radius.
* Burning ground and burn damage; heat rays whose damage falls with range; starburst missiles that climb, then fly; fake guns
  that never fire.
* Slow, EMP and disarm.  Each level is measured against current health, with overstun.  Disarm stops firing and reloading but
  not walking, has its own timer, and makes a disarmed unit a worse target.
* Manual-fire and stockpile weapons.  Felon's gun draws from its own shield.  Ronin and Phantom reload slower while moving.
* **Shields** stop a hostile shot at the first sphere on its path that can pay for it, and the shot explodes there.  Shells
  follow the engine's parabola, so they can fly over a small bubble.  Charging costs energy.  Touching shields share charge, as
  `unit_shield_link.lua` does.
* **Shots that go through** (`noExplode`) hit everything on their line, scaled by the chord through each.  `single_hit`
  weapons hit each unit once.
* **Bombs** (Limpet, Imp, Skuttle, Snitch) go off by themselves at their `kamikazeDist`.  They do so when told to attack, or
  on fire at will for any enemy in reach; at hold fire, their default, they don't.  Self-destruct is an order.
* **Jumping**: a unit walks into range, goes when charged, flies an arc, and reloads after landing.  Height in the air counts
  for reach and blasts.  Jugglenaut and Detriment have landing blasts.  Imp, Snitch, Flea and Blastwing dig in.
* **Death**: a death explosion, then a wreck (40% of the unit's metal) or a heap (20%), by its recent damage.  A frame over 5%
  built leaves a heap in proportion.  A wreck in a blast becomes a heap.

### Aircraft

* Made on the ground, an aircraft takes off when it has somewhere to be, and flies at its `wantedHeight`, climbing at
  `verticalSpeed`.  Under 10 elmos it's on the ground: ground blasts reach it, and it can't fire.
* **Planes** can't stop.  They turn at a fixed rate fitted to the engine, slowing in a hard turn, and come round after a pass.
  What a plane plant makes lands when idle; a gunship plant's hovers.
* **Bombers** spend a load a salvo.  Empty, they make for the nearest pad with room, rearm at 10 energy a second for
  `reammoseconds`, and repair there.  Bombs drop only on the target they were sent at.
* Thunderbird's run is 80 lightning bolts.  Odin's thermite is a burning area.
* **Strafing gunships** circle their target at 0.8 of their combat range.
* The strider hub is a factory here, and needs a grid of 50 energy like a Cerberus.

### Abilities

* **Grids**: what has `neededlink` (Cerberus, Lucifer, the strider hub) doesn't fire or build unless its pylon grid makes enough
  energy.  The unit token has no grid number.
* **Push and pull** (Newton, Sumo): a hit moves a light unit facing away by about an elmo.  Anything else skids against friction.
* **Throwing** (Lobster): what's near, of either side, flies a parabola to the aim point.
* **Silos**: one missile per 1,800 frames, on 4 pads.  A missile with a seen enemy in reach fires by itself, and losing the silo
  takes its missiles.
* **Teleporting** (Djinn): a beacon takes 270 frames to place.  Units waiting at the Djinn go over, the heaviest last.

### Unit AI

Units act on their own between orders, as in the game (`ZK_BEHAVIOUR_ORDERS`).  `ZK_BEHAVIOUR_SCRIPTED` does only what it's
told, for tests that measure one mechanic.

* **Attack-move**: a unit chases the closest enemy within its range plus 100 of its line.  Its leash is 500 plus its range.
* **Idle**: a unit attacks within its range plus 200, gives up after 150 frames, and walks home.
* **Fire states**: hold fire, return fire, and fire at will (the default).  Move states: hold position, maneuver (the default),
  roam.  The policy can't set either yet.
* **Tactical AI**, ported from `unit_tactical_ai.lua`: raiders run from riots, and skirmishers kite and step out of range while
  reloading.
* **Overkill prevention** from each script's `BlockShot`, and **bait prevention** at each unit's default level.
* **Radar**: an unidentified blip is aimed at with an error.  A weapon that hits only by touching can't hit a radar dot, unless
  the dot is a structure already seen (the user's rule, stricter than the game).

### Sensing

* **Line of sight** is a height-map ray per pair, within each unit's sight radius.  **Radar** is a plain circle.
* A blip's type is known only while it's in sight, or if it was seen and has stayed on radar since (the engine's rule).
* Enemy structures once seen stay as memories until seen gone.
* **Cloak**: a unit can cloak itself, or be inside an area cloak.  Cloaked, it's hidden from sight and from radar
  (`unit_stealth.lua`).  Cloak costs energy, more while moving.  A unit is shown by an enemy within `decloakDistance`, or by
  damage, firing or building, and is then blocked from cloaking for 45-240 frames.
* A unit's height in the frame has a 40-elmo dead zone, so it says "flying or not" the same way live and in the sim.

### Economy

A world runs the economy only if `zk_world_economy` turns it on.  Fight replays and the battery don't.

* **Start**: 350 metal and energy, and 500 storage.
* **Income**: 2 and 2 innate, plus 4 metal and 6 energy from the commander while it lives.  These figures were settled on 188
  logged games.  An extractor earns its spot's income and a solar 2 energy, and nothing while stunned.
* **Overdrive** runs once a second, as `unit_mex_overdrive.lua` does: spare energy is shared over pylon grids by each
  extractor's income squared, and an extractor earns income x (1 + 0.25 sqrt(energy)).  It matches the game exactly on full
  storage.  The income reported is the game's, so energy income falls when overdrive runs.
* **Building**: progress per second is build speed / build time, paid from both resources as it goes, and stalling on whichever
  runs out.  A side's first factory is free and instant.  ASSIST helps build; given to a factory, it helps with what the
  factory is building.  A factory repeats the type of its last PRODUCE.
* **Factory plates** (`unit_factory_plate.lua`) make their factory's roster, but only while a finished factory of their type and
  side, not stunned or disarmed, is within the plate's build range (420).  Off, a plate makes nothing and helping it does
  nothing (`World::plate_off`).  The catalog's `plate_parent` says whose plate is whose.
* **Caretakers** (`staticcon`): an idle one helps the nearest factory of its side in its reach (500) that's building, else the
  nearest frame of its side, as Zero-K's default widget has it patrol (`unit_auto_patrol_nanos.lua`; `World::run_caretaker`).
  It needs no orders, so a policy that can't give them one still gets its use.  There's no repair or reclaim for it to do.
* **Reclaim** runs at the builder's reclaim speed in metal a second and is paid as income.  **Upkeep** (radars, jammers, area
  cloak) is paid in energy once a second.  **Stockpile** pays for both resources as it goes.
* **Regeneration** (`unit_regeneration.lua`): `idle_regen` health a second, once `idleTime` has passed since the last hit.  It's
  the only healing; there's no repair.
* **Solars** close for 120 frames after a hit: no energy, armoured to a third.

### Group orders

One order to many units, as a player gives one to a box (`World::order_group`).  It names an anchor, a radius and a set of
roles.  The anchor, and every other alive, mobile, finished unit of its side within the radius whose role is in the set, get the
order.  Roles are the bot catalog's (`zkbot.policy.groups`).  Group orders apply before unit orders in a step.

## The game in the sim

`SimEnv` (`tools/policy_sim_env.py`) plays games of two sides, each one player:

* **Start**: commanders in two of the map's start boxes, drawn as Zero-K's `allshuffle` does.  The start is the box's start
  point half the time; otherwise it's anywhere level in the box.
* **End**: elimination; or from 2 minutes, one side at 2.5x the other's value and 10,000 metal ahead; or a draw at 20 minutes.
* **Every side** plays the opening script, and its constructors run the economy script.  Its fighters and factories are run
  by the army script or by the policy.
* **A policy's seat**: the scripts can still run there but only advise (`SimEnv.advise`), giving labels that are never carried
  out (libzkscripts' label seats).  In a share of games the policy runs its economy as well (`policy_economy_share`).
* **Variants** (`variants=True`): each side's army script draws its knobs at the start: a preset (standard, rush, turtle,
  skirmish), then each knob jittered.  The knobs are the role weights, how much metal a squad wants over the enemy's cover
  (`commit`), what the army gathers before going, the share of raiders guarding extractors, give-up timers, and the health at
  which a fighter retreats.
* **Group orders**: the army script sends identical orders as group orders wherever that's exact (`group_orders`, on by
  default).

### The opening (`tools/policy_opening.py`)

The same on both sides, in Python:

1. The factory goes about 160 elmos towards the enemy, and starts repeating its raider.  The factory and commander are drawn per
   side.  The training maps keep all 8 land factories, drawn evenly; the pool offers what suits each map (`Ground.suits`).
2. An extractor and a solar go at each of the nearest spots that are nearer us than them, 4 to 6 by map.
3. Then the commander assists the factory.

It finishes by a median of 152 s.  A step that takes over 2 minutes is given up.

### The army script (`sim/scripts/src/army.cpp`)

* **Production**: each unit is the cheapest of its role on the factory's roster.  A role starts when its share lags furthest
  behind its target share:
  * raiders only to 4 minutes, falling to 35% by 60;
  * riots from 10 minutes, assaults from 6, artillery from 18;
  * skirmishers once a riot or assault has been seen, AA once air has, and snipers once a slow heavy has (after 8 minutes);
  * one scout from 2 minutes, and a second from 10.
  
  Air plants have their own mix, which counts only what flies.
* **Raiders** move in squads of three.  A squad goes for extractors, then other buildings, and takes a target only when its
  metal beats the cover there.  Without a target, it probes towards the enemy and stops before defended ground.  A quarter of
  the raiders guard their own extractors.
* **The rest**: assaults lead to the enemy's start, then to the nearest building.  Riots hold behind them; skirmishers back off
  armed units; artillery stands at 0.75 of its range; AA follows the army; snipers hunt slow heavies; bombers go for defences.
* **Re-ordering**: a unit is re-ordered only when its job or goal changes.
* **Groups**: orders that are the same to the letter go as one group order (`Seat::group_plans`), and take in units already
  told that order.
* **Missiles and the Trinity** are aimed by the script's aimer: each missile at the most valuable building in reach, and the
  Trinity at the most enemy value within its blast.

### The economy script (`sim/scripts/src/eco.cpp`)

It runs every `ECO_EVERY` 6 slots, as the actor decides.  A free constructor (idle, helping, or stuck) takes the first job due:

1. finish what's being built nearby, or the big project;
2. the big project, once due: Cerberus or Lucifer and a silo from 16 minutes and 55 metal a second, when map control has
   stalled; the Trinity from 18 minutes and 60, by a 30% chance;
3. energy, while energy made is under 1.5x metal income;
4. expand to the nearest free metal spot on our half with no enemy near;
5. spend while metal floats over 50% of storage (the user, 2026-11-10): energy if short; else help a factory or plate that
   fewer than 2 are helping; else more production beside the first factory, one per 12 metal a second and 8 at most counting
   the factory, one started a minute: its plate, within 300 elmos, then caretakers within 380 of both, which help them; else
   help the nearest factory anyway.  Never another factory like the first, nor a second plate;
6. energy up to a share of income that grows over the game (solars, then fusions, then a Singularity);
7. radars and pylons;
8. help the factory.

Further rules:

* **Reclaim**: up to 2 constructors (one in 3) reclaim full time.  Others with nothing better reclaim within 1,200 elmos rather
  than assist.  They avoid enemies, and go home when hurt.
* **Constructors** come from the factory: the larger of a schedule by minute and one per 3 metal a second of income, 2 more
  while metal floats, 20 at most.
* **A second factory** (the user's rule, 2026-11-10): a plane plant, gunship plant or strider hub (0.5, 0.3, 0.2), at 40 metal a
  second not counting reclaim, or 20 minutes.  There's no third.  A factory a constructor was sent to start counts as there:
  the walk can take longer than the minute between starts, which once gave two.
* A constructor stuck for 30 s is given another job, and its spot is avoided for 3 minutes.

Metal lost at the storage cap is 2-13% of income by game phase (it was 62% before the spend step).  Reclaim is about 4% of a
side's metal.

Against the scripts before 2026-11-10's change, 48 games each (one seed, so the same maps and draws to start with):

| | before | after |
|---|---|---|
| factories at 10 / 15 / 19 minutes (median) | 3 / 4 / 5, up to 6 | 2 / 2 / 2 |
| plates, caretakers at 19 minutes | 0, 0 | 1, 3 |
| storage full at 10 / 15 / 19 minutes (mean) | 0.36 / 0.22 / 0.18 | 0.36 / 0.27 / 0.10 |
| a side's value at 10 / 15 / 19 minutes (median) | 12,300 / 24,500 / 27,800 | 14,600 / 25,400 / 31,700 |
| income at 10 / 15 / 19 minutes (median) | 48 / 66 / 50 | 56 / 77 / 66 |
| the second factory | 2 sides, at 19.7 minutes | 96 sides, at a median of 7.6 minutes and 49 metal a second |

The sides still in play thin out (84 at 10 minutes, 38 at 15, 20 at 19) as games end on a lead.

### What decides script games

Endings of `army-5`'s 114,223 script games: 62% on the lead, 34% at the time limit and 3% by elimination.

In raider-only games (an older script), the factory drawn decided 91-93% of decided games between different factories:

| Spider | Jump | Tank | Amph | Shield | Hover | Veh | Cloak |
|---|---|---|---|---|---|---|---|
| 0.91 | 0.74 | 0.70 | 0.57 | 0.45 | 0.30 | 0.22 | 0.16 |

That's overall score (wins plus half draws).  It was never checked against the engine.  With the army script's full mix, the
better factory won 63% (one batch of 64).

## Validation

Tools: `python tools/sim_engine.py facts|mechanics|overdrive|abilities|air` plays set-ups in a real game and logs them, and
`tools/sim_battery.py` (`engine` on Windows, then `compare` in WSL) runs the scripted battery.  The engine side is the gadgets
in `tools/simgame/`.  Mechanics are pinned by `sim/tests/test_*.py` with the engine's numbers in their docstrings.

* **Battery: 119 of 138.**  It covers straight lines, turning, duels, blasts, slope speed, re-order cost by class and rate,
  paths around walls, and status effects.  The misses:
  * routes: 9, mostly tank and hover detours and walk times;
  * the first Racketeer missile lands 35 frames late in the engine;
  * a unit held at the stun threshold flips over and under;
  * Bantha's manual salvo;
  * Scorcher at 90 elmos;
  * a kbot re-ordered every 0.5 s.
* **Mechanics checked in a game** were all within a tick or a few percent, except where noted:
  * overdrive, exact on full storage;
  * wrecks and heaps, reclaim, upkeep and stockpile;
  * the solar's closing; cloak and decloak timings;
  * shields: stops, charging and sharing;
  * the unit AI's probes, apart from the two Ronin cases;
  * bombs and jumps (blasts run 4-28% strong, probably height);
  * piercing shots: Pyro 0.9-1.1, Dagger 0.83;
  * push, throw, silos and teleporting;
  * aircraft: plane turns within 10%, rearm times within 3 frames, strafing circles within 8 elmos.
* **A live frame recreated in the sim** (`tools/policy_golden.py sim`) gives the same unit features.  Its fog is a little
  tighter: it saw 18 of the 23 enemies the live game saw, and none it didn't.
* **The replay gate failed.**  44,733 logged fights from the rules bot's games were replayed in the sim, and scored on who
  lost more in 20 s.  The sim got AUC 0.604 (advance) and 0.613 (the `orders` behaviour); counting value alone got 0.564, and a
  learned fight model, out of fold, 0.665.  The sim beats counting value and loses to the learned model at 20 s; at 10 s it
  ties.  Sim fights go much closer to the death than real ones (0.86 of a side's value lost where 0.46 was), probably because
  real units retreat and get re-ordered.  The user waived the gate on 2026-11-05.  `tools/sim_fight_gate.py` reruns it.
  **Don't tune the sim against these fights**, or they stop being a test.

## Known gaps

What's left out on purpose, so it isn't mistaken for modelled:

* **Movement:** terrain-type speed multipliers, crush, collision damage, and units spreading out on landing.  Raiders don't
  jink sideways as the game's swarm logic has them do.
* **Weapons:**
  * accuracy scatter up and down (sideways only);
  * Gauss rifles hitting everything along their line;
  * friends on a piercing shot's line (the game holds fire);
  * the first-shot timing of some heavy guns;
  * shots at a jumper in the air use its map position only.
* **Shields:** they all start full (several start empty in the game), a stun doesn't drain them, and there are no charge
  priorities when energy is short.
* **Unit AI:** ward fire, `fightOnlyOverride`, jammers, and the amphibians' sea table.  Builders on attack-move don't reclaim on
  the way.
* **Economy:** repair, Duck's faster healing under water, nanoframe rate limits, commander morph, capture, resurrect.  Reclaim is
  costed in metal alone.
* **Air:** no crashing, fighter pull-up or vertical swarm.  Transports carry nothing, and no drones are made.
* **Unmodelled weapons:** capture, spawned units and area impulse (`bomberassault`, `dronecarry`, `hoverminer`,
  `jumpblackhole`, `veharty`, `vehcapture`, `zenith`).  These units stand in frames without their weapon doing what it does,
  and are kept out of armies.
* **The frame:** disarm isn't shown (only paralysis), and nor are grid numbers.
