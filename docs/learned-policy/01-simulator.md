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
| `libzksim.so` | the sim (`sim/src`, `sim/include/zksim`, C ABI in `capi.h`, ABI 21) | `sim/py/zksim.py` (`Batch`), `sim/py/zknav.py` (routes) |
| `libzkscripts.so` | the army and economy scripts (`sim/scripts`, `zkscripts.h`, ABI 12) | `tools/policy_scripts_cpp.py` |
| `libzkrecord.so` | the game recorder (`sim/record`, `zkrecord.h`, ABI 2), needs `libzstd-dev` | `sim/py/zkrecord.py` |

`sim/include/zksim/rawframe.hpp` is generated from `bots/zkbot/policy/rawframe.py` (`python tools/policy_schema.py`), and a test
compares the two.  The game around the sim (starts, scripts, rewards, the policy's orders) is `tools/policy_sim_env.py`'s `SimEnv`.

## The batch

* A batch holds many games and steps them a slot at a time on a thread pool of its own.  The caller's thread works too, and
  games are handed out costliest first, by the last step's time.  Any thread count plays the same games bit for bit.
* A game has up to 8 teams (`ZK_MAX_TEAMS`), each in an allyteam ("Teams and alliances").  The batch's arrays are
  games x teams, a team's place being its seat (`game * slots + team`); a slot a game doesn't use gives empty frames.  With 2
  slots that's the `game * 2 + side` of a batch of 1v1s, which plays bit for bit as it did before there were teams.
* Several batches can step at once in one process.  A forked process with a live batch starts its own threads
  (`test_batch_threads.py`).  `zk_batch_step_begin` / `zk_batch_step_wait` let the caller work while the sim steps, and
  `zk_batch_set_threads` changes the pool.
* Each game has its own random generator, seeded by the seed and the game's index.  Sums run in handle order.
* Fog goes out from each allyteam's own units.  The spatial grid is split by allyteam, and a tick allocates nothing.  What a
  tick costs follows the units alive, not every unit there has been: an 80-minute 1v1 steps in 142 µs a game.

Speed, `tools/sim_step_bench.py`, 128 games on 8 threads, measured before the sim had teams (16-minute games have run about
12% faster since): 142,943 game-steps a second at 1 minute into games, 57,889 at 8 and 33,213 at 16.  With both sides scripted,
`SimEnv.step` runs 92,443 seat decisions a second at 8 minutes (2.8 ms a step).  Movement is the largest phase.

## Data

* **Units and weapons** come from an in-engine dump after Zero-K's post-processing (`tools/simgame/` gadgets, then
  `python tools/sim_data.py`), because `armordefs.lua` and `unitdefs_post.lua` change values the raw unit files don't show.
  `sim_data.py` also flags what isn't modelled (`ZK_W_UNMODELLED`: capture, spawned units, area impulse) and reads each
  `customParams` that changes a weapon's meaning.  A unit's record has its footprint (`foot_x`, `foot_z`: half its `xsize` and
  `zsize` in elmos) and `max_height_dif`, for "Where a building can go".
* **The game's rules** (`sim/data/rules.json`, from `python tools/sim_rules.py`, which needs a Python with `lupa`): the 42
  units that don't keep a team in the game, the reveal rule's numbers, the six chicken difficulties and the 21 egg features.
* **Maps** are exported by `tools/sim_data.py`, with their routes worked out at export (see "Movement").  `sim/data/maps` is
  gitignored: after a pull, run `python tools/sim_data.py maps`.  A map built by hand (`zksim.Map(...)` without routes) works
  them out when made, once for the move classes that move alike on it.
* **Map facts** come from a real game (`python tools/sim_engine.py facts MAP...`, into `sim/data/map_facts.json`): start boxes,
  metal spots and reclaimable features.  A game is asked because a map's layout file is Lua that may build its lists in code
  (Faultline makes half its spots that way).  A map may also draw its boxes differently on each load, so `facts` loads each map
  several times and keeps every layout it sees.
* **The five training maps** (`TRAINING_MAPS`): Agalta, Faultline, FrostyCove, Red Comet and TitanDuel.
* **The map pool** (`/mnt/d/zk-learned/maps`, `tools/map_gen.py`): 404 maps.  204 are real, 18 of them Violet Rampart; 200
  are made up (relief, water and symmetry drawn per map, with starts made reachable).  31 are held out (`held_out_maps`):
  TitanDuel and a tenth by the hash of the name.  The pool loads in 61 s to 10.8 GB (11.6 GB once games have run on every
  map), so load it once before forking workers, and never in two jobs at once.  `policy_sim_env.share_maps` builds each map's derived data before
  the fork too, so workers share it.
* **What each pool map can play** is its facts' `modes` (`tools/map_modes.py`; `python tools/map_gen.py modes` writes them):
  * `1v1`: any playable map with two start boxes or more (381 maps);
  * `teams`: players a side (332 maps).  Whether it's a team map is Zero-K's listing (`maps.json`), or two boxes on a map
    Zero-K doesn't list; how many a side is the map's area over 48 square map units, 2 at least and 4 at most, since Zero-K
    doesn't say;
  * `ffa`: alliances (64 maps): Zero-K's "up to N", or the boxes of an unlisted map with more than two, up to its boxes and 8;
  * `chickens`: a map named `Chicken...`, or listed for chicken defense and for no other kind of game, plays nothing else
    (5 maps).  "Special", Zero-K's mark for an unusual map, isn't a kind of game: Requiem Outpost has both, and is a
    chicken map.  `chickens_or_teams` adds the team maps, where chicken games are drawn too (337).
  
  A map Zero-K lists for no kind of game at all (18 of them) goes by its boxes, as an unlisted one does.
  
  Maps named `THISspace...` or `ZeroWars...` play nothing (the user, 2026-11-10).
* **Violet Rampart** makes its terrain at game start, by its number of bases and a seed.  `tools/map_rampart.py` is a port of
  the map's gadget: bit-identical to the Lua under a shared random generator (40 runs, 3 to 11 bases), and its 4-base map has
  exactly the 44 metal spots of a real game.  `python tools/map_gen.py rampart` adds the pool's 18: 3 to 8 bases, each at
  three seeds.  They're free-for-all maps only, and the maps of one base count share one map's weight in a draw.
* **New maps** need the engine: `zk-maps.ps1` to fetch them, `python tools/sim_engine.py facts ... --out
  D:/zk-learned/maps/real_facts.json` (which starts the engine six times a new map), then `python tools/map_gen.py real
  --facts ... --out D:/zk-learned/maps --workers 6`, then `python tools/map_gen.py modes`.
* **Not in the pool**, of the install's maps (2026-11-11, when 89 were added):
  * THISspace and ZeroWars Crossfire, which play nothing;
  * Violet Rampart Generator, which `rampart` makes instead, and five more that make their terrain when the game starts
    and have no port: Alphabet Siege Dry, Fickle Rivers, LimeQuat Republic, Rainbow Comet and Random Plateaus;
  * `Chicken_Roast_v1` and `duck`, whose facts have no layout of two start boxes;
  * `Valles_Marineris_v2`, whose facts timed out twice.
* **Every kind plays on every new map** (2026-11-11): one scripted game to 8 minutes of each kind a map holds.  They're no
  different from the old ones: in 1v1s a side still had 3 units or fewer at 8 minutes on 2 of 85 new maps, 1 of 96 old
  ones and 2 of 200 made ones, and the poorer side's income was a median 26 metal a second against 29.  Some of those
  were start points a commander can't stand on, since moved ("The game in the sim", Start): 12 games on Esker Creek
  then left no side stuck.  The rates weren't measured again.

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

### Transports (`sim/src/transport.cpp`)

The gunship plant's Charon and Hercules carry one unit each (the user, 2026-11-11), by Zero-K's
`unit_transport_load_unload.lua`, `unit_transport_speed.lua` and the engine's `MobileCAI`.

* **What's lifted** is by the load's class, which is by its cost (`unitdefs_post.lua`): light, medium
  (`requiremediumtrans`) or heavy (`requireheavytrans`).  A Charon flies at 0.7 of its speed with a light load and 0.4 with
  a medium one, and lifts no heavy one; a Hercules at 1.0, 0.75 and 0.5.  Nothing lifts a building, an aircraft, or a
  unit marked `cantBeTransported`.  The catalog has each type's `load_class`, `load_height` and `load_size`, and a
  transport's three shares and `lift_size`.
* **LOAD** (order 22, a unit): the transport flies to it, comes down to the load's height within 220 elmos, and picks it
  up within 16.  One of its own team stops for it once the transport is within 80.  Any other unit, an ally's too, is
  picked up only in sight, uncloaked and all but still (0.5 elmos a frame): so an enemy that stands can be taken.
* **Carried**, a unit does nothing and nothing can shoot it; it's seen when its transport is.  The frame flags it
  `CARRIED` and its transport `LOADED`.
* **UNLOAD** (order 23, a place the load can stand): the transport comes down within 198 elmos and sets it down unhurt.
* **DROP** (order 24, Drop Cargo): the load is let go at once, keeps the transport's speed and falls, hurt by
  `unit_fall_damage.lua` as anything thrown is: a Jack from a Charon in flight loses about a tenth of its health, an
  Ogre a fifth.  The button reloads for 7 s.  A transport that dies drops its load the same way.
* **Flight**: a transport on a LOAD or UNLOAD order heads straight for its goal.  The turning gunship model circled a goal
  inside its turning circle and never came down on it.

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
  runs out.  A team's first factory is free and instant.  ASSIST helps build, within an allyteam; given to a factory, it helps
  with what the factory is building.  A factory repeats the type of its last PRODUCE.  The sim builds wherever it's told:
  whether a place will do is decided by who orders it ("Where a building can go").
* **Allies** share as Zero-K's teams do, simplified.  Extractor income is pooled and split evenly among the allyteam's teams
  that have a live unit.  Overdrive runs per allyteam, its metal and energy shared by the spare energy each put in.  What
  overflows a team's storage is split among its allies with room, energy too.  There's no payback to whoever built the energy
  and no pooled storage, and innate and generator income stay with their owner.
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
roles.  The anchor, and every other alive, mobile, finished unit of its team within the radius whose role is in the set, get the
order: never an ally's.  Roles are the bot catalog's (`zkbot.policy.groups`).  Group orders apply before unit orders in a step.

### Teams and alliances

* **A team owns** its units and its purse (`team[u]`); **an allyteam** shares sight and can't be shot by its own (`side[u]`).
  A raw frame's `player` is the team and its `ally_team` the allyteam.  A frame lists the viewer's own units first, then its
  allies' (seen whole), then what it sees, radar blips, memories, and what's hidden (the critic's).
* **What keeps a team in the game** is any finished unit not on Zero-K's don't-count list (`game_over.lua`: the Flea, the
  bombs, terraform units and the like; 42 in `rules.json`).  The batch's four stats a team are its tracked value, its live
  units, its reclaim income, and its finished units that keep it in.
* **The reveal rule** (`game_over.lua`): with two allyteams left and one worth over 25 times the other, the weaker one's
  units are always visible, and any off the map are destroyed.  It's checked every 45 frames, never in chicken games, and
  doesn't end the game (`zk_world_set_reveal`, `zk_batch_set_reveal`; on by default).

### Chicken defense (`sim/src/chickens.cpp`)

A port of Zero-K's `unit_spawner.lua` with `spawn_defs.lua`'s numbers, at Zero-K's own timings (the user, 2026-11-10).
`Batch.reset(..., chickens=(team, difficulty))` gives a team to it: the sim then spawns and orders that team's units, and
`Batch.chicken_state(game)` says where things stand (the queen: not yet, here or dead; seconds until she's due; burrows alive
and killed; chickens; waves; anger; tech).

* **Burrows and waves** come up on the frames the Lua's formulas give (checked to wave 91).  A burrow killed angers the
  chickens and brings the queen sooner; each wave calms them.  The six difficulties are `rules.json`'s.
* **The queen** arrives when her time is up, morphs between her two forms, and her death ends the chickens: they vanish, and
  the players have won.
* **A chicken's tracked value** is its cost, or its build time where it has no cost.
* **Simplified**: no PvP, lava, eggs or endless modes.  Burrows come up on dry, empty land a kbot can walk.  The queen's extra
  health is damage divided.  A morph is a new unit of the other form with the same share of health.  A digger's ambush comes
  up above ground.
* **Cost**: about 1 ms a decision at 20 minutes, and 1.4 ms at 60 with 1,153 live units.

## The game in the sim

`SimEnv` (`tools/policy_sim_env.py`) plays games of four kinds (`kinds`, by weight; the user, 2026-11-10), each on a map whose
`modes` hold it ("Data"):

| Kind | Teams | 
|---|---|
| `1v1` | two alliances of one |
| `teams` | two alliances of 2 to 4 players, as many as the map holds a side |
| `ffa` | 3 to 8 alliances of one, as many as the map has boxes for |
| `chicken` | 1 to 4 players in one alliance against chicken defense, which the sim plays and no seat has |

A batch has room for the biggest game of its kinds (`slots`): 2 teams for 1v1s, 5 with chicken games, 8 with team games or
free-for-alls (the padding costs about 90 MB at 64 games, and no speed).  A seat is `game * slots + team`.

* **Start**: each alliance in one of the map's start boxes, drawn as Zero-K's `allshuffle` does; on a map with more boxes than
  alliances some stay empty.  An alliance of one starts at the box's start point half the time, otherwise anywhere level in
  the box.  Allies start at the box's start points, or apart in it.  A start point the commander can't stand on becomes the
  nearest place in the box with room for a base (`StartBox.at_point`), as Zero-K drops a commander at the nearest spot it
  can stand on (`start_unit_setup.lua`, `GetClosestValidSpawnSpot`): Esker Creek's, Pools of Ilys's and Acidic Quarry's
  points are on ground only spiders climb, and a commander started there never moved.  In a chicken game the players share one box and the
  burrows come up away from them.  On a chicken map that box is the first: it's the players' ("Robot Defenders", with the
  map's metal; the other is the hive's), and Zero-K, which doesn't shuffle a game of one player alliance, gives it to them.
* **Out**: an alliance with no team that has a unit keeping it in ("Teams and alliances") is out, and its seats finish at once.
  A dead team in a live alliance stops acting, and its seat finishes with its alliance.
* **End**: one alliance left; or the lead rule; or the time limit at 80 minutes (the user, 2026-11-10; it was 20).  A chicken
  game ends when the queen dies (the players win), when the players are out, or at the limit (a draw).
* **The lead rule**: from 2 minutes, an alliance at 2.5x the value of all the others together, and 10,000 metal ahead, has
  won.  Each game draws whether it has the rule (`lead_share`): PPO fades the share (03-training.md, "Reward"), tests,
  recordings and evals keep it in every game, frame collection in none, and chicken games never have it.  It rarely fires in
  a big free-for-all.
* **Every team** plays the opening script, and its constructors run the economy script.  Its fighters and factories are run
  by the army script or by the policy.  Against the scripts (`fight`) the policy plays team 0 and scripts play every other
  team, its allies too; in self-play (`mirror`) it plays every team.
* **A policy's seat**: the scripts can still run there but only advise (`SimEnv.advise`), giving labels that are never carried
  out (libzkscripts' label seats).  In a share of games the policy runs its economy as well (`policy_economy_share`).
* **Variants** (`variants=True`): each side's army script draws its knobs at the start: a preset (standard, rush, turtle,
  skirmish), then each knob jittered.  The knobs are the role weights, how much metal a squad wants over the enemy's cover
  (`commit`), what the army gathers before going, the share of raiders guarding extractors, give-up timers, and the health at
  which a fighter retreats.
* **Group orders**: the army script sends identical orders as group orders wherever that's exact (`group_orders`, on by
  default).
* **Chicken difficulty** is drawn each game (`difficulties`, `--chicken-difficulties`): Very Easy, Easy and Normal evenly by
  default.  Four scripted players won at Very Easy and Easy, 1 game in 6 at Normal and none at Hard (2026-11-10, a few games
  each); the default hasn't been set from that.

### Where a building can go

The engine's test (`CGameHelper::TestUnitBuildSquare`), in place of a circle round the building (the user, 2026-11-10).  The
sim itself builds wherever it's told, so the rule lives with whoever picks the place: `policy_opening.Ground.fits` and `site`
for the opening, the Python scripts and scenarios, `policy_sim_env.place_build` for the policy's orders, and `Ground::fits`
and `Seat::site` in the C++ economy script.  A test plays the C++ script's builds through the Python rule.

* **The footprint** is the def's `xsize` x `zsize` rectangle of 8-elmo squares (the catalog's `foot_x`, `foot_z` are its
  halves: a cloak factory 56 x 80, a solar 40 x 40, an extractor 24 x 24).  Two buildings' rectangles may touch and not
  overlap.  What's in the way is kept as rows of x, z, half x, half z.
* **Level ground**: every cell under the rectangle must be within the def's `maxHeightDif` of the height at its middle (a
  cloak factory 10.7 elmos, a solar 13, an extractor 21.3).  The engine tests its 8-elmo squares; this tests the sim map's
  16-elmo cells.  It's skipped for what floats or stands on the sea bed (the middle under water), and for a def with no limit.
* **And as before**: the rectangle on the map, on ground the def can stand on (land, or water of its depth).

### The opening (`tools/policy_opening.py`)

The same for every team, in Python:

1. The factory goes about 160 elmos towards the enemy, and starts repeating its raider.  The factory and commander are drawn per
   team.  The training maps keep all 8 land factories, drawn evenly; the pool offers what suits each map (`Ground.suits`).
2. An extractor and a solar go at each of the nearest spots that are nearer us than them, 4 to 6 by map.  Among allies a spot
   is whose start is nearest it.
3. Then the commander assists the factory.

It finishes by a median of 152 s.  A step that takes over 2 minutes is given up.  The chickens have no opening.

### The army script (`sim/scripts/src/army.cpp`)

* **Production**: each unit is the cheapest of its role on the factory's roster.  A role starts when its share lags furthest
  behind its target share:
  * raiders only to 4 minutes, falling to 35% by 60;
  * riots from 10 minutes, assaults from 6, artillery from 18;
  * skirmishers once a riot or assault has been seen, AA once air has, and snipers once a slow heavy has (after 8 minutes);
  * one scout from 2 minutes, and a second from 10 (a gunship plant has none: its Charon is a transport).
  
  Air plants have their own mix, which counts only what flies.
* **Raiders** move in squads of three.  A squad goes for extractors, then other buildings, and takes a target only when its
  metal beats the cover there.  Without a target, it probes towards the enemy and stops before defended ground.  A quarter of
  the raiders guard their own extractors.
* **The rest**: assaults lead to the start of the nearest enemy that still has units, then to the nearest building of any
  enemy, then to the nearest armed unit.  Riots, skirmishers, artillery and bombers hold behind assaults that are on the move
  (one that hasn't moved 100 elmos in 60 s doesn't count; with none moving they go for the goal themselves).  Skirmishers
  back off armed units; artillery stands at 0.75 of its range; AA follows the army; snipers hunt slow heavies; bombers go
  for defences.  Scouts' waypoints are kept on the map.
* **Several teams** (`zks_reset`'s `script_teams`): a script's enemies are the units of every team not allied to it, and its
  allies' buildings are in the way when it places its own.
* **Against chickens** (a game with a team the scripts don't play) the army guards.  It goes for chickens within 800 elmos of
  its buildings first, and hunts a burrow only while a kill is free: while the chickens' anger is at most 0, since a kill then
  brings the queen no sooner.  From minute 30 it hunts whatever the anger, to bring her on while it's strongest against them.
  Once she's come it goes for her, and plants add AA (0.35 of their mix) from 5 minutes before she's due.  Before this posture
  the scripts killed about 45 burrows in 9 minutes, met the queen at minute 8 to 10 and lost at Normal.
* **Transports** (`Seat::lifts`, the user, 2026-11-11).  A plant that can make them makes one for every two assaults or
  riots the side has that it could carry and no cheaper transport could, up to four of a kind: so a gunship plant makes
  Charons for Jacks, Knights, Reavers and Ogres, and a Hercules only for what a Charon can't lift (a Grizzly, a strider).
  * **The wing**: each free transport fetches one such unit, the dearest first, that isn't retreating and has no armed
    enemy within 600 elmos.  The loaded wait up to 45 s for those still fetching.
  * **Its target** is the enemy building worth most (cost, four times for what the aimer counts valuable) among those
    whose cover the wing's metal can take, as a raiding squad's target is taken: a high-value building, or an
    undefended one.
  * **Coming down**: a load that Drop Cargo would cost at most 15% of its health (`ScriptData.fall_loss`: a Jack 9%, a
    Knight 11%, a Dante 13%) is dropped at the last look before its transport is within 200 elmos of the target;
    any other (a Reaver 26%, an Ogre 20%) is set down 120 elmos short of it.
  * **Then** the unit attacks the target, then the nearest building within 1,000 elmos, one after another, and is the
    army's again when none is left.  The transport goes back for another.
  * With nothing the wing could take, it fetches nothing and sets down at the rally what it has.  It does nothing
    against chickens.
  * Left out: only the target's cover is weighed, not what could shoot the wing down on its straight way there.  It
    carries none of an ally's units and picks up no enemy.
  * The scripts' second factory is a gunship plant in 3 games of 10 that get one, and there's no third: so a script
    never has a strider hub and a Hercules together, and its Hercules carries a first factory's heavies.
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
* A constructor stuck for 30 s is given another job, and its spot is avoided for 3 minutes.  One sent home after fleeing is
  freed once it has sat within 300 elmos of home for 30 s.
* **Places** are tested as the game would ("Where a building can go"), from the catalog's footprints and the map's heights
  (`zks_data_desc`'s `half_x`, `half_z`, `height_dif`; `zks_map_desc.height`).
* **In long games** with the lead rule off (64 1v1s to the 80-minute limit): a median of 19.4 minutes, 7 past 40 minutes and 2
  at the limit, both slow sieges.

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
* **Air:** no crashing, fighter pull-up or vertical swarm, and no drones are made.
* **Transports:** a carried unit can't be shot (Zero-K's `targetableTransportedUnits` lets it be), a load isn't checked
  against what's in the way where it's set down, and there are no ground or sea transports.
* **Unmodelled weapons:** capture, spawned units and area impulse (`bomberassault`, `dronecarry`, `hoverminer`,
  `jumpblackhole`, `veharty`, `vehcapture`, `zenith`).  These units stand in frames without their weapon doing what it does,
  and are kept out of armies.
* **The frame:** disarm isn't shown (only paralysis), and nor are grid numbers.  A unit's `ord_age` (16-bit frames) stops
  counting at 36 minutes, and the global time feature (frame / 36,000) reaches 4.0 at 80 minutes, where it was made for 20.
* **Teams:** allies share income, overdrive and overflow only ("Economy"): units can't be given, and there's no pooled
  storage.
* **Chicken defense:** the simplifications listed under it.  A chicken counts as costing 1 in the policy's type table, so
  probes of enemy value undercount them.
