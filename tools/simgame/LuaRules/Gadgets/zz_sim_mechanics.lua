-- Plays out wrecks, reclaim, upkeep, cloak, stockpiling, a solar under fire, shields and what units do on their own in a real game and logs what happens, to compare with the simulator.
-- Runs only with the modoption sim_mode=mechanics, or pierce for that part alone: tools/sim_engine.py mechanics.  Lines start SIMMECH.
function gadget:GetInfo()
	return {name = "Sim mechanics", desc = "What the simulator's mechanics come to in a real game", layer = 1000000, enabled = true}
end

local mode = (Spring.GetModOptions() or {}).sim_mode
local active = mode == "mechanics" or mode == "pierce" or mode == "pierce_slow"
local SPEED = 40
local LEAD_IN = 300    -- frames before the first case: commanders exist
local SAFE = 1500      -- elmos from any other unit

if not gadgetHandler:IsSyncedCode() then
	if not active then
		return false
	end
	local started = false
	function gadget:GameFrame()
		if not started then
			started = true
			Spring.SendCommands("setmaxspeed " .. SPEED, "setminspeed " .. SPEED)
		end
		if Spring.GetGameRulesParam("sim_mechanics_done") == 1 then
			Spring.Echo("SIMDONE")
			Spring.SendCommands("quitforce")
		end
	end
	return
end

if not active then
	return false
end

local OWN, FOE, FOE_ALLY = 0, 1, 1
local here, made, fresh, watched = nil, {}, {}, nil

local function log(...)
	local parts = {"SIMMECH"}
	for i = 1, select("#", ...) do
		local v = select(i, ...)
		parts[#parts + 1] = type(v) == "number" and string.format("%.3f", v) or tostring(v)
	end
	Spring.Echo(table.concat(parts, ","))
end

local function wait(frames)
	for _ = 1, frames do
		coroutine.yield()
	end
end

local function add(name, team, dx, dz)
	local x, z = here.x + dx, here.z + dz
	local unit = Spring.CreateUnit(name, x, Spring.GetGroundHeight(x, z), z, 0, team)
	if not unit then
		log("FAILED", name)
	end
	made[#made + 1] = unit
	return unit
end

local function clear()
	for _, unit in ipairs(made) do
		if Spring.ValidUnitID(unit) then
			Spring.DestroyUnit(unit, false, true)
		end
	end
	wait(5)
	for _, feature in ipairs(fresh) do
		if Spring.ValidFeatureID(feature) then
			Spring.DestroyFeature(feature)
		end
	end
	made, fresh = {}, {}
	wait(30)
end

function gadget:FeatureCreated(featureID)
	fresh[#fresh + 1] = featureID
end

function gadget:UnitDamaged(unitID, _, _, damage)
	if unitID == watched then
		log("hit", Spring.GetGameFrame(), damage)
	end
end

local function showFeatures(case)
	for _, feature in ipairs(fresh) do
		if Spring.ValidFeatureID(feature) then
			local metal, whole, _, _, left = Spring.GetFeatureResources(feature)
			local hp, maxHp = Spring.GetFeatureHealth(feature)
			local x, y, z = Spring.GetFeaturePosition(feature)
			log(case, FeatureDefs[Spring.GetFeatureDefID(feature)].name, "metal", metal, "of", whole, "left", left, "hp", hp, "of", maxHp, "at", x - here.x, y - Spring.GetGroundHeight(x, z), z - here.z)
		end
	end
	if #fresh == 0 then
		log(case, "nothing")
	end
end

local function killed(case, how)
	local unit = add("cloakraid", FOE, 0, 0)
	wait(2)
	how(unit)
	wait(3)
	showFeatures(case)
	clear()
end

local function wrecks()
	killed("destroyed", function(u) Spring.DestroyUnit(u, false, false) end)
	killed("big_hit", function(u) Spring.AddUnitDamage(u, 5000) end)
	killed("last_of_it", function(u)
		Spring.SetUnitHealth(u, 5)
		Spring.AddUnitDamage(u, 10)
	end)
	killed("frame_half", function(u)
		Spring.SetUnitHealth(u, {build = 0.5})
		Spring.AddUnitDamage(u, 5000)
	end)
	killed("frame_barely", function(u)
		Spring.SetUnitHealth(u, {build = 0.03})
		Spring.AddUnitDamage(u, 5000)
	end)
	-- Recent damage fades by a tenth a frame: the heap line is half of 230, so what counts is how quick the last hits were.
	for _, gap in ipairs({1, 3, 10, 30}) do
		killed("three_of_a_hundred_" .. gap .. "_frames_apart", function(u)
			for _ = 1, 3 do
				Spring.AddUnitDamage(u, 100)
				wait(gap)
			end
		end)
	end
	killed("solar_or_a_building", function(u) Spring.DestroyUnit(add("energysolar", FOE, 200, 0), false, false) end)

	-- A wreck in a blast, with time for it to land, again and again.
	Spring.DestroyUnit(add("cloakraid", FOE, 0, 0), false, false)
	wait(3)
	local first = fresh[1]
	for i = 1, 3 do
		Spring.DestroyUnit(add("energysingu", FOE, 0, 200), false, false)
		wait(5)
		showFeatures("just_after_blast_" .. i)
		wait(150)
		showFeatures("settled_after_blast_" .. i)
	end
	log("first_wreck_still_there", Spring.ValidFeatureID(first))
	clear()

	-- What's taken from a wreck is gone from its heap.
	Spring.DestroyUnit(add("tankassault", FOE, 0, 0), false, false)
	wait(3)
	local partly = fresh[1]
	local builder = add("cloakcon", OWN, 60, 0)
	Spring.GiveOrderToUnit(builder, CMD.RECLAIM, {partly + Game.maxUnits}, 0)
	wait(900)
	showFeatures("partly_reclaimed")
	Spring.DestroyUnit(builder, false, true)
	Spring.DestroyUnit(add("energysingu", FOE, 0, 200), false, false)
	wait(5)
	showFeatures("partly_reclaimed_blasted")
	clear()
end

local function reclaimBy(case, builders)
	Spring.DestroyUnit(add("tankassault", FOE, 0, 0), false, false)
	wait(3)
	local feature = fresh[1]
	Spring.SetTeamResource(OWN, "ms", 100000)
	Spring.SetTeamResource(OWN, "metal", 100)
	for i = 1, builders do
		local unit = add("cloakcon", OWN, 60, 40 * i - 60)
		Spring.GiveOrderToUnit(unit, CMD.RECLAIM, {feature + Game.maxUnits}, 0)
	end
	for _ = 1, 6 do
		wait(30)
		local stored, _, _, income = Spring.GetTeamResources(OWN, "metal")
		local metal, whole, _, _, left = Spring.GetFeatureResources(feature)
		log(case, "stored", stored, "income", income, "wreck", metal, "of", whole, "left", left)
	end
	clear()
end

local function charge(case, setup, seconds)
	local unit = setup and setup()
	wait(60)
	for _ = 1, 2 do
		Spring.SetTeamResource(OWN, "energy", 5000)
		Spring.SetTeamResource(OWN, "metal", 5000)
		local energy, metal = Spring.GetTeamResources(OWN, "energy"), Spring.GetTeamResources(OWN, "metal")
		wait(30 * seconds)
		local energyNow, metalNow = Spring.GetTeamResources(OWN, "energy"), Spring.GetTeamResources(OWN, "metal")
		log(case, "energy_a_second", (energy - energyNow) / seconds, "metal_a_second", (metal - metalNow) / seconds, unit and "cloaked" or "", unit and Spring.GetUnitIsCloaked(unit) or "")
	end
	clear()
end

local function charges()
	Spring.SetTeamResource(OWN, "es", 100000)
	Spring.SetTeamResource(OWN, "ms", 100000)
	charge("nothing", nil, 5)
	charge("radar", function() return add("staticradar", OWN, 0, 0) end, 5)
	charge("frame_radar", function()
		local unit = add("staticradar", OWN, 0, 0)
		Spring.SetUnitHealth(unit, {build = 0.5})
		return unit
	end, 5)
	charge("jammer_cloak_on", function() return add("cloakjammer", OWN, 0, 0) end, 5)
	charge("heavy_raider_still", function() return add("cloakheavyraid", OWN, 0, 0) end, 5)
	charge("heavy_raider_moving", function()
		local unit = add("cloakheavyraid", OWN, 0, 0)
		Spring.GiveOrderToUnit(unit, CMD.MOVE, {here.x + 600, Spring.GetGroundHeight(here.x + 600, here.z), here.z}, 0)
		return unit
	end, 3)
	charge("stockpile", function()
		local unit = add("turretaaheavy", OWN, 0, 0)
		for _ = 1, 20 do
			Spring.GiveOrderToUnit(unit, CMD.STOCKPILE, {}, 0)
		end
		return unit
	end, 5)
end

local function radarGoesDark()
	-- Income comes in the frame before the upkeep's due, so only a store too small to hold it starves a radar.
	Spring.SetTeamResource(OWN, "es", 0.1)
	Spring.SetTeamResource(OWN, "energy", 0)
	local radar = add("staticradar", OWN, 0, 0)
	for frame = 1, 150 do
		wait(1)
		if frame % 30 == 0 then
			log("radar_no_energy", frame, "forced_off", Spring.GetUnitRulesParam(radar, "forcedOff") or "nil", "active", Spring.GetUnitIsActive(radar), "radius", Spring.GetUnitSensorRadius(radar, "radar") or "nil")
		end
	end
	Spring.SetTeamResource(OWN, "es", 100000)
	Spring.SetTeamResource(OWN, "energy", 5000)
	wait(60)
	log("radar_paid", "forced_off", Spring.GetUnitRulesParam(radar, "forcedOff") or "nil", "active", Spring.GetUnitIsActive(radar), "radius", Spring.GetUnitSensorRadius(radar, "radar") or "nil")
	clear()
end

local function solar()
	Spring.SetTeamResource(OWN, "es", 100000)
	local panel = add("energysolar", OWN, 0, 0)
	wait(30)
	log("solar_external_ai", Spring.GetUnitRulesParam(panel, "tacticalAi_external") or "nil", "income_param", Spring.GetUnitRulesParam(panel, "selfIncomeChange") or "nil", "armored", Spring.GetUnitArmored(panel))
	local gun = add("cloakraid", FOE, 150, 0)
	Spring.GiveOrderToUnit(gun, CMD.ATTACK, {panel}, 0)
	watched = panel
	local income, armored = nil, nil
	for frame = 1, 90 + 240 do
		wait(1)
		if frame == 90 then
			Spring.DestroyUnit(gun, false, true)
			log("gun_gone", Spring.GetGameFrame())
		end
		local nowIncome, nowArmored = Spring.GetUnitRulesParam(panel, "selfIncomeChange"), Spring.GetUnitArmored(panel)
		if nowIncome ~= income or nowArmored ~= armored then
			income, armored = nowIncome, nowArmored
			log("solar", Spring.GetGameFrame(), "income_param", income or "nil", "armored", armored)
		end
	end
	watched = nil
	clear()
end

local function untilCloaked(case, unit, most)
	local start = Spring.GetGameFrame()
	while Spring.GetGameFrame() - start < most do
		if Spring.GetUnitIsCloaked(unit) then
			log(case, "frames_until_cloaked", Spring.GetGameFrame() - start)
			return
		end
		wait(1)
	end
	log(case, "still_visible_after", most)
end

local function sight(case, unit)
	local state = Spring.GetUnitLosState(unit, FOE_ALLY) or {}
	log(case, "cloaked", Spring.GetUnitIsCloaked(unit), "in_los", state.los or false, "radar", state.radar or false)
end

local function cloak()
	local hider = add("cloakheavyraid", OWN, 0, 0)
	untilCloaked("own_starts", hider, 150)
	local watcher = add("cloakraid", FOE, 300, 0)
	wait(30)
	sight("watched", hider)
	local radar = add("staticradar", FOE, 300, 100)
	wait(30)
	sight("watched_on_radar_a_second", hider)
	wait(120)
	sight("watched_on_radar_five_seconds", hider)
	Spring.DestroyUnit(radar, false, true)
	Spring.DestroyUnit(watcher, false, true)
	wait(5)

	-- Near something, with no weapon to fire at it and show itself that way.
	local bomb = add("jumpbomb", OWN, 500, 0)
	untilCloaked("unarmed_starts", bomb, 150)
	local near = add("staticradar", FOE, 620, 0)
	wait(30)
	sight("enemy_close", bomb)
	Spring.DestroyUnit(near, false, true)
	untilCloaked("enemy_gone", bomb, 300)

	Spring.AddUnitDamage(hider, 10)
	local start = Spring.GetGameFrame()
	while Spring.GetUnitIsCloaked(hider) and Spring.GetGameFrame() - start < 60 do
		wait(1)
	end
	log("hurt_shown_after", Spring.GetGameFrame() - start)
	untilCloaked("after_hurt", hider, 300)
	clear()

	local cloaker = add("cloakjammer", OWN, 0, 0)
	local friend = add("cloakraid", OWN, 60, 0)
	untilCloaked("under_area_cloak", friend, 300)
	sight("area_cloak", friend)
	local foe = add("cloakraid", FOE, 200, 0)
	Spring.GiveOrderToUnit(friend, CMD.ATTACK, {foe}, 0)
	wait(30)
	sight("firing_from_area_cloak", friend)
	Spring.DestroyUnit(foe, false, true)
	untilCloaked("after_firing_from_area_cloak", friend, 400)
	log("jammer_itself", "cloaked", Spring.GetUnitIsCloaked(cloaker))
	clear()

	local sniper = add("cloaksnipe", OWN, 0, 0)
	untilCloaked("sniper_starts", sniper, 150)
	local tank = add("tankassault", FOE, 500, 0)
	Spring.GiveOrderToUnit(sniper, CMD.ATTACK, {tank}, 0)
	for _ = 1, 300 do
		wait(1)
		local hp, whole = Spring.GetUnitHealth(tank)
		if hp < whole then
			break
		end
	end
	local hp, whole = Spring.GetUnitHealth(tank)
	log("sniper_hit", hp, "of", whole, "cloaked", Spring.GetUnitIsCloaked(sniper))
	clear()
end

local function findSafePlace()
	for i = 1, Spring.GetGameRulesParam("mex_count") do
		local x, z = Spring.GetGameRulesParam("mex_x" .. i), Spring.GetGameRulesParam("mex_z" .. i)
		local clear = true
		for _, unit in ipairs(Spring.GetAllUnits()) do
			local ux, _, uz = Spring.GetUnitPosition(unit)
			clear = clear and math.sqrt((ux - x) ^ 2 + (uz - z) ^ 2) > SAFE
		end
		if clear then
			return {x = x, z = z}
		end
	end
end

local function plentyOfEnergy()
	for _, team in ipairs({OWN, FOE}) do
		Spring.SetTeamResource(team, "es", 100000)
		Spring.SetTeamResource(team, "e", 50000)
	end
end

-- A unit that does nothing but what it's told: no fire of its own, no running about.
local function still(unit)
	Spring.GiveOrderToUnit(unit, CMD.FIRE_STATE, {0}, 0)
	Spring.GiveOrderToUnit(unit, CMD.MOVE_STATE, {0}, 0)
	Spring.SetUnitRulesParam(unit, "disable_tac_ai", 1)
end

local function charge(unit)
	return select(2, Spring.GetUnitShieldState(unit, -1)) or 0
end

local function shielded(name, team, dx, dz, amount)
	local unit = add(name, team, dx, dz)
	still(unit)
	Spring.SetUnitShieldState(unit, -1, amount)
	return unit
end

-- Every hit on a shield or the target of the case under way, with its time: where a total alone doesn't explain itself.
local tracing, traceTarget, traceStart = nil, nil, 0

function gadget:ShieldPreDamaged(proID, proOwnerID, emitter, carrier, bounce, beamEmitter, beamCarrier)
	if tracing and proID and proID ~= -1 then
		local x, y, z = Spring.GetProjectilePosition(proID)
		log(tracing, "shield_hit_frame", Spring.GetGameFrame() - traceStart, "at", (x or 0) - here.x, y or 0, (z or 0) - here.z, "carrier_y", select(2, Spring.GetUnitPosition(carrier)))
	end
	return false
end

function gadget:UnitDamaged(unit, def, team, damage, paralyzer, weaponDef, projectile, attacker)
	if tracing and unit == traceTarget then
		local x, y, z = Spring.GetUnitPosition(unit)
		log(tracing, "target_hit_frame", Spring.GetGameFrame() - traceStart, "damage", damage, "weapon", weaponDef or -1, "attacker", attacker or -1, "y", y)
	end
end

-- A shooter told to shoot a target, with a shield of the other side's somewhere about.
local function shieldShot(case, shooter, targetX, targetZ, shield, shieldX, shieldZ, amount, seconds, trace)
	plentyOfEnergy()
	local gun = add(shooter, OWN, 0, 0)
	Spring.GiveOrderToUnit(gun, CMD.MOVE_STATE, {0}, 0)
	Spring.SetUnitRulesParam(gun, "disable_tac_ai", 1)
	local target = add("staticradar", FOE, targetX, targetZ)
	Spring.SetUnitMaxHealth(target, 1000000)
	Spring.SetUnitHealth(target, 1000000)
	local bubble = shielded(shield, FOE, shieldX, shieldZ, amount)
	wait(2)
	if trace then
		tracing, traceTarget, traceStart = case, target, Spring.GetGameFrame()
		log(case, "ground", Spring.GetGroundHeight(here.x, here.z), Spring.GetGroundHeight(here.x + shieldX, here.z + shieldZ), Spring.GetGroundHeight(here.x + targetX, here.z + targetZ))
	end
	Spring.GiveOrderToUnit(gun, CMD.ATTACK, {target}, 0)
	wait(math.floor(seconds * 30))
	tracing = nil
	log(case, "target_lost", 1000000 - (Spring.GetUnitHealth(target)), "shield", charge(bubble), "of", amount)
	clear()
	wait(150)       -- for what's still in the air, which would land on the next case's target
end

local function shieldLink(case, a, b, gap, chargeA, chargeB, times)
	plentyOfEnergy()
	local ua = shielded(a, OWN, 0, 0, chargeA)
	local ub = shielded(b, OWN, gap, 0, chargeB)
	local at = 0
	for _, t in ipairs(times) do
		wait((t - at) * 30)
		at = t
		log(case, t, "a", charge(ua), "b", charge(ub))
	end
	clear()
end

local function shields()
	shieldShot("shield_through_beam", "turretheavylaser", 500, 0, "shieldcon", 300, 0, 900, 10)
	shieldShot("shield_past_beam", "turretheavylaser", 500, 250, "shieldcon", 250, -100, 900, 10)
	shieldShot("shield_through_shell", "tankassault", 340, 0, "shieldcon", 170, 0, 900, 10)
	shieldShot("shield_splash_near", "turretriot", 225, 0, "shieldcon", 300, 0, 900, 1)
	shieldShot("shield_splash_deep", "turretriot", 350, 0, "shieldcon", 300, 0, 900, 1)
	shieldShot("shield_arc_over_small", "cloakarty", 800, 0, "shieldcon", 400, 0, 900, 20, true)
	shieldShot("shield_arc_big_in_way", "cloakarty", 800, 0, "shieldshield", 400, 0, 3600, 20, true)
	shieldShot("shield_arc_into", "cloakarty", 800, 0, "shieldshield", 800, 0, 3600, 20, true)

	-- What charging costs: the energy that goes while it charges, against the same time with it full.
	plentyOfEnergy()
	local aspis = shielded("shieldshield", OWN, 0, 0, 3600)
	local before = Spring.GetTeamResources(OWN, "energy")
	wait(150)
	local idle = (Spring.GetTeamResources(OWN, "energy") - before) / 5
	Spring.SetUnitShieldState(aspis, -1, 0)
	before = Spring.GetTeamResources(OWN, "energy")
	wait(150)
	local charging = (Spring.GetTeamResources(OWN, "energy") - before) / 5
	log("shield_recharge", "charge_after_5s", charge(aspis), "energy_per_s_idle", idle, "charging", charging)
	clear()

	plentyOfEnergy()
	aspis = shielded("shieldshield", OWN, 0, 0, 100)
	for _ = 1, 150 do
		Spring.SetTeamResource(OWN, "e", 0)
		coroutine.yield()
	end
	log("shield_no_energy", "charge_after_5s", charge(aspis), "from", 100)
	clear()
	plentyOfEnergy()

	shieldLink("shield_link_same", "shieldshield", "shieldshield", 400, 3000, 0, {2, 5, 10, 20})
	shieldLink("shield_link_rank", "shieldshield", "shieldcon", 300, 3000, 0, {2, 5, 10})
	shieldLink("shield_link_unlinked", "shieldshield", "statictempshield", 400, 3000, 0, {5})
	shieldLink("shield_link_far", "shieldshield", "shieldshield", 750, 3000, 0, {5})
end

-- Where a unit goes and what it hits, a line a time: a case's fight, idle, hold position, fire state and tactical behaviour.
local function track(case, units, seconds, every)
	for t = 1, math.floor(seconds / every) do
		wait(math.floor(every * 30))
		for name, unit in pairs(units) do
			if Spring.ValidUnitID(unit) then
				local x, _, z = Spring.GetUnitPosition(unit)
				log(case, t * every, name, x - here.x, z - here.z, (Spring.GetUnitHealth(unit)), Spring.GetUnitCurrentCommand(unit) or 0)
			else
				log(case, t * every, name, "dead")
			end
		end
	end
end

local function passive(unit)
	Spring.GiveOrderToUnit(unit, CMD.FIRE_STATE, {0}, 0)
	Spring.SetUnitRulesParam(unit, "disable_tac_ai", 1)
end

-- Something of the other side that takes all it's given and does nothing, at a health that lasts.
local function dummy(name, dx, dz, health)
	local unit = add(name, FOE, dx, dz)
	passive(unit)
	Spring.GiveOrderToUnit(unit, CMD.MOVE_STATE, {0}, 0)
	if health then
		Spring.SetUnitMaxHealth(unit, health)
		Spring.SetUnitHealth(unit, health)
	end
	return unit
end

local function states(unit, moveState, fireState)
	Spring.GiveOrderToUnit(unit, CMD.MOVE_STATE, {moveState}, 0)
	Spring.GiveOrderToUnit(unit, CMD.FIRE_STATE, {fireState}, 0)
end

local function fightOff(case, moveState, targetX, targetZ, seconds)
	local raider = add("cloakraid", OWN, 0, 0)
	states(raider, moveState, 2)
	local target = dummy("staticradar", targetX, targetZ)
	wait(2)
	Spring.GiveOrderToUnit(raider, CMD.FIGHT, {here.x, Spring.GetGroundHeight(here.x, here.z + 2000), here.z + 2000}, 0)
	track(case, {raider = raider, target = target}, seconds, 1)
	clear()
end

local function idleBy(case, moveState, gap, seconds)
	local raider = add("cloakraid", OWN, 0, 0)
	states(raider, moveState, 2)
	local target = dummy("staticradar", gap, 0)
	track(case, {raider = raider, target = target}, seconds, 1)
	clear()
end

local shotCounts = nil

function gadget:ProjectileCreated(proID, proOwnerID, weaponDefID)
	if shotCounts and proOwnerID then
		shotCounts[proOwnerID] = (shotCounts[proOwnerID] or 0) + 1
	end
end

-- How many shots a few guns fire in a few seconds at a target: how many overkill prevention let through.
local function shotsFrom(case, name, n, target, seconds)
	local guns = {}
	for i = 1, n do
		guns[i] = add(name, OWN, (i - 1) * 60 - (n - 1) * 30, 0)
		Spring.SetUnitRulesParam(guns[i], "disable_tac_ai", 1)
		Spring.GiveOrderToUnit(guns[i], CMD.MOVE_STATE, {0}, 0)
		Script.SetWatchWeapon(UnitDefNames[name].weapons[1].weaponDef, true)
	end
	shotCounts = {}
	local died
	for f = 1, math.floor(seconds * 30) do
		coroutine.yield()
		if not died and not Spring.ValidUnitID(target) then
			died = f
		end
	end
	local count = 0
	for i = 1, n do
		count = count + (shotCounts[guns[i]] or 0)
	end
	shotCounts = nil
	log(case, "shots", count, "guns", n, "target_died_frame", died or -1)
	clear()
	wait(150)
end

local function radarLance(case, target, seen, shooter, gap, seconds, radarAt)
	add("staticradar", OWN, radarAt or -900, 0)
	local gun = add(shooter, OWN, 0, 0)
	states(gun, 0, 0)
	Spring.SetUnitRulesParam(gun, "disable_tac_ai", 1)
	local enemy = dummy(target, gap, 0, 100000)
	if seen then
		local scout = add("cloakraid", OWN, gap - 30, 0)
		passive(scout)
		wait(45)
		Spring.DestroyUnit(scout, false, true)
		wait(5)
	else
		wait(30)
	end
	local state = Spring.GetUnitLosState(enemy, 0, true)
	Spring.GiveOrderToUnit(gun, CMD.FIRE_STATE, {2}, 0)
	wait(math.floor(seconds * 30))
	log(case, "target_lost", 100000 - (Spring.GetUnitHealth(enemy)), "los_state", state, "reload", Spring.GetUnitWeaponState(gun, 1, "reloadState") - Spring.GetGameFrame())
	clear()
	wait(90)
end

local function stardust(case, target, gap, seconds)
	local gun = add("turretriot", OWN, 0, 0)
	local enemy = dummy(target, gap, 0, 1000000)
	local last = 1000000
	local at = 0
	for _, t in ipairs({1, 2, 5, 10, seconds}) do
		wait((t - at) * 30)
		at = t
		local hp = Spring.GetUnitHealth(enemy)
		log(case, t, "target_lost", 1000000 - hp, "heat", Spring.GetUnitRulesParam(gun, "heat_bar") or 0)
	end
	clear()
	wait(90)
end

-- Bombs.  A case ends when the bomb's gone or the time's up, a moment after so that a slow or a stun has been written.
local function bombCase(case, bomb, target, seconds, friend)
	local start = Spring.GetUnitHealth(target)
	local friendStart = friend and Spring.GetUnitHealth(friend)
	local died, distance
	for f = 1, seconds * 30 do
		coroutine.yield()
		if not Spring.ValidUnitID(bomb) or Spring.GetUnitIsDead(bomb) then
			died = f
			break
		end
		local x, _, z = Spring.GetUnitPosition(bomb)
		local tx, _, tz = Spring.GetUnitPosition(target)
		distance = math.sqrt((x - tx) ^ 2 + (z - tz) ^ 2)
	end
	wait(20)
	local hp, maxHp, para = Spring.GetUnitHealth(target)
	log(case, "died_frame", died or -1, "distance", distance or -1, "target_lost", start - (hp or 0), "para", (para or 0) / (maxHp or 1),
		"slow", Spring.GetUnitRulesParam(target, "slowState") or 0, "friend_lost", friend and friendStart - (Spring.GetUnitHealth(friend) or 0) or -1)
	showFeatures(case)
	clear()
end

local function bombs()
	local TOUGH = 100000
	local function walking(gap, toward)
		local foe = add("cloakassault", FOE, 0, gap)
		passive(foe)
		Spring.GiveOrderToUnit(foe, CMD.MOVE, {here.x, Spring.GetGroundHeight(here.x, here.z + toward), here.z + toward}, 0)
		return foe
	end
	local function fireAtWill(bomb)
		Spring.GiveOrderToUnit(bomb, CMD.FIRE_STATE, {2}, 0)
	end
	for _, name in ipairs({"amphbomb", "cloakbomb", "jumpbomb", "shieldbomb"}) do
		-- 400 is out of sight of some of them, and an order at what can't be seen lapses.
		for _, gap in ipairs({400, 200}) do
			local bomb = add(name, OWN, 0, 0)
			local target = dummy("cloakassault", 0, gap, TOUGH)
			Spring.GiveOrderToUnit(bomb, CMD.ATTACK, {target}, 0)
			bombCase("bomb_attack_" .. name .. "_" .. gap, bomb, target, 25)
		end

		local bomb = add(name, OWN, 0, 0)
		local target = dummy("cloakassault", 0, 80, TOUGH)
		wait(30)
		Spring.GiveOrderToUnit(bomb, CMD.SELFD, {}, 0)
		bombCase("bomb_selfd_" .. name, bomb, target, 3)
	end
	-- What a blast does to an ordinary target, and to a friend beside it.
	for _, name in ipairs({"amphbomb", "cloakbomb", "shieldbomb"}) do
		local bomb = add(name, OWN, 0, 0)
		local target = dummy("cloakassault", 0, 60)
		local friend = add("cloakassault", OWN, 60, 0)
		passive(friend)
		wait(30)
		Spring.GiveOrderToUnit(bomb, CMD.SELFD, {}, 0)
		bombCase("bomb_blast_" .. name, bomb, target, 3, friend)
	end
	-- Left as it starts, hold fire: nothing near it sets it off.
	do
		local bomb = add("amphbomb", OWN, 0, 0)
		bombCase("bomb_hold_approaching", bomb, walking(500, -500), 12)
	end
	-- On fire at will: what's coming towards it does, and nothing else.
	do
		local bomb = add("amphbomb", OWN, 0, 0)
		fireAtWill(bomb)
		bombCase("bomb_faw_approaching", bomb, walking(500, -500), 12)
		bomb = add("amphbomb", OWN, 0, 0)
		fireAtWill(bomb)
		bombCase("bomb_faw_leaving", bomb, walking(100, 2000), 5)
		bomb = add("amphbomb", OWN, 0, 0)
		fireAtWill(bomb)
		bombCase("bomb_faw_still", bomb, dummy("cloakassault", 0, 100, TOUGH), 8)
	end
	-- It doesn't chase on a fight order or when idle, whatever its state.
	for _, faw in ipairs({false, true}) do
		local bomb = add("amphbomb", OWN, 0, 0)
		if faw then
			fireAtWill(bomb)
		end
		local target = dummy("cloakassault", 60, 300, TOUGH)
		Spring.GiveOrderToUnit(bomb, CMD.FIGHT, {here.x, Spring.GetGroundHeight(here.x, here.z + 800), here.z + 800}, 0)
		wait(90)
		local x, _, z = Spring.GetUnitPosition(bomb)
		log(faw and "bomb_fight_faw" or "bomb_fight_hold", "at", (x or 0) - here.x, (z or 0) - here.z)
		bombCase(faw and "bomb_fight_faw_end" or "bomb_fight_hold_end", bomb, target, 5)
	end
	-- Attacking the ground.
	do
		local bomb = add("amphbomb", OWN, 0, 0)
		local target = dummy("cloakassault", 200, 300, TOUGH)
		Spring.GiveOrderToUnit(bomb, CMD.ATTACK, {here.x, Spring.GetGroundHeight(here.x, here.z + 300), here.z + 300}, 0)
		bombCase("bomb_attack_ground", bomb, target, 10)
	end
	-- Stunned, it doesn't go off.
	do
		local bomb = add("amphbomb", OWN, 0, 0)
		local target = dummy("cloakassault", 0, 60, TOUGH)
		Spring.GiveOrderToUnit(bomb, CMD.ATTACK, {target}, 0)
		Spring.SetUnitHealth(bomb, {paralyze = 100000})
		bombCase("bomb_stunned_attack", bomb, target, 4)
		bomb = add("amphbomb", OWN, 0, 0)
		target = dummy("cloakassault", 0, 60, TOUGH)
		Spring.SetUnitHealth(bomb, {paralyze = 100000})
		wait(5)
		Spring.GiveOrderToUnit(bomb, CMD.SELFD, {}, 0)
		bombCase("bomb_stunned_selfd", bomb, target, 4)
	end
end

local CMD_JUMP = Spring.Utilities.CMD.JUMP

-- Somewhere flat that a walker can cross, with room along +z and to either side, for the sim's flat map to be a fair comparison.
local function findOpenPlace()
	local LANE, SIDE, STEP, CLEAR = 1300, 450, 64, 500
	local walker = UnitDefNames["cloakassault"].id
	local units = Spring.GetAllUnits()
	local flat = 0
	for z = 64, Game.mapSizeZ - LANE - 64, 64 do
		for x = SIDE + 64, Game.mapSizeX - SIDE - 64, 64 do
			local low, high, open = math.huge, -math.huge, true
			for dz = 0, LANE, STEP do
				for dx = -SIDE, SIDE, STEP do
					local h = Spring.GetGroundHeight(x + dx, z + dz)
					low, high = math.min(low, h), math.max(high, h)
					open = open and Spring.TestMoveOrder(walker, x + dx, h, z + dz, 0, 0, 0, true, false, true)     -- terrain only
				end
			end
			if open and high - low < 40 then
				flat = flat + 1
				for _, unit in ipairs(units) do
					local ux, _, uz = Spring.GetUnitPosition(unit)
					local away = math.max(math.abs(ux - x) - SIDE, 0)
					local along = math.max(z - uz, uz - z - LANE, 0)
					open = open and math.sqrt(away ^ 2 + along ^ 2) > CLEAR
				end
				if open then
					log("open_place", x, z, "heights", low, high)
					return {x = x, z = z}
				end
			end
		end
	end
	log("FAILED", "no open place", "flat", flat, "units", #units)
end

local function jumpTo(unit, dx, dz)
	local x, z = here.x + dx, here.z + dz
	Spring.GiveOrderToUnit(unit, CMD_JUMP, {x, Spring.GetGroundHeight(x, z), z}, 0)
end

-- Follows one jump: when it left, when it came down and where, and when it could go again.  during(f) runs each frame.
local function flight(case, unit, seconds, during)
	local began, landed, ready, top, beganAt, shown = nil, nil, nil, 0, nil, false
	for f = 1, seconds * 30 do
		coroutine.yield()
		if not Spring.ValidUnitID(unit) or Spring.GetUnitIsDead(unit) then
			log(case, "dead", f)
			break
		end
		if during then
			during(f)
		end
		local x, y, z = Spring.GetUnitPosition(unit)
		local jumping = Spring.GetUnitRulesParam(unit, "is_jumping") == 1
		if jumping and not began then
			began, beganAt = f, {x - here.x, z - here.z}
		end
		if jumping then
			top = math.max(top, y - Spring.GetGroundHeight(x, z))
			shown = shown or not Spring.GetUnitIsCloaked(unit)
		end
		if began and not jumping and not landed then
			landed = f
			log(case, "began", began, "at", beganAt[1], beganAt[2], "landed", landed, "x", x - here.x, "z", z - here.z, "top", top, "shown_in_flight", shown, "order", Spring.GetUnitCurrentCommand(unit) or 0, "charge", Spring.GetUnitRulesParam(unit, "jumpReload") or -1)
		end
		if landed and not ready and (Spring.GetUnitRulesParam(unit, "jumpReload") or 0) >= 1 then
			ready = f
		end
	end
	local x, _, z = Spring.GetUnitPosition(unit)
	log(case, "ready", ready or -1, "began", began or -1, "landed", landed or -1, "end", (x or 0) - here.x, (z or 0) - here.z, "order", Spring.GetUnitCurrentCommand(unit) or 0)
end

local function jumps()
	local TOUGH = 100000
	for _, name in ipairs({"jumpraid", "jumpassault", "jumpcon", "jumpsumo", "jumpbomb", "jumpaa", "shieldscout"}) do
		local unit = add(name, OWN, 0, 0)
		passive(unit)
		wait(10)
		jumpTo(unit, 0, 300)
		flight("jump_300_" .. name, unit, 25)
		clear()
	end
	do
		local unit = add("jumpraid", OWN, 0, 0)
		passive(unit)
		wait(10)
		jumpTo(unit, 0, 100)
		flight("jump_short", unit, 3)
		clear()
		unit = add("jumpraid", OWN, 0, 0)
		passive(unit)
		wait(10)
		jumpTo(unit, 0, 700)
		flight("jump_far", unit, 12)
		clear()
	end
	-- Told to go again as soon as it has landed: it waits for its charge.
	do
		local unit = add("jumpraid", OWN, 0, 0)
		passive(unit)
		wait(10)
		jumpTo(unit, 0, 300)
		flight("jump_twice_first", unit, 3)
		jumpTo(unit, 0, 0)
		flight("jump_twice_second", unit, 14)
		clear()
	end
	-- What it does with an order given in the air, and with being stunned.
	do
		local unit = add("jumpraid", OWN, 0, 0)
		passive(unit)
		wait(10)
		jumpTo(unit, 0, 300)
		flight("jump_move_in_air", unit, 15, function(f)
			if f == 30 then
				Spring.GiveOrderToUnit(unit, CMD.MOVE, {here.x + 300, Spring.GetGroundHeight(here.x + 300, here.z + 300), here.z + 300}, 0)
			end
		end)
		clear()
		for _, order in ipairs({"FIGHT", "MOVE_FAR", "ATTACK", "SECOND_JUMP"}) do
			unit = add("jumpraid", OWN, 0, 0)
			passive(unit)
			local foe = order == "ATTACK" and dummy("cloakassault", 200, 600, TOUGH)
			wait(10)
			jumpTo(unit, 0, 300)
			flight("jump_in_air_" .. order, unit, 10, function(f)
				if f == 30 then
					if order == "FIGHT" then
						Spring.GiveOrderToUnit(unit, CMD.FIGHT, {here.x + 300, Spring.GetGroundHeight(here.x + 300, here.z + 300), here.z + 300}, 0)
					elseif order == "MOVE_FAR" then
						Spring.GiveOrderToUnit(unit, CMD.MOVE, {here.x, Spring.GetGroundHeight(here.x, here.z + 1200), here.z + 1200}, 0)
					elseif order == "ATTACK" then
						Spring.GiveOrderToUnit(unit, CMD.ATTACK, {foe}, 0)
					else
						jumpTo(unit, 200, 300)
					end
				end
			end)
			clear()
		end
		unit = add("jumpraid", OWN, 0, 0)
		passive(unit)
		wait(10)
		jumpTo(unit, 0, 300)
		flight("jump_stunned_in_air", unit, 5, function(f)
			if f == 30 then
				Spring.SetUnitHealth(unit, {paralyze = 100000})
			end
		end)
		clear()
		unit = add("jumpraid", OWN, 0, 0)
		passive(unit)
		wait(10)
		Spring.SetUnitHealth(unit, {paralyze = 100000})
		wait(5)
		jumpTo(unit, 0, 300)
		flight("jump_stunned_first", unit, 4)
		clear()
	end
	-- Jugglenaut's landing.
	do
		local unit = add("jumpsumo", OWN, 0, 0)
		passive(unit)
		local near = dummy("cloakassault", 0, 360, TOUGH)
		local far = dummy("cloakassault", 0, 420, TOUGH)
		local friend = add("cloakassault", OWN, 100, 300)
		passive(friend)
		local friendStart = Spring.GetUnitHealth(friend)
		wait(10)
		jumpTo(unit, 0, 300)
		flight("jump_landing", unit, 6)
		log("jump_landing_hurt", "at60", TOUGH - Spring.GetUnitHealth(near), "at120", TOUGH - Spring.GetUnitHealth(far), "friend_at100", friendStart - Spring.GetUnitHealth(friend), "self", select(2, Spring.GetUnitHealth(unit)) - Spring.GetUnitHealth(unit))
		clear()
	end
	-- A blast under one at the top of its arc, and the same on the ground.
	for _, up in ipairs({true, false}) do
		local unit = add("jumpassault", OWN, 0, 0)
		passive(unit)
		Spring.SetUnitMaxHealth(unit, TOUGH)
		Spring.SetUnitHealth(unit, TOUGH)
		local bomb = add("shieldbomb", FOE, 0, 150)
		if up then
			wait(10)
			jumpTo(unit, 0, 300)
		else
			Spring.SetUnitPosition(unit, here.x, here.z + 140)
			wait(10)
		end
		wait(48)
		Spring.GiveOrderToUnit(bomb, CMD.SELFD, {}, 0)
		wait(40)
		local x, y, z = Spring.GetUnitPosition(unit)
		log(up and "jump_blast_under" or "jump_blast_ground", "hurt", TOUGH - Spring.GetUnitHealth(unit), "z", z - here.z, "height", y - Spring.GetGroundHeight(x, z))
		clear()
	end
	-- Shooting from the air, and for comparison on the ground.
	for _, name in ipairs({"jumpraid", "jumpcon"}) do
		local unit = add(name, OWN, 0, 150)
		local target = dummy("cloakassault", 40, 150, TOUGH)
		wait(90)
		log("jump_shooting_on_the_ground_" .. name, "lost", TOUGH - Spring.GetUnitHealth(target))
		clear()
	end
	for _, name in ipairs({"jumpraid", "jumpassault", "jumpcon"}) do
		local unit = add(name, OWN, 0, 0)
		local target = dummy("cloakassault", 40, 150, TOUGH)
		wait(5)
		jumpTo(unit, 0, 300)
		local hurtAt, jumpHurt = nil, 0
		flight("jump_shooting_" .. name, unit, 3, function(f)
			local lost = TOUGH - Spring.GetUnitHealth(target)
			if lost > 0 and not hurtAt then
				hurtAt = f
			end
			if Spring.GetUnitRulesParam(unit, "is_jumping") == 1 then
				jumpHurt = lost
			end
		end)
		log("jump_shooting_" .. name, "first_hurt", hurtAt or -1, "lost_in_air", jumpHurt, "lost_by_end", TOUGH - Spring.GetUnitHealth(target))
		clear()
	end
	-- Skuttle: stays cloaked, and what it passes over at height doesn't set it off.
	do
		local bomb = add("jumpbomb", OWN, 0, 0)
		Spring.GiveOrderToUnit(bomb, CMD.FIRE_STATE, {2}, 0)
		local target = dummy("cloakassault", 0, 150, TOUGH)
		wait(5)
		jumpTo(bomb, 0, 300)
		flight("jump_skuttle_over", bomb, 5)
		log("jump_skuttle_over_hurt", TOUGH - Spring.GetUnitHealth(target), Spring.ValidUnitID(bomb) and "alive" or "gone")
		clear()
		bomb = add("jumpbomb", OWN, 0, 0)
		Spring.GiveOrderToUnit(bomb, CMD.FIRE_STATE, {2}, 0)
		target = dummy("cloakassault", 0, 320, TOUGH)
		wait(5)
		jumpTo(bomb, 0, 300)
		flight("jump_skuttle_onto", bomb, 5)
		log("jump_skuttle_onto_hurt", TOUGH - Spring.GetUnitHealth(target), Spring.ValidUnitID(bomb) and "alive" or "gone")
		clear()
	end
end

-- When a unit's cloak changes, and how high and fast it is when it does.
local function cloakChanges(case, unit, seconds)
	local last = "none"
	for f = 1, seconds * 30 do
		coroutine.yield()
		if not Spring.ValidUnitID(unit) or Spring.GetUnitIsDead(unit) then
			log(case, "dead", f)
			return
		end
		local cloaked = Spring.GetUnitIsCloaked(unit) and true or false
		if cloaked ~= last then
			local x, y, z = Spring.GetUnitPosition(unit)
			local _, _, _, speed = Spring.GetUnitVelocity(unit)
			log(case, "frame", f, "cloaked", cloaked, "z", z - here.z, "height", y - Spring.GetGroundHeight(x, z), "speed", speed)
			last = cloaked
		end
	end
end

local function burrows()
	for _, name in ipairs({"cloakbomb", "shieldbomb", "spiderscout", "gunshipbomb"}) do
		local unit = add(name, OWN, 0, 0)
		Spring.GiveOrderToUnit(unit, CMD.FIRE_STATE, {0}, 0)
		cloakChanges("burrow_start_" .. name, unit, 6)
		Spring.GiveOrderToUnit(unit, CMD.MOVE, {here.x, Spring.GetGroundHeight(here.x, here.z + 300), here.z + 300}, 0)
		cloakChanges("burrow_move_" .. name, unit, 8)
		local x, y, z = Spring.GetUnitPosition(unit)
		local foe = dummy("cloakassault", 200, z - here.z, 100000)
		Spring.GiveOrderToUnit(foe, CMD.MOVE, {x, Spring.GetGroundHeight(x, z), z}, 0)
		cloakChanges("burrow_foe_comes_" .. name, unit, 6)
		local moved = Spring.ValidUnitID(foe) and Spring.GetUnitPosition(foe)
		log("burrow_foe_comes_" .. name, "foe_at", moved and moved - x or -1)
		clear()
	end
end

local function blastwingCase(case, bomb, target, seconds, friend)
	local start = Spring.GetUnitHealth(target)
	local friendStart = friend and Spring.GetUnitHealth(friend)
	local died, distance, speedAt, firstHurt
	for f = 1, seconds * 30 do
		coroutine.yield()
		local gone = not Spring.ValidUnitID(bomb) or Spring.GetUnitIsDead(bomb)
		if gone and not died then
			died = f
		end
		if not gone then
			local x, _, z = Spring.GetUnitPosition(bomb)
			local tx, _, tz = Spring.GetUnitPosition(target)
			distance = math.sqrt((x - tx) ^ 2 + (z - tz) ^ 2)
			speedAt = select(4, Spring.GetUnitVelocity(bomb))
		end
		if not firstHurt and Spring.GetUnitHealth(target) < start then
			firstHurt = f
		end
		if died and f > died + 45 then
			break
		end
	end
	local x, _, z = Spring.GetUnitPosition(target)
	log(case, "died_frame", died or -1, "distance", distance or -1, "speed", speedAt or -1, "hurt_frame", firstHurt or -1, "target_lost", start - (Spring.GetUnitHealth(target) or 0), "friend_lost", friend and friendStart - (Spring.GetUnitHealth(friend) or 0) or 0, "on_fire", Spring.GetUnitRulesParam(target, "on_fire") or 0)
	clear()
end

local function blastwing()
	local TOUGH = 100000
	local function goTo(bomb, dz)
		Spring.GiveOrderToUnit(bomb, CMD.MOVE, {here.x, Spring.GetGroundHeight(here.x, here.z + dz), here.z + dz}, 0)
	end
	for _, gap in ipairs({200, 400, 800}) do
		local bomb = add("gunshipbomb", OWN, 0, 0)
		blastwingCase("blastwing_attack_" .. gap, bomb, (function()
			local target = dummy("cloakassault", 0, gap, TOUGH)
			Spring.GiveOrderToUnit(bomb, CMD.ATTACK, {target}, 0)
			return target
		end)(), 12)
	end
	-- At a walking target.
	do
		local bomb = add("gunshipbomb", OWN, 0, 0)
		local target = add("cloakassault", FOE, -300, 400)
		passive(target)
		Spring.SetUnitMaxHealth(target, TOUGH)
		Spring.SetUnitHealth(target, TOUGH)
		Spring.GiveOrderToUnit(target, CMD.MOVE, {here.x + 800, Spring.GetGroundHeight(here.x + 800, here.z + 400), here.z + 400}, 0)
		Spring.GiveOrderToUnit(bomb, CMD.ATTACK, {target}, 0)
		blastwingCase("blastwing_attack_walking", bomb, target, 12)
	end
	-- Flying past on its default fire at will, then at hold fire.
	for _, off in ipairs({60, 120, 200, 300}) do
		local bomb = add("gunshipbomb", OWN, 0, 0)
		local target = dummy("cloakassault", off, 600, TOUGH)
		goTo(bomb, 1200)
		blastwingCase("blastwing_past_" .. off, bomb, target, 8)
	end
	do
		local bomb = add("gunshipbomb", OWN, 0, 0)
		Spring.GiveOrderToUnit(bomb, CMD.FIRE_STATE, {0}, 0)
		local target = dummy("cloakassault", 60, 600, TOUGH)
		goTo(bomb, 1200)
		blastwingCase("blastwing_past_hold", bomb, target, 8)
	end
	-- Sitting on the ground: what walks up to it, and going off where it is.
	do
		local bomb = add("gunshipbomb", OWN, 0, 0)
		wait(150)
		local target = add("cloakassault", FOE, 0, 300)
		passive(target)
		Spring.SetUnitMaxHealth(target, TOUGH)
		Spring.SetUnitHealth(target, TOUGH)
		Spring.GiveOrderToUnit(target, CMD.MOVE, {here.x + 40, Spring.GetGroundHeight(here.x + 40, here.z), here.z}, 0)
		blastwingCase("blastwing_sitting", bomb, target, 10)
		bomb = add("gunshipbomb", OWN, 0, 0)
		Spring.GiveOrderToUnit(bomb, CMD.FIRE_STATE, {0}, 0)
		wait(150)
		target = dummy("cloakassault", 0, 60, TOUGH)
		local friend = add("cloakassault", OWN, 60, 0)
		passive(friend)
		Spring.GiveOrderToUnit(bomb, CMD.SELFD, {}, 0)
		blastwingCase("blastwing_selfd_sitting", bomb, target, 4, friend)
	end
	-- Killed at full speed: it's thrown ahead.
	do
		local bomb = add("gunshipbomb", OWN, 0, 0)
		Spring.GiveOrderToUnit(bomb, CMD.FIRE_STATE, {0}, 0)
		goTo(bomb, 1200)
		wait(60)
		local _, _, z = Spring.GetUnitPosition(bomb)
		local target = dummy("cloakassault", 0, z - here.z + 170, TOUGH)
		wait(1)
		log("blastwing_killed_speed", select(4, Spring.GetUnitVelocity(bomb)))
		Spring.DestroyUnit(bomb, false, false)
		blastwingCase("blastwing_killed", bomb, target, 3)
	end
end

local function unitAI()
	-- Fight: a Glaive's reach is 185, so its search is 285 on maneuver and 185 on hold position.
	fightOff("ai_fight_near", 1, 245, 600, 25)
	fightOff("ai_fight_far", 1, 345, 600, 25)
	fightOff("ai_fight_hold_near", 0, 150, 600, 25)
	fightOff("ai_fight_hold_far", 0, 245, 600, 25)

	-- The leash: what it's after runs off sideways.
	do
		local raider = add("cloakraid", OWN, 0, 0)
		local scout = dummy("vehscout", 200, 300, 100000)
		Spring.GiveOrderToUnit(scout, CMD.MOVE_STATE, {1}, 0)
		wait(2)
		Spring.GiveOrderToUnit(raider, CMD.FIGHT, {here.x, Spring.GetGroundHeight(here.x, here.z + 2000), here.z + 2000}, 0)
		wait(30)
		Spring.GiveOrderToUnit(scout, CMD.MOVE, {here.x + 2500, Spring.GetGroundHeight(here.x + 2500, here.z + 300), here.z + 300}, 0)
		track("ai_fight_leash", {raider = raider, scout = scout}, 30, 1)
		clear()
	end

	-- Idle: its search is 385 on maneuver and 185 on hold position, and it goes back to where it was.
	idleBy("ai_idle_near", 1, 335, 25)
	idleBy("ai_idle_far", 1, 435, 12)
	idleBy("ai_idle_hold", 0, 335, 12)
	-- Having arrived somewhere, it has a place to go back to, and goes.
	do
		local raider = add("cloakraid", OWN, 0, 0)
		Spring.GiveOrderToUnit(raider, CMD.MOVE, {here.x, Spring.GetGroundHeight(here.x, here.z + 100), here.z + 100}, 0)
		wait(150)
		local target = dummy("staticradar", 335, 100)
		track("ai_idle_return", {raider = raider}, 25, 1)
		clear()
	end

	-- Fire states.
	do
		local gun = add("cloakassault", OWN, 0, 0)
		states(gun, 0, 0)
		local enemy = dummy("cloakraid", 150, 0, 100000)
		track("ai_hold_fire", {gun = gun, enemy = enemy}, 8, 8)
		clear()
		gun = add("cloakassault", OWN, 0, 0)
		states(gun, 0, 1)
		enemy = dummy("cloakraid", 150, 0, 100000)
		track("ai_return_unprovoked", {gun = gun, enemy = enemy}, 8, 8)
		clear()
		gun = add("cloakassault", OWN, 0, 0)
		states(gun, 0, 1)
		enemy = add("cloakraid", FOE, 150, 0)
		states(enemy, 0, 2)
		Spring.SetUnitRulesParam(enemy, "disable_tac_ai", 1)
		Spring.SetUnitMaxHealth(enemy, 100000)
		Spring.SetUnitHealth(enemy, 100000)
		track("ai_return_fire", {gun = gun, enemy = enemy}, 8, 8)
		clear()
	end

	-- Tactical: an idle raider and what it can't fight.
	do
		local raider = add("cloakraid", OWN, 0, 0)
		Spring.GiveOrderToUnit(raider, CMD.MOVE, {here.x, Spring.GetGroundHeight(here.x, here.z + 100), here.z + 100}, 0)
		wait(150)
		local riot = dummy("cloakriot", 700, 100, 100000)
		Spring.GiveOrderToUnit(riot, CMD.MOVE_STATE, {1}, 0)
		wait(2)
		Spring.GiveOrderToUnit(riot, CMD.MOVE, {here.x - 1500, Spring.GetGroundHeight(here.x - 1500, here.z + 100), here.z + 100}, 0)
		track("ai_raider_riot", {raider = raider, riot = riot}, 25, 1)
		clear()
	end
	-- Kiting: a skirmisher on a fight order, and an assault walking at it.
	do
		local skirm = add("cloakskirm", OWN, 0, 0)
		local knight = dummy("cloakassault", 800, 0, 100000)
		Spring.GiveOrderToUnit(knight, CMD.MOVE_STATE, {1}, 0)
		wait(2)
		Spring.GiveOrderToUnit(skirm, CMD.FIGHT, {here.x + 1500, Spring.GetGroundHeight(here.x + 1500, here.z), here.z}, 0)
		Spring.GiveOrderToUnit(knight, CMD.MOVE, {here.x - 1500, Spring.GetGroundHeight(here.x - 1500, here.z), here.z}, 0)
		track("ai_skirm_kite", {skirm = skirm, knight = knight}, 25, 1)
		clear()
	end
	-- Stepping back to reload: from a turret, and from another skirmisher.
	for _, against in ipairs({"turretlaser", "jumpskirm", "cloakskirm"}) do
		local skirm = add("cloakskirm", OWN, 0, 0)
		local other = dummy(against, 700, 0, 100000)
		wait(2)
		Spring.GiveOrderToUnit(skirm, CMD.FIGHT, {here.x + 1500, Spring.GetGroundHeight(here.x + 1500, here.z), here.z}, 0)
		track("ai_skirm_reload_" .. against, {skirm = skirm}, 20, 0.5)
		clear()
	end

	-- Overkill prevention, of a shell against a unit and against a building.
	shotsFrom("ai_okp_ogres", "tankriot", 5, dummy("cloakraid", 0, 250), 4)
	shotsFrom("ai_okp_knights", "tankriot", 5, dummy("cloakassault", 0, 250), 4)
	add("staticradar", OWN, 0, -100)
	shotsFrom("ai_okp_slings", "cloakarty", 4, dummy("staticradar", 0, 700), 4)
	-- Bait: a Lance won't pick a solar for itself, and will a Glaive.  The game leaves it to the player's side to turn this on.
	local CMD_PREVENT_BAIT = Spring.Utilities.CMD.PREVENT_BAIT
	do
		local gun = add("hoverarty", OWN, 0, 0)
		Spring.GiveOrderToUnit(gun, CMD_PREVENT_BAIT, {1}, 0)
		local solar = dummy("energysolar", 500, 0, 100000)
		wait(150)
		log("ai_bait_solar", "target_lost", 100000 - (Spring.GetUnitHealth(solar)), "reload", Spring.GetUnitWeaponState(gun, 1, "reloadState") - Spring.GetGameFrame())
		clear()
		gun = add("hoverarty", OWN, 0, 0)
		Spring.GiveOrderToUnit(gun, CMD_PREVENT_BAIT, {1}, 0)
		local raider = dummy("cloakraid", 500, 0, 100000)
		wait(150)
		log("ai_bait_raider", "target_lost", 100000 - (Spring.GetUnitHealth(raider)), "reload", Spring.GetUnitWeaponState(gun, 1, "reloadState") - Spring.GetGameFrame())
		clear()
	end

	-- What can be hit on radar alone.  A Lance's shots are 23 s apart.
	radarLance("ai_radar_lance_mobile", "cloakassault", false, "hoverarty", 900, 72)
	radarLance("ai_radar_lance_unseen", "turretlaser", false, "hoverarty", 900, 72)
	radarLance("ai_radar_lance_seen", "turretlaser", true, "hoverarty", 900, 72)
	radarLance("ai_radar_sling_mobile", "cloakassault", false, "cloakarty", 850, 30, -100)
	radarLance("ai_radar_sling_seen", "turretlaser", true, "cloakarty", 850, 30, -100)

	-- The Stardust: its scatter, by range and size of target, and its heat.
	for _, gap in ipairs({150, 250, 350, 400}) do
		stardust("ai_stardust_knight_" .. gap, "cloakassault", gap, 30)
		stardust("ai_stardust_glaive_" .. gap, "cloakraid", gap, 30)
	end
end

-- A noExplode shot isn't stopped by what it hits.  What it comes to by weapon, size of target and range, and for what stands behind, beside and in front of the target.
local function pierce()
	local PIERCE_TOUGH = 1000000
	local function lost(unit)
		return PIERCE_TOUGH - Spring.GetUnitHealth(unit)
	end
	local function own(name, dx, dz)
		local unit = add(name, OWN, dx, dz)
		Spring.GiveOrderToUnit(unit, CMD.MOVE_STATE, {0}, 0)
		Spring.SetUnitRulesParam(unit, "disable_tac_ai", 1)
		return unit
	end
	local cases = {
		{"jumpraid", 150, {60, 100, 140, 180, 220}},
		{"hoverraid", 450, {60, 100, 140, 180}},
		{"turretgauss", 450, {100, 200, 300, 400, 500}},
		{"striderdante", 150, {100, 150, 200, 250, 300}},
	}
	if mode == "pierce_slow" then
		-- A disintegrator: a slow shot, so many frames inside (weapon_noexplode_speed_damage.lua).
		cases = {{"striderantiheavy", 45, {100, 180}}}
	end
	for _, c in ipairs(cases) do
		for _, targetName in ipairs({"cloakraid", "cloakassault", "tankassault", "spiderassault", "turretlaser"}) do
			for _, gap in ipairs(c[3]) do
				own(c[1], 0, 0)
				local target = dummy(targetName, 0, gap, PIERCE_TOUGH)
				wait(c[2])
				log("pierce_" .. c[1] .. "_" .. targetName .. "_" .. gap, "lost", lost(target), "frames", c[2])
				clear()
			end
		end
	end
	if mode == "pierce_slow" then
		return
	end
	for _, name in ipairs({"jumpraid", "hoverraid", "turretgauss"}) do
		local frames = name == "jumpraid" and 150 or 450
		-- Behind: both on its line, the nearer the target.
		own(name, 0, 0)
		local near, behind = dummy("cloakassault", 0, 100, PIERCE_TOUGH), dummy("cloakassault", 0, 170, PIERCE_TOUGH)
		wait(frames)
		log("pierce_behind_" .. name, "near", lost(near), "behind", lost(behind))
		clear()
		-- Beside: off the line, within a flamer's blast.
		own(name, 0, 0)
		local aimed, beside = dummy("cloakassault", 0, 120, PIERCE_TOUGH), dummy("cloakassault", 45, 120, PIERCE_TOUGH)
		wait(frames)
		log("pierce_beside_" .. name, "a", lost(aimed), "b", lost(beside))
		clear()
		-- In front: a friend on the line, with the enemy past it.
		own(name, 0, 0)
		local friend = add("cloakassault", OWN, 0, 70)
		passive(friend)
		Spring.GiveOrderToUnit(friend, CMD.MOVE_STATE, {0}, 0)
		Spring.SetUnitMaxHealth(friend, PIERCE_TOUGH)
		Spring.SetUnitHealth(friend, PIERCE_TOUGH)
		local foe = dummy("cloakassault", 0, 150, PIERCE_TOUGH)
		wait(frames)
		log("pierce_friend_" .. name, "friend", lost(friend), "foe", lost(foe))
		clear()
	end
	-- A target that walks across, which a line can miss.
	own("jumpraid", 0, 0)
	local walker = dummy("cloakraid", -220, 150, PIERCE_TOUGH)
	Spring.GiveOrderToUnit(walker, CMD.MOVE, {here.x + 220, Spring.GetGroundHeight(here.x + 220, here.z + 150), here.z + 150}, 0)
	wait(150)
	log("pierce_walking_jumpraid_cloakraid", "lost", lost(walker))
	clear()
end

local script = coroutine.create(function()
	wait(LEAD_IN)
	if mode == "pierce" or mode == "pierce_slow" then
		here = findOpenPlace() or findSafePlace()
		pierce()
		Spring.SetGameRulesParam("sim_mechanics_done", 1)
		return
	end
	here = findSafePlace()
	log("at", here.x, here.z)
	wrecks()
	reclaimBy("one_builder", 1)
	reclaimBy("two_builders", 2)
	charges()
	radarGoesDark()
	solar()
	cloak()
	shields()
	unitAI()
	bombs()
	here = findOpenPlace() or here
	jumps()
	burrows()
	blastwing()
	pierce()
	Spring.SetGameRulesParam("sim_mechanics_done", 1)
end)

function gadget:GameFrame()
	if coroutine.status(script) ~= "dead" then
		local ok, problem = coroutine.resume(script)
		if not ok then
			log("ERROR", problem)
			Spring.SetGameRulesParam("sim_mechanics_done", 1)
		end
	end
end
