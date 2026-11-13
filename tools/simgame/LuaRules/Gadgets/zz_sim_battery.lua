-- Plays the scripted scenarios of tools/sim_battery.py one after another and logs each unit to
-- infolog.  Runs only with the modoption sim_mode=battery.  A scenario marked `fight` (tools/sim_fights.py) also
-- logs each shot its units fire and each hit they take, and ends when either team has no one left.
function gadget:GetInfo()
	return {name = "Sim battery", desc = "Scripted scenarios to compare with the simulator", layer = 1000000, enabled = true}
end

local active = (Spring.GetModOptions() or {}).sim_mode == "battery"
local SPEED = 40
local CORNER = 150
local LEAD_IN = 300    -- frames before the first scenario: commanders exist, path costs have caught up
local GAP = 45         -- frames between scenarios, for shots still in flight

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
		if Spring.GetGameRulesParam("sim_battery_done") == 1 then
			Spring.Echo("SIMDONE")
			Spring.SendCommands("quitforce")
		end
	end
	return
end

if not active then
	return false
end

local config = VFS.Include("LuaRules/Configs/sim_scenarios.lua")
local scenarios = config.list
local index, startFrame, units, nextStart = 0, nil, nil, LEAD_IN
local levelled = false

-- Before each scenario too: the shots of the one before leave craters, a few elmos deep, and a weapon told to fire at the
-- ground (ward fire) has no line of fire to a point behind a crater's rim.
local function levelLane()
	local p = config.patch
	Spring.LevelHeightMap(p.x1, p.z1, p.x2, p.z2, p.h)
end

local function ground(x, z)
	return Spring.GetGroundHeight(x, z)
end

-- Commanders can't be removed without ending the game: out of the way, and harmless.
local function parkOthers(own)
	local width, depth = Game.mapSizeX, Game.mapSizeZ
	for _, u in ipairs(Spring.GetAllUnits()) do
		if not own[u] then
			local team = Spring.GetUnitTeam(u)
			local x = team == 0 and CORNER or width - CORNER
			local z = team == 0 and CORNER or depth - CORNER
			Spring.SetUnitPosition(u, x, z)
			Spring.SetUnitNeutral(u, true)
			Spring.GiveOrderToUnit(u, CMD.FIRE_STATE, {0}, 0)
			Spring.GiveOrderToUnit(u, CMD.MOVE_STATE, {0}, 0)
			Spring.GiveOrderToUnit(u, CMD.STOP, {}, 0)
		end
	end
end

local function begin(scenario)
	units = {byKey = {}, order = {}, ids = {}, keyOf = {}, team = {}}
	levelLane()
	local pool = scenario.pool
	if pool then
		Spring.LevelHeightMap(pool.x1, pool.z1, pool.x2, pool.z2, pool.h)
	end
	for _, spec in ipairs(scenario.units) do
		local u = Spring.CreateUnit(spec.unit, spec.x, ground(spec.x, spec.z), spec.z, spec.facing or 0, spec.team)
		if u then
			units.byKey[spec.key] = u
			units.ids[u] = true
			units.keyOf[u] = spec.key
			units.team[spec.key] = spec.team
			units.order[#units.order + 1] = spec.key
			if scenario.fight then
				for _, w in ipairs(UnitDefNames[spec.unit].weapons) do
					Script.SetWatchProjectile(w.weaponDef, true)
				end
			end
			if spec.hold then
				Spring.GiveOrderToUnit(u, CMD.MOVE_STATE, {0}, 0)
			end
			Spring.GiveOrderToUnit(u, CMD.FIRE_STATE, {spec.fire or 2}, 0)
			if spec.hp then
				Spring.SetUnitMaxHealth(u, spec.hp)
				Spring.SetUnitHealth(u, spec.hp)
			end
			if spec.stock then
				Spring.SetUnitStockpile(u, spec.stock, spec.queue or 0)
			end
			if spec.shield then
				Spring.SetUnitShieldState(u, -1, true, spec.shield)
			end
		else
			Spring.Echo("SIMERROR," .. scenario.name .. ",cannot create " .. spec.unit)
		end
	end
	parkOthers(units.ids)
end

local function finish()
	for _, u in pairs(units.byKey) do
		if Spring.ValidUnitID(u) then
			Spring.DestroyUnit(u, false, true)
		end
	end
	for _, f in ipairs(Spring.GetAllFeatures()) do
		Spring.DestroyFeature(f)
	end
end

local function give(scenario, rel)
	for _, o in ipairs(scenario.orders or {}) do
		if o.f == rel then
			local u = units.byKey[o.u]
			if u and Spring.ValidUnitID(u) then
				if o.c == "move" then
					Spring.GiveOrderToUnit(u, CMD.MOVE, {o.x, ground(o.x, o.z), o.z}, 0)
				elseif o.c == "fight" then
					Spring.GiveOrderToUnit(u, CMD.FIGHT, {o.x, ground(o.x, o.z), o.z}, 0)
				elseif o.c == "kill" then
					Spring.DestroyUnit(u, false, false)
				elseif o.c == "attack" then
					Spring.GiveOrderToUnit(u, CMD.ATTACK, {units.byKey[o.t]}, 0)
				elseif o.c == "manual" then
					Spring.GiveOrderToUnit(u, CMD.MANUALFIRE, {units.byKey[o.t]}, 0)
				end
			end
		end
	end
end

local TWO_PI_OVER_CIRCLE = 2 * math.pi / 65536

local function log(scenario, rel)
	local anyDead = false
	local living = {[0] = 0, [1] = 0}
	local frameNow = Spring.GetGameFrame()
	for _, key in ipairs(units.order) do
		local u = units.byKey[key]
		if Spring.ValidUnitID(u) and not Spring.GetUnitIsDead(u) then
			living[units.team[key]] = living[units.team[key]] + 1
			if rel % scenario.log == 0 then
				local x, _, z = Spring.GetUnitBasePosition(u)
				local _, _, _, speed = Spring.GetUnitVelocity(u)
				local hp, maxHp, paralyze = Spring.GetUnitHealth(u)
				local disarmFrame = Spring.GetUnitRulesParam(u, "disarmframe") or -1
				-- As the sim's level: 1 is the threshold, 1200 frames to fall by 1.
				local disarm = disarmFrame > frameNow and (disarmFrame - frameNow) / 1200 or 0
				local _, charge = Spring.GetUnitShieldState(u)
				Spring.Echo(string.format("SIMLOG,%d,%d,%s,%.2f,%.2f,%.4f,%.3f,%.2f,%.4f,%.4f,%.1f", index, rel, key, x, z,
					Spring.GetUnitHeading(u) * TWO_PI_OVER_CIRCLE, speed or 0, hp, paralyze / maxHp, disarm, charge or 0))
			end
		else
			anyDead = true
			if not units[key .. "_dead"] then
				units[key .. "_dead"] = true
				Spring.Echo(string.format("SIMDEAD,%d,%d,%s", index, rel, key))
			end
		end
	end
	return anyDead, living[0] == 0 or living[1] == 0
end

function gadget:ProjectileCreated(proID, ownerID)
	local scenario = scenarios[index]
	if units and scenario and scenario.fight and startFrame and units.keyOf[ownerID] then
		Spring.Echo(string.format("SIMSHOT,%d,%d,%s", index, Spring.GetGameFrame() - startFrame, units.keyOf[ownerID]))
	end
end

function gadget:UnitDamaged(unitID, unitDefID, unitTeam, damage, paralyzer, weaponDefID, projectileID, attackerID)
	local scenario = scenarios[index]
	if units and scenario and scenario.fight and startFrame and units.keyOf[unitID] then
		Spring.Echo(string.format("SIMHIT,%d,%d,%s,%.2f,%s", index, Spring.GetGameFrame() - startFrame, units.keyOf[unitID], damage,
			attackerID and units.keyOf[attackerID] or "-"))
	end
end

function gadget:GameFrame(frame)
	if not levelled then
		levelled = true
		levelLane()
		local p = config.patch
		Spring.Echo(string.format("SIMLANE,%.1f,%.1f", Spring.GetGroundHeight((p.x1 + p.x2) / 2, (p.z1 + p.z2) / 2), p.h))
	end
	if nextStart then
		if frame >= nextStart then
			nextStart = nil
			index = index + 1
			if index > #scenarios then
				Spring.SetGameRulesParam("sim_battery_done", 1)
				return
			end
			begin(scenarios[index])
			startFrame = frame
			Spring.Echo(string.format("SIMBEGIN,%d,%s,%d", index, scenarios[index].name, frame))
		end
		return
	end
	local scenario = scenarios[index]
	if not scenario then
		return
	end
	local rel = frame - startFrame
	give(scenario, rel)
	local anyDead, sideDead = log(scenario, rel)
	if rel >= scenario.duration or (anyDead and scenario.stop_on_death) or (sideDead and scenario.fight) then
		finish()
		nextStart = frame + GAP
	end
end
