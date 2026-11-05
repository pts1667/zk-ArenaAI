-- Plays the scripted scenarios of tools/sim_battery.py one after another and logs each unit to
-- infolog.  Runs only with the modoption sim_mode=battery.
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

local function levelLane()
	local p = config.patch
	Spring.LevelHeightMap(p.x1, p.z1, p.x2, p.z2, p.h)
	Spring.Echo(string.format("SIMLANE,%.1f,%.1f", Spring.GetGroundHeight((p.x1 + p.x2) / 2, (p.z1 + p.z2) / 2), p.h))
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
	units = {byKey = {}, order = {}, ids = {}}
	for _, spec in ipairs(scenario.units) do
		local u = Spring.CreateUnit(spec.unit, spec.x, ground(spec.x, spec.z), spec.z, spec.facing or 0, spec.team)
		if u then
			units.byKey[spec.key] = u
			units.ids[u] = true
			units.order[#units.order + 1] = spec.key
			if spec.hold then
				Spring.GiveOrderToUnit(u, CMD.MOVE_STATE, {0}, 0)
			end
			Spring.GiveOrderToUnit(u, CMD.FIRE_STATE, {spec.fire or 2}, 0)
			if spec.hp then
				Spring.SetUnitMaxHealth(u, spec.hp)
				Spring.SetUnitHealth(u, spec.hp)
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
				end
			end
		end
	end
end

local TWO_PI_OVER_CIRCLE = 2 * math.pi / 65536

local function log(scenario, rel)
	local anyDead = false
	for _, key in ipairs(units.order) do
		local u = units.byKey[key]
		if Spring.ValidUnitID(u) and not Spring.GetUnitIsDead(u) then
			if rel % scenario.log == 0 then
				local x, _, z = Spring.GetUnitBasePosition(u)
				local _, _, _, speed = Spring.GetUnitVelocity(u)
				local hp = Spring.GetUnitHealth(u)
				Spring.Echo(string.format("SIMLOG,%d,%d,%s,%.2f,%.2f,%.4f,%.3f,%.2f", index, rel, key, x, z,
					Spring.GetUnitHeading(u) * TWO_PI_OVER_CIRCLE, speed or 0, hp))
			end
		else
			anyDead = true
			if not units[key .. "_dead"] then
				units[key .. "_dead"] = true
				Spring.Echo(string.format("SIMDEAD,%d,%d,%s", index, rel, key))
			end
		end
	end
	return anyDead
end

function gadget:GameFrame(frame)
	if not levelled then
		levelled = true
		levelLane()
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
	local anyDead = log(scenario, rel)
	if rel >= scenario.duration or (anyDead and scenario.stop_on_death) then
		finish()
		nextStart = frame + GAP
	end
end
