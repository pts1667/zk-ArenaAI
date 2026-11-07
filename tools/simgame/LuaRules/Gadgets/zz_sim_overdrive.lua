-- Builds overdrive layouts in a real game and logs what unit_mex_overdrive.lua makes of them, to compare with the simulator.
-- Runs only with the modoption sim_mode=overdrive: tools/sim_engine.py overdrive.  Lines start SIMOD.
function gadget:GetInfo()
	return {name = "Sim overdrive", desc = "Overdrive layouts to compare with the simulator", layer = 1000000, enabled = true}
end

local active = (Spring.GetModOptions() or {}).sim_mode == "overdrive"
local SPEED = 40
local LEAD_IN = 300    -- frames before the first case: commanders exist
local SETTLE = 150     -- frames per case: a few of the gadget's updates
local GAP = 60         -- frames between cases, for removals to reach its grids
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
		if Spring.GetGameRulesParam("sim_overdrive_done") == 1 then
			Spring.Echo("SIMDONE")
			Spring.SendCommands("quitforce")
		end
	end
	return
end

if not active then
	return false
end

local TEAM = 0
local rule = Spring.GetGameRulesParam

local function spot(i)
	return {x = rule("mex_x" .. i), z = rule("mex_z" .. i)}
end

local function distance(a, b)
	return math.sqrt((a.x - b.x) ^ 2 + (a.z - b.z) ^ 2)
end

-- Out of the commanders' reach: a solar that's being shot closes, and makes nothing.
local function isSafe(s)
	for _, u in ipairs(Spring.GetAllUnits()) do
		local x, _, z = Spring.GetUnitPosition(u)
		if distance(s, {x = x, z = z}) < SAFE then
			return false
		end
	end
	return true
end

-- Two spots near enough for one pylon to join, and one far from both.
local function pickSpots()
	local safe = {}
	for i = 1, rule("mex_count") do
		if isSafe(spot(i)) then
			safe[#safe + 1] = spot(i)
		end
	end
	for _, a in ipairs(safe) do
		for _, b in ipairs(safe) do
			if distance(a, b) > 300 and distance(a, b) < 900 then
				for _, c in ipairs(safe) do
					if distance(a, c) > 1500 and distance(b, c) > 1500 then
						return a, b, c
					end
				end
			end
		end
	end
end

local function ring(out, name, x, z, radius, count)
	for k = 1, count do
		local angle = k * 2 * math.pi / count
		out[#out + 1] = {name, x + radius * math.cos(angle), z + radius * math.sin(angle)}
	end
end

local function cases()
	local a, b, c = pickSpots()
	local list = {}
	local function add(name, energy, builds)
		list[#list + 1] = {name = name, energy = energy, builds = builds}
	end
	add("pair", 500, {{"staticmex", a.x, a.z}, {"energysolar", a.x + 100, a.z}})
	add("too_far", 500, {{"staticmex", a.x, a.z}, {"energysolar", a.x + 160, a.z}})
	local five = {{"staticmex", a.x, a.z}}
	ring(five, "energysolar", a.x, a.z, 120, 5)
	add("five_full", 500, five)
	add("five_half", 250, five)
	local grids = {{"staticmex", a.x, a.z}, {"energysolar", a.x + 100, a.z}, {"staticmex", c.x, c.z}}
	ring(grids, "energysolar", c.x, c.z, 120, 5)
	add("two_grids_half", 250, grids)
	local mx, mz = (a.x + b.x) / 2, (a.z + b.z) / 2
	local joined = {{"staticmex", a.x, a.z}, {"staticmex", b.x, b.z}, {"energypylon", mx, mz}}
	ring(joined, "energysolar", mx, mz, 240, 10)
	add("pylon_full", 500, joined)
	add("pylon_half", 250, joined)
	return list
end

local list, index, startFrame, made, nextStart = nil, 0, nil, nil, LEAD_IN

local function report(case)
	local parts = {}
	for _, u in ipairs(made) do
		if Spring.GetUnitDefID(u) == UnitDefNames.staticmex.id then
			parts[#parts + 1] = string.format("%.4f:%.4f", Spring.GetUnitRulesParam(u, "current_metalIncome") or -1, Spring.GetUnitRulesParam(u, "overdrive_energyDrain") or -1)
		end
	end
	local _, _, _, energyIncome = Spring.GetTeamResources(TEAM, "energy")
	local _, _, _, metalIncome = Spring.GetTeamResources(TEAM, "metal")
	Spring.Echo(string.format("SIMOD,%s,%g,%s,%.4f,%.4f", case.name, case.energy, table.concat(parts, ";"), metalIncome, energyIncome))
end

function gadget:GameFrame(f)
	if startFrame then
		local case = list[index]
		Spring.SetTeamResource(TEAM, "energy", case.energy)
		if f == startFrame + SETTLE then
			report(case)
			for _, u in ipairs(made) do
				Spring.DestroyUnit(u, false, true)
			end
			startFrame, nextStart = nil, f + GAP
		end
	elseif f >= nextStart then
		list = list or cases()
		index = index + 1
		if not list[index] then
			Spring.SetGameRulesParam("sim_overdrive_done", 1)
			nextStart = math.huge
			return
		end
		made = {}
		for _, b in ipairs(list[index].builds) do
			local u = Spring.CreateUnit(b[1], b[2], Spring.GetGroundHeight(b[2], b[3]), b[3], 0, TEAM)
			if u then
				made[#made + 1] = u
			else
				Spring.Echo("SIMOD_FAILED," .. list[index].name .. "," .. b[1])
			end
		end
		startFrame = f
	end
end
