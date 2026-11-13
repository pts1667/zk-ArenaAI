-- Writes UnitDefs, WeaponDefs, FeatureDefs and game rules as JSON after Zero-K's post-processing,
-- which the raw unit files don't show.  Runs only with the modoption sim_mode=dump.
-- With sim_mode=facts it writes the map's start boxes, metal spots and reclaimable features instead, as start_boxes.lua and mex_spot_finder.lua published them.
function gadget:GetInfo()
	return {name = "Sim data dump", desc = "UnitDefs and WeaponDefs to JSON", layer = 1000000, enabled = true}
end

if gadgetHandler:IsSyncedCode() then
	return
end

local function encode(v, depth, out)
	local t = type(v)
	if t == "number" then
		if v ~= v or v == math.huge or v == -math.huge then
			out[#out + 1] = "null"
		else
			out[#out + 1] = string.format("%.10g", v)
		end
	elseif t == "boolean" then
		out[#out + 1] = v and "true" or "false"
	elseif t == "string" then
		out[#out + 1] = '"' .. v:gsub('[%c"\]', function(c)
			return string.format("\u%04x", c:byte())
		end) .. '"'
	elseif t == "table" and depth < 6 then
		out[#out + 1] = "{"
		local first = true
		local iter = pairs
		for k, x in iter(v) do
			local kt = type(x)
			if kt == "number" or kt == "boolean" or kt == "string" or kt == "table" then
				if not first then
					out[#out + 1] = ","
				end
				first = false
				out[#out + 1] = '"' .. tostring(k):gsub('[%c"\]', "_") .. '":'
				encode(x, depth + 1, out)
			end
		end
		out[#out + 1] = "}"
	else
		out[#out + 1] = "null"
	end
end

local function dumpDefs(defs, f)
	f:write("{")
	local first = true
	for id, def in pairs(defs) do
		local row = {}
		for k, v in def:pairs() do
			row[k] = v
		end
		row.id = id
		local out = {}
		encode(row, 0, out)
		f:write(first and "" or ",\n", '"', tostring(def.name), '":', table.concat(out))
		first = false
	end
	f:write("}\n")
end

-- By id until one is missing: startbox_max_n is #config, which is one short of a config counted from 0.
local function dumpFacts()
	local rule = Spring.GetGameRulesParam
	local boxes = {}
	local id = 0
	while rule("startbox_n_" .. id) do
		local polygons = {}
		for i = 1, rule("startbox_n_" .. id) do
			local polygon = {}
			for j = 1, rule("startbox_polygon_" .. id .. "_" .. i) do
				polygon[j] = {rule("startbox_polygon_x_" .. id .. "_" .. i .. "_" .. j), rule("startbox_polygon_z_" .. id .. "_" .. i .. "_" .. j)}
			end
			polygons[i] = polygon
		end
		local points = {}
		for i = 1, rule("startpos_n_" .. id) or 0 do
			points[i] = {rule("startpos_x_" .. id .. "_" .. i), rule("startpos_z_" .. id .. "_" .. i)}
		end
		boxes[id + 1] = {polygons = polygons, startpoints = points}
		id = id + 1
	end
	local out = {}
	local spots = {}
	for i = 1, rule("mex_count") or 0 do
		spots[i] = {rule("mex_x" .. i), rule("mex_z" .. i), rule("mex_metal" .. i)}
	end
	local features = {}
	for _, id in ipairs(Spring.GetAllFeatures()) do
		local def = FeatureDefs[Spring.GetFeatureDefID(id)]
		if def.reclaimable and def.metal >= 1 then
			local x, _, z = Spring.GetFeaturePosition(id)
			features[#features + 1] = {x, z, def.metal, def.maxHealth, def.radius, def.id}
		end
	end
	encode({map = Game.mapName, width = Game.mapSizeX, height = Game.mapSizeZ, boxes = boxes, spots = spots, features = features}, 0, out)
	Spring.CreateDir("simdump")
	local f = io.open("simdump/map_facts.json", "w")
	f:write(table.concat(out), "\n")
	f:close()
	Spring.Echo("SIMDUMP done")
	Spring.SendCommands("quitforce")
end

-- What unit_tactical_ai.lua does by unit type, after its defaults, with its sets of unit defs as names, and as name:number where a
-- set gives each a number (wardFireTargets).  A table of parameters inside one (fightOnlyOverride) is a row of its own.
local function tacticsRow(behaviour)
	local row = {}
	for k, v in pairs(behaviour) do
		if type(v) == "table" then
			if type(next(v)) == "string" then
				row[k] = tacticsRow(v)
			else
				local names, values = {}, {}
				for id, on in pairs(v) do
					if on and type(id) == "number" and UnitDefs[id] then
						names[#names + 1] = UnitDefs[id].name
						if type(on) == "number" then
							values[#values + 1] = UnitDefs[id].name .. ":" .. on
						end
					end
				end
				row[k] = table.concat(names, " ")
				if #values > 0 then
					row[k .. "Values"] = table.concat(values, " ")
				end
			end
		elseif type(v) ~= "function" then
			row[k] = v
		end
	end
	return row
end

-- The lines of each unit's script that ask overkill prevention whether to fire: their arguments are only there.
local function overkillLines(ud)
	local source = ud.scriptName and VFS.LoadFile(ud.scriptName)
	if not source then
		return nil
	end
	local lines = {}
	for line in source:gmatch("[^\n]+") do
		if line:find("OverkillPrevention", 1, true) and not line:find("^%s*%-%-") then
			lines[#lines + 1] = line:gsub("%-%-.*$", "")
		end
	end
	return lines[1] and lines or nil
end

-- How many times each weapon refines the flight time it leads by (accurateLeading).  It's set where a unit's file mounts the weapon,
-- and neither def table shows it.  Read as text: some of the files don't load outside the game's own def loader.
local function accurateLeading()
	local found = {}
	for _, ud in pairs(UnitDefs) do
		local text = (VFS.LoadFile("units/" .. ud.name .. ".lua") or ""):gsub("%-%-[^\n]*", "")
		local mounts = text:match("weapons%s*=%s*(%b{})")
		for mount in (mounts and mounts:sub(2, -2) or ""):gmatch("%b{}") do
			local def, times = mount:match("def%s*=%s*%[%[([%w_]+)%]%]"), mount:match("accurateLeading%s*=%s*(%d+)")
			if def and times then
				found[ud.name .. "_" .. string.lower(def)] = tonumber(times)
			end
		end
	end
	return found
end

-- The weapons whose shots weapon_projectile_retarget.lua turns towards where their target will be: the engine's own lead is
-- limited to nothing for these, so that the turret points at the target.
local function gadgetLead()
	local found = {}
	local _, projectileLead = VFS.Include("LuaRules/Configs/retargetable_projectiles.lua", nil, VFS.GAME)
	for id in pairs(projectileLead or {}) do
		found[WeaponDefs[id].name] = true
	end
	return found
end

function dumpTactics(radarWobblePenalty, radarDotPenalty)
	local behaviours = VFS.Include("LuaRules/Configs/tactical_ai_defs.lua")
	local tactics, overkill, radar = {}, {}, {}
	for id, b in pairs(behaviours) do
		local row = tacticsRow(b.land or b)
		row.defaultAIState = b.defaultAIState
		if b.land and b.sea then
			-- GetUnitBehavior: the other table, for where the unit, or the ground under one that floats, is below the waterline.
			row.sea = tacticsRow(b.sea)
			row.waterline = b.waterline or 0
			row.floatWaterline = b.floatWaterline or false
		end
		tactics[UnitDefs[id].name] = row
	end
	for _, ud in pairs(UnitDefs) do
		overkill[ud.name] = overkillLines(ud)
	end
	for wid, wd in pairs(WeaponDefs) do
		if radarWobblePenalty[wid] or radarDotPenalty[wid] then
			radar[wd.name] = {wobble = radarWobblePenalty[wid] or 0, dot = radarDotPenalty[wid] or 0}
		end
	end
	local out = {}
	encode({tactics = tactics, overkill = overkill, radar = radar, accurate_leading = accurateLeading(), gadget_lead = gadgetLead()}, 0, out)
	local f = io.open("simdump/tactics.json", "w")
	f:write(table.concat(out), "\n")
	f:close()
end

-- A map's features don't exist yet when gadgets start.
local factsDone = false
function gadget:GameFrame()
	if not factsDone and (Spring.GetModOptions() or {}).sim_mode == "facts" then
		factsDone = true
		dumpFacts()
	end
end

function gadget:Initialize()
	local mode = (Spring.GetModOptions() or {}).sim_mode
	if mode == "facts" then
		return
	end
	if mode ~= "dump" then
		gadgetHandler:RemoveGadget()
		return
	end
	Spring.CreateDir("simdump")
	local f = io.open("simdump/units.json", "w")
	dumpDefs(UnitDefs, f)
	f:close()
	f = io.open("simdump/weapons.json", "w")
	dumpDefs(WeaponDefs, f)
	f:close()
	f = io.open("simdump/features.json", "w")
	dumpDefs(FeatureDefs, f)
	f:close()
	f = io.open("simdump/game.json", "w")
	local out = {}
	local rows = {}
	for k, v in pairs(Game) do
		rows[k] = v
	end
	encode(rows, 0, out)
	f:write(table.concat(out), "\n")
	f:close()
	-- Zero-K's own target table: weapon by target unit, lower is preferred.
	local GetPriority, _, _, _, _, _, velocityPenaltyDefs, radarWobblePenalty, radarDotPenalty, _, highAlpha = VFS.Include("LuaRules/Configs/target_priority_defs.lua")
	dumpTactics(radarWobblePenalty, radarDotPenalty)
	f = io.open("simdump/priority.txt", "w")
	local unitIds = {}
	for id in pairs(UnitDefs) do
		unitIds[#unitIds + 1] = id
	end
	table.sort(unitIds)
	local names = {}
	for i = 1, #unitIds do
		names[i] = UnitDefs[unitIds[i]].name
	end
	f:write("units ", table.concat(names, " "), "\n")
	for wid, wd in pairs(WeaponDefs) do
		local vel = velocityPenaltyDefs[wid]
		f:write(wd.name, " ", vel and string.format("%g %g %g", vel[1], vel[2], vel[3]) or "0 0 0", " ", tostring(highAlpha[wid] or 0))
		for i = 1, #unitIds do
			f:write(" ", string.format("%.3f", GetPriority(unitIds[i], wid) or 100))
		end
		f:write("\n")
	end
	f:close()
	Spring.Echo("SIMDUMP done")
	Spring.SendCommands("quitforce")
end
