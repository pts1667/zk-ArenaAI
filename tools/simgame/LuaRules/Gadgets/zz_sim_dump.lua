-- Writes UnitDefs, WeaponDefs and game rules as JSON after Zero-K's post-processing,
-- which the raw unit files don't show.  Runs only with the modoption sim_mode=dump.
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

function gadget:Initialize()
	if (Spring.GetModOptions() or {}).sim_mode ~= "dump" then
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
	local GetPriority, _, _, _, _, _, velocityPenaltyDefs, _, _, _, highAlpha = VFS.Include("LuaRules/Configs/target_priority_defs.lua")
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
