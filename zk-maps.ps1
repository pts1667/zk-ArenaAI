<#
.SYNOPSIS
Categorizes the maps in a Zero-K install (size, modes, terrain, metal, wind) into maps.json and maps.md.

.DESCRIPTION
Reads every map archive in <Zero-K>\maps with the engine's unitsync library (dimensions, start positions, heights
and water share, metal spots, wind, tidal strength), and takes the modes (1v1, teams, ffa, ...), terrain and
featured/matchmaker status from Zero-K's own map data in <Zero-K>\news\community.json, which the lobby downloads.
Maps Zero-K doesn't list get estimated categories.

  .\zk-maps.ps1                       # writes maps.json and maps.md next to this script
  .\zk-maps.ps1 -ZkDir D:\Zero-K -OutDir .\data

Takes a few minutes (every map archive is opened). Re-run it after adding maps.
#>
param(
    [Alias('ZkPath')][string]$ZkDir,
    [string]$Engine,
    [string]$OutDir
)

$ErrorActionPreference = 'Stop'
trap { Write-Host $_.Exception.Message -ForegroundColor Red; exit 1 }
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$utf8 = New-Object System.Text.UTF8Encoding($false)
if (-not $OutDir) { $OutDir = $scriptDir }

# The Zero-K folder: -ZkDir, ZK_DIR, the folder above this one, Steam, or a folder in Downloads, Desktop, Documents, C:\Games.
function Test-ZkDir([string]$dir) { $dir -and (Test-Path ([IO.Path]::Combine($dir, 'engine', 'win64'))) }
if (-not (Test-ZkDir $ZkDir)) {
    $candidates = @($env:ZK_DIR, (Split-Path $scriptDir)) + @(
        (Get-ItemProperty 'HKCU:\Software\Valve\Steam' -ErrorAction SilentlyContinue).SteamPath | Where-Object { $_ } | ForEach-Object { Join-Path $_ 'steamapps\common\Zero-K' }) +
        @((Join-Path $env:USERPROFILE 'Downloads'), [Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('MyDocuments'), 'C:\Games' |
            Where-Object { $_ -and (Test-Path $_) } | ForEach-Object { Get-ChildItem $_ -Directory -ErrorAction SilentlyContinue | ForEach-Object FullName })
    $ZkDir = $candidates | Where-Object { Test-ZkDir $_ } | Select-Object -First 1
    if (-not $ZkDir) { throw "Couldn't find your Zero-K folder; pass -ZkDir or set ZK_DIR." }
}
$ZkDir = (Resolve-Path $ZkDir).Path
$engineRoot = Join-Path $ZkDir 'engine\win64'
$engineDir = if ($Engine) { Join-Path $engineRoot $Engine } else {
    (Get-ChildItem $engineRoot -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'unitsync.dll') } |
        Sort-Object { [version]($_.Name -replace '[^\d.].*$', '') } | Select-Object -Last 1).FullName
}
if (-not $engineDir -or -not (Test-Path (Join-Path $engineDir 'unitsync.dll'))) { throw "No unitsync.dll in $engineRoot." }
if (-not [Environment]::Is64BitProcess) { throw 'Run this in 64-bit PowerShell (unitsync.dll is 64-bit).' }

# ---------------------------------------------------------------- unitsync

$env:PATH = "$engineDir;$env:PATH"   # unitsync.dll and the DLLs it needs
$env:SPRING_DATADIR = $ZkDir
$env:SPRING_WRITEDIR = $ZkDir        # reuse the install's archive cache instead of rescanning elsewhere
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class ZkUnitsync {
    const string Dll = "unitsync.dll";
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int Init([MarshalAs(UnmanagedType.I1)] bool isServer, int id);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern void UnInit();
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] static extern IntPtr GetNextError();
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int GetMapCount();
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetMapName")] static extern IntPtr GetMapNamePtr(int index);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int GetMapInfoCount(int index);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetInfoKey")] static extern IntPtr GetInfoKeyPtr(int index);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetInfoType")] static extern IntPtr GetInfoTypePtr(int index);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "GetInfoValueString")] static extern IntPtr GetInfoValueStringPtr(int index);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int GetInfoValueInteger(int index);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern float GetInfoValueFloat(int index);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] [return: MarshalAs(UnmanagedType.I1)] public static extern bool GetInfoValueBool(int index);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern float GetMapMinHeight(string mapName);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern float GetMapMaxHeight(string mapName);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] public static extern int GetInfoMapSize(string mapName, string name, out int width, out int height);
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)] static extern int GetInfoMap(string mapName, string name, byte[] data, int typeHint);

    static string Str(IntPtr p) { return p == IntPtr.Zero ? null : Marshal.PtrToStringAnsi(p); }
    public static string NextError() { return Str(GetNextError()); }
    public static string MapName(int i) { return Str(GetMapNamePtr(i)); }
    public static string InfoKey(int i) { return Str(GetInfoKeyPtr(i)); }
    public static string InfoType(int i) { return Str(GetInfoTypePtr(i)); }
    public static string InfoString(int i) { return Str(GetInfoValueStringPtr(i)); }

    // Share of the heightmap below 0 (water), from the 16-bit heightmap scaled between min and max height.
    public static double WaterShare(string map, float minHeight, float maxHeight) {
        if (minHeight >= 0) return 0;
        if (maxHeight <= 0) return 1;
        int w, h;
        if (GetInfoMapSize(map, "height", out w, out h) == 0 || w <= 0 || h <= 0) return -1;
        byte[] data = new byte[w * h * 2];
        if (GetInfoMap(map, "height", data, 2) <= 0) return -1;
        double sea = (0 - minHeight) / (maxHeight - minHeight) * 65535.0;
        long under = 0;
        for (int i = 0; i < w * h; i++) { if (data[2 * i] + (data[2 * i + 1] << 8) < sea) under++; }
        return (double)under / (w * h);
    }

    // Metal spots (4-connected blobs of metal) and the share of the metal map that has metal.
    public static double[] Metal(string map) {
        int w, h;
        if (GetInfoMapSize(map, "metal", out w, out h) == 0 || w <= 0 || h <= 0) return new double[] { -1, -1 };
        byte[] m = new byte[w * h];
        if (GetInfoMap(map, "metal", m, 1) <= 0) return new double[] { -1, -1 };
        bool[] seen = new bool[w * h];
        int[] stack = new int[w * h];
        int spots = 0; long covered = 0;
        for (int start = 0; start < w * h; start++) {
            if (m[start] == 0 || seen[start]) continue;
            int size = 0, top = 0;
            stack[top++] = start; seen[start] = true;
            while (top > 0) {
                int c = stack[--top]; size++;
                int x = c % w, y = c / w;
                if (x > 0 && m[c - 1] != 0 && !seen[c - 1]) { seen[c - 1] = true; stack[top++] = c - 1; }
                if (x < w - 1 && m[c + 1] != 0 && !seen[c + 1]) { seen[c + 1] = true; stack[top++] = c + 1; }
                if (y > 0 && m[c - w] != 0 && !seen[c - w]) { seen[c - w] = true; stack[top++] = c - w; }
                if (y < h - 1 && m[c + w] != 0 && !seen[c + w]) { seen[c + w] = true; stack[top++] = c + w; }
            }
            covered += size;
            if (size >= 2) spots++;  // single cells are noise
        }
        return new double[] { spots, (double)covered / (w * h) };
    }
}
'@

Write-Host "Reading maps with $engineDir\unitsync.dll (Zero-K folder $ZkDir)..." -ForegroundColor Cyan
if ([ZkUnitsync]::Init($false, 0) -eq 0) { throw "unitsync Init failed: $([ZkUnitsync]::NextError())" }

# Map name -> archive file in maps\, from the engine's archive cache (cache\ArchiveCache*.lua, which Init just updated).
$mapsDir = (Join-Path $ZkDir 'maps').TrimEnd('\')
$mapFiles = @{}
$cache = Get-ChildItem (Join-Path $ZkDir 'cache') -Filter 'ArchiveCache*.lua' | Sort-Object Name | Select-Object -Last 1
$file = $null; $dir = $null; $prev = ''; $inData = $false
foreach ($line in [IO.File]::ReadLines($cache.FullName)) {
    if ($line -match '^\s*path = "(.*)",$') {
        $dir = $Matches[1].Replace('/', '\').TrimEnd('\')
        $file = if ($prev -match '^\s*name = "(.*)",$') { $Matches[1] }
        $inData = $false
    } elseif ($line -match '^\s*archivedata = \{') { $inData = $true }
    elseif ($inData -and $file -and $line -match '^\s*name = "(.*)",$') {
        if ($dir -eq $mapsDir) { $mapFiles[$Matches[1]] = $file }
        $inData = $false
    }
    $prev = $line
}

$maps = [Collections.Generic.List[object]]::new()
try {
    $count = [ZkUnitsync]::GetMapCount()
    for ($i = 0; $i -lt $count; $i++) {
        $name = [ZkUnitsync]::MapName($i)
        if (-not $mapFiles.ContainsKey($name)) { continue }  # only the archives in maps\
        Write-Host "  $name" -ForegroundColor DarkGray
        try {
            $info = @{}; $starts = 0
            $n = [ZkUnitsync]::GetMapInfoCount($i)
            for ($k = 0; $k -lt $n; $k++) {
                $key = [ZkUnitsync]::InfoKey($k)
                $value = switch ([ZkUnitsync]::InfoType($k)) {
                    'integer' { [ZkUnitsync]::GetInfoValueInteger($k) }
                    'float' { [ZkUnitsync]::GetInfoValueFloat($k) }
                    'bool' { [ZkUnitsync]::GetInfoValueBool($k) }
                    default { [ZkUnitsync]::InfoString($k) }
                }
                if ($key -eq 'xPos') { $starts++ } elseif ($key) { $info[$key] = $value }
            }
            $minH = [ZkUnitsync]::GetMapMinHeight($name)
            $maxH = [ZkUnitsync]::GetMapMaxHeight($name)
            $water = [ZkUnitsync]::WaterShare($name, $minH, $maxH)
            $metal = [ZkUnitsync]::Metal($name)
            $maps.Add([pscustomobject]@{
                Name = $name; File = $mapFiles[$name]; Author = $info.author; Description = $info.description
                Width = [Math]::Round($info.width / 512, 1); Height = [Math]::Round($info.height / 512, 1)   # Zero-K map size units (512 elmos)
                StartPositions = $starts
                MinHeight = [Math]::Round($minH); MaxHeight = [Math]::Round($maxH)
                WaterShare = [Math]::Round([Math]::Max($water, 0), 3)
                MetalSpots = [int]$metal[0]; MetalCoverage = [Math]::Round([Math]::Max($metal[1], 0), 3)
                MinWind = [Math]::Round($info.minWind, 1); MaxWind = [Math]::Round($info.maxWind, 1); Tidal = [Math]::Round($info.tidalStrength, 1)
            })
        } catch { Write-Host "  couldn't read $($name): $($_.Exception.Message)" -ForegroundColor Yellow }
    }
} finally { [ZkUnitsync]::UnInit() }

# ---------------------------------------------------------------- categories

# Zero-K's own map data (what the lobby's map list shows), downloaded by the lobby into news\community.json.
$curated = @{}
$communityFile = Join-Path $ZkDir 'news\community.json'
if (Test-Path $communityFile) {
    foreach ($item in (Get-Content $communityFile -Raw | ConvertFrom-Json).MapItems) { $curated[$item.Name] = $item }
} else { Write-Host "No $communityFile (start the Zero-K lobby once to get it); categories are estimated." -ForegroundColor Yellow }

function Add-Categories($m) {
    $area = $m.Width * $m.Height
    $c = $curated[$m.Name]
    $m | Add-Member Size $(if ($area -le 144) { 'small' } elseif ($area -le 324) { 'medium' } elseif ($area -le 576) { 'large' } else { 'huge' })
    if ($c) {
        $modes = @()
        if ($c.Is1v1) { $modes += '1v1' }
        if ($c.IsTeams) { $modes += 'teams' }
        if ($c.IsFFA) { $modes += $(if ($c.FFAMaxTeams) { "ffa (up to $($c.FFAMaxTeams))" } else { 'ffa' }) }
        if ($c.IsChickens) { $modes += 'chickens' }
        if ($c.IsSpecial) { $modes += 'special' }
        $m | Add-Member Source 'zero-k'
        $m | Add-Member Modes $modes
        $m | Add-Member Hills $(switch ([int]$c.Hills) { 1 { 'flat' } 2 { 'hilly' } default { 'mountainous' } })
        $m | Add-Member Water $(switch ([int]$c.WaterLevel) { 1 { 'land' } 2 { 'mixed' } default { 'sea' } }) -Force
        $m | Add-Member Pool $(if ([int]$c.SupportLevel -ge 3) { 'matchmaker' } else { 'featured' })
        $m | Add-Member Asymmetrical ([bool]$c.IsAssymetrical)
    } else {
        # Estimates from the map file: the mode isn't knowable (maps have spare start positions), and the height
        # range only roughly matches Zero-K's flat/hilly/mountainous.
        $relief = $m.MaxHeight - [Math]::Max($m.MinHeight, 0)
        $m | Add-Member Source 'estimated'
        $m | Add-Member Modes @()
        $m | Add-Member Hills $(if ($relief -lt 300) { 'flat' } elseif ($relief -lt 700) { 'hilly' } else { 'mountainous' })
        $m | Add-Member Water $(if ($m.WaterShare -lt 0.1) { 'land' } elseif ($m.WaterShare -lt 0.6) { 'mixed' } else { 'sea' }) -Force
        $m | Add-Member Pool 'not featured'
        $m | Add-Member Asymmetrical $null
    }
    $m | Add-Member Metal $(if ($m.MetalCoverage -ge 0.25) { 'metal map' } elseif ($m.MetalSpots -le 0) { 'not in map file' } elseif ($m.MetalSpots / [Math]::Max($area, 1) -ge 0.25) { 'rich' } else { 'normal' })
    $wind = ($m.MinWind + $m.MaxWind) / 2
    $m | Add-Member Wind $(if ($wind -lt 5) { 'low' } elseif ($wind -lt 12) { 'medium' } else { 'high' })
    $m
}
$maps = @($maps | Sort-Object Name | ForEach-Object { Add-Categories $_ })

[void](New-Item $OutDir -ItemType Directory -Force)
$jsonPath = Join-Path $OutDir 'maps.json'
[IO.File]::WriteAllText($jsonPath, ($maps | ConvertTo-Json -Depth 4), $utf8)

$tick = [char]96
function Format-Names($list) {
    $items = @($list | Where-Object { $_ })
    if ($items.Count) { ($items | ForEach-Object { $tick + $_.Name + $tick }) -join ', ' } else { 'none' }
}

$md = [Collections.Generic.List[string]]::new()
$md.Add('# Maps')
$md.Add('')
$md.Add("The $($maps.Count) maps in ${tick}maps\${tick} of the Zero-K install, generated by ${tick}zk-maps.ps1${tick} on $(Get-Date -Format 'yyyy-MM-dd'); re-run it after adding maps. The same data is in [${tick}maps.json${tick}](maps.json). Use the names as ${tick}-Map${tick} values for ${tick}zk-startup.ps1${tick}.")
$md.Add('')
$md.Add("Modes, hills, water and pool come from Zero-K's own map data (${tick}news\community.json${tick}, what the lobby's map list shows) for $(@($maps | Where-Object Source -eq 'zero-k').Count) maps. For the other $(@($maps | Where-Object Source -eq 'estimated').Count) they're estimated from the map file and marked *estimated*.")
$md.Add('')
$md.Add('| Category | Meaning |')
$md.Add('| --- | --- |')
$md.Add('| Size | Zero-K map units (512 elmos): small up to 144 units² (12x12), medium up to 324 (18x18), large up to 576 (24x24), huge beyond. Smaller maps give shorter games. |')
$md.Add('| Modes | What Zero-K lists the map for: 1v1, teams, ffa, chickens, special. Unknown for maps Zero-K doesn''t list. |')
$md.Add('| Pool | matchmaker (in the ranked matchmaker''s pool), featured (in the lobby''s map list), not featured. |')
$md.Add('| Hills, water | Zero-K''s terrain type: flat, hilly or mountainous; land, mixed or sea. |')
$md.Add('| Metal | From the map''s metal map: normal, rich (at least one spot per 4 units²), metal map (metal everywhere), or not in map file (Zero-K sets the spots itself). Zero-K overrides the spots of some maps, so the count is approximate. |')
$md.Add('| Wind | Average wind speed, for wind generators: low (under 5), medium (under 12), high. Tidal strength is in the table. |')
$md.Add('')
$md.Add('## Suggested test maps')
$md.Add('')
$duel = @($maps | Where-Object { $_.Modes -contains '1v1' })
$md.Add("- **Quick 1v1 tests** (small 1v1 land maps from the matchmaker pool): $(Format-Names ($duel | Where-Object { $_.Size -eq 'small' -and $_.Water -eq 'land' -and $_.Pool -eq 'matchmaker' }))")
$md.Add("- **More small 1v1 land maps**: $(Format-Names ($duel | Where-Object { $_.Size -eq 'small' -and $_.Water -eq 'land' -and $_.Pool -ne 'matchmaker' }))")
$md.Add("- **Ranked 1v1 pool** (matchmaker 1v1 maps, what competitive play uses): $(Format-Names ($duel | Where-Object Pool -eq 'matchmaker'))")
$md.Add("- **Naval** (1v1 or teams maps with water): $(Format-Names ($maps | Where-Object { $_.Water -ne 'land' -and $_.Source -eq 'zero-k' -and ($_.Modes -contains '1v1' -or $_.Modes -contains 'teams') }))")
$md.Add("- **Asymmetrical** (sides differ: alternate sides when comparing AIs): $(Format-Names ($maps | Where-Object Asymmetrical))")
$md.Add('')
$md.Add('## By mode')
$md.Add('')
foreach ($mode in '1v1', 'teams', 'ffa', 'chickens', 'special') {
    $md.Add("- **$mode**: $(Format-Names ($maps | Where-Object { @($_.Modes | Where-Object { $_ -like "$mode*" }).Count }))")
}
$md.Add("- **unknown** (not in Zero-K's list): $(Format-Names ($maps | Where-Object Source -eq 'estimated'))")
$md.Add('')
foreach ($category in 'Size', 'Pool', 'Hills', 'Water', 'Metal', 'Wind') {
    $md.Add("## By $($category.ToLower())")
    $md.Add('')
    foreach ($g in $maps | Group-Object $category | Sort-Object Name) { $md.Add("- **$($g.Name)** ($($g.Count)): $(Format-Names $g.Group)") }
    $md.Add('')
}
$md.Add('## All maps')
$md.Add('')
$md.Add('| Map | Size | Modes | Pool | Terrain | Water share | Metal spots | Wind | Tidal | Start positions |')
$md.Add('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
foreach ($m in $maps) {
    $est = if ($m.Source -eq 'estimated') { ' *estimated*' } else { '' }
    $modes = if (@($m.Modes).Count) { $m.Modes -join ', ' } else { 'unknown' }
    $metal = if ($m.Metal -eq 'metal map') { 'metal map' } elseif ($m.Metal -eq 'not in map file') { '-' } else { $m.MetalSpots }
    $md.Add("| $($m.Name) | $($m.Width)x$($m.Height) $($m.Size) | $modes | $($m.Pool) | $($m.Hills) $($m.Water)$est$(if ($m.Asymmetrical) { ', asymmetrical' }) | $([Math]::Round($m.WaterShare * 100))% | $metal | $($m.MinWind)-$($m.MaxWind) | $($m.Tidal) | $($m.StartPositions) |")
}
[IO.File]::WriteAllText((Join-Path $OutDir 'maps.md'), ($md -join "`n") + "`n", $utf8)

Write-Host "$($maps.Count) maps: $jsonPath and $(Join-Path $OutDir 'maps.md')" -ForegroundColor Cyan
