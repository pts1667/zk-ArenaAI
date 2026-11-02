<#
.SYNOPSIS
Starts Zero-K matches between AIs (and optionally you) from a Zero-K install, for AI comparison games.

.DESCRIPTION
Each argument that isn't a -Parameter is a side (an ally team). A side is one or more members joined with '+':

  ControllerAI[@version]      ControllerAI (newest installed version unless @version); see -Client
  CircuitAI[:difficulty]      Zero-K's CircuitAI: Beginner, Novice, Easy, Normal (default), Hard, Brutal
  Player[:name]               you, playing (not with -Headless); otherwise you watch as a spectator
  <ShortName>[@version]       any other installed Skirmish AI, e.g. NullAI

  .\zk-startup.ps1 ControllerAI CircuitAI:Hard -Headless -Client 'python my_bot.py {team}' -Games 10 -AlternateSides
  .\zk-startup.ps1 ControllerAI Player -Client 'python my_bot.py {team}'
  .\zk-startup.ps1 CircuitAI:Hard+CircuitAI:Hard ControllerAI+ControllerAI -Map 'Comet Catcher Redux' -Headless -Client 'python all_teams.py'
  .\zk-startup.ps1 -List

The engine reports the game's start and end to this script over its autohost (UDP) interface, which is how it
records the winner, sets the game speed and ends headless games. Each match's start script, infolog, autohost and
client logs go to -OutDir (results\<date-time> next to this script), with results.csv and summary.txt.

ControllerAI waits for a client every frame (Synchronous Mode), so a match with ControllerAI needs one: -Client
starts it for you (placeholders: {master} {team} {ally} {game} {out}; the same values are in the environment as
CONTROLLERAI_MASTER, CONTROLLERAI_TEAM, CONTROLLERAI_ALLYTEAM, ZK_ARENA_GAME, ZK_ARENA_OUT). Give one -Client per
ControllerAI team, in the order they appear; a single -Client containing {team} runs once per ControllerAI team, and
one without {team} runs once per match (e.g. a client that drives every team it finds in /list).

The Zero-K folder (containing engine\) is -ZkDir, else ZK_DIR, else found in Steam libraries, Program Files,
%LOCALAPPDATA% or a folder in Downloads, Desktop, Documents or C:\Games (like zk-direct).

.PARAMETER Teams
The sides, as described above. Default: ControllerAI vs CircuitAI:Normal.

.PARAMETER Map
One or more maps; match i is played on map i mod (number of maps). Names are matched loosely ("titanduel" is
"TitanDuel 2.2") and missing maps are downloaded.

.PARAMETER AlternateSides
Reverse the order of the sides (so they swap start boxes) on every other round of maps, e.g. with one map: matches
1, 3, 5 as given, 2, 4, 6 swapped. Results are counted per side as given.

.PARAMETER Speed
Game speed to ask the engine for once the game starts, or 'max': as fast as the machine allows. The engine slows
down to what the CPU (or a Synchronous-Mode client) can keep up with. Default max with -Headless, else 1 (normal).
0 leaves it alone.

.PARAMETER TimeoutMinutes
Wall-clock limit per match, after which it is ended and counted as a timeout. Default 60 with -Headless, else 0 (none).
#>
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]]$Teams,
    [string[]]$Map = @('TitanDuel 2.2'),
    [string]$Game = 'zk:stable',
    [string]$Engine,
    [Alias('ZkPath')][string]$ZkDir,
    [switch]$Headless,
    [int]$Games = 1,
    [switch]$AlternateSides,
    [string[]]$Client,
    [int]$ControllerPort = 3017,
    $ControllerOptions,
    $ModOptions,
    [ValidateSet('Fixed', 'Random', 'Choose')][string]$StartPos = 'Choose',
    [string]$Speed,
    [double]$TimeoutMinutes = -1,
    [string]$Name,
    [string]$OutDir,
    [string]$WriteDir,
    [int]$HostPort = 8459,
    [switch]$DryRun,
    [switch]$NoDownload,
    [switch]$List
)

$ErrorActionPreference = 'Stop'
trap { Write-Host $_.Exception.Message -ForegroundColor Red; exit 1 }
$utf8 = New-Object System.Text.UTF8Encoding($false)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$onLinux = [bool]$IsLinux  # PowerShell 7; $IsLinux doesn't exist in Windows PowerShell
$exeSuffix = if ($onLinux) { '' } else { '.exe' }
$engineSubdir = [IO.Path]::Combine('engine', $(if ($onLinux) { 'linux64' } else { 'win64' }))
$aiDifficulties = 'Beginner', 'Novice', 'Easy', 'Normal', 'Hard', 'Brutal'
$startPosTypes = @{ Fixed = 0; Random = 1; Choose = 2 }

function Note([string]$msg, [string]$color = 'Gray') { Write-Host $msg -ForegroundColor $color }

# ---------------------------------------------------------------- content (as in zk-direct)

# [IO.Path]::Combine rather than Join-Path, which throws on a missing drive (e.g. an unplugged Steam library)
function Test-ZkDir([string]$dir) { $dir -and (Test-Path ([IO.Path]::Combine($dir, $engineSubdir))) }

function Get-SteamZkDirs {
    $steams = if ($onLinux) { '.local/share/Steam', '.steam/steam', '.var/app/com.valvesoftware.Steam/.local/share/Steam' | ForEach-Object { Join-Path $HOME $_ } }
              else { (Get-ItemProperty 'HKCU:\Software\Valve\Steam' -ErrorAction SilentlyContinue).SteamPath }
    foreach ($steam in $steams) {
        if (-not $steam -or -not (Test-Path $steam)) { continue }
        $libraries = @($steam)
        $vdf = [IO.Path]::Combine($steam, 'steamapps', 'libraryfolders.vdf')
        if (Test-Path $vdf) { $libraries += [regex]::Matches((Get-Content $vdf -Raw), '"path"\s+"([^"]+)"') | ForEach-Object { $_.Groups[1].Value -replace '\\\\', '\' } }
        $libraries | ForEach-Object { [IO.Path]::Combine($_, 'steamapps', 'common', 'Zero-K') }
    }
}

# Lazily, so later (slower) places are only searched when the earlier ones have nothing.
function Get-ZkDirCandidates {
    $env:ZK_DIR
    $scriptDir
    Split-Path $scriptDir
    Get-SteamZkDirs
    foreach ($root in ${env:ProgramFiles(x86)}, $env:ProgramFiles, $env:LOCALAPPDATA) { if ($root) { Join-Path $root 'Zero-K' } }
    $parents = if ($onLinux) { $HOME, (Join-Path $HOME 'Downloads'), (Join-Path $HOME 'Games'), (Join-Path $HOME '.local/share') }
               else { (Join-Path $env:USERPROFILE 'Downloads'), [Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('MyDocuments'), 'C:\Games' }
    foreach ($parent in $parents) {
        if ($parent -and (Test-Path $parent)) { Get-ChildItem $parent -Directory -ErrorAction SilentlyContinue | ForEach-Object FullName }
    }
}

function Find-ZkDir {
    if ($ZkDir) {
        if (-not (Test-ZkDir $ZkDir)) { throw "'$ZkDir' isn't a Zero-K folder (no $engineSubdir inside)." }
        return (Resolve-Path $ZkDir).Path
    }
    $found = Get-ZkDirCandidates | Where-Object { Test-ZkDir $_ } | Select-Object -First 1
    if (-not $found) { throw "Couldn't find your Zero-K folder (the one containing $engineSubdir); pass -ZkDir or set ZK_DIR." }
    (Resolve-Path $found).Path
}

# The name you log in to Zero-K with, from the lobby's saved settings (LuaMenu\Config\<lobby>_data.lua).
function Get-ZkLoginName {
    foreach ($file in Get-ChildItem ([IO.Path]::Combine($ZkDir, 'LuaMenu', 'Config')) -Filter '*_data.lua' -ErrorAction SilentlyContinue) {
        $m = Select-String -LiteralPath $file.FullName -Pattern '^\s*userName = "(.+)",\s*$' | Select-Object -First 1
        if ($m -and $m.Matches[0].Groups[1].Value -match '^[A-Za-z0-9_\-\[\]]{1,30}$') { return $m.Matches[0].Groups[1].Value }
    }
}

function Get-EngineExe([string]$exe, [string]$engineDir = $script:EngineDir) { Join-Path $engineDir "$exe$exeSuffix" }

function Get-EngineDir([string]$version) {
    $root = Join-Path $ZkDir $engineSubdir
    if ($version) { return Join-Path $root $version }
    $dirs = Get-ChildItem $root -Directory | Where-Object { Test-Path (Get-EngineExe 'spring' $_.FullName) }
    ($dirs | Sort-Object { [version]($_.Name -replace '[^\d.].*$', '') } | Select-Object -Last 1).FullName
}

function Get-RapidEntries {
    $file = Join-Path $ZkDir 'rapid\repos.springrts.com\zk\versions.gz'
    if (-not (Test-Path $file)) { return }
    $fs = [IO.File]::OpenRead($file)
    try {
        $gz = New-Object IO.Compression.GZipStream($fs, [IO.Compression.CompressionMode]::Decompress)
        $text = (New-Object IO.StreamReader($gz)).ReadToEnd()
    } finally { $fs.Dispose() }
    foreach ($line in $text -split "`n") {
        $f = $line.Trim() -split ','
        if ($f.Count -ge 4) { [pscustomobject]@{ Tag = $f[0]; Hash = $f[1]; Name = $f[3] } }
    }
}

# Archives the engine has scanned (cache\ArchiveCache*.lua): name, file and modtype (1 = game, 3 = map).
function Get-CachedArchives {
    $cache = Get-ChildItem (Join-Path $ZkDir 'cache') -Filter 'ArchiveCache*.lua' -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
    if (-not $cache) { return }
    $file = $null; $dir = $null; $prev = ''; $inData = $false; $type = $null
    foreach ($line in [IO.File]::ReadLines($cache.FullName)) {
        if ($line -match '^\s*path = "(.*)",$') {
            $dir = $Matches[1]
            $file = if ($prev -match '^\s*name = "(.*)",$') { $Matches[1] }
            $inData = $false; $type = $null
        } elseif ($line -match '^\s*archivedata = \{') { $inData = $true }
        elseif ($inData -and $line -match '^\s*modtype = (\d+)') { $type = [int]$Matches[1] }
        elseif ($inData -and $file -and $line -match '^\s*name = "(.*)",$') {
            [pscustomobject]@{ Name = $Matches[1]; Type = $type; File = $file; Path = [IO.Path]::Combine($dir, $file) }
            $inData = $false
        }
        $prev = $line
    }
}

function Get-InstalledMaps {
    if ($null -eq $script:mapCache) { $script:mapCache = @(Get-CachedArchives | Where-Object { $_.Type -eq 3 -and (Test-Path $_.Path) }) }
    $script:mapCache
}

# 'zk:stable' -> 'Zero-K v1.14.10.3'; full names pass through
function Resolve-GameName([string]$name) {
    $entry = Get-RapidEntries | Where-Object Tag -eq $name | Select-Object -First 1
    if ($entry) { return $entry.Name }
    $name
}

function Test-GameInstalled([string]$gameName) {
    $entry = Get-RapidEntries | Where-Object Name -eq $gameName | Select-Object -First 1
    if ($entry -and (Test-Path (Join-Path $ZkDir "packages\$($entry.Hash).sdp"))) { return $true }
    [bool](Get-CachedArchives | Where-Object { $_.Type -eq 1 -and $_.Name -eq $gameName -and (Test-Path $_.Path) })
}

function Get-NameKey([string]$s) { ($s.ToLower() -replace '\.(sd7|sdz)$', '') -replace '[^a-z0-9]', '' }

# What was typed -> the map's exact name, which the engine needs: "TitanDuel_2.2", "titanduel 2.2", "titanduel_2.2.sd7"
# and "titanduel" all mean "TitanDuel 2.2". Installed maps first, then springfiles. Throws if it isn't clear which.
function Resolve-MapName([string]$typed) {
    $typed = $typed.Trim().Trim('"')
    $key = Get-NameKey $typed
    $maps = @(Get-InstalledMaps)
    $hit = $maps | Where-Object { $_.Name -eq $typed } | Select-Object -First 1
    if (-not $hit) { $hit = $maps | Where-Object { (Get-NameKey $_.Name) -eq $key -or (Get-NameKey $_.File) -eq $key } | Select-Object -First 1 }
    if ($hit) { return $hit.Name }
    $part = @($maps | Where-Object { $_.Name.IndexOf($typed, [StringComparison]::OrdinalIgnoreCase) -ge 0 })
    if ($part.Count -eq 1) { return $part[0].Name }

    # springfiles: as typed ('_' matches any character there), then with '*' for separators, then anything containing it
    $query = $typed -replace '\.(sd7|sdz)$', ''
    $loose = $query -replace '[\s_.-]+', '*'
    $found = @()
    if (-not $NoDownload) {
        foreach ($q in $query, $loose, "*$loose*") {
            try { $found = @(Invoke-RestMethod "https://springfiles.springrts.com/json.php?nosensitive=on&category=map&springname=$([Uri]::EscapeDataString($q))" -TimeoutSec 10 | ForEach-Object { $_ }) }
            catch { Note "Couldn't search springfiles for '$typed': $($_.Exception.Message)" Yellow; break }
            if ($found.Count) { break }
        }
    }
    $exact = $found | Where-Object { (Get-NameKey $_.springname) -eq $key } | Select-Object -First 1
    if ($exact) { return $exact.springname }
    if ($found.Count -eq 1) { return $found[0].springname }
    $suggest = @(@($part | ForEach-Object Name) + @($found | ForEach-Object springname) | Select-Object -Unique -First 8)
    if ($suggest.Count) { throw "Which map did you mean by '$typed'? $($suggest -join ', ')" }
    throw "There's no map called '$typed' installed$(if (-not $NoDownload) { ' or on springfiles' }). -List shows the installed maps."
}

function Invoke-Download([string]$kind, [string]$what) {
    if ($NoDownload) { throw "The $kind '$what' isn't installed (and -NoDownload is set)." }
    Note "Downloading $kind '$what'..." Yellow
    & (Get-EngineExe 'pr-downloader') --filesystem-writepath $ZkDir "--download-$kind" $what | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Couldn't download $kind '$what'." }
    $script:mapCache = $null
}

# ---------------------------------------------------------------- AIs

function Get-InstalledAIs {
    if ($null -ne $script:aiCache) { return $script:aiCache }
    $roots = @($WriteDir, $ZkDir, $script:EngineDir) | Where-Object { $_ } | ForEach-Object { [IO.Path]::Combine($_, 'AI', 'Skirmish') } |
        Where-Object { Test-Path $_ } | Select-Object -Unique
    $script:aiCache = @(foreach ($root in $roots) {
        foreach ($ai in Get-ChildItem $root -Directory) {
            foreach ($v in Get-ChildItem $ai.FullName -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'AIInfo.lua') }) {
                [pscustomobject]@{ ShortName = $ai.Name; Version = $v.Name; Path = $v.FullName }
            }
        }
    })
    $script:aiCache
}

function Get-AIVersion([string]$shortName, [string]$version) {
    $found = @(Get-InstalledAIs | Where-Object ShortName -eq $shortName)
    if (-not $found.Count) {
        $names = Get-InstalledAIs | ForEach-Object ShortName | Where-Object { $_ -notmatch '^(\d+|Dev)?Circuit(AI)?.' } | Sort-Object -Unique
        throw "No Skirmish AI '$shortName' is installed. Installed: $($names -join ', ') (and CircuitAI:<difficulty>)."
    }
    if ($version) {
        $v = $found | Where-Object Version -eq $version | Select-Object -First 1
        if (-not $v) { throw "$shortName $version isn't installed; there is $(($found | ForEach-Object Version | Sort-Object -Unique) -join ', ')." }
        return $v.Version
    }
    # newest, by the numeric part of the version
    ($found | Sort-Object { try { [version](($_.Version -replace '[^\d.]', '') + '.0').TrimStart('.') } catch { [version]'0.0' } } | Select-Object -Last 1).Version
}

# The CircuitAI build Zero-K's lobby would pick for a difficulty: the newest version prefix (as in zk-direct).
function Get-CircuitShortName([string]$difficulty) {
    $match = Get-InstalledAIs | Where-Object ShortName -match "^\d*CircuitAI$difficulty\d*$" |
        Sort-Object { [long]('0' + ($_.ShortName -replace 'CircuitAI.*$', '')) } -Descending | Select-Object -First 1
    if ($match) { return $match.ShortName }
    'CircuitAI'
}

# One member of a side ('ControllerAI', 'CircuitAI:Hard', 'Player:Tom', 'NullAI@0.1') -> what goes in the start script.
function ConvertTo-Member([string]$spec) {
    if ($spec -notmatch '^\s*([^:@\s]+)(?::([^@]*))?(?:@(.+))?\s*$') { throw "Can't read the side member '$spec'." }
    $kind = $Matches[1]; $arg = $Matches[2]; $version = $Matches[3]
    switch -regex ($kind) {
        '^(player|human|me|you)$' {
            return [pscustomobject]@{ Kind = 'Player'; Spec = $spec; Name = $arg; ShortName = $null; Version = $null }
        }
        '^circuit(ai)?$' {
            $difficulty = if ($arg) { $aiDifficulties | Where-Object { $_ -eq $arg } | Select-Object -First 1 } else { 'Normal' }
            if (-not $difficulty) { throw "CircuitAI difficulty must be one of: $($aiDifficulties -join ', ') (in '$spec')." }
            $short = Get-CircuitShortName $difficulty
            return [pscustomobject]@{ Kind = 'CircuitAI'; Spec = $spec; Name = "Circuit$difficulty"; ShortName = $short; Version = $(if ($version) { Get-AIVersion $short $version }) }
        }
    }
    $short = Get-InstalledAIs | Where-Object { $_.ShortName -eq $kind } | Select-Object -First 1 -ExpandProperty ShortName
    if (-not $short) { $short = $kind }
    if ($arg) { throw "Only CircuitAI takes a ':' argument (in '$spec')." }
    $v = Get-AIVersion $short $version
    $k = if ($short -eq 'ControllerAI') { 'ControllerAI' } else { 'AI' }
    [pscustomobject]@{ Kind = $k; Spec = $spec; Name = $short; ShortName = $short; Version = $v }
}

# ---------------------------------------------------------------- options

# A hashtable, or 'key=value' strings (also comma-separated, for cmd.exe), -> ordered key/value pairs.
function ConvertTo-OptionTable($value, [string]$what) {
    $table = [ordered]@{}
    if ($null -eq $value) { return $table }
    if ($value -is [Collections.IDictionary]) { foreach ($k in $value.Keys) { $table[[string]$k] = [string]$value[$k] }; return $table }
    foreach ($item in @($value) | ForEach-Object { ([string]$_) -split ',' } | Where-Object { $_.Trim() }) {
        if ($item -notmatch '^\s*([^=\s]+)\s*=\s*(.*?)\s*$') { throw "$what must be key=value (got '$item')." }
        $table[$Matches[1]] = $Matches[2]
    }
    $table
}

# ---------------------------------------------------------------- start script

# $sides: arrays of members, in ally team order. Returns the script text and every member with its team numbers.
function New-StartScript($sides, [string]$mapName, $order) {
    $l = [Collections.Generic.List[string]]::new()
    $human = $null
    $members = [Collections.Generic.List[object]]::new()
    $team = 0
    for ($ally = 0; $ally -lt $order.Count; $ally++) {
        $side = $order[$ally]
        foreach ($m in $sides[$side]) {
            $members.Add([pscustomobject]@{ Side = $side; Ally = $ally; Team = $team; Member = $m })
            if ($m.Kind -eq 'Player') { $human = $members[$members.Count - 1] }
            $team++
        }
    }
    $playerName = if ($human -and $human.Member.Name) { $human.Member.Name } else { $script:PlayerName }

    $l.Add('[GAME]'); $l.Add('{')
    $l.Add("  GameType=$($script:GameName);")
    $l.Add("  MapName=$mapName;")
    $l.Add("  StartPosType=$($startPosTypes[$StartPos]);")
    $l.Add('  IsHost=1;')
    $l.Add('  HostIP=127.0.0.1;')
    $l.Add("  HostPort=$HostPort;")
    $l.Add("  MyPlayerName=$playerName;")
    $l.Add('  [PLAYER0]'); $l.Add('  {')
    $l.Add("    Name=$playerName;")
    if ($human) { $l.Add('    Spectator=0;'); $l.Add("    Team=$($human.Team);") } else { $l.Add('    Spectator=1;'); $l.Add('    Team=0;') }
    $l.Add('  }')

    $ai = 0
    $names = @{}
    foreach ($e in $members) {
        $m = $e.Member
        if ($m.Kind -ne 'Player') {
            $n = 1
            while ($names.ContainsKey("$($m.Name)$n")) { $n++ }
            $e | Add-Member AIName "$($m.Name)$n"
            $names[$e.AIName] = $true
            $l.Add("  [AI$ai]"); $l.Add('  {')
            $l.Add("    Name=$($e.AIName);")
            $l.Add("    ShortName=$($m.ShortName);")
            if ($m.Version) { $l.Add("    Version=$($m.Version);") }
            $l.Add("    Team=$($e.Team);")
            $l.Add('    Host=0;')
            if ($m.Kind -eq 'ControllerAI') {
                $l.Add('    [Options]'); $l.Add('    {')
                foreach ($o in $script:ControllerOptionTable.GetEnumerator()) { $l.Add("      $($o.Key)=$($o.Value);") }
                $l.Add('    }')
            }
            $l.Add('  }')
            $ai++
        } else { $e | Add-Member AIName $playerName }
        $l.Add("  [TEAM$($e.Team)]"); $l.Add('  {'); $l.Add('    TeamLeader=0;'); $l.Add("    AllyTeam=$($e.Ally);"); $l.Add('  }')
    }
    for ($a = 0; $a -lt $order.Count; $a++) { $l.Add("  [ALLYTEAM$a]"); $l.Add('  {'); $l.Add('    NumAllies=0;'); $l.Add('  }') }
    $l.Add('  [MODOPTIONS]'); $l.Add('  {')
    foreach ($o in $script:ModOptionTable.GetEnumerator()) { $l.Add("    $($o.Key)=$($o.Value);") }
    $l.Add('  }')
    $l.Add('}')
    [pscustomobject]@{ Text = ($l -join "`r`n"); Members = $members }
}

# The engine's settings plus the autohost address, for --config (the engine only takes AutohostPort from settings).
function New-EngineConfig([string]$path, [int]$autohostPort) {
    $lines = @()
    $base = Join-Path $WriteDir 'springsettings.cfg'
    if (Test-Path $base) { $lines = @(Get-Content $base | Where-Object { $_ -notmatch '^\s*Autohost(IP|Port)\s*=' }) }
    $lines += 'AutohostIP = 127.0.0.1', "AutohostPort = $autohostPort"
    [IO.File]::WriteAllText($path, ($lines -join "`r`n") + "`r`n", $utf8)
}

# ---------------------------------------------------------------- processes

function Start-ClientProcess([string]$command, [hashtable]$vars, [string]$logBase) {
    foreach ($k in $vars.Keys) { $command = $command.Replace("{$k}", [string]$vars[$k]) }
    $saved = @{}
    $envVars = @{
        CONTROLLERAI_MASTER = $vars.master; CONTROLLERAI_TEAM = $vars.team; CONTROLLERAI_ALLYTEAM = $vars.ally
        ZK_ARENA_GAME = $vars.game; ZK_ARENA_OUT = $vars.out; PYTHONUNBUFFERED = '1'
    }
    foreach ($k in $envVars.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, [string]$envVars[$k]) }
    try {
        $start = @{
            PassThru = $true; NoNewWindow = $true; WorkingDirectory = (Get-Location).Path
            RedirectStandardOutput = "$logBase.log"; RedirectStandardError = "$logBase.err.log"
        }
        if ($onLinux) { $start.FilePath = '/bin/sh'; $start.ArgumentList = @('-c', $command) }
        else { $start.FilePath = $env:ComSpec; $start.ArgumentList = "/d /s /c `"$command`"" }
        $p = Start-Process @start
        [pscustomobject]@{ Process = $p; Command = $command; Log = "$logBase.log"; ErrLog = "$logBase.err.log"; Reported = $false }
    } finally {
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
    }
}

# A process and everything it started (a client's cmd.exe -> python).
function Stop-ProcessTree($p) {
    if (-not $p -or $p.HasExited) { return }
    if ($onLinux) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    else {
        # taskkill writes to stderr when a child already exited; under
        # ErrorActionPreference=Stop that aborted the whole multi-game run.
        $eap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try { & taskkill.exe /T /F /PID $p.Id 2>&1 | Out-Null } catch { }
        finally { $ErrorActionPreference = $eap }
    }
}

function Get-LogTail([string]$path, [int]$lines = 8) {
    if (Test-Path $path) { (Get-Content $path -Tail $lines) -join "`n" }
}

# ---------------------------------------------------------------- one match

function Invoke-Match([int]$index, [string]$mapName, [bool]$swapped) {
    $number = $index + 1
    $gameDir = Join-Path $OutDir ('game-{0:000}' -f $number)
    [void](New-Item $gameDir -ItemType Directory -Force)
    $order = @(0..($sides.Count - 1))
    if ($swapped) { [array]::Reverse($order) }
    $built = New-StartScript $sides $mapName $order
    $scriptPath = Join-Path $gameDir 'script.txt'
    [IO.File]::WriteAllText($scriptPath, $built.Text, $utf8)

    $lineup = ($order | ForEach-Object { $sideLabels[$_] }) -join '  vs  '
    Note ''
    Note "Match $number/$Games on $mapName$(if ($swapped) { ' (sides swapped)' }): $lineup" Cyan

    $result = [pscustomobject]@{
        Game = $number; Map = $mapName; Swapped = $swapped; Outcome = 'no result'; WinnerSides = @(); Seconds = 0
        Demo = ''; Dir = $gameDir; Note = ''
    }

    $udp = New-Object Net.Sockets.UdpClient (New-Object Net.IPEndPoint ([Net.IPAddress]::Loopback, 0))
    $autohostPort = $udp.Client.LocalEndPoint.Port
    $configPath = Join-Path $gameDir 'springsettings.cfg'
    New-EngineConfig $configPath $autohostPort
    $exe = Get-EngineExe $(if ($Headless) { 'spring-headless' } else { 'spring' })
    $arguments = @('--write-dir', "`"$WriteDir`"", '--config', "`"$configPath`"", "`"$scriptPath`"")

    # Clients for the ControllerAI teams, in the order the sides were given (so they stay with their side when swapped).
    $controllers = @($built.Members | Where-Object { $_.Member.Kind -eq 'ControllerAI' } | Sort-Object Side, Team)
    $clientPlan = @()
    if ($Client.Count -eq 1 -and $Client[0] -notmatch '\{team\}') { $clientPlan = @([pscustomobject]@{ Command = $Client[0]; Target = $null }) }
    elseif ($Client.Count -eq 1) { $clientPlan = @($controllers | ForEach-Object { [pscustomobject]@{ Command = $Client[0]; Target = $_ } }) }
    elseif ($Client.Count) { for ($i = 0; $i -lt $controllers.Count; $i++) { $clientPlan += [pscustomobject]@{ Command = $Client[$i]; Target = $controllers[$i] } } }

    if ($DryRun) {
        Note "  $exe $($arguments -join ' ')" DarkGray
        foreach ($c in $clientPlan) {
            $shown = $c.Command
            $vars = @{ master = $masterUrl; team = $(if ($c.Target) { $c.Target.Team }); ally = $(if ($c.Target) { $c.Target.Ally }); game = $number; out = $gameDir }
            foreach ($k in $vars.Keys) { $shown = $shown.Replace("{$k}", [string]$vars[$k]) }
            Note "  client$(if ($c.Target) { " for team $($c.Target.Team)" }): $shown" DarkGray
        }
        $udp.Dispose()
        $result.Outcome = 'dry run'
        return $result
    }

    $savedDataDir = $env:SPRING_DATADIR
    if ($WriteDir -ne $ZkDir) { $env:SPRING_DATADIR = $ZkDir }  # games, maps and AIs from the install
    $engine = $null
    $clients = @()
    $ahLog = [IO.StreamWriter]::new((Join-Path $gameDir 'autohost.log'), $false, $utf8)
    $ahLog.AutoFlush = $true
    $started = Get-Date
    try {
        $start = @{ FilePath = $exe; ArgumentList = $arguments; WorkingDirectory = $WriteDir; PassThru = $true }
        if ($Headless) { $start.NoNewWindow = $true; $start.RedirectStandardOutput = Join-Path $gameDir 'engine.log'; $start.RedirectStandardError = Join-Path $gameDir 'engine.err.log' }
        $engine = Start-Process @start
        Note "  engine pid $($engine.Id); logs in $gameDir" DarkGray

        foreach ($c in $clientPlan) {
            $t = $c.Target
            $vars = @{ master = $masterUrl; team = $(if ($t) { $t.Team } else { '' }); ally = $(if ($t) { $t.Ally } else { '' }); game = $number; out = $gameDir }
            $logBase = Join-Path $gameDir $(if ($t) { "client-team$($t.Team)" } else { 'client' })
            $cp = Start-ClientProcess $c.Command $vars $logBase
            Note "  client$(if ($t) { " for team $($t.Team)" }): $($cp.Command)" DarkGray
            $clients += $cp
        }

        $remote = New-Object Net.IPEndPoint ([Net.IPAddress]::Loopback, 0)
        $engineEndpoint = $null
        $gameOverAt = $null; $killSentAt = $null; $playing = $false; $clientFailedAt = $null
        $send = {
            param([string]$text)
            if (-not $engineEndpoint) { return }
            $bytes = $utf8.GetBytes($text)
            try { [void]$udp.Send($bytes, $bytes.Length, $engineEndpoint) } catch {}
            $ahLog.WriteLine("$(Get-Date -Format HH:mm:ss) > $text")
            Start-Sleep -Milliseconds 100  # the server reads one message per tick
        }

        while (-not $engine.HasExited) {
            while ($udp.Available -gt 0) {
                try { $b = $udp.Receive([ref]$remote) } catch { break }
                $engineEndpoint = New-Object Net.IPEndPoint ($remote.Address, $remote.Port)
                $text = if ($b.Length -gt 1) { $utf8.GetString($b, 1, $b.Length - 1) } else { '' }
                switch ([int]$b[0]) {
                    0 { $ahLog.WriteLine("$(Get-Date -Format HH:mm:ss) server started") }
                    2 {
                        $playing = $true
                        if ($b.Length -gt 21) { $result.Demo = $utf8.GetString($b, 21, $b.Length - 21).TrimEnd([char]0) }
                        $ahLog.WriteLine("$(Get-Date -Format HH:mm:ss) game started; demo: $($result.Demo)")
                        Note "  game started$(if ($result.Demo) { "; demo $($result.Demo)" })" DarkGray
                        if ($script:SpeedValue -gt 0 -and $script:SpeedValue -ne 1) {
                            & $send "/setmaxspeed $($script:SpeedValue)"
                            & $send "/setminspeed $($script:SpeedValue)"
                        }
                    }
                    3 {
                        if (-not $gameOverAt) {
                            $allies = if ($b.Length -gt 3) { @($b[3..($b.Length - 1)] | ForEach-Object { [int]$_ }) } else { @() }
                            $result.WinnerSides = @($allies | Where-Object { $_ -lt $order.Count } | ForEach-Object { $order[$_] })
                            $result.Outcome = if ($result.WinnerSides.Count) { 'win' } else { 'draw' }
                            $gameOverAt = Get-Date
                            $ahLog.WriteLine("$(Get-Date -Format HH:mm:ss) game over; winning ally teams: $($allies -join ',')")
                            $who = if ($result.WinnerSides.Count) { ($result.WinnerSides | ForEach-Object { $sideLabels[$_] }) -join ', ' } else { 'nobody (draw)' }
                            Note "  game over: $who won" Green
                        }
                    }
                    4 { $ahLog.WriteLine("$(Get-Date -Format HH:mm:ss) message: $text") }
                    5 { $ahLog.WriteLine("$(Get-Date -Format HH:mm:ss) warning: $text") }
                    13 { if ($b.Length -gt 3) { $ahLog.WriteLine("$(Get-Date -Format HH:mm:ss) chat from player $($b[1]): $($utf8.GetString($b, 3, $b.Length - 3))") } }
                    14 { $ahLog.WriteLine("$(Get-Date -Format HH:mm:ss) player $($b[1]) defeated") }
                    default { $ahLog.WriteLine("$(Get-Date -Format HH:mm:ss) event $($b[0]) ($($b.Length) bytes)") }
                }
            }

            foreach ($c in $clients) {
                if ($c.Process.HasExited -and -not $c.Reported) {
                    $c.Reported = $true
                    $code = $c.Process.ExitCode
                    if ($code -ne 0 -and -not $gameOverAt) {
                        Note "  client exited with code ${code}: $($c.Command)" Red
                        $tail = Get-LogTail $c.ErrLog
                        if ($tail) { Note ($tail -replace '(?m)^', '    ') DarkGray }
                        # A client also fails when its team is destroyed (the AI's server goes away), which can come a few
                        # seconds before the game over; wait for that before calling it a failure.
                        if ($Headless -and -not $clientFailedAt) { $clientFailedAt = Get-Date; $result.Note = "client exited with code $code" }
                    } elseif (-not $gameOverAt) { Note "  client finished: $($c.Command)" DarkGray }
                }
            }

            if ($clientFailedAt -and -not $gameOverAt -and ((Get-Date) - $clientFailedAt).TotalSeconds -ge 15) {
                $result.Outcome = 'client failed'
                $gameOverAt = Get-Date
                Note '  no game over followed; ending the match' Yellow
            }
            $elapsed = ((Get-Date) - $started).TotalMinutes
            if (-not $gameOverAt -and $script:TimeoutValue -gt 0 -and $elapsed -ge $script:TimeoutValue) {
                $result.Outcome = 'timeout'
                $gameOverAt = Get-Date
                Note "  timed out after $($script:TimeoutValue) minutes" Yellow
            }
            # Headless matches end at game over; graphical ones when you close the window.
            $ending = $gameOverAt -and ($Headless -or $result.Outcome -eq 'timeout')
            if ($ending -and -not $killSentAt -and ((Get-Date) - $gameOverAt).TotalSeconds -ge 2) {
                & $send '/kill'
                $killSentAt = Get-Date
                if (-not $engineEndpoint) { Stop-ProcessTree $engine }
            }
            # A ControllerAI waiting for its (gone) client blocks the engine's main thread, so /kill can't finish it.
            if ($killSentAt -and ((Get-Date) - $killSentAt).TotalSeconds -ge 10) { Note '  the engine is still running; stopping it' DarkGray; Stop-ProcessTree $engine; $killSentAt = $null }
            Start-Sleep -Milliseconds 200
        }

        $engine.WaitForExit()
        if ($result.Outcome -eq 'no result') {
            $code = $engine.ExitCode
            $result.Note = if ($playing) { "engine exited before the game ended (code $code)" } else { "engine exited before the game started (code $code)" }
            Note "  $($result.Note)" $(if ($Headless) { 'Red' } else { 'Yellow' })
            $tail = Get-LogTail (Join-Path $WriteDir 'infolog.txt') 6
            if ($tail -and $Headless) { Note ($tail -replace '(?m)^', '    ') DarkGray }
        }
    } finally {
        $result.Seconds = [int]((Get-Date) - $started).TotalSeconds
        foreach ($c in $clients) { Stop-ProcessTree $c.Process }
        if ($engine -and -not $engine.HasExited) { Stop-ProcessTree $engine }
        $udp.Dispose()
        $ahLog.Dispose()
        $env:SPRING_DATADIR = $savedDataDir
        $infolog = Join-Path $WriteDir 'infolog.txt'
        if ($engine -and (Test-Path $infolog) -and (Get-Item $infolog).LastWriteTime -ge $started) {
            Copy-Item $infolog (Join-Path $gameDir 'infolog.txt')
            # the local client records the demo, so the server (autohost) usually doesn't know its name
            $m = Select-String -LiteralPath $infolog -Pattern 'recording demo to "(.+)"' | Select-Object -First 1
            if ($m -and -not $result.Demo) { $result.Demo = $m.Matches[0].Groups[1].Value }
        }
        if ($result.Demo -and -not [IO.Path]::IsPathRooted($result.Demo)) { $result.Demo = [IO.Path]::GetFullPath([IO.Path]::Combine($WriteDir, $result.Demo)) }
    }
    $result
}

# ---------------------------------------------------------------- main

$ZkDir = Find-ZkDir
$script:EngineDir = Get-EngineDir $Engine
if (-not $script:EngineDir -or -not (Test-Path (Get-EngineExe 'spring'))) { throw "No spring$exeSuffix in '$(if ($script:EngineDir) { $script:EngineDir } else { Join-Path $ZkDir $engineSubdir })'." }
$WriteDir = if ($WriteDir) { [void](New-Item $WriteDir -ItemType Directory -Force); (Resolve-Path $WriteDir).Path } else { $ZkDir }

if ($List) {
    Note "Zero-K folder: $ZkDir" Cyan
    Note "Engines (newest is the default): $((Get-ChildItem (Join-Path $ZkDir $engineSubdir) -Directory | Where-Object { Test-Path (Get-EngineExe 'spring' $_.FullName) } | ForEach-Object Name) -join ', ')"
    Note "Game zk:stable: $(Resolve-GameName 'zk:stable')"
    Note 'Skirmish AIs:' Cyan
    Get-InstalledAIs | Where-Object { $_.ShortName -notmatch '^(\d+|Dev)?Circuit(AI)?.' } | Group-Object ShortName | Sort-Object Name |
        ForEach-Object { Note "  $($_.Name) $(($_.Group | ForEach-Object Version | Sort-Object -Unique) -join ', ')" }
    Note "  CircuitAI:<difficulty> ($($aiDifficulties -join ', ')) uses $(Get-CircuitShortName 'Normal') and its siblings" DarkGray
    Note 'Maps:' Cyan
    Get-InstalledMaps | Sort-Object Name -Unique | ForEach-Object { Note "  $($_.Name)" }
    return
}

if (-not $Teams -or -not $Teams.Count) { $Teams = @('ControllerAI', 'CircuitAI') }
$sides = @(foreach ($t in $Teams) { , @($t -split '\+' | Where-Object { $_.Trim() } | ForEach-Object { ConvertTo-Member $_ }) })
$sideLabels = @(for ($i = 0; $i -lt $sides.Count; $i++) { "$($i + 1):$($Teams[$i])" })
$all = @($sides | ForEach-Object { $_ })
$players = @($all | Where-Object Kind -eq 'Player')
if ($players.Count -gt 1) { throw 'Only one side member can be the Player.' }
if ($players.Count -and $Headless) { throw "Player can't play a -Headless match (there's no window); leave out -Headless, or replace Player with an AI." }
if ($sides.Count -lt 1) { throw 'Give at least one side.' }
if ($Games -lt 1) { throw '-Games must be at least 1.' }

$controllerCount = @($all | Where-Object Kind -eq 'ControllerAI').Count
if ($Client.Count -gt 1 -and $Client.Count -ne $controllerCount) { throw "Give one -Client per ControllerAI team ($controllerCount), or a single one; got $($Client.Count)." }
if ($Client.Count -and -not $controllerCount) { throw '-Client is for ControllerAI teams, and there are none.' }

$script:ControllerOptionTable = ConvertTo-OptionTable $ControllerOptions '-ControllerOptions'
if (-not $script:ControllerOptionTable.Contains('ip')) { $script:ControllerOptionTable.Insert(0, 'ip', '127.0.0.1') }
if ($script:ControllerOptionTable.Contains('port')) { $ControllerPort = [int]$script:ControllerOptionTable['port'] } else { $script:ControllerOptionTable['port'] = [string]$ControllerPort }
$masterUrl = "http://$($script:ControllerOptionTable['ip']):$ControllerPort"
$script:ModOptionTable = ConvertTo-OptionTable $ModOptions '-ModOptions'
# 'max': the engine has no speed limit of its own, it adjusts the speed to keep the CPU busy (see CGameServer's
# speed control), so a very high wanted speed means "as fast as this machine can".
$maxSpeed = 1000
if (-not $Speed) { $Speed = if ($Headless) { 'max' } else { '1' } }
$script:SpeedValue = if ($Speed -eq 'max') { $maxSpeed } else {
    $parsed = 0.0
    if (-not [double]::TryParse($Speed, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$parsed) -or $parsed -lt 0) { throw "-Speed must be a number (0 leaves the speed alone) or 'max'." }
    $parsed
}
$script:TimeoutValue = if ($TimeoutMinutes -ge 0) { $TimeoutMinutes } elseif ($Headless) { 60 } else { 0 }
$script:PlayerName = if ($Name) { $Name } elseif ($players.Count -and $players[0].Name) { $players[0].Name } elseif (-not $Headless) { Get-ZkLoginName }
if (-not $script:PlayerName) { $script:PlayerName = if ($players.Count) { 'Player' } else { 'Spectator' } }

$script:GameName = Resolve-GameName $Game
if (-not (Test-GameInstalled $script:GameName)) { Invoke-Download 'game' $script:GameName }
$maps = @($Map | ForEach-Object { $_ -split ',' } | Where-Object { $_.Trim() } | ForEach-Object {
    $resolved = Resolve-MapName $_
    if (-not (Get-InstalledMaps | Where-Object Name -eq $resolved)) { Invoke-Download 'map' $resolved }
    $resolved
})

if (-not $OutDir) { $OutDir = Join-Path $scriptDir ('results\' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
[void](New-Item $OutDir -ItemType Directory -Force)
$OutDir = (Resolve-Path $OutDir).Path

Note "Zero-K $ZkDir, engine $(Split-Path -Leaf $script:EngineDir)$(if ($Headless) { ' (headless)' }), $($script:GameName)" Cyan
Note "Sides: $($sideLabels -join '  vs  ')"
if ($controllerCount) { Note "ControllerAI master: $masterUrl/list" }
if ($controllerCount -and -not $Client.Count) { Note "No -Client: the ControllerAI team(s) wait every frame for a client to connect to $masterUrl (Synchronous Mode)." Yellow }
Note "Results: $OutDir"

$csv = Join-Path $OutDir 'results.csv'
$results = [Collections.Generic.List[object]]::new()
for ($i = 0; $i -lt $Games; $i++) {
    $mapName = $maps[$i % $maps.Count]
    $swapped = [bool]($AlternateSides -and $sides.Count -gt 1 -and ([Math]::Floor($i / $maps.Count) % 2 -eq 1))
    $r = Invoke-Match $i $mapName $swapped
    $results.Add($r)
    [pscustomobject]@{
        game = $r.Game; map = $r.Map; swapped = $r.Swapped; outcome = $r.Outcome
        winner = ($r.WinnerSides | ForEach-Object { $_ + 1 }) -join ' '; winner_sides = ($r.WinnerSides | ForEach-Object { $Teams[$_] }) -join ' '
        seconds = $r.Seconds; demo = $r.Demo; note = $r.Note; dir = $r.Dir
    } | Export-Csv $csv -NoTypeInformation -Append -Encoding UTF8
}

if ($DryRun) { Note "Dry run: start scripts are in $OutDir" Cyan; return }

$summary = [Collections.Generic.List[string]]::new()
$summary.Add("$($results.Count) match(es): $($sideLabels -join '  vs  ') on $(($maps | Select-Object -Unique) -join ', ')")
for ($s = 0; $s -lt $sides.Count; $s++) {
    $wins = @($results | Where-Object { $_.Outcome -eq 'win' -and $_.WinnerSides -contains $s }).Count
    $summary.Add(('  side {0,-28} {1} win{2}' -f $sideLabels[$s], $wins, $(if ($wins -eq 1) { '' } else { 's' })))
}
foreach ($o in 'draw', 'timeout', 'client failed', 'no result') {
    $n = @($results | Where-Object Outcome -eq $o).Count
    if ($n) { $summary.Add("  $($o): $n") }
}
[IO.File]::WriteAllText((Join-Path $OutDir 'summary.txt'), ($summary -join "`r`n") + "`r`n", $utf8)
Note ''
$summary | ForEach-Object { Note $_ Cyan }
Note "Details: $csv"
