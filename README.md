# zk-arena

A template workspace for AI agents (LLM coding agents) developing Zero-K AIs. Copy it, put your AI next to these scripts, and use `zk-startup.ps1` to play it against Zero-K's CircuitAI, other AIs or a person, straight from a Zero-K install without the lobby.

AIs are written as [ControllerAI](https://github.com/pts1667/ControllerAI) clients: ControllerAI is a Skirmish AI that exposes the game over HTTP/WebSocket, so your AI is an ordinary program (Python, or anything that speaks HTTP) that reads observations and sends commands.

## Rules for agents

1. **Test games run headless at max speed.** Use `-Headless`; the speed then defaults to `max`, as fast as the machine allows. Don't lower it, and don't open a game window for testing.

   ```powershell
   .\zk-startup.ps1 ControllerAI CircuitAI:Normal -Headless -Client 'python bots\mybot\main.py {team}'
   ```

2. **Windowed games only when the user asks for one,** at the default speed (normal, 1x): leave out `-Headless` and `-Speed`. That's for the user to watch the AI (`ControllerAI CircuitAI:Hard`) or play against it (`ControllerAI Player`).

   ```powershell
   .\zk-startup.ps1 ControllerAI Player -Client 'python bots\mybot\main.py {team}'
   ```

3. **Judge changes by results over several games,** not one: `-Games N -AlternateSides` against a fixed opponent and map set, then compare `summary.txt` between versions.
4. **Pick maps from [`maps.md`](maps.md)** (installed maps, by size, mode and terrain) and pass `-NoDownload` so a typo doesn't download a map into the user's Zero-K install.
5. **When a match ends badly, read its logs** in `results\<run>\game-NNN\`: `client-team<N>.err.log` (your AI's errors), `infolog.txt` (the engine and game), `autohost.log` (start, end, Zero-K's end-of-game stats).
6. **Keep your AI in its own folder** (e.g. `bots\<name>\`), and don't edit the Zero-K install.

## Setup

- A Zero-K install (Steam or standalone), found automatically or given with `-ZkDir`.
- ControllerAI built and installed into it: in the ControllerAI repo, `.\build.ps1 -InstallDir "<Zero-K folder>"`.
- For Python AIs: `pip install requests websocket-client`.

Check with `.\zk-startup.ps1 -List`, which shows the engines, AIs and maps it found.

## Writing an AI

The full API documentation is served by ControllerAI itself while a match runs, on the master port (3017 by default):

```powershell
# a match with an undriven ControllerAI waits for its client: long enough to read the docs (ends after 2 minutes)
Start-Job { & "$using:PWD\zk-startup.ps1" ControllerAI NullAI -Headless -TimeoutMinutes 2 }
curl.exe http://127.0.0.1:3017/SKILL.md          # a guide to the API, written for agents
curl.exe http://127.0.0.1:3017/docs/             # the full reference
curl.exe http://127.0.0.1:3017/docs/examples/    # example clients to start from
```

The same docs are in the ControllerAI repository (`docs/`, `examples/`). The essentials:

- Your AI finds its team's server through the master's `/list`, using the team ID that `{team}` gives it.
- ControllerAI runs in Synchronous Mode: the game waits for your AI every frame. For each observation, send commands and then `{"type": "finish_frame"}`. The WebSocket (`/ws`) is much faster than polling HTTP.
- Before the game starts, choose a commander (`set_commander`) and, when `game_info.canChooseStartPos` is true, a start position (`set_start_pos`). Otherwise Zero-K places you.
- Unit and game definitions come from queries (`/query?type=...`), e.g. `unit_def_by_unit_id`.
- CircuitAI, Zero-K's own AI, is the reference opponent. SKILL.md links its source and the engine and game source for this match.

Suggested opponents, in order: `NullAI` (does nothing: checks that your AI can win at all), then `CircuitAI:Beginner`, `Novice`, `Easy`, `Normal`, `Hard`, `Brutal`.

## Running matches: zk-startup.ps1

```powershell
# your AI against CircuitAI Hard, 10 headless matches, swapping start boxes every other match
.\zk-startup.ps1 ControllerAI CircuitAI:Hard -Headless -Games 10 -AlternateSides -Client 'python bots\mybot\main.py {team}'

# two versions of your AI against each other, on two maps
.\zk-startup.ps1 ControllerAI ControllerAI -Headless -Games 4 -AlternateSides -Map 'TitanDuel 2.2','Fairyland 1.31' `
    -Client 'python bots\v1\main.py {team}', 'python bots\v2\main.py {team}'

# 2v2, one client process driving both ControllerAI teams
.\zk-startup.ps1 CircuitAI:Hard+CircuitAI:Hard ControllerAI+ControllerAI -Headless -Client 'python bots\mybot\all_teams.py'
```

From `cmd.exe`, use `zk-startup.cmd` with the same arguments (double quotes instead of single).

### Sides

Every argument that isn't a `-Parameter` is a side (an ally team); join members of the same side with `+`.

| Member | |
| --- | --- |
| `ControllerAI[@version]` | ControllerAI, the newest installed version unless given. It needs a client: see `-Client`. |
| `CircuitAI[:difficulty]` | Zero-K's AI, the same build the Zero-K lobby uses: `Beginner`, `Novice`, `Easy`, `Normal` (default), `Hard`, `Brutal`. |
| `Player[:name]` | The user, playing. Not with `-Headless`. A windowed match without a `Player` has the user watching as a spectator. |
| `<ShortName>[@version]` | Any other installed Skirmish AI, e.g. `NullAI`. |

Default: `ControllerAI CircuitAI` (Normal).

### Clients

ControllerAI waits for its client every frame, so a match with ControllerAI stalls until something drives it. `-Client` starts that for you, after the engine, and stops it when the match ends:

- One `-Client` per ControllerAI team, in the order they appear in the sides; or
- a single `-Client` containing `{team}`, which runs once per ControllerAI team; or
- a single `-Client` without `{team}`, which runs once per match (for a client that drives every team it finds in `/list`).

Placeholders in the command, also set as environment variables:

| Placeholder | Variable | |
| --- | --- | --- |
| `{master}` | `CONTROLLERAI_MASTER` | the master server, e.g. `http://127.0.0.1:3017` (its `/list` gives each team's URLs) |
| `{team}` | `CONTROLLERAI_TEAM` | the engine team ID of the ControllerAI this client drives |
| `{ally}` | `CONTROLLERAI_ALLYTEAM` | its ally team |
| `{game}` | `ZK_ARENA_GAME` | match number, from 1 |
| `{out}` | `ZK_ARENA_OUT` | the match's output folder (for your AI's own logs) |

The command runs through `cmd.exe` in the current folder with `PYTHONUNBUFFERED=1`. Its output goes to `client-team<N>.log` and `.err.log` in the match folder. In a headless match, a client that exits with an error ends the match (`client failed`), unless game over follows within 15 seconds: a client also fails when its team is destroyed.

### Results

The engine reports the game's start and end over its autohost (UDP) interface. That's how the script records the winner, sets the speed and closes headless matches. Windowed matches end when the user closes the window.

Each run writes to `results\<date-time>\` (or `-OutDir`):

- `results.csv`: one row per match with the map, whether the sides were swapped, the outcome (`win`, `draw`, `timeout`, `client failed`, `no result`), the winning side (numbered as given), the wall-clock seconds, and the demo file.
- `summary.txt`: wins per side. It's also printed at the end.
- `game-NNN\`: the start script, the engine settings used, `infolog.txt`, `autohost.log`, and the client and engine logs.

Demos (replays) go to the Zero-K folder's `demos\`, so the user can watch them from the Zero-K lobby.

### Parameters

| Parameter | Default | |
| --- | --- | --- |
| `-Map` | `TitanDuel 2.2` | One or more maps. Match *i* uses map *i* mod the number of maps. Matched loosely (`titanduel`), and downloaded if missing (unless `-NoDownload`). |
| `-Games` | 1 | Number of matches. |
| `-AlternateSides` | | Reverses the side order (swapping start boxes) on every other round through the maps. |
| `-Headless` | | Runs `spring-headless`: no window and no sound. |
| `-Speed` | `max` headless, otherwise 1 | Game speed, or `max` (as fast as the machine allows). The engine slows down to what the CPU, or a slow client, can keep up with. 0 leaves it alone. |
| `-TimeoutMinutes` | 60 headless, otherwise none | Wall-clock limit per match, after which it's ended as a timeout. |
| `-Client` | | ControllerAI client command(s), see above. |
| `-ControllerPort` | 3017 | ControllerAI's master port (its `port` AI option). |
| `-ControllerOptions` | | More ControllerAI AI options: `@{ recoil_engine_path = 'C:\src\RecoilEngine' }`, or `key=value,...`. |
| `-ModOptions` | | Game options: `@{ metalmult = 2 }`, or `metalmult=2,energymult=2`. The full list is in the game's `ModOptions.lua`. |
| `-StartPos` | `Choose` | `Choose` (in game, in the map's start boxes), `Fixed` or `Random`. |
| `-Game` | `zk:stable` | Rapid tag or full game name (`Zero-K v1.14.10.3`). Downloaded if missing. |
| `-Engine` | newest installed | Engine version folder under `engine\win64`. |
| `-ZkDir` | found automatically | The Zero-K folder. Otherwise `ZK_DIR`, Steam libraries, Program Files, `%LOCALAPPDATA%`, or a folder in Downloads, Desktop, Documents or `C:\Games`. |
| `-WriteDir` | the Zero-K folder | The engine's write folder (settings, cache, demos, infolog). Content still comes from the Zero-K folder. |
| `-Name` | the user's Zero-K login | The player name. |
| `-HostPort` | 8459 | The game's local port. |
| `-OutDir` | `results\<date-time>` | |
| `-DryRun` | | Writes the start scripts and shows the commands without running anything. |
| `-NoDownload` | | Fails instead of downloading a missing game or map. |
| `-List` | | Lists the installed engines, AIs and maps. |

The engine is started with `--config`, pointing at a copy of the install's `springsettings.cfg` plus the autohost port. Settings changed during a match therefore aren't saved to the install.

## Maps: zk-maps.ps1

[`maps.md`](maps.md) sorts the installed maps by size, mode (1v1, teams, ffa, chickens), matchmaker/featured pool, hills, water, metal and wind, and suggests maps for quick tests. [`maps.json`](maps.json) has the same data for scripts. Modes, terrain and pool come from Zero-K's own map data (`news\community.json` in the install, downloaded by the Zero-K lobby). Maps that Zero-K doesn't list are estimated from the map file and marked as such.

They describe one particular install, so regenerate them after adding maps or in a copy of this template on another machine (it takes a few minutes):

```powershell
.\zk-maps.ps1
```

## Files

| | |
| --- | --- |
| `zk-startup.ps1`, `zk-startup.cmd` | Start matches; see above. |
| `zk-maps.ps1`, `maps.md`, `maps.json` | Map categories for the installed maps. |
| `results\` | Match output (not committed). |
| `bots\<name>\` | Your AI (suggested layout). |
