"""The learned policy's host in the engine: keeps what a raw frame is read from (policy/rawframe.py `from_host`) and runs the
policy runtime (policy/runtime.py) once a decision.

It plays nothing itself.  The rules bot that used to play here, and host the policy beside its own behaviours, was removed on
2026-11-08 (git tag `rules-bot-final`).  Until the policy's hand-off is built (its decoded orders sent as commands, and the
opening and the economy script as the sim plays them), a game with this host is the commander standing still, and the flag
`rawframe` records frames for the golden tests.

What it keeps, per frame:

  units                our units, as UnitInfo
  commander_id         our commander, the healthiest commander-like unit
  base_pos             where the commander stood at the first frame
  enemy_structures     enemy buildings once seen, until a unit of ours sees the place empty: the frame's remembered units
  blip_tracks          each radar blip's sustained top speed, for the frame
  map.spots            metal spots, by whose extractor stands on them
  last_enemy_seen_frame
  unit_order           orders this side gave, by unit: none, since it gives none
"""
from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING, Any

from . import units
from .hazards import Hazards
from .mapdata import MapData
from .models import UnitInfo
from .policy.runtime import PolicyMixin

if TYPE_CHECKING:
    from .client import Client      # needs requests, which the training environment hasn't

COMMANDER = "dyntrainer_support_base"
SPOT_REACH = 105.0          # elmos: an extractor this near a spot stands on it
SPEED_WINDOW = 60           # frames: a blip's sustained speed is the least over this window
FORGET_TRACK = 1800         # frames without a blip before its track is dropped
SIGHT_SHARE = 0.7           # of a unit's sight radius: an enemy building not seen within it is gone
BLOCK_FRAMES = 6            # observations come every this many frames


class Host(PolicyMixin):
    def __init__(self, client: Client, team_id: int, log):
        self.client = client
        self.team_id = int(team_id)
        self.log = log
        self.map = MapData()
        self.hazards: Hazards | None = None
        self.game_info: dict[str, Any] = {}
        self.ally_team_id = 0
        self.frame = -1
        self.commander_id: int | None = None
        self.base_pos: tuple[float, float] = (0.0, 0.0)
        self.units: dict[int, UnitInfo] = {}
        self.enemies: dict[int, UnitInfo] = {}
        self.enemy_structures: dict[int, tuple[UnitInfo, int]] = {}
        self.blip_tracks: dict[int, dict[str, Any]] = {}
        self.unit_order: dict[int, tuple] = {}
        self.last_enemy_seen_frame = -10**9
        self.economy: dict[str, Any] = {}
        self.cmds: list[dict] = []
        self.commander_sent = False
        self.settings_sent = False
        self.policy_init()

    # The connection --------------------------------------------------------------------------------------------------------

    def take_commands(self) -> list[dict]:
        out, self.cmds = self.cmds, []
        return out

    def on_bootstrap(self, typ: str, data: Any) -> None:
        if not isinstance(data, dict):
            return
        if typ == "game_info":
            self.game_info = data
            self.ally_team_id = int(data.get("allyTeamId") or 0)
            self.map.width = float(data.get("mapWidthElmos") or 0.0)
            self.map.height = float(data.get("mapHeightElmos") or 0.0)
        elif typ == "spawn_boxes":
            self.map.start_boxes = data
        elif typ == "metadata":
            # Def ids as the running game numbers them.
            for id_s, d in data.items():
                known = units.BY_NAME.get((d or {}).get("name") or "")
                if known is not None and str(id_s).isdigit():
                    known["id"] = int(id_s)
                    units.BY_ID[int(id_s)] = known
        elif typ == "map_features":
            self.map.set_map_features(data)

    def on_startup(self, obs: dict[str, Any]) -> None:
        if not self.commander_sent:
            self.client.send({"type": "set_commander", "name": COMMANDER})
            self.commander_sent = True
        self.client.send({"type": "finish_frame"})

    def on_frame(self, obs: dict[str, Any]) -> None:
        self.frame = int(obs.get("frame") or 0)
        self.client.frame = self.frame
        self.economy = obs.get("economy") or {}
        self.policy_events(obs)
        if not self.settings_sent and self.frame >= 2:
            self.settings_sent = True
            self.client.set_settings({"block_n_frames": BLOCK_FRAMES})
        self._read_units(obs)
        self._read_enemies(obs)
        self._read_spots()
        self.policy_step(obs)

    # What a frame is read from ---------------------------------------------------------------------------------------------

    def _unit(self, uid: int, raw: dict[str, Any], previous: UnitInfo | None = None) -> UnitInfo:
        did = int(raw.get("defId") or 0)
        d = units.by_id(did) or {}
        pos = raw.get("pos") or (0.0, 0.0, 0.0)
        u = UnitInfo(uid=uid, def_id=did, name=d.get("name") or raw.get("name") or f"def{did}", data=d,
                     x=float(pos[0]), y=float(pos[1]), z=float(pos[2]), health=float(raw.get("health") or 0.0),
                     max_health=float(raw.get("maxHealth") or 1.0), build_progress=float(raw.get("buildProgress") or 0.0),
                     is_being_built=bool(raw.get("isBeingBuilt") or False),
                     first_seen=previous.first_seen if previous else self.frame)
        u.finished_frame = (previous.finished_frame if previous and previous.finished_frame >= 0 else self.frame) if u.is_finished else -1
        return u

    def _read_units(self, obs: dict[str, Any]) -> None:
        self.units = {int(k): self._unit(int(k), raw, self.units.get(int(k))) for k, raw in (obs.get("units") or {}).items()}
        if self.commander_id not in self.units:
            commanders = [u for u in self.units.values() if "dyntrainer" in u.name or u.name.startswith("commander")]
            self.commander_id = max(commanders, key=lambda u: u.health).uid if commanders else None
            if self.commander_id is not None and self.base_pos == (0.0, 0.0):
                self.base_pos = self.units[self.commander_id].pos

    def _read_enemies(self, obs: dict[str, Any]) -> None:
        now = self.frame
        self.enemies = {int(k): self._unit(int(k), raw) for k, raw in (obs.get("enemies") or {}).items()}
        for u in self.enemies.values():
            if not u.is_mobile:
                self.enemy_structures[u.uid] = (u, now)
        blips = obs.get("radarBlips") or {}
        for uid_s, raw in blips.items():
            self._track_blip(int(uid_s), raw)
        if self.enemies or blips:
            self.last_enemy_seen_frame = now
        for uid in [k for k, t in self.blip_tracks.items() if now - t["last"] > FORGET_TRACK]:
            del self.blip_tracks[uid]
        # A remembered building whose place one of ours sees, without it there, is gone.
        viewers = [(u.x, u.z, float(u.data.get("losRadius") or 0.0) * SIGHT_SHARE) for u in self.units.values() if u.is_finished]
        for uid, (e, _seen) in list(self.enemy_structures.items()):
            if uid not in self.enemies and any(r > 50.0 and (x - e.x) ** 2 + (z - e.z) ** 2 < r * r for x, z, r in viewers):
                del self.enemy_structures[uid]

    def _track_blip(self, uid: int, raw: dict[str, Any]) -> None:
        now = self.frame
        track = self.blip_tracks.setdefault(uid, {"speeds": deque(), "smax": 0.0, "last": now})
        v = raw.get("vel") or (0.0, 0.0, 0.0)
        speeds = track["speeds"]
        speeds.append((now, math.hypot(float(v[0]), float(v[2])) * 30.0))
        while speeds and now - speeds[0][0] > SPEED_WINDOW:
            speeds.popleft()
        if len(speeds) >= 2 and speeds[-1][0] - speeds[0][0] >= SPEED_WINDOW * 0.6:
            track["smax"] = max(track["smax"], min(s for _f, s in speeds))
        track["last"] = now

    def _read_spots(self) -> None:
        if self.frame % 15:
            return
        ours = [u.pos for u in self.units.values() if u.name == "staticmex" and u.is_finished]
        theirs = [u.pos for u in list(self.enemies.values()) + [e for e, _f in self.enemy_structures.values()] if u.name == "staticmex"]
        for s in self.map.spots:
            near = lambda p: (p[0] - s.x) ** 2 + (p[1] - s.z) ** 2 <= SPOT_REACH ** 2  # noqa: E731
            s.status = "ours" if any(map(near, ours)) else "enemy" if any(map(near, theirs)) else "free"

    # The economy, as rawframe reads it -------------------------------------------------------------------------------------

    def metal(self) -> dict:
        return self.economy.get("Metal") or {}

    def energy(self) -> dict:
        return self.economy.get("Energy") or {}

    def metal_current(self) -> float:
        return float(self.metal().get("current") or 0.0)

    def metal_income(self) -> float:
        return float(self.metal().get("income") or 0.0)

    def energy_current(self) -> float:
        return float(self.energy().get("current") or 0.0)

    def energy_income(self) -> float:
        return float(self.energy().get("income") or 0.0)

    @staticmethod
    def effective_storage(resource: dict) -> float:
        """Zero-K keeps 10,000 hidden storage, so that caretakers can spend in chunks: the visible cap is what's above it."""
        raw = float(resource.get("storage") or 0.0)
        return max(1.0, raw - 10000.0 if raw > 10000.0 else raw)
