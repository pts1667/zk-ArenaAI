"""Bot orchestration: startup, frame loop and the mixin composition."""
from __future__ import annotations

import math
import os
import time
from typing import Any

from . import units
from .client import Client, QueryError
from .construction import ConstructionMixin
from .defense import DefenseMixin
from .economy import EconomyMixin
from .intel import IntelMixin
from .mapdata import MapData
from .military import MilitaryMixin
from .models import Task, UnitInfo
from .production import ProductionMixin
from .scouting import ScoutingMixin
from .strategy import StrategyMixin
from .tactics import TacticsMixin
from .threat import ThreatMixin
from .air import AirMixin
from .burrow import BurrowMixin


class Brain(
    ThreatMixin,
    IntelMixin,
    EconomyMixin,
    ConstructionMixin,
    ProductionMixin,
    StrategyMixin,
    TacticsMixin,
    DefenseMixin,
    ScoutingMixin,
    AirMixin,
    BurrowMixin,
    MilitaryMixin,
):
    def __init__(self, client: Client, team_id: int, log):
        self.client = client
        self.team_id = int(team_id)
        self.log = log
        self.map = MapData()
        self.game_info: dict[str, Any] = {}
        self.spawn_boxes: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.settings: dict[str, Any] = {}
        self.ally_team_id = 0
        self.start_box_id: int | None = None

        self.frame = -1
        self.commander_id: int | None = None
        self.commander: UnitInfo | None = None
        self.base_pos: tuple[float, float] = (0.0, 0.0)
        self.sectors: list = []
        self.sector_of_spot: dict[int, int] = {}
        self.sector_built_frame = -10**9
        self.sector_update_frame = -10**9
        self.factory_def_id: int | None = None
        self.factory_name = ""
        self.factory_plop_sent = False
        self.first_game_frame = True
        self.startup_commander_sent = False
        self.startup_queries_done = False
        self.game_over = False

        self.units: dict[int, UnitInfo] = {}
        self.enemies: dict[int, UnitInfo] = {}
        self.radar: dict[int, UnitInfo] = {}
        self.enemy_memory: dict[int, tuple[UnitInfo, float]] = {}
        self.enemy_commander_id: int | None = None
        self.enemy_commander_dead = False
        self.enemy_base: tuple[float, float] | None = None
        self.enemy_base_seen_frame = -10**9
        self.enemy_army_pos: tuple[float, float] | None = None
        self.enemy_army_frame = -10**9

        self.tasks: dict[str, Task] = {}
        self.builder_task: dict[int, str] = {}
        self.builder_idle: dict[int, bool] = {}
        self.builder_job_frame: dict[int, int] = {}
        self.builder_fallback_frame: dict[int, int] = {}
        self.builder_fallback_until: dict[int, int] = {}
        self.factory_queue: dict[int, int] = {}
        self.factory_query_frame: dict[int, int] = {}
        self.caretaker_guarded: set[int] = set()
        self.production_counter = 0
        self.energy_task_seq = 0
        self.factory_task_seq = 0

        self.army: set[int] = set()
        self.retreating: set[int] = set()
        self.unit_order: dict[int, tuple[str, float, float, int | None, int]] = {}
        self.raiders: set[int] = set()

        self.last_summary_frame = -10**9
        self.last_economy_frame = -10**9
        self.last_military_frame = -10**9
        self.last_query_frame = -10**9
        self.last_assign_frame = -10**9
        self.last_enemy_seen_frame = -10**9
        self.enemy_peak = 0
        self.enemy_peak_frame = -10**9
        self.blocked_spot_until: dict[int, int] = {}
        self.last_attack_frame = -10**9
        self.last_commander_retreat_frame = -10**9
        self.last_threat_frame = -10**9
        self.threat_pos: tuple[float, float] | None = None
        self.threat_frame = -10**9
        self.defense_built_near: list[tuple[float, float]] = []
        self.announced: set[str] = set()
        self.cmds: list[dict] = []
        self._feature_cache: dict = {}
        self.debug_queries = 0
        self.t0 = time.time()
        self.stuck_releases = 0
        self.hunt_commander = False
        self.air_counter = 0
        # Shared raider squad target.  Raiders are much more effective when
        # two or three arrive together than when they trickle in one by one.
        self.raid_group_target: tuple[float, float] | None = None
        self.raid_group_frame = -10**9
        self.raid_group_size = 0
        self.last_composition_log_frame = -10**9
        self.home_guard: set[int] = set()
        self.stride = max(1, int(os.environ.get("ZKBOT_STRIDE", "1")))
        self.observation_setting_sent = False
        self.block_frames = 6
        try:
            _f, _c = (int(v) for v in os.environ.get("ZKBOT_BLOCK", "6,10").split(","))
            self.block_setting = (max(1, _f), max(1, _c))
        except ValueError:
            self.block_setting = (6, 10)
        self.last_combat_frame = -10**9
        self.last_spot_frame = -10**9
        self.last_threat_frame = -10**9
        self.last_commander_safe_frame = -10**9
        self.last_feature_frame = -10**9
        # Defence intelligence state.  Kept on the Brain so it survives all
        # mixin calls and is cheap to inspect from the economy.
        self.air_seen_frame = -10**9
        self.air_seen_defs: set[str] = set()
        self.recent_attacker_range = 0.0
        self.recent_attacker_frame = -10**9
        self.recent_attacker_name = ""
        self.sector_attack_pressure: dict[int, float] = {}
        self.sector_last_attack: dict[int, int] = {}
        self.sector_attacker_range: dict[int, tuple[float, int]] = {}
        self.sector_artillery_damage: dict[int, float] = {}
        self.artillery_damage_log: list[tuple[int, float]] = []
        self._defense_spend_records: list[tuple[int, float, str]] = []
        self.gremlin_scouts: set[int] = set()
        self.air_scout_id: int | None = None
        self.air_scout_pending = False
        self.gremlin_pending = 0
        self.gremlin_fire_state: set[int] = set()
        self.gremlin_last_order: dict[int, int] = {}
        self.last_defense_log_frame = -10**9
        self.last_od_query_frame = -10**9
        self.od_metal = 0.0
        self.od_energy = 0.0
        self.od_waste = 0.0
        self.recent_placements: list[tuple[float, float, int, str]] = []

    # --------------------------------------------------------------- utilities

    def say(self, once_key: str, message: str) -> None:
        if once_key in self.announced:
            return
        self.announced.add(once_key)
        self.log(message)

    def queue(self, cmd: dict) -> None:
        self.cmds.append(cmd)

    def take_commands(self) -> list[dict]:
        out = self.cmds
        self.cmds = []
        return out

    def frame_time(self, frame: int | None = None) -> float:
        return (self.frame if frame is None else frame) / 30.0

    def dist(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # --------------------------------------------------------------- bootstrap

    def on_bootstrap(self, typ: str, data: Any) -> None:
        if not isinstance(data, dict):
            return
        if typ == "game_info":
            self.game_info = data
            self.ally_team_id = int(data.get("allyTeamId") or 0)
            self.map.width = float(data.get("mapWidthElmos") or 0.0)
            self.map.height = float(data.get("mapHeightElmos") or 0.0)
        elif typ == "spawn_boxes":
            self.spawn_boxes = data
            self.map.start_boxes = data
        elif typ == "metadata":
            self.metadata = data
            self._sync_metadata(data)
        elif typ == "map_features":
            self.map.set_map_features(data)
        elif typ == "heightmap":
            self.map.set_heightmap(data)
        elif typ == "settings":
            self.settings = data

    def _sync_metadata(self, data: dict[str, Any]) -> None:
        """Keep names/ids in sync with the running game version."""
        for id_s, d in data.items():
            try:
                did = int(id_s)
                name = d.get("name")
                if not name:
                    continue
                known = units.BY_NAME.get(name)
                if known is not None:
                    known["id"] = did
                    units.BY_ID[did] = known
            except Exception:
                continue

    # ----------------------------------------------------------------- startup

    def on_startup(self, obs: dict[str, Any]) -> None:
        if not self.startup_commander_sent:
            # 12 buildpower and 232 build range; the strongest economic
            # commander.  It still has a 600-range weapon for base defense.
            self.client.send({"type": "set_commander", "name": "dyntrainer_support_base"})
            self.startup_commander_sent = True
            self.say("commander", "commander: dyntrainer_support_base")
        if not self.startup_queries_done:
            self.startup_queries_done = True
            try:
                res = self.client.query(
                    {"type": "team_rules_param", "key": "start_box_id"}, timeout=4.0
                )
                self.debug_queries += 1
                if res:
                    self.start_box_id = int(float(res.get("floatValue") or 0))
            except Exception:
                pass
        self.client.send({"type": "finish_frame"})

    # -------------------------------------------------------------- frame loop

    def on_frame(self, obs: dict[str, Any]) -> None:
        self.frame = int(obs.get("frame") or 0)
        self.client.frame = self.frame
        self.store_economy(obs)
        # Events arrive with every observation; consume them before the
        # stride return so damage/destroyed events are never lost.
        if self.units:
            self.read_battle_events(obs)
        # Synchronous ControllerAI blocks the engine for every observation.
        # Processing every Nth observation (while still acknowledging every
        # frame) roughly halves the Python work without meaningfully changing
        # command latency at max speed.
        if self.stride > 1 and self.frame % self.stride != 0:
            return
        if self.frame - self.last_feature_frame >= 10:
            self.last_feature_frame = self.frame
            self.read_features(obs)
        if not self.commander_id:
            self.find_commander(obs)

        if self.first_game_frame:
            self.setup_first_frame(obs)
            self.first_game_frame = False
        if not self.sectors:
            self.ensure_strategy()

        if not self.observation_setting_sent and self.frame >= 2:
            # Fewer synchronous observations means the engine can run several
            # frames between Python round-trips.  Six frames is 0.2 s of game
            # time; macro decisions do not need finer granularity, and this
            # removes most of the websocket/JSON overhead.
            self.observation_setting_sent = True
            try:
                self.client.set_settings({"block_n_frames": 6})
                self.log("requested block_n_frames=6")
            except Exception:
                pass

        self.read_units(obs)
        self.read_enemies(obs)
        if self.frame - self.last_spot_frame >= 15:
            self.last_spot_frame = self.frame
            self.update_spot_status()
        if self.frame - self.last_threat_frame >= 5:
            self.last_threat_frame = self.frame
            self.update_threat()
        if self.frame - getattr(self, "threat_frame_built", -10**9) >= 15:
            self.update_threat_map()
            self.builder_safety_step()

        if self.frame - self.last_commander_safe_frame >= 3:
            self.last_commander_safe_frame = self.frame
            self.ensure_commander_safe()
        if self.frame - getattr(self, "_last_repair_frame", -10**9) >= 30:
            self._last_repair_frame = self.frame
            self.repair_step()
        self.ensure_caretakers_guarding()

        if self.frame - self.last_economy_frame >= 15:
            self.last_economy_frame = self.frame
            self.economy_step()

        self.update_observation_rate()

        if self.frame - self.last_military_frame >= 15:
            self.last_military_frame = self.frame
            self.military_step()
            if hasattr(self, "gremlin_step"):
                self.gremlin_step()

        if self.frame - self.last_query_frame >= 60:
            self.last_query_frame = self.frame
            self.query_step()

        if self.frame - self.last_summary_frame >= 300:
            self.last_summary_frame = self.frame
            self.summary()
            self.write_metrics()
            self.burrow_check()
            self.builder_watchdog()
            if self.frame % 5400 < 300 and getattr(self, "spend_by_name", None):
                top = sorted(self.spend_by_name.items(), key=lambda kv: -kv[1])[:14]
                total = sum(self.spend_by_name.values())
                self.log(
                    "spend " + f"total={total:.0f} dup_deaths={getattr(self, 'duplicate_death_events', 0)} "
                    + " ".join(f"{k}={v:.0f}" for k, v in top)
                )
                for label, led in (
                    ("lost", getattr(self, "lost_by_name", {})),
                    ("lost_branch", getattr(self, "lost_by_branch", {})),
                    ("dmg_from", getattr(self, "dmg_from", {})),
                    ("enemy_seen", {k: len(v) for k, v in getattr(self, "enemy_seen_ids", {}).items()}),
                ):
                    top = sorted(led.items(), key=lambda kv: -kv[1])[:14]
                    self.log(f"{label} " + " ".join(f"{k}={v:.0f}" for k, v in top))

    # ------------------------------------------------------------- first frame

    def setup_first_frame(self, obs: dict[str, Any]) -> None:
        if self.commander is None:
            self.find_commander(obs)
        if self.commander is None:
            self.log("WARNING: no commander found at frame 0")
            return
        self.base_pos = self.commander.pos
        self.choose_enemy_base()
        self.ensure_strategy()
        self.factory_def_id, self.factory_name = self.choose_factory()
        self.log(
            f"start base={tuple(round(v) for v in self.base_pos)} "
            f"map={self.map.width:.0f}x{self.map.height:.0f} "
            f"water={self.map.water_fraction:.2%} "
            f"factory={self.factory_name} enemy_guess={tuple(round(v) for v in self.enemy_base or (0,0))}"
        )
        self.plop_factory()
        # Energy first: construction is limited by energy in the opening.
        self.ensure_mex_tasks(limit=4)
        self.ensure_energy_tasks(force=3)

    def choose_enemy_base(self) -> None:
        boxes = self.spawn_boxes or {}
        candidates: list[tuple[float, float]] = []
        for key, box in boxes.items():
            if str(key) == str(self.ally_team_id):
                continue
            try:
                candidates.append(
                    (
                        (float(box["left"]) + float(box["right"])) * 0.5,
                        (float(box["top"]) + float(box["bottom"])) * 0.5,
                    )
                )
            except Exception:
                pass
        bx, bz = self.base_pos
        w, h = self.map.width, self.map.height
        if w <= 0 or h <= 0:
            self.enemy_base = (bx, bz)
            return
        if candidates:
            # The ally-team key of the spawn boxes does not always match ours
            # (swapped sides): our own start box then became the "enemy
            # base" and raids, scouting and burrowing all aimed at home.
            real = [
                p
                for p in candidates
                if self.dist(p, (w * 0.5, h * 0.5)) > min(w, h) * 0.1
                and self.dist(p, self.base_pos) > min(w, h) * 0.3
            ]
            if real:
                self.enemy_base = max(real, key=lambda p: self.dist(p, self.base_pos))
                return
        options = [
            (w - bx, h - bz),
            (w - bx, bz),
            (bx, h - bz),
            (w * 0.15, h * 0.15),
            (w * 0.85, h * 0.85),
            (w * 0.5, h * 0.5),
        ]
        self.enemy_base = max(options, key=lambda p: self.dist(p, self.base_pos))

    def choose_factory(self) -> tuple[int, str]:
        ground = self.map.ground_height(*self.base_pos)
        water = self.map.water_fraction
        name = "factorycloak"
        if ground < -1.0 or water > 0.25:
            name = "factoryhover"
        did = units.id_of(name)
        if did is None:
            did = units.id_of("factorycloak") or 409
            name = "factorycloak"
        return int(did), name

    def plop_factory(self) -> None:
        if not self.commander or not self.factory_def_id:
            return
        bx, bz = self.commander.pos
        pos = None
        facing = 0
        try:
            res = self.client.query(
                {
                    "type": "closest_build_site",
                    "unitDefId": int(self.factory_def_id),
                    "searchRadius": 420,
                    "minDist": 100,
                    "pos": [bx, 0.0, bz],
                    "facing": 0,
                }
            )
            self.debug_queries += 1
            if res and res.get("found"):
                pos = res.get("pos")
                facing = int(res.get("facing") or 0)
        except QueryError as exc:
            self.log(f"factory site query failed: {exc}")
        if not pos:
            for radius in (120, 160, 200):
                for angle in (0, 45, 90, 135, 180, 225, 270, 315):
                    a = math.radians(angle)
                    cand = [bx + math.cos(a) * radius, 0.0, bz + math.sin(a) * radius]
                    try:
                        res = self.client.query(
                            {
                                "type": "can_build_at",
                                "unitDefId": int(self.factory_def_id),
                                "pos": cand,
                                "facing": 0,
                            }
                        )
                        self.debug_queries += 1
                        if res and res.get("possible"):
                            pos = res.get("pos") or cand
                            facing = int(res.get("facing") or 0)
                            break
                    except QueryError:
                        continue
                if pos:
                    break
        if not pos:
            self.log("ERROR: no valid free-factory site found")
            return
        self.queue(
            {
                "type": "build",
                "unitId": int(self.commander.uid),
                "defId": int(self.factory_def_id),
                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                "facing": facing,
            }
        )
        self.factory_plop_sent = True
        self.log(f"plop {self.factory_name} at {tuple(round(float(v)) for v in pos)}")

    # --------------------------------------------------------------- unit state

    def summary(self) -> None:
        m, e = self.metal(), self.energy()
        self.log(
            "t={:.0f}s f={} M={:.0f}/{:.1f}/{:.0f} E={:.0f}/{:.1f}/{:.0f} use={:.1f} "
            "units={} army={} builders={} mex={} fac={} tasks={} enemy={} mode={} "
            "odM={:.1f} odE={:.1f} odW={:.1f} queries={} stuck={} "
            "solar={} fusion={} pylon={} storage={} caretaker={} air={} radar={} seismic={} bombs={}".format(
                self.frame_time(),
                self.frame,
                float(m.get("current") or 0),
                float(m.get("income") or 0),
                self.effective_storage(m),
                float(e.get("current") or 0),
                float(e.get("income") or 0),
                self.effective_storage(e),
                float(e.get("usage") or 0),
                len(self.units),
                len(self.army),
                self.builder_count(),
                self.mex_count(),
                len(self.factory_list()),
                len(self.tasks),
                len(self.enemies),
                self.army_mode(),
                self.od_metal,
                self.od_energy,
                self.od_waste,
                self.debug_queries,
                self.stuck_releases,
                sum(1 for u in self.units.values() if u.name == "energysolar" and u.is_finished),
                sum(1 for u in self.units.values() if u.name in ("energyfusion", "energysingu") and u.is_finished),
                sum(1 for u in self.units.values() if u.name == "energypylon" and u.is_finished),
                sum(1 for u in self.units.values() if u.name == "staticstorage" and u.is_finished),
                sum(1 for u in self.units.values() if u.name == "staticcon" and u.is_finished),
                sum(1 for u in self.units.values() if u.is_air and u.is_combat),
                sum(1 for u in self.units.values() if u.name == "staticradar" and u.is_finished),
                sum(1 for u in self.units.values() if u.name == "seismic" and u.is_finished),
                sum(1 for u in self.units.values() if units.role_of(u.data) == "bomb"),
            )
        )

    # ---------------------------------------------------------------- metrics

    _METRIC_COLS = (
        "t", "m_income", "e_income", "m_cur", "e_cur", "mex", "builders",
        "builders_tasked", "army", "army_value", "factories", "caretakers",
        "turrets", "radar", "enemy_vis", "enemy_mex_known", "tasks", "mode",
        "flees", "builder_deaths", "own_lost_value", "enemy_killed_value",
    )

    def write_metrics(self) -> None:
        """Append one row of the per-game timeline (every 10 s game time)."""
        if not hasattr(self, "_metrics_file"):
            out = os.environ.get("ZK_ARENA_OUT", ".")
            try:
                path = os.path.join(out, f"metrics-team{self.team_id}.csv")
                self._metrics_file = open(path, "w", encoding="utf8", buffering=1)
                self._metrics_file.write(",".join(self._METRIC_COLS) + "\n")
            except Exception:
                self._metrics_file = None
        f = self._metrics_file
        if f is None:
            return
        mobile_builders = [u for u in self.units.values() if u.is_builder and u.is_mobile and u.is_finished]
        tasked = sum(1 for u in mobile_builders if self.builder_task.get(u.uid) in self.tasks)
        row = (
            f"{self.frame_time():.0f}",
            f"{self.metal_income():.1f}",
            f"{self.energy_income():.1f}",
            f"{self.metal_current():.0f}",
            f"{self.energy_current():.0f}",
            str(self.mex_count()),
            str(len(mobile_builders)),
            str(tasked),
            str(len(self.army)),
            f"{sum(units.cost(self.units[u].data) for u in self.army if u in self.units):.0f}",
            str(len(self.factory_list())),
            str(sum(1 for u in self.units.values() if u.name == "staticcon" and u.is_finished)),
            str(sum(1 for u in self.units.values() if units.is_defense(u.data) and not u.is_mobile and u.is_finished)),
            str(sum(1 for u in self.units.values() if u.name == "staticradar" and u.is_finished)),
            str(len(self.enemies)),
            str(sum(1 for s in self.map.spots if s.status == "enemy")),
            str(len(self.tasks)),
            self.army_mode() if self.army else "none",
            str(getattr(self, "builder_flee_count", 0)),
            str(getattr(self, "builder_death_count", 0)),
            f"{getattr(self, 'own_lost_value', 0.0):.0f}",
            f"{getattr(self, 'enemy_killed_value', 0.0):.0f}",
        )
        try:
            f.write(",".join(row) + "\n")
        except Exception:
            pass

    def builder_watchdog(self) -> None:
        """Flag constructors that have had no job for a long time while
        unassigned jobs exist.  A healthy bot should never print this."""
        since = getattr(self, "_builder_free_since", {})
        self._builder_free_since = since
        open_tasks = [t for t in self.tasks.values() if not t.assignees and not t.unplaceable]
        for u in self.units.values():
            if not (u.is_builder and u.is_mobile and u.is_finished):
                continue
            if self.builder_task.get(u.uid) in self.tasks:
                since.pop(u.uid, None)
                continue
            start = since.setdefault(u.uid, self.frame)
            if open_tasks and self.frame - start > 1800:
                self.log(
                    f"WATCHDOG builder {u.name}#{u.uid} without a job for "
                    f"{(self.frame - start) / 30:.0f}s at {tuple(round(v) for v in u.pos)}; "
                    f"open tasks={len(open_tasks)} fallback_until={self.builder_fallback_until.get(u.uid)}"
                )
                since[u.uid] = self.frame
        for uid in list(since):
            if uid not in self.units:
                since.pop(uid, None)
