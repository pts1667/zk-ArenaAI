"""What the policy knows about unit types, read from catalog data.  Nothing here branches on a unit's name.

Row T of each array is the unknown type (a blip not identified, or a type that isn't in the catalog): all zeros.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import numpy as np

from .. import catalog, units
from .bundle import MOVE_CLASSES

ROLES = ("raid", "scout", "riot", "skirm", "assault", "arty", "snipe", "aa", "antiheavy", "bomb", "builder", "static", "other")
_DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
TAGS = ("area_cloak", "capture", "cloak", "disarm", "drones", "emp", "fire", "gravity", "jammer", "jump", "kamikaze", "manualfire",
        "morph", "radar", "regen", "shield", "slow", "stealth", "stockpile", "teleport", "torpedo", "transport", "underwater")
TYPE_DIM = 16 + len(ROLES) + len(MOVE_CLASSES) + len(TAGS)
NO_CLASS = -1


class TypeTable:
    def __init__(self) -> None:
        names = sorted(units.BY_NAME)
        self.names = tuple(names)
        n = len(names)
        moves = json.loads((_DATA / "moveclass.json").read_text("utf8"))
        self.jump_range = np.zeros(n + 1, np.float32)
        for name, reach in json.loads((_DATA / "jumprange.json").read_text("utf8")).items():
            if name in names:
                self.jump_range[names.index(name)] = reach
        self.stats = np.zeros((n + 1, TYPE_DIM), np.float32)
        self.cost = np.zeros(n + 1, np.float32)
        self.speed = np.zeros(n + 1, np.float32)         # elmos/s
        self.range = np.zeros(n + 1, np.float32)
        self.dps_ground = np.zeros(n + 1, np.float32)
        self.dps_air = np.zeros(n + 1, np.float32)
        self.manual_range = np.zeros(n + 1, np.float32)
        self.manual_reload = np.zeros(n + 1, np.float32)  # frames
        self.move_class = np.full(n + 1, NO_CLASS, np.int8)
        self.mobile = np.zeros(n + 1, bool)
        self.air = np.zeros(n + 1, bool)
        self.armed = np.zeros(n + 1, bool)
        self.can_hit_ground = np.zeros(n + 1, bool)
        self.can_hit_air = np.zeros(n + 1, bool)
        self.builder = np.zeros(n + 1, bool)
        self.factory = np.zeros(n + 1, bool)
        self.combat = np.zeros(n + 1, bool)
        self.manualfire = np.zeros(n + 1, bool)
        self.jump = np.zeros(n + 1, bool)
        self.cloak = np.zeros(n + 1, bool)
        for row, name in enumerate(names):
            self._fill(row, name, units.BY_NAME[name], moves.get(name))
        self._fill_specials(names)
        top = max(int(k) for k in units.BY_ID)
        self.row_of_def = np.full(top + 2, n, np.int32)   # the last entry stands for -1
        for def_id, d in units.BY_ID.items():
            self.row_of_def[int(def_id)] = names.index(d["name"])
        self.unknown = n
        self.hash = _catalog_hash()

    def _fill(self, row: int, name: str, d: dict, move: str | None) -> None:
        s = catalog.stats(name) or {}
        tags = catalog.tags(name)
        cost, hp = float(s.get("cost", 1.0)), float(s.get("hp", 0.0))
        dps_g, dps_a = float(s.get("dps_ground", 0.0)), float(s.get("dps_air", 0.0))
        rng, speed, aoe = float(s.get("range", 0.0)), float(s.get("speed", 0.0)), float(s.get("aoe", 0.0))
        mobile, air = bool(d.get("isMobile")), bool(d.get("isAir")) or bool(s.get("air"))
        armed = rng > 0.0 and not s.get("unarmed")
        builder, factory = units.is_builder(d), bool(d.get("isFactory"))
        manual = catalog.manual_weapon(name) or {}
        self.cost[row], self.speed[row], self.range[row] = cost, speed, rng
        self.dps_ground[row], self.dps_air[row] = dps_g, dps_a
        self.manual_range[row] = float(manual.get("range") or 0.0)
        self.manual_reload[row] = float(manual.get("reload") or 0.0) * 30.0
        self.move_class[row] = MOVE_CLASSES.index(move) if move in MOVE_CLASSES else NO_CLASS
        self.mobile[row], self.air[row], self.armed[row] = mobile, air, armed
        # Some weapons (torpedoes, gravity guns) show no DPS in the dump: an armed unit with none for air can hit the ground.
        self.can_hit_ground[row] = armed and (dps_g > 0.0 or dps_a <= 0.0)
        self.can_hit_air[row] = armed and dps_a > 0.0
        self.builder[row], self.factory[row], self.combat[row] = builder, factory, units.is_combat(d)
        self.manualfire[row], self.jump[row], self.cloak[row] = "manualfire" in tags, "jump" in tags, "cloak" in tags
        role = catalog.infer_role(name)
        v = [
            math.log1p(cost) / 8.0, math.log1p(hp) / 10.0, math.log1p(dps_g) / 7.0, math.log1p(dps_a) / 7.0,
            rng / 1000.0, speed / 150.0, aoe / 200.0,
            float(air), float(mobile), float(builder), float(factory), float(units.is_defense(d)), float(not armed),
            float(d.get("buildSpeed") or 0.0) / 20.0, float(d.get("buildDistance") or 0.0) / 500.0, float(d.get("losRadius") or 0.0) / 1000.0,
        ]
        v += [float(role == r) for r in ROLES]
        v += [float(move == c) for c in MOVE_CLASSES]
        v += [float(t in tags) for t in TAGS]
        self.stats[row] = v

    def _fill_specials(self, names: list[str]) -> None:
        n = len(names) + 1
        row = {name: r for r, name in enumerate(names)}
        special = catalog.SPECIALS
        self.throw_radius, self.throw_reach = np.zeros(n, np.float32), np.zeros(n, np.float32)
        for name, facts in special["throwers"].items():
            if name in row:
                self.throw_radius[row[name]], self.throw_reach[row[name]] = facts["radius"], facts["reach"]
        self.switches, self.teleports, self.beacon = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        self.switches[[row[name] for name in special["switches"] if name in row]] = True
        for name, beacon in special["teleporters"].items():
            if name in row:
                self.teleports[row[name]], self.beacon[row[beacon]] = True, True
        # What a PRODUCE can point at, and who can make which: a silo's missiles, and a factory's roster.  For a factory it says what to build from now on.
        makers = {row[name]: [row[o] for o in options if o in row] for name, options in special["silos"].items() if name in row}
        for name in names:
            if self.factory[row[name]]:
                roster = [row[o] for o in catalog.roster(name) if o in row]
                if roster:
                    makers[row[name]] = roster
        self.producible = np.array(sorted({o for options in makers.values() for o in options}), np.int64)
        self.produces = np.zeros((n, len(self.producible)), bool)
        for maker, options in makers.items():
            self.produces[maker, np.searchsorted(self.producible, options)] = True
        # Those the policy gives orders to though they aren't fighters that walk.
        self.acts_in_place = self.switches | self.produces.any(axis=1)

    def rows(self, def_ids: np.ndarray) -> np.ndarray:
        """Catalog row per def id; -1 and anything unheard of give the unknown row."""
        ids = np.where((def_ids < 0) | (def_ids >= len(self.row_of_def) - 1), len(self.row_of_def) - 1, def_ids)
        return self.row_of_def[ids]


def _catalog_hash() -> str:
    h = hashlib.sha256()
    for name in ("units.json", "weapons.json", "abilities.json", "moveclass.json", "jumprange.json", "specials.json"):
        h.update((_DATA / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


_TABLE: TypeTable | None = None


def table() -> TypeTable:
    global _TABLE
    if _TABLE is None:
        _TABLE = TypeTable()
    return _TABLE
