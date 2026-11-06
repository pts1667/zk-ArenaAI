"""The simulator as an environment for tools/policy_train.py, with the same interface as the toy's.

For now this is the binding (T-M1): two mirrored armies march at each other from far-apart metal spots.  The kinds it obeys are
the four the sim has, until S-M7 adds the rest and the proper army game of 03-training.md section 3 replaces this start.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim" / "py"))
sys.path.insert(0, str(ROOT / "bots"))
import zksim  # noqa: E402
from zkbot import catalog as bot_catalog  # noqa: E402
from zkbot.policy import bundle as bd, rawframe as rf, tables  # noqa: E402

KINDS = (rf.KEEP, rf.MOVE, rf.FIGHT, rf.ATTACK_UNIT, rf.MANUAL_FIRE)
TICKS_PER_DECISION = 3
TIME_LIMIT_DECISIONS = 2400         # 20 minutes
VALUE_CONSTANT = 1000.0
SCRIPT_EVERY = 6                    # decisions: as often as the policy's commit lock lets it re-order
SPREAD = 220.0
MOST_COST = 600.0
EARLY_END_DECISIONS = 240           # 2 minutes: from then on, a side this far ahead in metal value has won
EARLY_END_RATIO = 2.5
# 03-training.md section 3.  Other exported maps stay out of training: playing pads every seat to the map with the most regions.
TRAINING_MAPS = ("agalta_v1_0", "faultline_v1_0", "frostycove_v1_13", "red_comet_remake_1_7", "titanduel_2_2")

_POOL: list[tuple[zksim.Map, bd.Bundle, tuple[float, float, float, float]]] = []
_CATALOG: zksim.Catalog | None = None
_TYPES: dict[str, np.ndarray] = {}


def catalog() -> zksim.Catalog:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = zksim.Catalog()
    return _CATALOG


def _starts(bundle: bd.Bundle) -> tuple[float, float, float, float]:
    """The two spots farthest apart, or opposite corners on a map with too few."""
    spots = bundle.spot_list
    if len(spots) < 2:
        return bundle.map_w * 0.1, bundle.map_h * 0.1, bundle.map_w * 0.9, bundle.map_h * 0.9
    d = np.hypot(spots[:, None, 0] - spots[None, :, 0], spots[:, None, 1] - spots[None, :, 1])
    a, b = np.unravel_index(int(d.argmax()), d.shape)
    return float(spots[a, 0]), float(spots[a, 1]), float(spots[b, 0]), float(spots[b, 1])


def pool() -> list[tuple[zksim.Map, bd.Bundle, tuple[float, float, float, float]]]:
    """The training maps, exported by tools/sim_data.py maps.  One object per map, so forked workers can key on it."""
    if not _POOL:
        for slug in TRAINING_MAPS:
            bundle = bd.Bundle.load(zksim.DATA / "maps" / f"{slug}.bundle.npz")
            _POOL.append((zksim.Map.load(zksim.DATA / "maps" / f"{slug}.npz"), bundle, _starts(bundle)))
    return _POOL


def army_types(table: tables.TypeTable) -> dict[str, np.ndarray]:
    """Ground and water fighters of ordinary cost, by the data rather than by name.  Ships need a sea this start doesn't look for."""
    if not _TYPES:
        cat = catalog()
        # The factory roster of 00-overview, not every def there is: no chickens, and no test units.
        built = {n for r, f in enumerate(table.names) if table.factory[r] for n in bot_catalog.roster(f)}
        rows = [r for r, name in enumerate(table.names) if name in cat.index and name in built and table.combat[r] and table.mobile[r] and not table.air[r]
                and table.can_hit_ground[r] and table.cost[r] <= MOST_COST and table.move_class[r] != bd.MOVE_CLASSES.index("ship") and modelled(cat, name)]
        _TYPES["rows"] = np.array(rows)
        _TYPES["index"] = np.array([cat.index[table.names[r]] for r in rows], np.int32)
        _TYPES["classes"] = table.move_class[rows].astype(np.int32)
    return _TYPES


def modelled(cat: zksim.Catalog, name: str) -> bool:
    """Not a unit whose point is something the sim doesn't do yet (capture, mines, pushing, throwing)."""
    return not any(cat.weapons[w]["flags"] & zksim.W_UNMODELLED for w in cat.units[cat.index[name]]["mount_weapon"])


def place(bundle: bd.Bundle, rng: np.random.Generator, classes: np.ndarray, cx: float, cz: float) -> tuple[np.ndarray, np.ndarray]:
    """Round a point, each unit somewhere its class can stand."""
    n = len(classes)
    x, z = np.full(n, cx, np.float32), np.full(n, cz, np.float32)
    todo = np.arange(n)
    rows, cols = bundle.passable.shape
    for _ in range(12):
        if not len(todo):
            break
        tx = np.clip(cx + rng.normal(0.0, SPREAD, len(todo)), 8.0, bundle.map_w - 8.0)
        tz = np.clip(cz + rng.normal(0.0, SPREAD, len(todo)), 8.0, bundle.map_h - 8.0)
        i = np.minimum((tz / bd.PASS_CELL).astype(np.int32), rows - 1)
        j = np.minimum((tx / bd.PASS_CELL).astype(np.int32), cols - 1)
        good = (bundle.passable[i, j] >> classes[todo]) & 1 == 1
        x[todo[good]], z[todo[good]] = tx[good], tz[good]
        todo = todo[~good]
    return x, z


class SimEnv:
    eval_opponents = ("fight",)

    def __init__(self, games: int, seed: int, modes: list[str], units: int = 40, threads: int = 16, decide_every: int = 1):
        self.count, self.modes, self.units = games, modes, units
        # In slots of TICKS_PER_DECISION.  Game time stays what it was: the time limit, and how often the script re-orders.
        self.ticks = TICKS_PER_DECISION * decide_every
        self.time_limit = TIME_LIMIT_DECISIONS // decide_every
        self.early_end = EARLY_END_DECISIONS // decide_every
        self.script_every = max(1, SCRIPT_EVERY // decide_every)
        self.rng = np.random.default_rng(seed)
        self.seed_stream = np.random.SeedSequence(seed)
        self.batch = zksim.Batch(catalog(), games, threads=threads)
        self.table = tables.table()
        self.seat_count = games * 2
        self.max_regions = max(b.regions for _, b, _ in pool())
        self.mode = [modes[0]] * games
        self.bundle = [pool()[0][1]] * games
        self.starts = np.zeros((games, 4), np.float32)
        self.decisions = np.zeros(games, np.int32)
        self.last_raw = np.zeros(games)
        self.delta = np.zeros(games)
        self.result = np.zeros(games)
        self.pending: dict[int, list] = {}
        self.done: list[tuple[int, str, float]] = []
        self._replace: list[int] = []

    def bundles(self) -> list[bd.Bundle]:
        return [b for _, b, _ in pool()]

    def begin(self) -> None:
        for g in range(self.count):
            self._reset(g)
        self.batch.step(ticks=0)
        self.last_raw[:] = self._raw()

    def _reset(self, g: int) -> None:
        world_map, bundle, start = pool()[int(self.rng.integers(len(pool())))]
        types = army_types(self.table)
        self.mode[g] = self.modes[int(self.rng.integers(len(self.modes)))]
        pick = self.rng.integers(0, len(types["index"]), self.units)
        x0, z0, x1, z1 = start if self.rng.random() < 0.5 else (start[2], start[3], start[0], start[1])
        ax, az = place(bundle, self.rng, types["classes"][pick], x0, z0)
        bx, bz = place(bundle, self.rng, types["classes"][pick], x1, z1)
        self.batch.reset(g, world_map, int(self.rng.integers(1 << 62)), np.tile(types["index"][pick], 2), np.repeat([0, 1], self.units),
                         np.concatenate((ax, bx)), np.concatenate((az, bz)), base=(x0, z0, x1, z1), sites=bundle.sites())
        self.bundle[g] = bundle
        self.starts[g] = (x0, z0, x1, z1)
        self.decisions[g] = 0

    def _is_policy(self, seat: int) -> bool:
        return seat % 2 == 0 or self.mode[seat // 2] == "mirror"

    def policy_seats(self) -> list[int]:
        return list(range(self.seat_count))

    def policy_seats_to_act(self) -> list[int]:
        return [s for s in range(self.seat_count) if self._is_policy(s)]

    def bundle_of(self, seat: int) -> bd.Bundle:
        return self.bundle[seat // 2]

    def frame(self, seat: int) -> rf.Frame:
        return self.batch.frame(seat // 2, seat % 2)

    def give_orders(self, seat: int, orders: list) -> None:
        self.pending[seat] = orders

    def script_orders(self, g: int, side: int) -> np.ndarray:
        """Everyone fights towards the enemy's start."""
        if self.decisions[g] % self.script_every:
            return np.zeros(0, zksim.ORDER_DTYPE)
        units = self.batch.frame(g, side).units
        own = units[units["player"] == side]["uid"]
        out = np.zeros(len(own), zksim.ORDER_DTYPE)
        out["unit"], out["kind"], out["target"] = own, rf.FIGHT, -1
        out["x"], out["z"] = self.starts[g, 2 - 2 * side], self.starts[g, 3 - 2 * side]
        return out

    def step_scripted_only(self) -> None:
        self.step(0.0)

    def _raw(self) -> np.ndarray:
        a, b = self.batch.stats[:, 0, 0], self.batch.stats[:, 1, 0]
        return (a - b) / (a + b + VALUE_CONSTANT)

    def step(self, progress: float) -> None:
        parts, offsets = [], [0]
        for g in range(self.count):
            total = 0
            for side in (0, 1):
                seat = g * 2 + side
                orders = self.pending.pop(seat, None)
                if orders is not None:
                    part = np.array([(o.uid, o.kind, o.x, o.z, o.target) for o in orders], zksim.ORDER_DTYPE)
                elif not self._is_policy(seat):
                    part = self.script_orders(g, side)
                else:
                    continue
                parts.append(part)
                total += len(part)
            offsets.append(offsets[-1] + total)
        all_orders = np.concatenate(parts) if parts else np.zeros(0, zksim.ORDER_DTYPE)
        self.batch.step(all_orders, np.array(offsets, np.int32), ticks=self.ticks)
        self.decisions += 1
        raw = self._raw()
        self.delta = raw - self.last_raw
        self.last_raw = raw
        self.done, self._replace = [], []
        self.result[:] = 0.0
        alive_a, alive_b = self.batch.stats[:, 0, 1] > 0, self.batch.stats[:, 1, 1] > 0
        value_a, value_b = self.batch.stats[:, 0, 0], self.batch.stats[:, 1, 0]
        late = self.decisions >= self.early_end
        wins_a = (alive_a & ~alive_b) | (late & (value_a > EARLY_END_RATIO * value_b))
        wins_b = (alive_b & ~alive_a) | (late & (value_b > EARLY_END_RATIO * value_a))
        for g in np.flatnonzero(~(alive_a & alive_b) | wins_a | wins_b | (self.decisions >= self.time_limit)):
            g = int(g)
            won_a, won_b = bool(wins_a[g]), bool(wins_b[g])
            self.result[g] = 1.0 if won_a else (-1.0 if won_b else 0.0)
            for side, won, lost in ((0, won_a, won_b), (1, won_b, won_a)):
                seat = g * 2 + side
                if self._is_policy(seat):
                    self.done.append((seat, self.mode[g], 1.0 if won else (0.0 if lost else 0.5)))
            self._replace.append(g)

    def reward(self, seat: int, beta: float) -> float:
        g = seat // 2
        sign = 1.0 if seat % 2 == 0 else -1.0
        return float(sign * ((1.0 - beta) * self.delta[g] + beta * self.result[g]))

    def finished_seats(self) -> list[tuple[int, str, float]]:
        """Seats whose game just ended.  The call also starts those games afresh, so read the reward first."""
        out = self.done
        for g in self._replace:
            self._reset(g)
            self.batch.refresh(g)
            self.last_raw[g] = self._raw()[g]
        self._replace, self.done = [], []
        return out
