"""The toy for the PPO core (T-M0): 8 against 8 on an open plain, with the real raw frame, featurizer and network.

It has no terrain and no fog, and damage lands at once.  What it does have is a known answer: shooting one enemy at a time ends
fights sooner than each unit shooting its nearest, so a working learner must find focus fire.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bots"))
from zkbot import terrain, units as unit_data  # noqa: E402
from zkbot.policy import bundle as bd, featurize as fz, rawframe as rf, tables  # noqa: E402

SIZE = 2048.0
UNITS = 8
TYPE = "cloakskirm"
RANGE = 420.0
LINE_GAP = 340.0
DAMAGE_PER_SECOND = 40.0
TICK_SECONDS = 5.0 / 30.0
DECISION_TICKS = 3
TIME_LIMIT_DECISIONS = 240
VALUE_CONSTANT = 1000.0

_BUNDLE: bd.Bundle | None = None


def plain() -> bd.Bundle:
    global _BUNDLE
    if _BUNDLE is None:
        n = int(SIZE / terrain.SQUARE)
        _BUNDLE = bd.from_terrain(terrain.TerrainMap(np.full((n, n), 50.0, np.float32)))
    return _BUNDLE


class ToyGame:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        table = tables.table()
        row = table.names.index(TYPE)
        self.def_id = unit_data.id_of(TYPE)
        self.hp_max = float(unit_data.by_name(TYPE)["health"])
        self.speed = float(table.speed[row]) / 30.0
        self.cost = float(table.cost[row])
        self.bundle = plain()
        n = UNITS * 2
        self.side = np.repeat([0, 1], UNITS)
        along = np.tile(np.arange(UNITS) * 60.0 - 210.0, 2) + rng.normal(0.0, 12.0, n)
        across = np.where(self.side == 0, -LINE_GAP / 2.0, LINE_GAP / 2.0) + rng.normal(0.0, 12.0, n)
        self.x = SIZE / 2.0 + along
        self.z = SIZE / 2.0 + across
        self.hp = np.full(n, self.hp_max)
        self.target = np.full(n, -1)         # uid it was told to attack
        self.goal = np.full((n, 2), np.nan)
        self.order_kind = np.full(n, -1)
        self.order_frame = np.zeros(n, np.int64)
        self.damaged = np.zeros(n, bool)
        self.vx = np.zeros(n)
        self.vz = np.zeros(n)
        self.frame = 0
        self.decisions = 0
        self.start_raw = self.raw_value()

    @property
    def alive(self) -> np.ndarray:
        return self.hp > 0.0

    def raw_value(self) -> float:
        a = self.cost * float(self.alive[self.side == 0].sum())
        b = self.cost * float(self.alive[self.side == 1].sum())
        return (a - b) / (a + b + VALUE_CONSTANT)

    def winner(self) -> int | None:
        """0 or 1, 2 for a draw, or None while it's going."""
        a, b = self.alive[self.side == 0].any(), self.alive[self.side == 1].any()
        if a and b:
            return 2 if self.decisions >= TIME_LIMIT_DECISIONS else None
        return 0 if a else (1 if b else 2)

    def frame_for(self, seat: int) -> rf.Frame:
        live = np.flatnonzero(self.alive)
        out = np.zeros(len(live), rf.UNIT_DTYPE)
        out["uid"] = live + 1
        out["def_id"] = self.def_id
        out["x"], out["z"], out["y"] = self.x[live], self.z[live], 50.0
        out["vx"], out["vz"] = self.vx[live], self.vz[live]
        out["hp_frac"] = self.hp[live] / self.hp_max
        out["build_frac"] = 1.0
        mine = self.side[live] == seat
        out["player"], out["ally_team"] = self.side[live], self.side[live]
        flags = np.full(len(live), rf.IN_LOS | rf.TYPE_KNOWN, np.uint16)
        idle = self.order_kind[live] < 0
        flags[mine & idle] |= rf.IDLE
        flags[mine & self.damaged[live]] |= rf.DAMAGED
        out["flags"] = flags
        kinds = self.order_kind[live]
        out["ord_kind"] = np.where(mine, kinds, -1)
        out["ord_age"] = np.where(mine & (kinds >= 0), self.frame - self.order_frame[live], 0)
        goals = np.nan_to_num(self.goal[live])
        out["ord_x"], out["ord_z"] = np.where(mine, goals[:, 0], 0.0), np.where(mine, goals[:, 1], 0.0)
        out["ord_target"] = np.where(mine & (self.target[live] >= 0), self.target[live], -1)
        glob = np.zeros(len(rf.GLOBAL_FIELDS), np.float32)
        glob[rf.G["frame"]] = self.frame
        glob[rf.G["map_w"]], glob[rf.G["map_h"]] = SIZE, SIZE
        glob[rf.G["commander_uid"]] = -1
        glob[rf.G["player"]], glob[rf.G["ally_team"]] = seat, seat
        glob[rf.G["base_x"]] = SIZE / 2.0
        glob[rf.G["base_z"]] = SIZE / 2.0 + (LINE_GAP if seat else -LINE_GAP)
        return rf.Frame(out, np.zeros(0, rf.SITE_DTYPE), glob)

    def apply(self, seat: int, orders: list) -> None:
        for o in orders:
            u = int(o.uid) - 1
            if u < 0 or not self.alive[u] or self.side[u] != seat:
                continue
            self.order_kind[u] = o.kind
            self.order_frame[u] = self.frame
            self.target[u] = o.target if o.kind == rf.ATTACK_UNIT else -1
            self.goal[u] = (o.x, o.z) if o.kind in (rf.MOVE, rf.FIGHT, rf.ATTACK_GROUND, rf.JUMP) else (np.nan, np.nan)
            if o.kind == rf.STOP:
                self.order_kind[u] = -1

    def distances(self) -> np.ndarray:
        return np.hypot(self.x[:, None] - self.x[None, :], self.z[:, None] - self.z[None, :])

    def nearest_enemies(self, d: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Per unit, the index of the nearest live enemy and how far it is, inf if there is none."""
        d = self.distances() if d is None else d
        apart = np.where((self.side[:, None] != self.side[None, :]) & self.alive[None, :], d, np.inf)
        near = apart.argmin(axis=1)
        return near, apart[np.arange(len(near)), near]

    def step(self) -> None:
        self.damaged[:] = False
        for _ in range(DECISION_TICKS):
            self._tick()
        self.frame += DECISION_TICKS * 5
        self.decisions += 1

    def _tick(self) -> None:
        """Everything is decided from where units stood at the start of the tick, so neither side moves first."""
        n = len(self.hp)
        every = np.arange(n)
        alive = self.alive
        d = self.distances()
        near, near_d = self.nearest_enemies(d)
        shot = np.full(n, -1)
        walk_x, walk_z = np.full(n, np.nan), np.full(n, np.nan)

        attacking = alive & (self.order_kind == rf.ATTACK_UNIT)
        target = np.where(self.target > 0, self.target - 1, 0)
        valid = attacking & (self.target > 0) & alive[target]
        self.order_kind[attacking & ~valid] = -1        # the target's gone: idle
        reach = valid & (d[every, target] <= RANGE)
        shot[reach] = target[reach]
        chase = valid & ~reach
        walk_x[chase], walk_z[chase] = self.x[target[chase]], self.z[target[chase]]

        going = alive & np.isin(self.order_kind, (rf.MOVE, rf.FIGHT, rf.ATTACK_GROUND))
        left = np.hypot(self.goal[:, 0] - self.x, self.goal[:, 1] - self.z)
        arriving = going & (left > 24.0)
        walk_x[arriving], walk_z[arriving] = self.goal[arriving, 0], self.goal[arriving, 1]
        self.order_kind[going & ~arriving] = -1

        # Every unit left to itself shoots its nearest enemy in range.
        free = alive & (shot < 0) & (near_d <= RANGE)
        shot[free] = near[free]

        self.vx[:], self.vz[:] = 0.0, 0.0
        walking = np.flatnonzero(~np.isnan(walk_x))
        if len(walking):
            dx, dz = walk_x[walking] - self.x[walking], walk_z[walking] - self.z[walking]
            length = np.maximum(np.hypot(dx, dz), 1e-6)
            self.vx[walking], self.vz[walking] = dx / length * self.speed, dz / length * self.speed
            self.x[walking] = np.clip(self.x[walking] + self.vx[walking] * 5.0, 0.0, SIZE - 1.0)
            self.z[walking] = np.clip(self.z[walking] + self.vz[walking] * 5.0, 0.0, SIZE - 1.0)

        firing = shot[shot >= 0]
        damage = np.bincount(firing, minlength=n) * DAMAGE_PER_SECOND * TICK_SECONDS
        self.damaged |= damage > 0.0
        self.hp = np.maximum(self.hp - damage, 0.0)


def _order(uid: int, target: int) -> object:
    from zkbot.policy.actions import Order

    return Order(uid, rf.ATTACK_UNIT, 0.0, 0.0, target)


def script_nearest(game: ToyGame, seat: int, rng: np.random.Generator) -> list:
    """Each unit attacks its nearest enemy."""
    near, far = game.nearest_enemies()
    return [_order(int(u) + 1, int(near[u]) + 1) for u in np.flatnonzero(game.alive & (game.side == seat) & np.isfinite(far))]


def script_random(game: ToyGame, seat: int, rng: np.random.Generator) -> list:
    """Each unit, when free to, picks an enemy at random or walks somewhere."""
    from zkbot.policy.actions import Order

    orders = []
    enemies = np.flatnonzero(game.alive & (game.side != seat))
    for u in np.flatnonzero(game.alive & (game.side == seat)):
        if not len(enemies):
            break
        if game.order_kind[u] >= 0 and rng.random() < 0.7:
            continue
        if rng.random() < 0.5:
            orders.append(_order(int(u) + 1, int(rng.choice(enemies)) + 1))
        else:
            orders.append(Order(int(u) + 1, rf.MOVE, float(rng.uniform(0, SIZE)), float(rng.uniform(0, SIZE))))
    return orders


def script_focus(game: ToyGame, seat: int, rng: np.random.Generator) -> list:
    """The known answer: everyone shoots the enemy nearest our centre, one at a time."""
    mine = np.flatnonzero(game.alive & (game.side == seat))
    enemies = np.flatnonzero(game.alive & (game.side != seat))
    if not len(mine) or not len(enemies):
        return []
    cx, cz = game.x[mine].mean(), game.z[mine].mean()
    t = enemies[np.argmin((game.x[enemies] - cx) ** 2 + (game.z[enemies] - cz) ** 2)]
    return [_order(int(u) + 1, int(t) + 1) for u in mine]


SCRIPTS = {"nearest": script_nearest, "random": script_random, "focus": script_focus}


def play_scripts(a: str, b: str, games: int, seed: int) -> dict[str, int]:
    """To check that the toy has the answer it claims: `focus` must beat `nearest`."""
    out = {"a": 0, "b": 0, "draw": 0}
    for g in range(games):
        rng = np.random.default_rng(seed + g)
        game = ToyGame(rng)
        while game.winner() is None:
            game.apply(0, SCRIPTS[a](game, 0, rng))
            game.apply(1, SCRIPTS[b](game, 1, rng))
            game.step()
        out[{0: "a", 1: "b", 2: "draw"}[game.winner()]] += 1
    return out


def featurized_sizes() -> tuple[int, int]:
    t = fz.featurize(ToyGame(np.random.default_rng(0)).frame_for(0), plain(), tables.table())
    return len(t.tok_kind), plain().regions


if __name__ == "__main__":
    print("tokens, regions:", featurized_sizes())
    for a, b in (("focus", "nearest"), ("nearest", "random"), ("focus", "random"), ("nearest", "nearest")):
        print(a, "vs", b, play_scripts(a, b, 1000, 1))
