"""Where the sim's tick time goes, per phase, single thread, 400 units on Titan Duel.

In WSL, from the repo:
    cmake -S sim -B ~/zksim/build-prof -DZKSIM_PROFILE=ON && cmake --build ~/zksim/build-prof
    ~/zkml/bin/python tools/sim_profile.py
"""
from __future__ import annotations

import ctypes as ct
import os
import pathlib
import sys
import time

import numpy as np

os.environ.setdefault("ZKSIM_LIB", os.path.expanduser("~/zksim/build-prof/libzksim.so"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "sim" / "py"))
import zksim  # noqa: E402

UNITS = 400
TICKS = 200
ARMY = ("cloakraid", "cloakassault", "shieldraid", "tankassault", "spiderassault", "jumpraid", "hoverraid", "vehassault")
GROUP_GOALS = [(float(x), float(z)) for x in (1000, 2500, 4000) for z in (1000, 2500, 4000)]
GROUP_SIZE = 8


def two_armies(rng, side):
    return 1500 + 2000 * side + rng.uniform(-700, 700), 2500 + rng.uniform(-700, 700)


def whole_map(rng, _side):
    return rng.uniform(800, 4300), rng.uniform(800, 4300)


def shared_goal(_rng, unit):
    return 3500 - 2000 * (unit % 2), 2500


def group_goal(_rng, unit):
    return GROUP_GOALS[(unit // GROUP_SIZE) % len(GROUP_GOALS)]


def own_goal(rng, _unit):
    return rng.uniform(800, 4300), rng.uniform(800, 4300)


SCENARIOS = (
    ("fight orders, one goal per side", zksim.ORDERS, two_armies, shared_goal),
    (f"fight orders, {len(GROUP_GOALS)} group goals", zksim.ORDERS, two_armies, group_goal),
    ("fight orders, every unit its own goal", zksim.ORDERS, two_armies, own_goal),
    ("advance, two armies", zksim.ADVANCE, two_armies, None),
    ("advance, units over the whole map", zksim.ADVANCE, whole_map, None),
)


class Counters:
    def __init__(self):
        lib = zksim._lib
        try:
            self.size = lib.zk_prof_count()
        except AttributeError:
            sys.exit(f"{os.environ['ZKSIM_LIB']} wasn't built with -DZKSIM_PROFILE=ON")
        lib.zk_prof_name.restype = ct.c_char_p
        self.names = [lib.zk_prof_name(i).decode() for i in range(self.size)]
        self.nanos = (ct.c_longlong * self.size)()
        self.calls = (ct.c_longlong * self.size)()

    def take(self):
        zksim._lib.zk_prof_read(self.nanos, self.calls, 1)
        return list(self.nanos), list(self.calls)


def run(counters, cat, world_map, label, behaviour, place, goal):
    rng = np.random.default_rng(3)
    w = zksim.World(cat, world_map, behaviour=behaviour)
    units = []
    for i in range(UNITS):
        px, pz = place(rng, i % 2)
        units.append(w.add(ARMY[rng.integers(len(ARMY))], i % 2, px, pz))
    if goal:
        for i, u in enumerate(units):
            w.fight(u, *goal(rng, i))
    counters.take()
    t0 = time.perf_counter()
    w.step(TICKS)
    per_tick = (time.perf_counter() - t0) / TICKS * 1e6
    nanos, calls = counters.take()
    print(f"\n{label}: {per_tick:.0f} us per tick, {sum(w.get(u).alive for u in units)} of {UNITS} alive at the end")
    print(f"  {'phase':32s} {'us/tick':>9s} {'share':>6s} {'calls/tick':>11s}")
    for name, ns, c in zip(counters.names, nanos, calls):
        if c:
            print(f"  {name:32s} {ns / 1e3 / TICKS:9.1f} {ns / 10 / TICKS / per_tick:5.1f}% {c / TICKS:11.1f}")


def main() -> None:
    counters = Counters()
    cat = zksim.Catalog()
    world_map = zksim.Map.load(sys.argv[1] if len(sys.argv) > 1 else zksim.DATA / "maps/titanduel_2_2.npz")
    for label, behaviour, place, goal in SCENARIOS:
        run(counters, cat, world_map, label, behaviour, place, goal)


if __name__ == "__main__":
    main()
