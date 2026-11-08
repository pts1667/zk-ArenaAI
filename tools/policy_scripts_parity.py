"""Order-by-order parity of the C++ scripts (policy_scripts_cpp.py) with the Python ones they port.

    ~/zkml/bin/python tools/policy_scripts_parity.py --games 16 --decisions 2400 [--seed 1] [--threads 8]

A `SimEnv` plays its games with the Python scripts.  At every decision, the C++ scripts are given the same frames, the same
decision count, the same seats and the same commander-busy flags, just before the sim steps, and what each says is compared,
seat by seat, with what the Python said.  The C++ state is its own: it's reset when the Python's is and then only ever sees its
own past orders, so a difference can carry over to later decisions (a "cascade"), which the report counts apart.

Random numbers: each Python economy draws from a recording wrapper of its generator (`TapeRng`), and the C++ economy of the same
seat is handed those draws for that decision (`Scripts.tape`), so draws are no source of difference while both make the same
calls.  A C++ draw of the wrong kind, or one too many or too few, is a "tape miss": it shows the two took different paths.
Calls that drew nothing are also reported apart.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import policy_sim_env as se  # noqa: E402
import policy_scripts_cpp as psc  # noqa: E402
import zksim  # noqa: E402
from policy_eco import ECO_EVERY  # noqa: E402

KIND_NAMES = {1: "move", 2: "fight", 3: "attack", 6: "manual_fire", 10: "build", 11: "assist", 14: "produce"}


class TapeRng:
    """A numpy Generator that keeps what it draws, as the scripts call it: random() and normal(loc, scale), one value or many."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.values: list[float] = []
        self.kinds: list[int] = []

    def random(self, size=None):
        v = self.rng.random(size)
        if size is None:
            self.values.append(float(v))
            self.kinds.append(0)
        else:
            self.values += np.ravel(v).tolist()
            self.kinds += [0] * int(np.size(v))
        return v

    def normal(self, loc=0.0, scale=1.0, size=None):
        assert size is None, "the scripts draw normals one at a time"
        z = float(self.rng.standard_normal())       # what Generator.normal does: loc + scale * a standard normal
        self.values.append(z)
        self.kinds.append(1)
        return loc + scale * z

    def take(self) -> tuple[list[float], list[int]]:
        out = (self.values, self.kinds)
        self.values, self.kinds = [], []
        return out


def _key(o) -> tuple:
    return int(o["unit"]), int(o["kind"]), float(o["x"]), float(o["z"]), int(o["target"])


class Parity:
    """Hooks a Python-script `SimEnv` so that a C++ runner sees what its scripts see, and tallies how their orders compare."""

    def __init__(self, env: se.SimEnv, examples: int = 20):
        assert env.cpp is None, "the environment plays the Python scripts; the C++ ones shadow them"
        self.env = env
        self.cpp = psc.Scripts(psc.ScriptsData(env.script_data, env.eco_data), env.count, threads=1)
        self.examples_left = examples
        self.examples: list[dict] = []
        self.stats = {part: Counter() for part in ("army", "eco")}
        self.diff_kinds = {part: Counter() for part in ("army", "eco")}
        self.coverage = {part: Counter() for part in ("army", "eco")}     # the Python orders compared, by kind and what for
        self.diverged: set[tuple[int, int, str]] = set()      # (seat, game serial, part) that have differed once
        self.serial = np.zeros(env.count, np.int64)
        self.misses = 0
        self._py_army: dict[int, np.ndarray] = {}
        self._py_eco: dict[int, np.ndarray] = {}
        self._busy = np.full(2 * env.count, -1, np.int32)
        self._tapes: dict[int, tuple[list[float], list[int]]] = {}
        start, script_orders, economy_orders, step = env._start_commanders, env.script_orders, env._economy_orders, env.batch.step

        def start_commanders(g, world_map, bundle, box0, box1, x0, z0, x1, z1):
            start(g, world_map, bundle, box0, box1, x0, z0, x1, z1)
            seats = env.seats[g]
            if env.economies[g] is not None:
                for e in env.economies[g]:
                    e.rng = TapeRng(e.rng)
            self.cpp.reset(g, env.grounds[id(bundle)], (x0, z0, x1, z1), (*seats[0].plan.rally, *seats[1].plan.rally),
                           economy=env.economies[g] is not None, aimer=True)
            self.serial[g] += 1

        def scripts(g, side):
            out = script_orders(g, side)
            if env.decisions[g] % env.script_every == 0:
                self._py_army[g * 2 + side] = out
            return out

        def economy(g, side):
            seats = env.seats[g]
            self._busy[g * 2 + side] = side if seats is not None and not seats[side].done else -1
            out = economy_orders(g, side)
            if env.economies[g] is not None and env.decisions[g] % ECO_EVERY == 0:
                self._py_eco[g * 2 + side] = out if out is not None else np.zeros(0, zksim.ORDER_DTYPE)
                self._tapes[g * 2 + side] = env.economies[g][side].rng.take()
            return out

        def batch_step(orders=None, order_offsets=None, games=None, ticks=3):
            self._compare()
            step(orders, order_offsets, games, ticks)

        env._start_commanders, env.script_orders, env._economy_orders, env.batch.step = start_commanders, scripts, economy, batch_step

    def _compare(self) -> None:
        if not self._py_army and not self._py_eco:
            return      # SimEnv.begin's first look, or a decision where no script runs
        env, n = self.env, self.env.count
        flags = np.zeros(2 * n, np.int32)
        for seat in self._py_army:
            flags[seat] = psc.SEAT_ARMY
        for seat, (values, kinds) in self._tapes.items():
            self.cpp.tape(seat // 2, seat % 2, values, kinds)
        before = sum(self.cpp.tape_misses(s // 2, s % 2) for s in range(2 * n))
        orders, offsets, army, army_offsets = self.cpp.step(env.batch, env.decisions, flags, self._busy, env.ticks, env.script_every, ECO_EVERY)
        self.misses += sum(self.cpp.tape_misses(s // 2, s % 2) for s in range(2 * n)) - before
        parts = self.cpp.seat_counts(n)
        for seat in range(2 * n):
            g = seat // 2
            start = int(offsets[g]) + (int(parts[seat - 1].sum()) if seat % 2 else 0)
            mine_army = army[army_offsets[seat]:army_offsets[seat + 1]]
            mine_eco = orders[start + parts[seat, 0] + parts[seat, 1] + parts[seat, 2]:start + parts[seat].sum()]
            if seat in self._py_army:
                self._tally("army", seat, self._py_army[seat], mine_army)
            if seat in self._py_eco:
                drew = bool(self._tapes[seat][0])
                self._tally("eco", seat, self._py_eco[seat], mine_eco, drew)
        self._py_army, self._py_eco, self._tapes = {}, {}, {}

    def _tally(self, part: str, seat: int, py: np.ndarray, cpp: np.ndarray, drew: bool | None = None) -> None:
        s = self.stats[part]
        key = (seat, int(self.serial[seat // 2]), part)
        cascade = key in self.diverged
        a, b = [_key(o) for o in py], [_key(o) for o in cpp]
        same = a == b
        matched = sum((Counter(a) & Counter(b)).values())
        near = sum((Counter(k[:2] + k[4:] for k in a) & Counter(k[:2] + k[4:] for k in b)).values())
        tag = "after" if cascade else "before"
        s["calls"] += 1
        s["calls_identical"] += same
        s[f"calls_{tag}"] += 1
        s[f"calls_identical_{tag}"] += same
        s["orders_py"] += len(a)
        s["orders_cpp"] += len(b)
        s["orders_identical"] += matched
        s[f"orders_py_{tag}"] += len(a)
        s[f"orders_identical_{tag}"] += matched
        s["orders_same_unit_kind_target"] += near
        names = se.catalog().units
        for k in a:
            what = KIND_NAMES.get(k[1], str(k[1]))
            if k[1] in (10, 14):        # what's built or made
                what += " " + names[k[4]]["name"]
            elif k[1] in (3, 6) and part == "army":
                own = self.env.frame(seat).units
                row = own["def_id"][own["uid"] == k[0]]
                what += " by " + (names[self.env.script_data.catalog_row[self.env.table.rows(row)[0]]]["name"] if len(row) else "?")
            self.coverage[part][what] += 1
        if drew is not None:
            d = "drew" if drew else "no_draw"
            s[f"calls_{d}"] += 1
            s[f"calls_identical_{d}"] += same
            s[f"orders_py_{d}"] += len(a)
            s[f"orders_identical_{d}"] += matched
        if not same:
            for k in (Counter(a) - Counter(b)) + (Counter(b) - Counter(a)):
                self.diff_kinds[part][f"{tag}:{KIND_NAMES.get(k[1], k[1])}"] += 1
            if not cascade:
                self.diverged.add(key)
                s["first_differences"] += 1
                if self.examples_left > 0:
                    self.examples_left -= 1
                    self.examples.append({"part": part, "seat": seat, "decision": int(self.env.decisions[seat // 2]), "drew": drew,
                                          "python": [list(k) for k in a if k not in b][:6], "cpp": [list(k) for k in b if k not in a][:6]})

    def report(self) -> dict:
        out = {"tape_misses": self.misses}
        for part, s in self.stats.items():
            r = dict(s)
            r["share_identical_orders"] = s["orders_identical"] / max(s["orders_py"], 1)
            r["share_identical_calls"] = s["calls_identical"] / max(s["calls"], 1)
            r["share_identical_orders_before_first_difference"] = s["orders_identical_before"] / max(s["orders_py_before"], 1)
            r["differences_by_kind"] = dict(self.diff_kinds[part].most_common())
            r["orders_compared_by_kind"] = dict(self.coverage[part].most_common())
            out[part] = r
        out["examples"] = self.examples
        return out


def run(games: int, seed: int, decisions: int, threads: int = 8, examples: int = 20) -> dict:
    env = se.SimEnv(games, seed, ["scripts"], threads=threads, script_impl="python")
    parity = Parity(env, examples)
    env.begin()
    for _ in range(decisions):
        env.step(0.0)
        env.finished_seats()
    return parity.report()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--decisions", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--examples", type=int, default=20)
    args = ap.parse_args()
    print(json.dumps(run(args.games, args.seed, args.decisions, args.threads, args.examples), indent=1))


if __name__ == "__main__":
    main()
