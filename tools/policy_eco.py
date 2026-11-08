"""A side's economy and its big buildings, when nobody else runs its constructors: the economy script.

It runs for every side of a `SimEnv` game, the policy's as well, since the policy is given no constructors.  Its orders aren't
recorded as the script's (`SimEnv.script_record`): they aren't the policy's to give.  Constructors come from the factory: the army
script builds them (policy_mix.py), and so may the policy.  Once the opening is over the commander is one of them, though it takes
no extractor farther than COMMANDER_REACH from the base.  It has to be: builders are paid first come, first served, as in the
engine, and a commander helping the factory leaves nothing for the constructors.

Each idle constructor, every few seconds, takes the first of these it can do:

  finish    help with something of ours that's being built near it, or the big project wherever it is
  project   start the big project, if one's due and nobody has
  energy    first, while energy made is short of ENERGY_FIRST x metal income: building costs as much energy as metal, and
            overdrive only gets what's left, so short of energy, everything slows and metal piles up unspent
  expand    an extractor on the nearest free spot on our half that no enemy defence or army is near; while metal floats, only
            while fewer than FLOAT_EXPANDERS are on their way to one
  spend     while metal floats (FLOATING of storage, until FLOATING_UNTIL): energy if that's what's short, else another factory
            like our first, while income is past FACTORY_PER_INCOME for each, else help the nearest factory
  energy    then, while energy made is short of a share of metal income that grows over the game, for overdrive: the best small
            generator for its metal beside extractors, then fusions, then a Singularity, the big ones at the base and on its grid
  radar     a radar tower behind the front from RADAR_FROM, and a second once there's a silo: what a missile is aimed at has to be seen
  grid      a pylon from an extractor or turret that's on no grid with the base, towards it, since overdrive is capped by what each
            grid makes and Cerberus and Lucifer fire only on a grid making 50
  assist    help the factory, which is what a constructor does when there's nothing else: so one helping a factory counts as free

Metal in storage is income not yet spent, and past the cap it's lost: the user wants it spent almost all the time (2026-11-08).
The army script makes a constructor for each INCOME_PER_CONSTRUCTOR of income (policy_mix.py) to have the build power.

A constructor that hasn't got nearer the place it was sent to build for STUCK_AFTER seconds is given something else, and that
place is left alone for a while: some spots can't be reached by every kind of constructor.

The big projects, as the user set them out (2026-11-08):

  Cerberus or Lucifer  from 16 minutes with 55 metal a second, while neither side's extractors have changed, net, by more than a
                       tenth of the map's spots (two at least) over the last two minutes.  Cerberus where an enemy building is in its reach from our side of the front,
                       Lucifer at the front otherwise.  One at a time, and no more than MAX_TURRETS.
  missile silo         from 16 minutes with 55 metal a second, once an enemy Singularity, Cerberus, Lucifer, Trinity or Big Bertha,
                       or a second fusion, stands where a silo on our half would reach it.  One silo.
  Trinity              from 18 minutes with more than 60 metal a second, by a chance drawn once, when that first holds.

Missiles and the Trinity are the policy's units (2026-11-08), so this script doesn't aim them.  `Aimer` does, for the army script,
whose orders are labels: each missile at the most valuable building we can see in its reach, and the Trinity at the enemy building
in sight or on radar with the most enemy value around it.  Only what's in sight or on radar: that's what the policy can point at.  What the silo makes is the army script's to choose too.
"""
from __future__ import annotations

import math
import pathlib
import sys
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim" / "py"))
sys.path.insert(0, str(ROOT / "bots"))
import zksim  # noqa: E402
from zkbot import units as bot_units  # noqa: E402
from zkbot.policy import rawframe as rf, tables  # noqa: E402

from policy_opening import Footing, Ground  # noqa: E402
from policy_script import HURTS_NOT  # noqa: E402

ECO_EVERY = 4               # decisions between passes: 2 s
INNATE_ENERGY = 2.0         # sim/src/economy.cpp: what a side makes with nothing
# Energy made wanted, as a share of metal income, by minute: overdrive turns what's spare into metal.
ENERGY_SHARE = ((0.0, 1.25), (4.0, 1.75), (8.0, 2.5), (14.0, 3.0))
ENERGY_FIRST = 1.5          # of metal income: energy comes before anything else while it makes less.  Spending metal takes as
                            # much energy, and overdrive only gets what's left (sim/src/economy.cpp), so at 1.0 metal can't all be spent
SMALL_ENERGY_BELOW = 10.0   # energy a second: below this a generator is a small one (solar, wind), at or above it a big one
FUSION_FROM = (240.0, 15.0)         # seconds, metal income: big generators from then on
SINGULARITY_FROM = (720.0, 45.0)    # the biggest
SMALL_PER_PASS = 8
LINK_SLACK = 0.85           # of two pylon ranges, so that a new building is safely on the grid it's meant for
BEHIND_BASE = 250.0         # elmos from the base away from the enemy, where big generators and the Trinity go
SITE_SPREAD = 120.0
FINISH_REACH = 1200.0       # elmos a constructor goes to help with something being built
SAFE_FROM_ARMY = 600.0      # elmos between a spot and an enemy armed unit, for it to be taken
COMMANDER_REACH = 1500.0    # elmos from the base: the commander takes no spot farther
SAFE_MARGIN = 100.0         # elmos past an enemy defence's range
OUR_HALF = 1.0              # a spot is ours to take if it's no farther from our base than this times its distance from theirs
FIRE_AGAIN = 60.0           # seconds before a missile or the Trinity is given a new target for the same one

BIG_FROM = (960.0, 55.0)            # seconds, metal income: Cerberus, Lucifer and the silo
STALE_FOR = 120.0                   # seconds over which map control is judged
STALE_SHARE = 0.1                   # of the map's metal spots: what either side may gain or lose over that time for it not to be moving
STALE_ENDS = 30.0                   # seconds at each end of that time, averaged, so that an extractor raided and rebuilt doesn't count
MAX_TURRETS = 3
FRONT_BACK = 250.0                  # elmos behind the front a Lucifer goes
REACH_SHARE = 0.85                  # of a weapon's range: where a Cerberus or silo stands from what it's for
NUKE_FROM = (1080.0, 60.0)          # seconds, metal income
NUKE_CHANCE = 0.3
NUKE_VALUE_REACH = 960.0            # elmos: its blast, over which enemy value is summed to choose its target
PENDING_FOR = 60.0                  # seconds an ordered project counts as begun before anything stands there
STUCK_AFTER = 30.0                  # seconds without getting nearer
STUCK_NEARER = 50.0                 # elmos it must have got nearer in that time
AVOID_FOR = 180.0                   # seconds a place a constructor got stuck going to is left alone
FLOATING = 0.5                      # of metal storage: past this, income is being lost, and constructors are told to spend it
FLOATING_UNTIL = 0.15               # of metal storage: and until it's below this
FLOAT_ASSIST_REACH = 3000.0         # elmos from a factory: a constructor this near helps it while metal floats
ENERGY_LOW = 0.3                    # of energy storage: below this, metal that floats is for want of energy, not of builders
FLOAT_EXPANDERS = 4                 # constructors going to take metal spots, at most, while metal floats: the rest spend it
FACTORY_PER_INCOME = 12.0           # metal a second for each factory
MOST_FACTORIES = 6
RADAR_FROM = 180.0                  # seconds
RADAR = "staticradar"

CERBERUS, LUCIFER, SILO, NUKE = "staticarty", "turretantiheavy", "staticmissilesilo", "staticnuke"
VALUABLE = ("energysingu", "staticarty", "turretantiheavy", "staticnuke", "staticheavyarty")
FUSION = "energyfusion"             # valuable from the second one on
PYLON = "energypylon"


def _interp(points: tuple[tuple[float, float], ...], x: float) -> float:
    return float(np.interp(x, [p for p, _ in points], [v for _, v in points]))


def _along(frm: tuple[float, float], to: tuple[float, float], dist: float) -> tuple[float, float]:
    d = math.hypot(to[0] - frm[0], to[1] - frm[1])
    if d < 1e-6:
        return float(frm[0]), float(frm[1])
    return float(frm[0] + (to[0] - frm[0]) / d * dist), float(frm[1] + (to[1] - frm[1]) / d * dist)


class EcoData:
    """What the data says, by row of the type table.  Shared by all the economies of an environment."""

    def __init__(self, table: tables.TypeTable, cat: zksim.Catalog):
        self.table, self.cat = table, cat
        n = len(table.names) + 1
        self.catalog_row = np.full(n, -1, np.int32)
        self.energy = np.zeros(n, np.float32)
        self.pylon = np.zeros(n, np.float32)
        self.needs_link = np.zeros(n, np.float32)
        self.extractor = np.zeros(n, bool)
        self.missile = np.zeros(n, bool)
        self.commander = np.zeros(n, bool)
        self.footing: dict[int, Footing] = {}
        self.radius = np.zeros(n, np.float32)
        for row, name in enumerate(table.names):
            self.extractor[row] = bool(bot_units.is_mex(bot_units.by_name(name)))
            if name not in cat.index:
                continue
            u = cat.units[cat.index[name]]
            self.catalog_row[row] = cat.index[name]
            self.energy[row] = u["energy_make"]
            self.pylon[row] = u["pylon_range"]
            self.needs_link[row] = u["needed_link"]
            self.missile[row] = bool(u["flags"] & zksim.MISSILE)
            self.commander[row] = bool(u["flags"] & zksim.COMMANDER)
            self.footing[row] = Footing.of(u)
            self.radius[row] = u["footprint_radius"]
        self.armed = table.armed & ~table.mobile         # defences, which keep constructors off a spot
        # Generators that need a place the sim doesn't have, a geothermal vent: never chosen.
        self.needs_place = np.array([bool((bot_units.by_name(name) or {}).get("needGeo")) or "geo" in name for name in table.names] + [False])
        self.row = {name: table.names.index(name) for name in (CERBERUS, LUCIFER, SILO, NUKE, FUSION, PYLON, RADAR) if name in table.names}
        self.valuable = np.zeros(n, bool)
        for name in VALUABLE:
            if name in table.names:
                self.valuable[table.names.index(name)] = True
        self.reach = {name: float(table.range[self.row[name]]) for name in (CERBERUS,) if name in self.row}
        # The reach of the missile a silo makes: the one that hurts most, as the army script chooses (ScriptData.best_missile).
        hurts = {r: max((float(np.max(cat.weapons[int(w)]["damage"])) for w in cat.units[self.catalog_row[r]]["mount_weapon"]
                         if w >= 0 and not cat.weapons[int(w)]["flags"] & HURTS_NOT), default=0.0) for r in np.flatnonzero(self.missile)}
        self.missile_reach = float(table.range[max(hurts, key=hurts.get)]) if hurts else 0.0
        self._options: dict[int, np.ndarray] = {}

    def options(self, builder_row: int) -> np.ndarray:
        """The rows of what this constructor can build, among what the sim has."""
        if builder_row not in self._options:
            named = {o.get("name") if isinstance(o, dict) else bot_units.name_of(o)
                     for o in (bot_units.by_name(self.table.names[builder_row]) or {}).get("buildOptions") or []}
            self._options[builder_row] = np.array(sorted(self.table.names.index(n) for n in named
                                                         if n in self.table.names and self.catalog_row[self.table.names.index(n)] >= 0), np.int64)
        return self._options[builder_row]

    def generators(self, builder_row: int) -> tuple[int | None, int | None, int | None]:
        """The small, big and biggest generators this constructor can build: most energy for the metal, for the first two."""
        opts = self.options(builder_row)
        makes = opts[(self.energy[opts] > 0.0) & ~self.table.mobile[opts] & ~self.needs_place[opts]]
        if not len(makes):
            return None, None, None
        per_metal = self.energy[makes] / np.maximum(self.table.cost[makes], 1.0)
        small, big = makes[self.energy[makes] < SMALL_ENERGY_BELOW], makes[self.energy[makes] >= SMALL_ENERGY_BELOW]
        pick = lambda rows: int(rows[np.argmax(per_metal[np.searchsorted(makes, rows)])]) if len(rows) else None  # noqa: E731
        biggest = int(big[np.argmax(self.energy[big])]) if len(big) else None
        return pick(small), pick(big[big != biggest]) if biggest is not None and len(big) > 1 else pick(big), biggest


@dataclass
class Grid:
    """A side's pylon grids: which of its finished buildings link up, and what each grid makes."""
    x: np.ndarray
    z: np.ndarray
    reach: np.ndarray
    label: np.ndarray           # which grid each is on
    makes: dict[int, float]

    @classmethod
    def of(cls, data: EcoData, own: np.ndarray, rows: np.ndarray) -> "Grid":
        on = ((own["flags"] & rf.BEING_BUILT) == 0) & (data.pylon[rows] > 0.0)
        x, z, reach = own["x"][on].astype(float), own["z"][on].astype(float), data.pylon[rows[on]].astype(float)
        n = len(x)
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        if n:
            d = np.hypot(x[:, None] - x[None, :], z[:, None] - z[None, :])
            for a, b in zip(*np.nonzero(np.triu(d < reach[:, None] + reach[None, :], 1))):
                parent[find(int(a))] = find(int(b))
        label = np.array([find(a) for a in range(n)], np.int64)
        makes: dict[int, float] = {}
        for a, e in zip(label.tolist(), data.energy[rows[on]].tolist()):
            makes[a] = makes.get(a, 0.0) + e
        return cls(x, z, reach, label, makes)

    @property
    def main(self) -> int:
        """The grid that makes the most."""
        return max(self.makes, key=lambda g: (self.makes[g], -g)) if self.makes else -1

    def links(self, x: float, z: float, reach: float, grid: int) -> bool:
        on = self.label == grid
        return bool((np.hypot(self.x[on] - x, self.z[on] - z) < LINK_SLACK * (self.reach[on] + reach)).any())

    def nearest(self, x: float, z: float, grid: int) -> int:
        on = np.flatnonzero(self.label == grid)
        return int(on[np.argmin(np.hypot(self.x[on] - x, self.z[on] - z))])


@dataclass
class Economy:
    """One side of one game."""
    data: EcoData
    ground: Ground
    side: int
    base: tuple[float, float]
    enemy_base: tuple[float, float]
    rng: np.random.Generator
    floating: bool = False
    energy_low: bool = False
    checks: Counter = field(default_factory=Counter)      # passes past BIG_FROM, and of those, with map control stalled: for a measurement
    counts: deque = field(default_factory=deque)        # (seconds, our extractors, theirs)
    nuke_drawn: bool | None = None
    pending: dict[str, float] = field(default_factory=dict)                # big project -> when a constructor was last sent to begin one
    finished: dict[str, set] = field(default_factory=dict)                 # big building or missile -> the uids of ours seen finished, for a measurement
    log: list[tuple[float, str]] = field(default_factory=list)             # (seconds, what): projects begun and shots, for a measurement
    helps: dict[int, int] = field(default_factory=dict)                    # builder uid -> the uid it was last told to help
    going: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)   # builder uid -> (goal x, z, since, nearest yet)
    avoid: dict[tuple[int, int], float] = field(default_factory=dict)      # (x, z) / 64 of a place given up on -> until when

    def orders(self, own: np.ndarray, enemies: np.ndarray, metal_income: float, seconds: float, busy: int = -1,
               metal: float = 0.0, storage: float = 1.0, energy: float = 1.0, energy_storage: float = 1.0) -> np.ndarray:
        """own: this side's records.  enemies: the other's, all of them: like the army script, this one knows where things are."""
        data, t = self.data, self.data.table
        rows = t.rows(own["def_id"])
        erows = t.rows(enemies["def_id"])
        built = (own["flags"] & rf.BEING_BUILT) == 0
        self._count(own, rows, enemies, erows, seconds)
        self._note(own, rows, built)
        out: list[tuple] = []
        # The frame doesn't say what an ASSIST is helping, so any helper may be given something better, and is given the same
        # thing again only if it's told something else first.
        helping = own["ord_kind"] == rf.ASSIST
        builders = built & (t.builder[rows] | data.commander[rows]) & t.mobile[rows]
        stuck = self._stuck(own, builders, seconds)
        idle = np.flatnonzero(builders & (((own["flags"] & rf.IDLE) != 0) | helping | stuck) & (own["uid"] != busy))
        self.floating = metal >= FLOATING * storage or (self.floating and metal > FLOATING_UNTIL * storage)
        self.energy_low = energy < ENERGY_LOW * energy_storage
        if not len(idle):
            return np.array(out, zksim.ORDER_DTYPE) if out else np.zeros(0, zksim.ORDER_DTYPE)
        grid = Grid.of(data, own, rows)
        made = INNATE_ENERGY + float(data.energy[rows[built]].sum())
        taken = self._taken(own, rows)
        pass_ = _Pass(made=made, small=0, taken=taken, expanding=self._expanding(own, builders, idle))
        project = self._project(own, rows, enemies, erows, grid, metal_income, seconds)
        for i in idle:
            job = self._job(own, rows, i, enemies, erows, grid, project, pass_, metal_income, seconds)
            uid = int(own["uid"][i])
            if job is None or (job[1] == rf.ASSIST and own["ord_kind"][i] == rf.ASSIST and self.helps.get(uid) == job[4]):
                continue
            self.helps[uid] = job[4] if job[1] == rf.ASSIST else -1
            out.append(job)
            if job[1] == rf.BUILD:
                row = int(np.flatnonzero(data.catalog_row == job[4])[0])
                pass_.taken = np.concatenate((pass_.taken, np.array([[job[2], job[3], data.radius[row]]], np.float32)))
        return np.array(out, zksim.ORDER_DTYPE) if out else np.zeros(0, zksim.ORDER_DTYPE)

    def _stuck(self, own, builders: np.ndarray, seconds: float) -> np.ndarray:
        """Constructors going to build somewhere that haven't got nearer it lately; the place is then avoided for a while."""
        out = np.zeros(len(own), bool)
        alive = set()
        for i in np.flatnonzero(builders & (own["ord_kind"] == rf.BUILD)):
            uid = int(own["uid"][i])
            alive.add(uid)
            gx, gz = float(own["ord_x"][i]), float(own["ord_z"][i])
            d = math.hypot(float(own["x"][i]) - gx, float(own["z"][i]) - gz)
            was = self.going.get(uid)
            if was is None or math.hypot(was[0] - gx, was[1] - gz) > 1.0 or d < was[3] - STUCK_NEARER:
                self.going[uid] = (gx, gz, seconds, d)
            elif seconds - was[2] >= STUCK_AFTER and d > 300.0:
                out[i] = True
                self.avoid[(int(gx // 64), int(gz // 64))] = seconds + AVOID_FOR
                del self.going[uid]
        for uid in [u for u in self.going if u not in alive]:
            del self.going[uid]
        for key in [k for k, until in self.avoid.items() if until <= seconds]:
            del self.avoid[key]
        return out

    def _expanding(self, own, builders: np.ndarray, idle: np.ndarray) -> int:
        """Constructors not being re-tasked that are on their way to build on a metal spot."""
        spots = self.ground.bundle.spot_list
        going = builders & (own["ord_kind"] == rf.BUILD)
        going[idle] = False
        if not len(spots) or not going.any():
            return 0
        d = np.hypot(own["ord_x"][going, None] - spots[None, :, 0], own["ord_z"][going, None] - spots[None, :, 1]).min(axis=1)
        return int((d < 64.0).sum())

    def _avoided(self, x: float, z: float) -> bool:
        return (int(x // 64), int(z // 64)) in self.avoid

    def _note(self, own, rows, built) -> None:
        data = self.data
        for name, r in data.row.items():
            if name not in (PYLON, RADAR):
                self.finished.setdefault(name, set()).update(own["uid"][built & (rows == r)].tolist())
        biggest = np.flatnonzero(data.energy >= 100.0)
        self.finished.setdefault("singularity", set()).update(own["uid"][built & np.isin(rows, biggest)].tolist())
        self.finished.setdefault("missile", set()).update(own["uid"][built & data.missile[rows]].tolist())

    def _have(self, own, rows, name: str, seconds: float) -> int:
        """How many of these we have, standing or being built, counting one ordered lately that isn't there yet."""
        n = int((rows == self.data.row[name]).sum())
        return max(n, 1) if seconds - self.pending.get(name, -1e9) < PENDING_FOR and n == 0 else n

    # Map control.

    def _count(self, own, rows, enemies, erows, seconds: float) -> None:
        self.counts.append((seconds, int(self.data.extractor[rows].sum()), int(self.data.extractor[erows].sum())))
        while self.counts and self.counts[0][0] < seconds - STALE_FOR - 10.0:
            self.counts.popleft()

    def stalled(self, seconds: float) -> bool:
        """Whether map control hasn't moved much for either side, by extractors held, over the last STALE_FOR seconds."""
        window = [c for c in self.counts if c[0] >= seconds - STALE_FOR]
        if not window or seconds - window[0][0] < STALE_FOR - 10.0:
            return False
        start = np.array([c[1:] for c in window if c[0] <= window[0][0] + STALE_ENDS], float).mean(axis=0)
        end = np.array([c[1:] for c in window if c[0] >= seconds - STALE_ENDS], float).mean(axis=0)
        allowed = max(2.0, STALE_SHARE * len(self.ground.bundle.spot_list))
        return bool((np.abs(end - start) <= allowed).all())

    # Big projects.

    def _project(self, own, rows, enemies, erows, grid: Grid, metal_income: float, seconds: float) -> tuple[int, float, float] | None:
        """The big project due now, as (row, x, z), whether started or not; None if there's none."""
        data = self.data
        for name in (NUKE, SILO, CERBERUS, LUCIFER):
            if name not in data.row:
                continue
            being = np.flatnonzero((rows == data.row[name]) & ((own["flags"] & rf.BEING_BUILT) != 0))
            if len(being):
                return data.row[name], float(own["x"][being[0]]), float(own["z"][being[0]])
        if seconds >= NUKE_FROM[0] and metal_income > NUKE_FROM[1] and NUKE in data.row:
            if self.nuke_drawn is None:
                self.nuke_drawn = bool(self.rng.random() < NUKE_CHANCE)
            if self.nuke_drawn and not self._have(own, rows, NUKE, seconds):
                return data.row[NUKE], *_along(self.base, self.enemy_base, -BEHIND_BASE)
        if seconds < BIG_FROM[0] or metal_income < BIG_FROM[1]:
            return None
        self.checks["eligible"] += 1
        self.checks["stalled"] += self.stalled(seconds)
        if SILO in data.row and not self._have(own, rows, SILO, seconds):
            site = self._silo_site(enemies, erows)
            if site is not None:
                return data.row[SILO], *site
        turrets = sum(self._have(own, rows, name, seconds) for name in (CERBERUS, LUCIFER) if name in data.row)
        if self.stalled(seconds) and turrets < MAX_TURRETS and not any(seconds - self.pending.get(n, -1e9) < PENDING_FOR for n in (CERBERUS, LUCIFER)):
            return self._turret(own, rows, enemies, erows)
        return None

    def _ours(self, x: float, z: float) -> bool:
        return math.hypot(x - self.base[0], z - self.base[1]) <= OUR_HALF * math.hypot(x - self.enemy_base[0], z - self.enemy_base[1])

    def _valuable(self, enemies, erows) -> np.ndarray:
        return valuable(self.data, enemies, erows)

    def _silo_site(self, enemies, erows) -> tuple[float, float] | None:
        """On our half, within reach of the most valuable target there's a place for, if any."""
        valuable = np.flatnonzero(self._valuable(enemies, erows))
        reach = REACH_SHARE * self.data.missile_reach
        for i in valuable[np.argsort(-self.data.table.cost[erows[valuable]], kind="stable")]:
            target = (float(enemies["x"][i]), float(enemies["z"][i]))
            away = math.hypot(target[0] - self.base[0], target[1] - self.base[1])
            site = _along(target, self.base, min(reach, away))
            if self._ours(*site):
                return site
        return None

    def _front(self, own, rows) -> tuple[float, float]:
        """Our extractor nearest the enemy's base, or our base when we have none."""
        ext = np.flatnonzero(self.data.extractor[rows] & ((own["flags"] & rf.BEING_BUILT) == 0))
        if not len(ext):
            return self.base
        i = ext[np.argmin(np.hypot(own["x"][ext] - self.enemy_base[0], own["z"][ext] - self.enemy_base[1]))]
        return float(own["x"][i]), float(own["z"][i])

    def _turret(self, own, rows, enemies, erows) -> tuple[int, float, float] | None:
        """A Cerberus where an enemy building is in its reach from our side of the front, else a Lucifer behind the front."""
        data = self.data
        front = self._front(own, rows)
        buildings = np.flatnonzero(~data.table.mobile[erows] & ((enemies["flags"] & rf.BEING_BUILT) == 0) & (erows != data.table.unknown))
        if CERBERUS in data.row and len(buildings):
            d = np.hypot(enemies["x"][buildings] - front[0], enemies["z"][buildings] - front[1])
            i = buildings[int(np.argmin(d))]
            target = (float(enemies["x"][i]), float(enemies["z"][i]))
            site = _along(target, self.base, REACH_SHARE * data.reach[CERBERUS])
            behind_front = math.hypot(site[0] - self.enemy_base[0], site[1] - self.enemy_base[1]) >= \
                math.hypot(front[0] - self.enemy_base[0], front[1] - self.enemy_base[1]) - FRONT_BACK
            if behind_front and self._ours(*site):
                return data.row[CERBERUS], *site
        if LUCIFER in data.row:
            return data.row[LUCIFER], *_along(front, self.base, FRONT_BACK)
        return None

    # Constructors.

    def _taken(self, own, rows) -> np.ndarray:
        """Where nothing new can stand: our buildings, and metal spots, which are kept for extractors."""
        data = self.data
        still = ~data.table.mobile[rows]
        spots = self.ground.bundle.spot_list
        ext_r = data.radius[np.flatnonzero(data.extractor)[0]] if data.extractor.any() else 24.0
        return np.concatenate((np.column_stack((own["x"][still], own["z"][still], data.radius[rows[still]])),
                               np.column_stack((spots[:, 0], spots[:, 1], np.full(len(spots), ext_r))) if len(spots) else np.zeros((0, 3)))).astype(np.float32)

    def _build(self, uid: int, row: int, near: tuple[float, float], taken: np.ndarray, spread: float = SITE_SPREAD,
               grid: Grid | None = None, on: int = -1, by: tuple[int, float, float] | None = None) -> tuple | None:
        """A BUILD of this type near this point, where it fits, on that grid if one is given, and where the builder (its move
        class and where it is) can get to."""
        footing = self.data.footing.get(row)
        if footing is None:
            return None
        for _ in range(4):
            place = self.ground.site(self.rng, footing, near[0], near[1], taken, spread)
            if place is None:
                return None
            if self._avoided(*place) or (by is not None and by[0] != tables.NO_CLASS and not self.ground.joined(by[0], (by[1], by[2]), place)):
                continue
            if grid is None or on < 0 or grid.links(place[0], place[1], float(self.data.pylon[row]), on):
                return uid, rf.BUILD, float(place[0]), float(place[1]), int(self.data.catalog_row[row])
        return None

    def _job(self, own, rows, i: int, enemies, erows, grid: Grid, project, pass_: "_Pass", metal_income: float, seconds: float) -> tuple | None:
        data, t = self.data, self.data.table
        uid, x, z, row = int(own["uid"][i]), float(own["x"][i]), float(own["z"][i]), int(rows[i])
        opts = set(data.options(row).tolist())
        by = (int(t.move_class[row]), x, z)
        being = np.flatnonzero(((own["flags"] & rf.BEING_BUILT) != 0) & ~t.mobile[rows])
        # The big project, wherever it is, and anything of ours being built near.
        if project is not None:
            prow, px, pz = project
            standing = [j for j in being if rows[j] == prow]
            if standing:
                return uid, rf.ASSIST, 0.0, 0.0, int(own["uid"][standing[0]])
            if prow in opts:
                name = t.names[prow]
                needs_grid = data.needs_link[prow] > 0.0
                job = self._build(uid, prow, (px, pz), pass_.taken, grid=grid if needs_grid else None, on=grid.main if needs_grid else -1, by=by)
                if job is None and needs_grid:
                    job = self._build(uid, prow, (px, pz), pass_.taken, by=by)     # a pylon will link it
                if job is not None and seconds - self.pending.get(name, -1e9) >= PENDING_FOR:
                    self.pending[name] = seconds
                    self.log.append((seconds, f"start {name}"))
                    return job
        near = [j for j in being if math.hypot(own["x"][j] - x, own["z"][j] - z) <= FINISH_REACH]
        if near:
            j = min(near, key=lambda j: math.hypot(own["x"][j] - x, own["z"][j] - z))
            return uid, rf.ASSIST, 0.0, 0.0, int(own["uid"][j])
        if pass_.made < ENERGY_FIRST * metal_income:
            energy = self._energy(uid, row, own, rows, grid, pass_, metal_income, seconds, being)
            if energy is not None:
                return energy
        if not self.floating or pass_.expanding < FLOAT_EXPANDERS:
            spot = self._spot(uid, row, x, z, own, rows, enemies, erows, pass_)
            if spot is not None:
                pass_.expanding += 1
                return spot
        if self.floating and self.energy_low:
            # Metal's being lost for want of energy to spend it with.
            energy = self._energy(uid, row, own, rows, grid, pass_, metal_income, seconds, being, force=True)
            if energy is not None:
                return energy
        if self.floating:
            # Metal's being lost: spend it where it can be spent at once.  Another factory, or help one that's near.
            more = self._factory(uid, row, own, rows, pass_, metal_income, seconds, by)
            if more is not None:
                return more
            factories = np.flatnonzero(t.factory[rows])
            if len(factories):
                d = np.hypot(own["x"][factories] - x, own["z"][factories] - z)
                if d.min() <= FLOAT_ASSIST_REACH:
                    return uid, rf.ASSIST, 0.0, 0.0, int(own["uid"][factories[int(np.argmin(d))]])
        energy = self._energy(uid, row, own, rows, grid, pass_, metal_income, seconds, being)
        if energy is not None:
            return energy
        radar = self._radar(uid, row, own, rows, pass_, seconds, by)
        if radar is not None:
            return radar
        link = self._link(uid, row, own, rows, grid, pass_)
        if link is not None:
            return link
        more = self._factory(uid, row, own, rows, pass_, metal_income, seconds, by)
        if more is not None:
            return more
        factories = np.flatnonzero(t.factory[rows] & ((own["flags"] & rf.BEING_BUILT) == 0))
        if len(factories):
            j = factories[np.argmin(np.hypot(own["x"][factories] - x, own["z"][factories] - z))]
            return uid, rf.ASSIST, 0.0, 0.0, int(own["uid"][j])
        return None

    def _energy(self, uid, row, own, rows, grid: Grid, pass_: "_Pass", metal_income: float, seconds: float, being,
                force: bool = False) -> tuple | None:
        """force: whatever energy is made already."""
        data = self.data
        wanted = _interp(ENERGY_SHARE, seconds / 60.0) * metal_income
        if pass_.made >= wanted and not force:
            return None
        small, big, biggest = data.generators(row)
        big_being = any(data.energy[rows[j]] >= SMALL_ENERGY_BELOW for j in being)
        choice = None
        if not big_being:
            if biggest is not None and seconds >= SINGULARITY_FROM[0] and metal_income >= SINGULARITY_FROM[1]:
                choice = biggest
            elif big is not None and seconds >= FUSION_FROM[0] and metal_income >= FUSION_FROM[1]:
                choice = big
        if choice is not None:
            main = grid.main
            near = self.base if main < 0 else (float(grid.x[grid.nearest(*self.base, main)]), float(grid.z[grid.nearest(*self.base, main)]))
            near = _along(near, self.enemy_base, -BEHIND_BASE * 0.5)
            job = self._build(uid, choice, near, pass_.taken, grid=grid, on=main) or self._build(uid, choice, near, pass_.taken)
            if job is not None:
                pass_.made += float(data.energy[choice])
                self.log.append((seconds, f"start {data.table.names[choice]}"))
            return job
        if small is None or pass_.small >= SMALL_PER_PASS:
            return None
        # Beside one of our extractors with the least energy near it, so that it overdrives.
        ext = np.flatnonzero(data.extractor[rows] & ((own["flags"] & rf.BEING_BUILT) == 0))
        if len(ext):
            makers = np.flatnonzero(data.energy[rows] > 0.0)
            near_energy = np.array([(np.hypot(own["x"][makers] - own["x"][j], own["z"][makers] - own["z"][j]) < 200.0).sum() for j in ext])
            j = ext[int(np.argmin(near_energy + self.rng.random(len(ext)) * 0.5))]
            spot = (float(own["x"][j]), float(own["z"][j]))
        else:
            spot = self.base
        job = self._build(uid, small, spot, pass_.taken, spread=float(data.pylon[small]) * 0.6)
        if job is not None:
            pass_.made += float(data.energy[small])
            pass_.small += 1
        return job

    def _factory(self, uid, row, own, rows, pass_: "_Pass", metal_income: float, seconds: float, by) -> tuple | None:
        """Another of the factory we have, when income is being lost for want of somewhere to spend it."""
        t = self.data.table
        if not self.floating or pass_.factory or seconds - self.pending.get("factory", -1e9) < PENDING_FOR:
            return None
        ours = np.flatnonzero(t.factory[rows])
        if not len(ours) or len(ours) >= min(MOST_FACTORIES, 1 + int(metal_income / FACTORY_PER_INCOME)):
            return None
        kind = int(rows[ours[0]])
        if kind not in set(self.data.options(row).tolist()):
            return None
        job = self._build(uid, kind, _along(self.base, self.enemy_base, -BEHIND_BASE), pass_.taken, spread=300.0, by=by)
        if job is not None:
            pass_.factory = True
            self.pending["factory"] = seconds
            self.log.append((seconds, f"start {t.names[kind]}"))
        return job

    def _radar(self, uid, row, own, rows, pass_: "_Pass", seconds: float, by) -> tuple | None:
        data = self.data
        if seconds < RADAR_FROM or RADAR not in data.row or data.row[RADAR] not in set(data.options(row).tolist()) or pass_.radar:
            return None
        want = 1 + int(SILO in data.row and (rows == data.row[SILO]).any())
        if self._have(own, rows, RADAR, seconds) >= want:
            return None
        job = self._build(uid, data.row[RADAR], _along(self._front(own, rows), self.base, FRONT_BACK), pass_.taken, by=by)
        if job is not None:
            pass_.radar = True
            self.pending[RADAR] = seconds
        return job

    def _link(self, uid, row, own, rows, grid: Grid, pass_: "_Pass") -> tuple | None:
        """A pylon from the nearest of our grids that has an extractor or a turret needing power on it, but isn't the main one, towards the main one."""
        data = self.data
        if PYLON not in data.row or data.row[PYLON] not in set(data.options(row).tolist()) or pass_.pylons:
            return None
        main = grid.main
        if main < 0:
            return None
        on = ((own["flags"] & rf.BEING_BUILT) == 0) & (data.pylon[rows] > 0.0)
        wants = (data.extractor[rows[on]] | (data.needs_link[rows[on]] > 0.0))
        loose = np.flatnonzero(wants & (grid.label != main))
        if not len(loose):
            return None
        bx, bz = self.base
        k = int(loose[np.argmin(np.hypot(grid.x[loose] - bx, grid.z[loose] - bz))])
        m = grid.nearest(float(grid.x[k]), float(grid.z[k]), main)
        pylon = float(data.pylon[data.row[PYLON]])
        gap = math.hypot(grid.x[m] - grid.x[k], grid.z[m] - grid.z[k])
        step = min(gap * 0.5, LINK_SLACK * (grid.reach[k] + pylon))
        point = _along((float(grid.x[k]), float(grid.z[k])), (float(grid.x[m]), float(grid.z[m])), step)
        job = self._build(uid, data.row[PYLON], point, pass_.taken, spread=40.0)
        if job is not None:
            pass_.pylons += 1
        return job

    def _spot(self, uid, row, x, z, own, rows, enemies, erows, pass_: "_Pass") -> tuple | None:
        """The nearest free metal spot on our half that this constructor can get to and nothing of the enemy's covers."""
        data, t = self.data, self.data.table
        ext_rows = [r for r in data.options(row).tolist() if data.extractor[r]]
        if not ext_rows:
            return None
        spots = self.ground.bundle.spot_list
        if not len(spots):
            return None
        if data.commander[row]:
            spots = spots[np.hypot(spots[:, 0] - self.base[0], spots[:, 1] - self.base[1]) <= COMMANDER_REACH]
            if not len(spots):
                return None
        holders = np.concatenate((np.column_stack((own["x"], own["z"]))[data.extractor[rows]],
                                  np.column_stack((enemies["x"], enemies["z"]))[data.extractor[erows]],
                                  pass_.claimed.reshape(-1, 2)))
        free = np.ones(len(spots), bool)
        if len(holders):
            free = np.hypot(spots[:, 0, None] - holders[:, 0], spots[:, 1, None] - holders[:, 1]).min(axis=1) > 64.0
        mine = np.hypot(spots[:, 0] - self.base[0], spots[:, 1] - self.base[1]) <= OUR_HALF * np.hypot(spots[:, 0] - self.enemy_base[0], spots[:, 1] - self.enemy_base[1])
        cand = np.flatnonzero(free & mine)
        if not len(cand):
            return None
        defence = data.armed[erows] & ((enemies["flags"] & rf.BEING_BUILT) == 0)
        army = t.armed[erows] & t.mobile[erows]
        reach = t.range[erows] + SAFE_MARGIN
        for j in cand[np.argsort(np.hypot(spots[cand, 0] - x, spots[cand, 1] - z))][:8]:
            sx, sz = float(spots[j, 0]), float(spots[j, 1])
            if self._avoided(sx, sz):
                continue
            d = np.hypot(enemies["x"] - sx, enemies["z"] - sz)
            if (defence & (d <= reach)).any() or (army & (d <= SAFE_FROM_ARMY)).any():
                continue
            c = int(t.move_class[row])
            if c != tables.NO_CLASS and not self.ground.joined(c, (x, z), (sx, sz)):
                continue
            pass_.claimed = np.concatenate((pass_.claimed, [sx, sz]))
            return uid, rf.BUILD, sx, sz, int(data.catalog_row[ext_rows[0]])
        return None



def valuable(data: EcoData, enemies: np.ndarray, erows: np.ndarray) -> np.ndarray:
    """The enemy's buildings worth a silo: VALUABLE, and fusions from the second one."""
    built = (enemies["flags"] & rf.BEING_BUILT) == 0
    out = data.valuable[erows] & built
    if FUSION in data.row:
        fusions = (erows == data.row[FUSION]) & built
        if fusions.sum() >= 2:
            out |= fusions
    return out


@dataclass
class Aimer:
    """Where the army script sends a side's missiles and Trinity.  Its orders are the script's, so they're labels."""
    data: EcoData
    aimed: dict[int, tuple[int, float]] = field(default_factory=dict)      # missile or Trinity uid -> (target uid, when)
    log: list[tuple[float, str]] = field(default_factory=list)             # (seconds, what): shots, for a measurement

    def orders(self, own: np.ndarray, enemies: np.ndarray, seen: np.ndarray, seconds: float) -> list[tuple]:
        """seen: the uids of the enemy we can see or have on radar, which is what a missile or the Trinity can be aimed at."""
        t = self.data.table
        rows, erows = t.rows(own["def_id"]), t.rows(enemies["def_id"])
        built = (own["flags"] & rf.BEING_BUILT) == 0
        return self._aim(own, rows, built, enemies, erows, seen, seconds)

    def _valuable(self, enemies, erows) -> np.ndarray:
        return valuable(self.data, enemies, erows)

    def _aim(self, own, rows, built, enemies, erows, seen: np.ndarray, seconds: float) -> list[tuple]:
        data, t = self.data, self.data.table
        out: list[tuple] = []
        for uid in [u for u in self.aimed if u not in set(own["uid"].tolist())]:
            del self.aimed[uid]
        alive = set(enemies["uid"].tolist())
        missiles = np.flatnonzero(built & data.missile[rows])
        if len(missiles):
            visible = np.isin(enemies["uid"], seen) & ~t.mobile[erows] & ((enemies["flags"] & rf.BEING_BUILT) == 0) & (erows != t.unknown)
            worth = t.cost[erows] * np.where(self._valuable(enemies, erows), 4.0, 1.0)
            for i in missiles:
                uid = int(own["uid"][i])
                if not self._due(uid, alive, seconds):
                    continue
                d = np.hypot(enemies["x"] - own["x"][i], enemies["z"] - own["z"][i])
                ok = np.flatnonzero(visible & (d <= t.range[rows[i]]))
                if not len(ok):
                    continue
                j = ok[int(np.argmax(worth[ok]))]
                self.aimed[uid] = (int(enemies["uid"][j]), seconds)
                self.log.append((seconds, f"missile at {t.names[erows[j]]}"))
                out.append((uid, rf.ATTACK_UNIT, float(enemies["x"][j]), float(enemies["z"][j]), int(enemies["uid"][j])))
        if NUKE in data.row:
            for i in np.flatnonzero(built & (rows == data.row[NUKE])):
                uid = int(own["uid"][i])
                if own["ord_kind"][i] == rf.MANUAL_FIRE and int(own["ord_target"][i]) in alive:
                    continue
                if uid in self.aimed and self.aimed[uid][0] in alive and seconds - self.aimed[uid][1] < FIRE_AGAIN:
                    continue
                j = self._nuke_target(enemies, erows, seen)
                if j is None:
                    continue
                self.aimed[uid] = (int(enemies["uid"][j]), seconds)
                self.log.append((seconds, f"nuke at {t.names[erows[j]]}"))
                out.append((uid, rf.MANUAL_FIRE, float(enemies["x"][j]), float(enemies["z"][j]), int(enemies["uid"][j])))
        return out

    def _due(self, uid: int, alive: set[int], seconds: float) -> bool:
        if uid not in self.aimed:
            return True
        target, when = self.aimed[uid]
        return target not in alive or seconds - when >= FIRE_AGAIN

    def _nuke_target(self, enemies, erows, seen: np.ndarray) -> int | None:
        """The enemy building in sight or on radar with the most enemy value within the blast of it."""
        t = self.data.table
        buildings = np.flatnonzero(~t.mobile[erows] & (erows != t.unknown) & np.isin(enemies["uid"], seen))
        if not len(buildings):
            return None
        d = np.hypot(enemies["x"][buildings, None] - enemies["x"][None, :], enemies["z"][buildings, None] - enemies["z"][None, :])
        value = ((d <= NUKE_VALUE_REACH) * t.cost[erows][None, :]).sum(axis=1)
        return int(buildings[int(np.argmax(value))])


@dataclass
class _Pass:
    """What this pass has already set going, so that two constructors don't do the same."""
    made: float
    small: int
    taken: np.ndarray
    pylons: int = 0
    radar: bool = False
    factory: bool = False
    expanding: int = 0          # constructors going to take metal spots
    claimed: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
