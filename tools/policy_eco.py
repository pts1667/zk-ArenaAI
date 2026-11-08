"""The economy script's tables.  The script is C++ (sim/scripts/src/eco.cpp, through policy_scripts_cpp.py), and its numbers
are there; this is what it does.

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

Missiles and the Trinity are the policy's units (2026-11-08), so this script doesn't aim them.  The aimer does, for the army script,
whose orders are labels: each missile at the most valuable building we can see in its reach, and the Trinity at the enemy building
in sight or on radar with the most enemy value around it.  Only what's in sight or on radar: that's what the policy can point at.  What the silo makes is the army script's to choose too.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim" / "py"))
sys.path.insert(0, str(ROOT / "bots"))
import zksim  # noqa: E402
from zkbot import units as bot_units  # noqa: E402
from zkbot.policy import tables  # noqa: E402

from policy_opening import Footing  # noqa: E402
from policy_script import HURTS_NOT  # noqa: E402

ECO_EVERY = 4               # decisions between passes: 2 s
SMALL_ENERGY_BELOW = 10.0   # energy a second: below this a generator is a small one (solar, wind), at or above it a big one
RADAR = "staticradar"
CERBERUS, LUCIFER, SILO, NUKE = "staticarty", "turretantiheavy", "staticmissilesilo", "staticnuke"
VALUABLE = ("energysingu", "staticarty", "turretantiheavy", "staticnuke", "staticheavyarty")
FUSION = "energyfusion"             # valuable from the second one on
PYLON = "energypylon"


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
