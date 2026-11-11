"""Which kinds of game a map holds: a 1v1, a team game of how many a side, a free-for-all of how many alliances, and Zero-K's
chicken defense.

The rules, the user's (2026-11-10), by a map's names and Zero-K's listing rather than by a list of maps, as more maps come:
* A map whose name begins with one of UNPLAYABLE plays nothing, and stays out of the pool.
* A chicken map plays chicken games only: one whose name begins with one of CHICKEN_MAPS, or that Zero-K lists for chickens and
  for no other kind of game ("special", its mark for an unusual map, isn't one: Requiem Outpost, a survival outpost, has both).
* Any other map with two boxes or more is a 1v1 map.  How many a side a team game has, and how many alliances a free-for-all,
  is from Zero-K's list (maps.json's `Modes`, the lobby's map data) where it has the map, else from the boxes: two make a team
  map, more a free-for-all of up to that many.  Zero-K doesn't say how many a side a team map holds, so that's from the map's
  area (`team_size`).
* Chicken games are drawn on team maps too (`chickens_or_teams`); `chickens` alone is the chicken maps.

A map's facts file holds the result as `"modes"` (tools/map_gen.py writes it), and tools/policy_sim_env.py works it out from the
boxes and the name where a file has none.
"""
from __future__ import annotations

import functools
import json
import pathlib
import re
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
SIDE = 512.0                # elmos in one of a map's size units
UNPLAYABLE = ("thisspace", "zerowars")      # name prefixes, in lower case
CHICKEN_MAPS = ("chicken",)
NO_KIND = ("special",)                      # in Zero-K's listing, a mark for an unusual map rather than a kind of game
MOST_A_SIDE = 4             # players a side of a team game
MOST_ALLIANCES = 8          # of a free-for-all: the sim's ZK_MAX_TEAMS
# Map units² each player of a team game has at least: half what a 1v1 on a 10 x 10 map gives each, so 2 a side there, 3 on
# 12 x 12 and 4 from 14 x 14 up.
AREA_A_PLAYER = 24.0
UP_TO = re.compile(r"ffa \(up to (\d+)\)")
NOTHING = {"1v1": False, "teams": 0, "ffa": 0, "chickens": False, "chickens_or_teams": False}
CHICKENS_ONLY = {"1v1": False, "teams": 0, "ffa": 0, "chickens": True, "chickens_or_teams": True}


def slug(name: str) -> str:
    """As tools/sim_data.py names a map's files."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@functools.lru_cache(maxsize=4)
def catalog(path: pathlib.Path = ROOT / "maps.json") -> dict[str, tuple[str, list[str]]]:
    """Each map Zero-K's install has: its name and Modes, by every name a pool map may go by: its archive's (which a real map's
    facts have as `name`), and the slugs of that and of the map's own name."""
    out: dict[str, tuple[str, list[str]]] = {}
    for m in json.loads(pathlib.Path(path).read_text("utf8")):
        stem = pathlib.Path(m["File"]).stem
        for key in (stem.lower(), slug(stem), slug(m["Name"])):
            out.setdefault(key, (m["Name"], list(m["Modes"])))
    return out


def lookup(name: str, table: dict | None = None) -> tuple[str, list[str]] | None:
    """The map's own name and its Modes in `catalog`, or None if the install hasn't it.  Modes may be empty: a map Zero-K has
    but doesn't list for anything."""
    table = catalog() if table is None else table
    found = table.get(name.lower())
    return found if found is not None else table.get(slug(name))


def unplayable(*names: str) -> bool:
    return any(n.lower().startswith(UNPLAYABLE) for n in names)


@dataclass(frozen=True)
class Known:
    """What says which kinds of game a map holds."""
    boxes: int                                  # the most start boxes a layout of the map has
    points: int = 1                             # the most start points a box lists
    area: float = 0.0                           # map units², 0 if not known
    listing: tuple[str, ...] | None = None      # Zero-K's Modes for the map, None (or empty) where it lists none
    names: tuple[str, ...] = ()                 # every name the map goes by: its archive's, its own, its generator's
    variant: bool = False                       # one of several maps a generator made of one real map

    @property
    def unplayable(self) -> bool:
        return unplayable(*self.names)

    @property
    def chicken_map(self) -> bool:
        return any(n.lower().startswith(CHICKEN_MAPS) for n in self.names) or (set(self.listing or ()) - set(NO_KIND) == {"chickens"})


def team_size(area: float, points: int = 1) -> int:
    """Players a side of a team game on a map of this area (map units², 0 if not known): as many as AREA_A_PLAYER each allows, or
    as many start points as a box lists if that's more, 2 at least and MOST_A_SIDE at most."""
    return int(min(MOST_A_SIDE, max(2, area // (2.0 * AREA_A_PLAYER), points)))


def modes(known: Known) -> dict:
    """`teams` is the most players a side and `ffa` the most alliances, 0 where the map isn't one.  `chickens` is the chicken
    maps; `chickens_or_teams` adds every team map, where chicken games are drawn too (the user, 2026-11-10): a game goes by
    whichever of the two it's told to.

    A variant a generator made isn't a 1v1 map: one map's many variants would crowd out the rest of 1v1's maps."""
    if known.unplayable:
        return dict(NOTHING)
    if known.chicken_map:
        return dict(CHICKENS_ONLY)
    boxes, listing = known.boxes, known.listing
    if listing:
        teams, chickens = "teams" in listing, "chickens" in listing
        ffa = max([int(n) for m in listing for n in UP_TO.findall(m)] + [boxes if "ffa" in listing else 0])
    else:
        teams, chickens = boxes == 2, False
        ffa = boxes if boxes > 2 else 0
    # A game needs a box an alliance, and a free-for-all of two is a 1v1: as on maps whose facts came from a game with fewer
    # players than they're listed for, which made only two boxes.
    ffa = min(ffa, MOST_ALLIANCES, boxes)
    size = team_size(known.area, known.points) if teams and boxes >= 2 else 0
    return {"1v1": boxes >= 2 and not known.variant, "teams": size, "ffa": int(ffa) if ffa >= 3 else 0, "chickens": chickens,
            "chickens_or_teams": chickens or size > 0}


def known(facts: dict, area: float = 0.0, table: dict | None = None) -> Known:
    """What a facts file (tools/map_gen.py write_entry) says, with Zero-K's listing for a real map when `table` (`catalog()`)
    is given.  A map a generator made goes by its generator's name too (`style["generator"]`)."""
    names = [facts.get("name", "")]
    generator = (facts.get("style") or {}).get("generator")
    if generator:
        names.append(generator)
    listing = None
    if table is not None and facts.get("kind") == "real":
        found = next((f for f in (lookup(n, table) for n in reversed(names)) if f is not None), None)
        if found is not None:
            names.append(found[0])
            listing = tuple(found[1])
    layouts = facts.get("layouts") or []
    return Known(boxes=max((len(b) for b in layouts), default=0), points=max((len(box["startpoints"]) for b in layouts for box in b), default=1),
                 area=area, listing=listing, names=tuple(names), variant=bool(generator))


def of_facts(facts: dict, area: float = 0.0, table: dict | None = None) -> dict:
    return modes(known(facts, area, table))
