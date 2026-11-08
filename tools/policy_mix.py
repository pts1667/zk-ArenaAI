"""What a side's factory builds next, and when that changes.  The army script is C++ (sim/scripts/src/army.cpp), and its
numbers are there; this is what it does, and each factory's menu, which the C++ is given (policy_scripts_cpp.py).

The sim's factory keeps building the type it was last told to, with no queue, and a PRODUCE order only sets the type for the unit
after the one in progress.  So a choice is made each time a unit starts: the script works out what the factory should build next,
and orders a PRODUCE only if that's not the type the factory is on.  Each such order is a choice a recorder can read
(`SimEnv.script_record`), and no other order changes what the factory builds.

The shares follow CircuitAI's factory.json, which gives each factory its unit mix by time, with the user's Zero-K rules on top.
Raiders come first and in numbers: riots only start at about four minutes, however many raiders there are.  Skirmishers
wait until slow units that can be kited (riots and assaults) have been seen.  AA waits until the enemy has been seen with air.
Artillery comes late, and a scout every few minutes from two.  Each unit is chosen by its catalog role, never by name: the
cheapest unit on the factory's roster with that role that the sim models.

A role is built when it's furthest behind its share of the units the side has started.  Losses don't change the mix, so a side that
loses its raiders keeps building raiders in the same proportion as before.

Constructors aren't in the shares: the factory makes one whenever the side has fewer than CONSTRUCTORS says for the minute, or
than one for each INCOME_PER_CONSTRUCTOR of metal income, lost ones included, for the economy script (policy_eco.py) to use.
Without enough of them to spend it, income piles up in storage and is lost.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from zkbot import catalog as bot_catalog

if TYPE_CHECKING:
    from policy_script import ScriptData

SCOUT = "scout"
BUILDER = "builder"
ROLE_ORDER = ("raid", "riot", "assault", "skirm", "arty", "aa", "snipe")     # ties in the deficit go to the first


def menu(data: "ScriptData", factory_row: int) -> dict[str, int]:
    """The table row of the cheapest unit this factory builds for each role, among those the sim models."""
    t = data.table
    best: dict[str, int] = {}
    for name in bot_catalog.roster(t.names[factory_row]):
        row = data.row_of_name.get(name)
        if row is None or not data.modelled[row]:
            continue
        role = data.role_of[row]
        if role not in best or t.cost[row] < t.cost[best[role]]:
            best[role] = row
    return best
