"""Compare stat-inferred unit roles (bots/zkbot/catalog.py) with the
hand-written roles for every factory roster.

  python tools/catalog_check.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "bots"))
from zkbot import catalog, units  # noqa: E402

FACTORIES = sorted(n for n in units.BY_NAME if n.startswith("factory") or n == "striderhub")
agree = total = 0
for f in FACTORIES:
    for n in catalog.roster(f):
        hand = units.role_of(units.by_name(n))
        inf = catalog.infer_role(n)
        st = catalog.stat_role(n)
        if hand in ("combat", "", "other", "support", "heavy"):
            mark = "~"  # no specific hand role to compare with
        else:
            total += 1
            agree += hand == inf
            mark = " " if hand == inf else "X"
        tg = ",".join(sorted(catalog.tags(n)))
        print(f"{mark} {f:15} {n:20} hand={hand:8} inferred={inf:8} stats={st:8} "
              f"desc={catalog.abilities(n).get('description', '')[:38]!r} tags={tg}")
print(f"agreement on specific hand roles: {agree}/{total}")
