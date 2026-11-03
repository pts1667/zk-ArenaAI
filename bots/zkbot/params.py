"""Feature flags and tunable parameters, in one place.

FLAGS  the comma-separated tokens of ZKBOT_FLAGS, as an exact set.  (Several
       modules used the raw string, where "emergency" in "noemergency" is
       True.)

P(name, default, lo, hi, doc)
       registers a tunable and returns its value: the default, or an
       override from ZKBOT_PARAMS="resign.low_share=0.25,raid.offense_share=0.4".
       lo/hi document the sensible range for tuners (SPSA/CLOP, see
       docs/ARCHITECTURE.md); overrides outside it are clamped.

describe() returns the overrides in effect, for the game log.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

FLAGS: frozenset[str] = frozenset(
    t.strip() for t in os.environ.get("ZKBOT_FLAGS", "").split(",") if t.strip()
)


def flag(name: str) -> bool:
    return name in FLAGS


@dataclass
class Param:
    name: str
    default: float
    lo: float | None
    hi: float | None
    doc: str
    value: float


REGISTRY: dict[str, Param] = {}


def _overrides() -> dict[str, float]:
    out = {}
    for item in os.environ.get("ZKBOT_PARAMS", "").split(","):
        k, sep, v = item.partition("=")
        if sep and k.strip():
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


_OVERRIDES = _overrides()


def P(name: str, default: float, lo: float | None = None, hi: float | None = None, doc: str = "") -> float:
    value = _OVERRIDES.get(name, default)
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    if isinstance(default, int) and not isinstance(default, bool):
        value = int(round(value))
    REGISTRY[name] = Param(name, default, lo, hi, doc, value)
    return value


def describe() -> str:
    changed = [f"{p.name}={p.value:g}" for p in REGISTRY.values() if p.value != p.default]
    unknown = [k for k in _OVERRIDES if k not in REGISTRY]
    s = f"flags={','.join(sorted(FLAGS)) or '-'} params={','.join(changed) or 'defaults'}"
    if unknown:
        s += f" UNKNOWN params={','.join(unknown)}"
    return s
