"""Variants of the army script: each side's knobs for a game, so that the games recorded for cloning aren't one script against itself.

The knobs are the army script's constants that change how a game plays (sim/scripts/src/army.cpp takes them as a `zks_style`):

  role_weight   how a factory's mix shares each role, by ROLES: 1 is the standard share, and a role with weight 0 is never started
  commit        what a squad needs before it goes at a target: its metal at least commit times the cover on it.  Higher is more cautious.
  gather        the army waits until its metal is gather times the group metal (policy_script.py's), or LONGEST_WAIT.  Lower goes sooner.
  patrol_share  the share of raiders past the first few that guard our extractors
  patience      scales how long a squad keeps at a target: getting there, and killing it once there
  retreat_below a fighter that regenerates (every fighting unit of the sim does) goes back to the rally below this share of its
                health, until it's at 90%.  0.4 standard.

A preset is a named set of knobs.  A draw takes the standard or one of PRESETS, chosen at random, and jitters each knob about it
within BOUNDS: multiplicatively, by a normal of standard deviation JITTER in log, for all but patrol_share and retreat_below, which
move additively.  The draw takes the environment's generator, so a run's games are reproducible from its seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROLES = ("raid", "riot", "assault", "skirm", "arty", "aa", "snipe")     # the order of role_weight; policy_mix.ROLE_ORDER, then the rest


@dataclass(frozen=True)
class Variant:
    name: str
    role_weight: tuple[float, ...] = (1.0,) * len(ROLES)
    commit: float = 1.0
    gather: float = 1.0
    patrol_share: float = 0.25
    patience: float = 1.0
    retreat_below: float = 0.4

    def as_dict(self) -> dict:
        """What a recording says about the side's script: its name and every knob."""
        return {"name": self.name, "role_weight": dict(zip(ROLES, self.role_weight)), "commit": self.commit, "gather": self.gather,
                "patrol_share": self.patrol_share, "patience": self.patience, "retreat_below": self.retreat_below}


STANDARD = Variant("standard")

# Raiders first and quick; turtling in numbers; skirmishers in numbers.  Rush goes at a target with less metal than the cover on it,
# gathers less, gives up on a target later, and fights on to lower health; turtle does the reverse and keeps more raiders at home.
# Arty, aa and snipe barely change a game of this length (arty starts at 12 minutes, snipers need a slow heavy seen, aa needs air),
# so the presets lean on raid, riot, assault and skirm.
PRESETS = (
    Variant("rush", role_weight=(1.6, 0.6, 1.4, 0.7, 0.5, 1.0, 1.0), commit=0.7, gather=0.5, patrol_share=0.10, patience=1.5,
            retreat_below=0.25),
    Variant("turtle", role_weight=(0.6, 1.4, 0.8, 1.5, 1.5, 1.5, 1.0), commit=1.6, gather=2.0, patrol_share=0.45, patience=0.6,
            retreat_below=0.6),
    Variant("skirmish", role_weight=(0.9, 0.8, 0.9, 2.0, 2.0, 1.0, 1.0), commit=1.2, gather=1.0, patrol_share=0.25, patience=1.0),
)

BOUNDS = {"role_weight": (0.25, 4.0), "commit": (0.6, 2.0), "gather": (0.4, 2.5), "patrol_share": (0.05, 0.5), "patience": (0.4, 2.0),
          "retreat_below": (0.1, 0.7)}
JITTER = {"role_weight": 0.3, "commit": 0.15, "gather": 0.25, "patrol_share": 0.05, "patience": 0.25, "retreat_below": 0.05}
# log-scale, except patrol_share and retreat_below


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def draw(rng: np.random.Generator) -> Variant:
    """One side's knobs for a game: one of the standard and PRESETS, jittered.  Uses `rng` in a fixed order."""
    base = (STANDARD, *PRESETS)[int(rng.integers(1 + len(PRESETS)))]
    roles = tuple(_clip(w * float(np.exp(rng.normal(0.0, JITTER["role_weight"]))), BOUNDS["role_weight"]) for w in base.role_weight)
    commit = _clip(base.commit * float(np.exp(rng.normal(0.0, JITTER["commit"]))), BOUNDS["commit"])
    gather = _clip(base.gather * float(np.exp(rng.normal(0.0, JITTER["gather"]))), BOUNDS["gather"])
    patrol = _clip(base.patrol_share + float(rng.normal(0.0, JITTER["patrol_share"])), BOUNDS["patrol_share"])
    patience = _clip(base.patience * float(np.exp(rng.normal(0.0, JITTER["patience"]))), BOUNDS["patience"])
    retreat = _clip(base.retreat_below + float(rng.normal(0.0, JITTER["retreat_below"])), BOUNDS["retreat_below"])
    return Variant(base.name, roles, commit, gather, patrol, patience, retreat)
