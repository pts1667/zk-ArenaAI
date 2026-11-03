"""Resign a lost game (ResignMixin).

User rule: resign when the current metal *extraction* income is below 30%
of its peak for 1-2 minutes, where both are 2-minute averages and reclaim
(and overdrive) is excluded.  Extraction income is the summed `income` of
the metal spots holding our finished extractors.

There is no resign command in the ControllerAI API.  Zero-K disables its
"self-destruct everything = resign" blocker in 1v1, so the bot resigns by
self-destructing every unit; the game ends when the team has no units.
After that, the brain issues no further orders (a second self-destruct
order would cancel the countdown).

Guards: not before 6 minutes, only after the 2-minute peak reached
4 metal/s, and not while our army power is >= 1.5x the enemy's estimate.  Flag `noresign` disables it.
"""
from __future__ import annotations

import os
from collections import deque

from .params import FLAGS, P

WINDOW = P("resign.window", 3600, 1800, 7200, "frames averaged (2 min)")
LOW_SHARE = P("resign.low_share", 0.30, 0.1, 0.6, "resign below this share of peak extraction")
LOW_FOR = P("resign.low_for", 2700, 900, 7200, "frames below the threshold before resigning (90 s)")
MIN_TIME = P("resign.min_time", 360.0, 120.0, 1800.0, "seconds")
MIN_PEAK = P("resign.min_peak", 4.0, 0.0, 20.0, "metal/s")
ARMY_GUARD = P("resign.army_guard", 1.5, 1.0, 5.0, "no resign while our army power >= this x enemy's")
# Test override: ZKBOT_RESIGN="share,low_frames,min_seconds,min_peak".
if os.environ.get("ZKBOT_RESIGN"):
    _v = [float(x) for x in os.environ["ZKBOT_RESIGN"].split(",")]
    LOW_SHARE, LOW_FOR, MIN_TIME, MIN_PEAK = _v[0], int(_v[1]), _v[2], _v[3]


class ResignMixin:
    def extraction_income(self) -> float:
        spots = self.map.spots
        if not spots:
            return 0.0
        total = 0.0
        for u in self.units.values():
            if u.name != "staticmex" or not u.is_finished:
                continue
            best = None
            bd = 80.0 * 80.0
            for s in spots:
                d = (s.x - u.x) ** 2 + (s.z - u.z) ** 2
                if d < bd:
                    best, bd = s, d
            if best is not None:
                total += float(best.income)
        return total

    def resign_step(self) -> bool:
        """True once resigned: the caller stops issuing orders."""
        if getattr(self, "resigned", False):
            return True
        if "noresign" in FLAGS:
            return False
        if self.frame - getattr(self, "_resign_sample", -10**9) < 30:
            return False
        self._resign_sample = self.frame
        hist = self.__dict__.setdefault("_extract_hist", deque())
        hist.append((self.frame, self.extraction_income()))
        while hist and self.frame - hist[0][0] > WINDOW:
            hist.popleft()
        avg = sum(v for _f, v in hist) / len(hist)
        # The peak only counts full 2-minute windows.
        if self.frame >= min(WINDOW, MIN_TIME * 30):
            self.extract_peak = max(getattr(self, "extract_peak", 0.0), avg)
        peak = getattr(self, "extract_peak", 0.0)
        if self.frame % 1800 < 30 and peak > 0:
            self.log(f"extraction avg2m={avg:.1f} peak={peak:.1f}")
        if self.frame_time() < MIN_TIME or peak < MIN_PEAK or avg >= LOW_SHARE * peak:
            self._low_since = None
            return False
        if getattr(self, "_low_since", None) is None:
            self._low_since = self.frame
            self.log(f"extraction {avg:.1f} below {LOW_SHARE:.0%} of peak {peak:.1f}")
        if self.frame - self._low_since < LOW_FOR:
            return False
        # Low extraction is not a lost game while the army still dominates
        # (a 67-minute game resigned at 5x the enemy's army power).
        cache = getattr(self, "_power_cache", None)
        if cache is not None and cache[1] >= ARMY_GUARD * max(1.0, cache[2]):
            if self.frame - getattr(self, "_resign_guard_log", -10**9) >= 1800:
                self._resign_guard_log = self.frame
                self.log(f"not resigning: army power {cache[1]:.0f} vs enemy {cache[2]:.0f}")
            return False
        self.resigned = True
        self.log(
            f"resigning: extraction {avg:.1f}/s < {LOW_SHARE:.0%} of peak {peak:.1f}/s "
            f"for {(self.frame - self._low_since) / 30:.0f}s; self-destructing {len(self.units)} units"
        )
        for uid in list(self.units):
            self.queue({"type": "self_destruct", "unitId": int(uid)})
        return True
