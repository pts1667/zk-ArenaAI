"""Burrowed metal extractors.

A heavily contested metal spot is dug ~55 elmos down before the extractor is
built in the pit.  Raiders, skirmishers and flat-trajectory artillery then
have to come right up to the rim to hit it.  The dig uses Zero-K's own
"level to height" terraform (the same orders the build-height widget sends):

  1. CMD_TERRAFORM_INTERNAL (39801) sets up the terraform over the footprint,
  2. CMD_LEVEL (39736) with the same tag makes the constructor work on it,
  3. the extractor build order is appended behind it.

At 55 elmos over a 48x48 footprint the gadget's cost formula gives about 45
terraform (volume ~17, perimeter ~18, base 6, slope ~4), inside the 30-100
that is worth paying.
"""
from __future__ import annotations

from .client import QueryError
from .models import Task, UnitInfo
from .params import FLAGS

CMD_TERRAFORM_INTERNAL = 39801
CMD_LEVEL = 39736
BURROW_DEPTH = 55.0
MEX_HALF = 23.9  # 3x3 footprint, as the build-height widget sizes it


class BurrowMixin:
    def note_mex_lost(self, u: UnitInfo) -> None:
        idx = self.spot_index_at(u.pos)
        if idx is None:
            return
        losses = getattr(self, "spot_mex_losses", None)
        if losses is None:
            losses = self.spot_mex_losses = {}
        losses[idx] = losses.get(idx, 0) + 1

    def spot_index_at(self, pos) -> int | None:
        best, best_d = None, 110.0
        for i, s in enumerate(self.map.spots):
            d = self.dist((s.x, s.z), pos)
            if d < best_d:
                best, best_d = i, d
        return best

    def spot_contested(self, idx: int) -> bool:
        if "burrowall" in FLAGS:
            return True  # test mode
        lost = getattr(self, "spot_mex_losses", {}).get(idx, 0)
        if lost >= 2:
            return True
        if lost >= 1 and self.enemy_base is not None:
            s = self.map.spots[idx]
            # Frontier half of the map: one loss is enough.
            return self.dist((s.x, s.z), self.enemy_base) < self.dist((s.x, s.z), self.base_pos) * 1.15
        return False

    def try_burrow_mex(self, u: UnitInfo, t: Task) -> bool:
        if "noburrow" in FLAGS:
            return False
        if t.spot_index is None or not self.spot_contested(int(t.spot_index)):
            return False
        done = getattr(self, "burrowed_spots", None)
        if done is None:
            done = self.burrowed_spots = {}
        x, z = float(t.x), float(t.z)
        try:
            res = self.client.query({"type": "elevation_at", "pos": [x, 0.0, z]})
            self.debug_queries += 1
        except QueryError:
            return False
        h = None
        if isinstance(res, (int, float)):
            h = float(res)
        elif isinstance(res, dict):
            for k in ("elevation", "height", "y", "value"):
                if res.get(k) is not None:
                    h = float(res[k])
                    break
        if h is None:
            if not getattr(self, "_elev_logged", False):
                self._elev_logged = True
                self.log(f"burrow: unexpected elevation_at result {res!r}")
            return False
        idx = int(t.spot_index)
        attempts = getattr(self, "burrow_attempts", None)
        if attempts is None:
            attempts = self.burrow_attempts = {}
        if idx in done:
            # The target is fixed the first time: an interrupted dig must be
            # finished to the same height, not restarted 55 elmos deeper.
            target = done[idx]
            if h <= target + 8.0 or attempts.get(idx, 0) >= 3:
                return False
        else:
            depth = min(BURROW_DEPTH, h - 2.0)  # never dig below the waterline
            if depth < 25.0:
                return False
            target = h - depth
        attempts[idx] = attempts.get(idx, 0) + 1
        self.terra_tag = getattr(self, "terra_tag", 0) + 1
        tag = float(self.terra_tag)
        a = MEX_HALF
        params = [
            1.0, float(self.team_id), x, z, tag, 1.0, target, 5.0, 1.0, 0.0,
            x + a, z + a, x + a, z - a, x - a, z - a, x - a, z + a, x + a, z + a,
            float(u.uid),
        ]
        self.queue({"type": "custom", "unitId": int(u.uid), "cmdId": CMD_TERRAFORM_INTERNAL, "params": params})
        self.queue({"type": "custom", "unitId": int(u.uid), "cmdId": CMD_LEVEL, "params": [x, h, z, tag]})
        self.queue(
            {
                "type": "build",
                "unitId": int(u.uid),
                "defId": int(t.def_id),
                "pos": [x, 0.0, z],
                "facing": int(t.facing),
                "options": 32,
            }
        )
        done[idx] = target
        self.burrow_count = getattr(self, "burrow_count", 0) + 1
        self.log(
            f"burrow mex spot {t.spot_index} at ({x:.0f},{z:.0f}) ground={h:.0f} -> {target:.0f} "
            f"(losses {getattr(self, 'spot_mex_losses', {}).get(int(t.spot_index), 0)}) by {u.name}#{u.uid}"
        )
        return True

    def burrow_check(self) -> None:
        done = getattr(self, "burrowed_spots", {})
        seen = getattr(self, "_burrow_seen", None)
        if seen is None:
            seen = self._burrow_seen = set()
        for u in self.units.values():
            if u.name != "staticmex" or not u.is_finished or u.uid in seen:
                continue
            idx = self.spot_index_at(u.pos)
            if idx in done:
                seen.add(u.uid)
                self.log(f"burrowed mex {u.uid} spot {idx} finished at y={u.y:.0f} (target {done[idx]:.0f})")
