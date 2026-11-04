"""Constructors that finish things (flags `unstick`, `finishfirst`).

`unstick`: a constructor told to assist a nanoframe guards it, and a guard order
never ends.  The queue is never empty, so the idle check never saw it, and one
constructor stood beside a finished solar for a whole game.  Guards are now
tracked and let go as soon as the building is finished or gone.

`finishfirst`: with metal short, every new nanoframe slows all the others.  Above a
limit that grows with income, constructors help what's already started instead of
starting more.
"""
from __future__ import annotations

import math

from .models import UnitInfo
from .params import FLAGS, P

FRAME_INCOME = P("focus.frame_income", 15.0, 5.0, 40.0, "metal income per extra unfinished building allowed")
MIN_FRAMES = P("focus.min_frames", 2, 1, 6, "unfinished buildings always allowed")
MAX_FRAMES = P("focus.max_frames", 8, 2, 20, "unfinished buildings ever limited to")
SPARE_METAL = P("focus.spare_metal", 150.0, 0.0, 1000.0, "with this much metal banked there is no limit")
URGENT_PRIORITY = P("focus.urgent", 1600.0, 1000.0, 3000.0, "a job this urgent is started anyway")
HELP_REACH = P("focus.reach", 2000.0, 500.0, 5000.0, "walk this far at most to help")
MAX_HELPERS = P("focus.helpers", 4, 1, 12, "constructors sent to help one building")
WAITING_JOBS = ("mex", "mexllt", "pylon")
JOBS_PER_CONSTRUCTOR = P("focus.jobs_per_con", 3.0, 1.0, 10.0, "safe waiting expansion jobs that justify a constructor")
MAX_FOR_JOBS = P("focus.max_for_jobs", 4, 0, 10, "constructors added for waiting jobs, at most")
JOBS_COUNT_AFTER = P("focus.jobs_after", 120.0, 0.0, 600.0, "seconds before waiting jobs count: the opening stays lean")
HELP_FRAMES = 900             # frames a helper stays off the job board; released sooner when it's done
DANGER_RADIUS = 350.0


class BuildFocusMixin:
    def constructors_for_waiting_jobs(self) -> int:
        """Income can't be the only measure: it needs extractors, which need constructors."""
        if self.frame_time() < JOBS_COUNT_AFTER:
            return 0
        waiting = sum(1 for t in self.tasks.values()
                      if t.kind in WAITING_JOBS and not t.assignees and not t.unplaceable
                      and self.site_safe_for_builder((t.x, t.z), t.kind))
        return int(min(MAX_FOR_JOBS, math.ceil(waiting / JOBS_PER_CONSTRUCTOR)))

    def note_guard(self, uid: int, target: int, key: str | None) -> None:
        if "unstick" in FLAGS or "finishfirst" in FLAGS:
            self.__dict__.setdefault("builder_guards", {})[uid] = (target, key)

    def guard_is_current(self, uid: int, key: str | None) -> bool:
        """False once the constructor has been given other work."""
        if key:
            return self.builder_task.get(uid) == key
        return self.frame < self.builder_fallback_until.get(uid, -1)

    def release_finished_guards(self) -> None:
        guards = self.__dict__.get("builder_guards")
        if not guards:
            return
        for uid, (target, key) in list(guards.items()):
            u = self.units.get(uid)
            if u is None or not self.guard_is_current(uid, key):
                guards.pop(uid)
                continue
            b = self.units.get(target)
            if b is not None and not b.is_finished:
                # Off the job board, so builder_safety_step doesn't look after helpers.
                danger = self.enemy_near(u.pos, DANGER_RADIUS) if key is None else None
                if danger is not None:
                    guards.pop(uid)
                    self.retreat_builder(u, danger.pos)
                continue
            guards.pop(uid)
            self.release_guard(uid, key, b)

    def release_guard(self, uid: int, key: str | None, building: UnitInfo | None) -> None:
        t = self.tasks.get(key) if key else None
        if t is not None and building is not None and building.def_id == t.def_id:
            self.finish_task(key)
        elif t is not None:
            t.assignees.discard(uid)
        self.builder_task.pop(uid, None)
        self.builder_idle[uid] = True
        self.builder_job_frame.pop(uid, None)
        self.builder_fallback_until.pop(uid, None)
        self.log(f"guard over {uid}: {'done' if building is not None else 'building gone'} ({key})")

    def unfinished_structures(self) -> list[UnitInfo]:
        return [u for u in self.units.values() if u.is_being_built and not u.is_mobile]

    def frame_limit(self) -> int:
        return int(max(MIN_FRAMES, min(MAX_FRAMES, 1 + self.metal_income() / FRAME_INCOME)))

    def too_many_frames(self, frames: list[UnitInfo]) -> bool:
        return len(frames) >= self.frame_limit() and self.metal_current() < SPARE_METAL

    def frame_to_help(self, u: UnitInfo, frames: list[UnitInfo]) -> UnitInfo | None:
        helpers: dict[int, int] = {}
        for target, _key in self.__dict__.get("builder_guards", {}).values():
            helpers[target] = helpers.get(target, 0) + 1
        best, best_d = None, HELP_REACH
        for b in frames:
            d = self.dist(u.pos, b.pos)
            if d >= best_d or helpers.get(b.uid, 0) >= MAX_HELPERS:
                continue
            if not self.site_safe_for_builder(b.pos, "") or self.path_threat(u.pos, b.pos) > 60.0:
                continue
            if self.hazards is not None and not self.hazards.can_walk(u.name, u.pos, b.pos):
                continue
            best, best_d = b, d
        return best

    def focus_on_frames(self, u: UnitInfo, best_priority: float | None) -> bool:
        """Sends this constructor to help, if too much is under way already."""
        if "finishfirst" not in FLAGS or not u.is_mobile:
            return False
        if best_priority is not None and best_priority >= URGENT_PRIORITY:
            return False
        frames = self.unfinished_structures()
        if not self.too_many_frames(frames):
            return False
        b = self.frame_to_help(u, frames)
        if b is None:
            return False
        self.queue({"type": "guard", "unitId": int(u.uid), "targetId": int(b.uid)})
        self.builder_fallback_until[u.uid] = self.frame + HELP_FRAMES
        self.note_guard(u.uid, b.uid, None)
        self.log(f"help {u.name}#{u.uid} -> {b.name}#{b.uid} ({len(frames)} unfinished, limit {self.frame_limit()})")
        return True
