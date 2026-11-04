"""Constructors put to reclaiming wrecks (flag `reclaimcons`).

Reclaim used to be what an idle constructor did when no job was left, which on a
busy board meant never.  Now, with at least 700 metal lying safe, one or two
constructors are told off for it, and so is any constructor beyond what metal
income calls for.  The commander stays out of it.
"""
from __future__ import annotations

from .models import UnitInfo
from .params import FLAGS, P

WORTH_METAL = P("reclaim.worth", 700.0, 100.0, 3000.0, "safe wreck metal that always gets a crew")
SURPLUS_METAL = P("reclaim.surplus", 150.0, 30.0, 1000.0, "safe wreck metal that surplus constructors go for")
MIN_WRECK = P("reclaim.min_wreck", 10.0, 1.0, 100.0, "metal in a wreck worth walking to")
SAFE_RADIUS = P("reclaim.safe_radius", 250.0, 100.0, 600.0, "no threat this close to a wreck")
ENEMY_RADIUS = P("reclaim.enemy_radius", 500.0, 200.0, 1200.0, "no enemy seen this close to a wreck")
REACH = P("reclaim.reach", 3500.0, 800.0, 8000.0, "walk this far at most")
HOME_SIDE = P("reclaim.home_side", 1.3, 0.8, 3.0, "a wreck may be this many times further from our base than the enemy's")
FULL_SHARE = P("reclaim.full", 0.8, 0.3, 1.0, "storage this full and metal is no use")
FLEE_RADIUS = 350.0
CLUSTER_RADIUS = 300.0        # wrecks this close count together when choosing where to go
STEP_FRAMES = 30
JOB_FRAMES = 1800             # a wreck that takes longer than this is unreachable
BAD_WRECK_FRAMES = 3600
BUSY_JOBS = ("mex", "factory", "defense", "mexdef", "mexllt")


class ReclaimMixin:
    def safe_wrecks(self) -> list[tuple[int, tuple[float, float], float]]:
        """(feature id, position, metal) of wrecks on our side with no threat around."""
        bad = self.__dict__.setdefault("reclaim_bad", {})
        out = []
        for fid, feat in (getattr(self, "_feature_cache", None) or {}).items():
            fid = int(fid)
            contained = (feat.get("definition") or {}).get("containedResources") or {}
            metal = float(contained.get("Metal") or contained.get("metal") or 0.0)
            if metal < MIN_WRECK or bad.get(fid, -1) > self.frame:
                continue
            p = feat.get("pos") or [0.0, 0.0, 0.0]
            pos = (float(p[0]), float(p[2]))
            if self.enemy_base is not None and self.dist(pos, self.base_pos) > HOME_SIDE * self.dist(pos, self.enemy_base):
                continue
            if self.threat_in_radius(pos, SAFE_RADIUS) > 0.0 or self.enemy_near(pos, ENEMY_RADIUS) is not None:
                continue
            if self.hazards is not None and self.hazards.site_bad(pos[0], pos[1]):
                continue
            out.append((fid, pos, metal))
        return out

    def reclaim_crew(self, metal: float, constructors: int) -> int:
        """How many constructors should be reclaiming this much metal."""
        if constructors <= 0 or self.metal_current() > FULL_SHARE * self.effective_storage(self.metal()):
            return 0
        want = 0
        if metal >= SURPLUS_METAL:
            want = max(0, self.builder_count() - self.desired_builder_count())
        # A crew at work carries on as the wrecks run down, instead of leaving at 699.
        working = bool(self.__dict__.get("reclaim_jobs")) and metal >= SURPLUS_METAL
        if metal >= WORTH_METAL or working:
            want = max(want, 2 if constructors >= 3 else 1)
        return min(want, max(1, constructors // 2))

    def reclaim_constructors(self) -> list[UnitInfo]:
        return [u for u in self.units.values()
                if u.is_builder and u.is_mobile and u.is_finished and u.uid != self.commander_id]

    def best_wreck(self, u: UnitInfo, wrecks: list) -> tuple | None:
        """The most metal for the walk, counting the wrecks around it, and not one
        another constructor is already on."""
        taken = {fid for fid, _started in self.reclaim_jobs.values()}
        best, best_score = None, 0.0
        for wreck in wrecks:
            fid, pos, _metal = wreck
            d = self.dist(u.pos, pos)
            if fid in taken or d > REACH or self.path_threat(u.pos, pos) > 60.0:
                continue
            if self.hazards is not None and not self.hazards.can_walk(u.name, u.pos, pos):
                continue
            pile = sum(m for _f, p, m in wrecks if self.dist(p, pos) < CLUSTER_RADIUS)
            score = pile / (d + 300.0)
            if score > best_score:
                best, best_score = wreck, score
        return best

    def delegate_cost(self, u: UnitInfo, wrecks: list) -> float:
        """Idle constructors first, then those on jobs that can most easily wait."""
        key = self.builder_task.get(u.uid)
        t = self.tasks.get(key) if key else None
        job = 0.0 if t is None else (3000.0 if t.kind in BUSY_JOBS else 1000.0)
        return job + min(self.dist(u.pos, pos) for _fid, pos, _metal in wrecks)

    def start_reclaim(self, u: UnitInfo, wreck: tuple) -> None:
        fid, pos, metal = wreck
        self.reclaim_jobs[u.uid] = (fid, self.frame)
        key = self.builder_task.pop(u.uid, None)
        if key and key in self.tasks:
            self.tasks[key].assignees.discard(u.uid)
        self.builder_job_frame.pop(u.uid, None)
        self.queue({"type": "reclaim", "unitId": int(u.uid), "featureId": int(fid)})
        self.log(f"reclaim crew {u.name}#{u.uid} -> {metal:.0f}M d={self.dist(u.pos, pos):.0f}")

    def end_reclaim(self, uid: int) -> None:
        self.reclaim_jobs.pop(uid, None)
        self.builder_fallback_until.pop(uid, None)
        self.builder_idle[uid] = True

    def reclaim_flee(self) -> None:
        """Off the job board, so builder_safety_step doesn't look after the crew."""
        for uid in list(self.reclaim_jobs):
            u = self.units.get(uid)
            if u is None:
                self.reclaim_jobs.pop(uid)
                continue
            danger = self.enemy_near(u.pos, FLEE_RADIUS)
            if danger is not None:
                self.reclaim_jobs.pop(uid)
                self.retreat_builder(u, danger.pos)

    def reclaim_step(self) -> None:
        if "reclaimcons" not in FLAGS or self.frame - self.__dict__.get("reclaim_frame", -10**9) < STEP_FRAMES:
            return
        self.reclaim_frame = self.frame
        self.reclaim_jobs = self.__dict__.get("reclaim_jobs", {})
        self.reclaim_flee()
        jobs = self.reclaim_jobs
        wrecks = self.safe_wrecks()
        crew = self.reclaim_constructors()
        want = self.reclaim_crew(sum(w[2] for w in wrecks), len(crew))
        # Too many: the ones furthest from any wreck go back to the board.
        nearest = lambda uid: min((self.dist(self.units[uid].pos, w[1]) for w in wrecks), default=0.0)
        for uid in sorted(jobs, key=nearest)[want:]:
            self.end_reclaim(uid)
        if wrecks:
            free = sorted((u for u in crew if u.uid not in jobs), key=lambda u: self.delegate_cost(u, wrecks))
            for u in free[:max(0, want - len(jobs))]:
                wreck = self.best_wreck(u, wrecks)
                if wreck is not None:
                    self.start_reclaim(u, wreck)
        self.keep_reclaiming(wrecks)

    def keep_reclaiming(self, wrecks: list) -> None:
        """On to the next wreck when this one's gone, or giving up on one that can't be reached."""
        for uid, (fid, started) in list(self.reclaim_jobs.items()):
            # A wreck stays the job while it exists.  Going by the safe list had the
            # crew turn around each time a threat flickered near it; real danger to the
            # constructor is reclaim_flee's business.
            overdue = self.frame - started > JOB_FRAMES
            if overdue:
                self.reclaim_bad[fid] = self.frame + BAD_WRECK_FRAMES
            gone = str(fid) not in self._feature_cache and fid not in self._feature_cache
            if overdue or gone:
                self.reclaim_jobs.pop(uid)
                wreck = self.best_wreck(self.units[uid], [w for w in wrecks if w[0] != fid])
                if wreck is None:
                    self.end_reclaim(uid)
                    continue
                self.start_reclaim(self.units[uid], wreck)
            self.builder_fallback_until[uid] = self.frame + 2 * STEP_FRAMES
