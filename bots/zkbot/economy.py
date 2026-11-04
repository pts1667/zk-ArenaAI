"""EconomyMixin: the build task board.

economy_step runs the task generators (build_resources.py, build_facilities.py,
build_support.py, build_defense.py), then production and builder assignment.
Tasks live in self.tasks (add_task/finish_task/drop_task/cleanup_tasks).
"""
from __future__ import annotations


from .models import Task


class EconomyMixin:
    def economy_step(self) -> None:
        self.defense_step()
        self.cleanup_tasks()
        self.ensure_mex_tasks()
        self.ensure_mex_defense_tasks()
        self.ensure_cluster_defense_tasks()
        self.ensure_energy_tasks()
        self.ensure_storage_tasks()
        self.ensure_pylon_tasks()
        self.ensure_factory_tasks()
        self.ensure_strider_tasks()
        self.ensure_airpad_tasks()
        self.ensure_caretaker_tasks()
        self.ensure_radar_tasks()
        self.ensure_defense_tasks()
        self.ensure_front_defense_tasks()
        self.ensure_silo_tasks()
        self.ensure_seismic_tasks()
        self.production_step()
        self.release_finished_guards()
        self.reclaim_step()
        self.assign_builders()

    # -------------------------------------------------------------- task setup

    def add_task(
        self,
        key: str,
        kind: str,
        def_id: int,
        x: float,
        z: float,
        priority: float,
        *,
        anchor: tuple[float, float] | None = None,
        spot_index: int | None = None,
        max_assignees: int = 1,
    ) -> Task:
        t = Task(
            key=key,
            kind=kind,
            def_id=int(def_id),
            x=float(x),
            z=float(z),
            priority=float(priority),
            created=self.frame_time(),
            created_frame=self.frame,
            anchor=anchor,
            spot_index=spot_index,
            max_assignees=max_assignees,
        )
        self.tasks[key] = t
        return t

    def cleanup_tasks(self) -> None:
        for key, t in list(self.tasks.items()):
            # Drop assignees that no longer point at this task.
            for uid in list(t.assignees):
                if self.builder_task.get(uid) != key or uid not in self.units:
                    t.assignees.discard(uid)
            # Completed?
            if self.task_completed(t):
                self.finish_task(key)
                continue
            # If a building for this task already exists (even unfinished),
            # keep the task alive; abandoning it would leave a half-built
            # factory or reactor.
            for u in self.units.values():
                if u.def_id == t.def_id and not u.is_finished:
                    if self.dist(u.pos, (t.x, t.z)) < 220:
                        t.created_frame = self.frame
                        break
            age = self.frame - t.created_frame
            if t.unplaceable and age > 120:
                self.drop_task(key, "unplaceable")
            elif t.kind in ("energy", "fusion") and not t.assignees and age > 600:
                self.drop_task(key, "unassigned")
            elif not t.assignees and age > 6000:
                # Long walks and a busy constructor queue are normal; dropping
                # an expansion job after 60 seconds caused constant churn.
                self.drop_task(key, "stale")
            elif age > 9000 and not t.assignees:
                self.drop_task(key, "too old")

    def task_completed(self, t: Task) -> bool:
        # An unassigned energy job cannot be completed by a generator that
        # happened to finish nearby.  That was creating a stream of "task
        # done" messages while the solar count and income stayed flat.
        if t.kind in ("energy", "fusion") and not t.assignees:
            return False
        if t.kind == "mex" and t.spot_index is not None:
            if self.map.spots[t.spot_index].status == "ours":
                return True
            # Also accept an already built extractor at the task position.
            # This handles the case where the extractor was completed before
            # the task was created (for example after a status update race).
            for u in self.units.values():
                if u.name != "staticmex" or not u.is_finished:
                    continue
                if self.dist(u.pos, (t.x, t.z)) < 130.0:
                    return True
        if not t.placed:
            return False
        if t.kind == "energy":
            radius = 40.0
        elif t.kind == "fusion":
            radius = 120.0
        elif t.kind == "factory":
            radius = 190.0
        else:
            radius = 160.0
        for u in self.units.values():
            if not u.is_finished or u.def_id != t.def_id:
                continue
            # Only a building that appeared after the task was created can
            # complete it.  Using finished_frame here let an old generator
            # complete a long stream of new energy tasks as soon as its
            # finish was first observed, so the economy never grew.
            if t.kind in ("energy", "fusion"):
                if u.first_seen < t.created_frame:
                    continue
            elif u.finished_frame < t.created_frame:
                continue
            if self.dist(u.pos, (t.x, t.z)) < radius:
                return True
        return False

    def finish_task(self, key: str) -> None:
        t = self.tasks.pop(key, None)
        if not t:
            return
        for uid in list(t.assignees):
            if self.builder_task.get(uid) == key:
                self.builder_task.pop(uid, None)
                self.builder_idle[uid] = True
                self.builder_job_frame.pop(uid, None)
        if t.kind == "mex" and t.spot_index is not None:
            s = self.map.spots[t.spot_index]
            s.status = "ours"
            s.task_id = None
        if t.kind == "defense":
            self.defense_built_near.append((t.x, t.z))
        if t.kind in ("energy", "fusion", "factory", "caretaker", "radar"):
            self.log(f"task done {key}")

    def drop_task(self, key: str, reason: str) -> None:
        t = self.tasks.pop(key, None)
        if not t:
            return
        for uid in list(t.assignees):
            if self.builder_task.get(uid) == key:
                self.builder_task.pop(uid, None)
                self.builder_idle[uid] = True
                self.builder_job_frame.pop(uid, None)
        if t.kind == "mex" and t.spot_index is not None:
            s = self.map.spots[t.spot_index]
            s.status = "free"
            s.task_id = None
        self.log(f"drop task {key} ({reason}), attempts={t.placement_attempts}")
