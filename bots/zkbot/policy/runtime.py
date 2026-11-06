"""The policy's place in the live bot, behind flags.  With none of them set this does nothing, and imports nothing heavy.

Flags:  rawframe      record raw frames and the static bundle, for golden tests
        policyshadow  build frames and run the whole forward pass, logging its cost, and send no commands

The flag `policy` itself, where the network commands the army, comes with stage 2's hand-off (P-M4).
"""
from __future__ import annotations

import os
import pathlib
import time
import traceback
from typing import Any

from ..params import FLAGS

DECISION_FRAMES = 15
RECORD_EVERY = 20             # decisions: a frame every 10 s
SAVE_EVERY_FRAMES = 1800      # game frames between writes of the recording, since a game can end without warning
LOG_EVERY_FRAMES = 1800
BUDGET_MS = 60.0


def _enabled() -> bool:
    return "rawframe" in FLAGS or "policyshadow" in FLAGS


class PolicyMixin:
    def policy_init(self) -> None:
        self.policy_on = _enabled()
        if not self.policy_on:
            return
        from .rawframe import LiveMemory

        self.policy_memory = LiveMemory()
        self.policy_bundle = None
        self.policy_table = None
        self.policy_net = None
        self.policy_state = None
        self.policy_frame_at = -10**9
        self.policy_decisions = 0
        self.policy_recorded: list[Any] = []
        self.policy_saved_at = 0
        self.policy_logged_at = 0
        self.policy_times: list[tuple[float, float, int]] = []
        self.policy_failures = 0
        if "policyshadow" in FLAGS:
            # Decide on every 15th frame: this is the rate policy mode runs at, so it's the rate to measure.
            self.block_setting = (5, 5)

    def policy_events(self, obs: dict[str, Any]) -> None:
        if self.policy_on:
            self.policy_memory.note_events(obs)

    def policy_setup(self) -> bool:
        """Once per game, at the first decision.  The bundle reads live heights: gadgets can move the terrain after load."""
        from .bundle import from_terrain
        from .tables import table
        from ..hazards import Hazards

        hazards = self.hazards or Hazards.build(self.client, self.map, self.log)
        if hazards is None:
            self.policy_on = False
            self.log("policy: no terrain, so no policy this game")
            return False
        self.policy_table = table()
        self.policy_bundle = from_terrain(hazards.terrain)
        out = pathlib.Path(os.environ.get("ZK_ARENA_OUT", "."))
        if "rawframe" in FLAGS:
            self.policy_bundle.save(out / f"bundle-team{self.team_id}.npz")
        if "policyshadow" in FLAGS:
            self._policy_load_net()
        self.log(f"policy: {self.policy_bundle.regions} regions, catalog {self.policy_table.hash}, net {'yes' if self.policy_net else 'no'}")
        return True

    def _policy_load_net(self) -> None:
        import numpy as np

        from . import net

        path = os.environ.get("ZKBOT_POLICY") or str(pathlib.Path(__file__).resolve().parent.parent / "data" / "policy.npz")
        try:
            weights = net.load(path, None)
        except (OSError, ValueError, KeyError) as exc:
            self.log(f"policy: no weights from {path} ({exc!r}); shadow mode off")
            return
        self.policy_net = net.Policy(weights)
        self.policy_state = net.State.initial(self.policy_bundle.regions)
        self.policy_rng = np.random.default_rng(self.team_id)

    def policy_step(self, obs: dict[str, Any]) -> None:
        """Called every frame; acts once per decision.  Nothing here can stop the frame loop."""
        if not self.policy_on or self.frame - self.policy_frame_at < DECISION_FRAMES:
            return
        self.policy_frame_at = self.frame
        try:
            if self.policy_bundle is None and not self.policy_setup():
                return
            self._policy_decision(obs)
            self.policy_failures = 0
        except Exception:
            self.policy_failures += 1
            self.log("policy error:\n" + traceback.format_exc())
            if self.policy_failures >= 2:
                self.policy_on = False
                self.log("policy: two errors in a row, off for this game")

    def _policy_decision(self, obs: dict[str, Any]) -> None:
        from . import rawframe

        t0 = time.perf_counter()
        bundle = self.policy_bundle
        frame = rawframe.from_brain(self, obs, self.policy_memory, region_of=lambda x, z: bundle.region_at(x, z))
        self.policy_decisions += 1
        if "rawframe" in FLAGS and self.policy_decisions % RECORD_EVERY == 1:
            self.policy_recorded.append(frame)
            if self.frame - self.policy_saved_at >= SAVE_EVERY_FRAMES:
                self._policy_save()
        if self.policy_net is None:
            return
        from . import actions
        from . import featurize as fz

        t1 = time.perf_counter()
        tensors = fz.featurize(frame, bundle, self.policy_table)
        t2 = time.perf_counter()
        decision = self.policy_net.decide(tensors, self.policy_state, self.policy_rng)
        self.policy_state = decision.state
        x, z = actions.world_positions(tensors, bundle)
        orders = actions.decode(decision, tensors, bundle, x, z)     # read, and thrown away: shadow mode sends nothing
        t3 = time.perf_counter()
        self.policy_times.append(((t3 - t0) * 1000.0, (t2 - t1) * 1000.0, len(tensors.tok_kind)))
        if self.frame - self.policy_logged_at >= LOG_EVERY_FRAMES:
            self._policy_log(len(orders), tensors.controllable)

    def _policy_log(self, orders: int, controllable: int) -> None:
        import numpy as np

        self.policy_logged_at = self.frame
        rows = np.array(self.policy_times, np.float64)
        if not len(rows):
            return
        total = rows[:, 0]
        self.log(f"policy shadow frame={self.frame} decisions={self.policy_decisions} ms mean={total.mean():.1f} p99={np.percentile(total, 99):.1f} "
                 f"max={total.max():.1f} featurize={rows[:, 1].mean():.1f} tokens mean={rows[:, 2].mean():.0f} max={rows[:, 2].max():.0f} "
                 f"over_budget={(total > BUDGET_MS).mean():.3f} controllable={controllable} last_orders={orders}")

    def _policy_save(self) -> None:
        from . import rawframe

        self.policy_saved_at = self.frame
        out = pathlib.Path(os.environ.get("ZK_ARENA_OUT", "."))
        temp = out / f"rawframes-team{self.team_id}.tmp.npz"
        rawframe.save_frames(temp, self.policy_recorded)
        os.replace(temp, out / f"rawframes-team{self.team_id}.npz")
