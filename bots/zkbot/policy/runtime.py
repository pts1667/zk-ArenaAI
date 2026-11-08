"""The policy's place in the live bot, behind flags.  With none of them set this does nothing, and imports nothing heavy.

Flags:  rawframe      record raw frames and the static bundle, for golden tests (tools/policy_golden.py)

The policy itself plays here once its hand-off is built: the latent actor and its encoder as numpy, its decoded orders sent as
commands.  The shadow mode that timed the first actor's numpy forward went with that actor on 2026-11-08 (git tag
`rules-bot-final`).
"""
from __future__ import annotations

import os
import pathlib
import traceback
from typing import Any

from ..params import FLAGS

DECISION_FRAMES = 15
RECORD_EVERY = 20             # decisions: a frame every 10 s
SAVE_EVERY_FRAMES = 1800      # game frames between writes of the recording, since a game can end without warning


class PolicyMixin:
    def policy_init(self) -> None:
        self.policy_on = "rawframe" in FLAGS
        if not self.policy_on:
            return
        from .rawframe import LiveMemory

        self.policy_memory = LiveMemory()
        self.policy_bundle = None
        self.policy_frame_at = -10**9
        self.policy_decisions = 0
        self.policy_recorded: list[Any] = []
        self.policy_saved_at = 0
        self.policy_failures = 0

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
            self.log("policy: no terrain, so no recording this game")
            return False
        self.policy_bundle = from_terrain(hazards.terrain)
        self.policy_bundle.save(pathlib.Path(os.environ.get("ZK_ARENA_OUT", ".")) / f"bundle-team{self.team_id}.npz")
        self.log(f"policy: {self.policy_bundle.regions} regions, catalog {table().hash}")
        return True

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

        bundle = self.policy_bundle
        frame = rawframe.from_host(self, obs, self.policy_memory, region_of=lambda x, z: bundle.region_at(x, z))
        self.policy_decisions += 1
        if self.policy_decisions % RECORD_EVERY == 1:
            self.policy_recorded.append(frame)
            if self.frame - self.policy_saved_at >= SAVE_EVERY_FRAMES:
                self._policy_save()

    def _policy_save(self) -> None:
        from . import rawframe

        self.policy_saved_at = self.frame
        out = pathlib.Path(os.environ.get("ZK_ARENA_OUT", "."))
        temp = out / f"rawframes-team{self.team_id}.tmp.npz"
        rawframe.save_frames(temp, self.policy_recorded)
        os.replace(temp, out / f"rawframes-team{self.team_id}.npz")
