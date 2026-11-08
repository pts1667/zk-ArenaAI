"""Shared pieces: a small real map, frames made by hand, frames from a script game in the sim, and a small frame encoder."""
from __future__ import annotations

import functools
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "bots"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "tools"))

from zkbot.policy import bundle as bd, rawframe as rf, tables  # noqa: E402
from zkbot import terrain, units  # noqa: E402


@pytest.fixture(scope="session")
def table():
    return tables.table()


@pytest.fixture(scope="session")
def small_map():
    """Two plains joined by a ramp, with a metal spot on each."""
    n = 320
    h = np.full((n, n), 50.0, np.float32)
    h[:, n // 2:] = 250.0
    for x in range(n // 2 - 40, n // 2 + 40):
        h[140:180, x] = 50.0 + 200.0 * (x - (n // 2 - 40)) / 80.0
    t = terrain.TerrainMap(h, spots=[(400.0, 1280.0, 2.0), (2200.0, 1280.0, 2.0)])
    return bd.from_terrain(t)


# Who a hand-made unit is to the viewer, who is player 0 of ally team 0: knowledge, player and ally team.
OWN, ENEMY, BLIP, MEMORY, HIDDEN = "own", "enemy", "blip", "memory", "hidden"
_ROLES = {OWN: (rf.KNOW_LOS, 0, 0), ENEMY: (rf.KNOW_LOS, 1, 1), BLIP: (rf.KNOW_RADAR, 1, 1), MEMORY: (rf.KNOW_MEMORY, 1, 1), HIDDEN: (rf.KNOW_HIDDEN, 1, 1)}


def make_units(*rows):
    """Each row: dict of field overrides.  `name` picks the type, and `side` one of the roles above."""
    out = np.zeros(len(rows), rf.UNIT_DTYPE)
    for i, row in enumerate(rows):
        row = dict(row)
        name = row.pop("name", "cloakraid")
        rec = out[i]
        rec["uid"] = row.pop("uid", i + 1)
        rec["def_id"] = -1 if name is None else units.id_of(name)
        role = row.pop("side", OWN)
        rec["knowledge"], rec["player"], rec["ally_team"] = _ROLES[role]
        rec["x"], rec["z"] = row.pop("x", 500.0), row.pop("z", 500.0)
        rec["y"] = 50.0
        rec["hp_frac"] = row.pop("hp_frac", 1.0)
        rec["build_frac"] = 1.0
        rec["ord_kind"], rec["ord_target"] = -1, -1
        default_flags = (rf.IN_LOS | rf.TYPE_KNOWN) if role in (OWN, ENEMY) else (rf.TYPE_KNOWN if name else 0)
        rec["flags"] = row.pop("flags", default_flags)
        for key, value in row.items():
            rec[key] = value
    return out


def make_frame(units_array, frame=900, bundle=None):
    glob = np.zeros(len(rf.GLOBAL_FIELDS), np.float32)
    glob[rf.G["frame"]] = frame
    glob[rf.G["map_w"]], glob[rf.G["map_h"]] = 2560.0, 2560.0
    return rf.Frame(units_array, np.zeros(0, rf.SITE_DTYPE), glob)


@functools.lru_cache(maxsize=None)
def sim_frames(count: int, seed: int = 5, start: int = 120) -> tuple[tuple[rf.Frame, ...], bd.Bundle]:
    """Seat 0's frames at `count` decisions in a row of a script game in the sim, from `start` decisions in, and its map.  The same
    every time for the same arguments.  Needs the built sim library: a test module that calls it skips without one."""
    import policy_sim_env as se

    env = se.SimEnv(1, seed, ["scripts"], threads=2)
    env.begin()
    for _ in range(start):
        env.step(0.0)
    frames = []
    for _ in range(count):
        f = env.frame(0)
        frames.append(rf.Frame(f.units.copy(), f.sites.copy(), f.glob.copy(), f.features.copy()))
        env.step(0.0)
    return tuple(frames), env.bundle_of(0)


@pytest.fixture(scope="session")
def encoder_path(tmp_path_factory) -> pathlib.Path:
    """A small frame encoder, untrained, saved as tools/frame_vae.py saves one, for an actor to read."""
    import torch

    import frame_model as fm
    import frame_vae as fv

    torch.manual_seed(1)
    path = tmp_path_factory.mktemp("encoder") / "vae.pt"
    cfg = fv.VaeConfig(latent=8, decoder_d=32, decoder_heads=2, decoder_rounds=1, variational=False, two_part=True, questions=True)
    fv.save_vae(path, fv.FrameVae(cfg, fm.EncoderConfig(d=32, heads=2, queries=4, rounds=1, z=0, region_pool=6), tables.table()), 1)
    return path
