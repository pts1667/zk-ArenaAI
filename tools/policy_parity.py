"""Does the numpy forward match PyTorch?  Random frames, and 10 recurrent steps; the bound is 1e-4 (02-policy.md section 6).

  ~/zkml/bin/python tools/policy_parity.py        # in WSL
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bots"))
sys.path.insert(0, str(ROOT / "tools"))
import policy_model as pm  # noqa: E402
from zkbot.policy import bundle as bd, featurize as fz, net, rawframe as rf, tables  # noqa: E402
from zkbot import terrain, units as unit_data  # noqa: E402

BOUND = 1e-4
NAMES = ("cloakraid", "cloakskirm", "cloakriot", "cloakassault", "cloakcon", "staticmex", "tankassault", "jumpraid", "cloaksnipe", "spiderassault", "gunshipraid")


def random_frame(rng: np.random.Generator, size: float) -> rf.Frame:
    n = int(rng.integers(1, 260))
    u = np.zeros(n, rf.UNIT_DTYPE)
    u["uid"] = np.arange(n) + 1
    # Three alliances: ours with an allied player, and two more with two players each.
    u["player"] = rng.choice(6, n, p=(0.45, 0.1, 0.15, 0.1, 0.1, 0.1))
    u["ally_team"] = np.array((0, 0, 1, 1, 2, 2))[u["player"]]
    u["knowledge"] = np.where(u["ally_team"] == 0, rf.KNOW_LOS, rng.choice((rf.KNOW_LOS, rf.KNOW_RADAR, rf.KNOW_MEMORY), n, p=(0.6, 0.3, 0.1)))
    names = [str(rng.choice(NAMES)) for _ in range(n)]
    u["def_id"] = [unit_data.id_of(x) for x in names]
    u["x"], u["z"] = rng.uniform(0, size, n), rng.uniform(0, size, n)
    u["y"] = rng.uniform(0, 300, n)
    u["vx"], u["vz"] = rng.normal(0, 1.5, n), rng.normal(0, 1.5, n)
    u["hp_frac"] = rng.uniform(0.05, 1, n)
    u["build_frac"] = 1.0
    u["flags"] = rf.IN_LOS | rf.TYPE_KNOWN
    blips = u["knowledge"] == rf.KNOW_RADAR
    u["def_id"][blips] = -1
    u["hp_frac"][blips] = np.nan
    u["flags"][blips] = 0
    u["ord_kind"], u["ord_target"] = -1, -1
    aimable = u["uid"][(u["ally_team"] != 0) & (u["knowledge"] != rf.KNOW_MEMORY)]
    shooters = np.flatnonzero((u["player"] == 0) & (rng.random(n) < 0.5))
    if len(aimable) and len(shooters):
        u["ord_kind"][shooters] = rf.ATTACK_UNIT
        u["ord_target"][shooters] = rng.choice(aimable, len(shooters))
    glob = rng.uniform(0, 100, len(rf.GLOBAL_FIELDS)).astype(np.float32)
    glob[rf.G["player"]], glob[rf.G["ally_team"]] = 0, 0
    sites = np.zeros(int(rng.integers(0, 10)), rf.SITE_DTYPE)
    sites["x"], sites["z"], sites["income"] = rng.uniform(0, size, len(sites)), rng.uniform(0, size, len(sites)), 2.0
    features = np.zeros(int(rng.integers(0, 2 * fz.CAP_FEATURES)), rf.FEATURE_DTYPE)
    features["feat_id"], features["def_id"] = np.arange(len(features)), -1
    features["x"], features["z"], features["metal"] = rng.uniform(0, size, len(features)), rng.uniform(0, size, len(features)), rng.uniform(1, 400, len(features))
    return rf.Frame(u, sites, glob, features)


def main() -> None:
    torch.manual_seed(0)
    table = tables.table()
    vocabulary = len(table.names) + 1
    actor = pm.Actor(vocabulary).eval()
    for p in actor.parameters():
        if p.dim() > 1:
            torch.nn.init.normal_(p, std=0.1)
    weights = {k: v.detach().numpy().astype(np.float32) for k, v in actor.state_dict().items()}
    print("actor, critic parameters:", pm.parameter_counts(vocabulary))
    missing = set(net.weight_shapes(vocabulary)) ^ set(weights)
    assert not missing, missing
    policy = net.Policy(weights)

    n = 320
    h = np.full((n, n), 50.0, np.float32)
    h[:, n // 2:] = 250.0
    for x in range(n // 2 - 40, n // 2 + 40):
        h[140:180, x] = 50.0 + 200.0 * (x - (n // 2 - 40)) / 80.0
    bundle = bd.from_terrain(terrain.TerrainMap(h, spots=[(400.0, 1280.0, 2.0), (2200.0, 1280.0, 2.0)]))
    rng = np.random.default_rng(1)
    worst = 0.0
    state = net.State.initial(bundle.regions)
    glob, regions = pm.initial_state(1, bundle.regions, "cpu")
    for frame_number in range(100):
        t = fz.featurize(random_frame(rng, bundle.map_w), bundle, table)
        if frame_number % 10 == 0:                         # 10 recurrent steps from a fresh state, ten times
            state = net.State.initial(bundle.regions)
            glob, regions = pm.initial_state(1, bundle.regions, "cpu")
        enc = net.encode(weights, policy.bias_table, t, state)
        b = pm.collate([t]).to("cpu")
        with torch.no_grad():
            _, units, reg_h, new_glob, new_regions = actor.encode(b, glob, regions)
            kind, region, enemy, enemy_h = actor.heads(b, units, reg_h)
        pairs = [
            (enc.units, units[0, :len(t.unit_x)]), (enc.regions, reg_h[0, :len(t.reg_x)]),
            (enc.state.glob, new_glob[0]), (enc.state.regions, new_regions[0]),
        ]
        if t.controllable:
            np_kind, np_region = net.kind_logits(weights, enc, t), net.region_logits(weights, enc, t)
            np_jump = net.region_logits(weights, enc, t, np.full(t.controllable, rf.JUMP))
            walk, jump = region.masked_fill(~b.ptr_region, pm.ILLEGAL), region.masked_fill(~b.ptr_jump, pm.ILLEGAL)
            pairs += [(np_kind, kind[0, :t.controllable]), (np_region, walk[0, :t.controllable, :len(t.reg_x)]), (np_jump, jump[0, :t.controllable, :len(t.reg_x)])]
            if len(t.enemy_tokens):
                np_enemy = net.enemy_logits(weights, enc, t)
                pairs.append((np_enemy, enemy[0, :, :t.controllable, :len(t.enemy_tokens)]))
            target = enc.regions[np.zeros(t.controllable, np.int64)]
            np_fine = net.fine_logits(weights, enc.units[:t.controllable], target)
            with torch.no_grad():
                fine = actor.fine(units[0, :t.controllable], reg_h[0, np.zeros(t.controllable, np.int64)])
            pairs.append((np_fine, fine))
        for a, tensor in pairs:
            worst = max(worst, float(np.abs(a - tensor.numpy()).max()))
        state, glob, regions = enc.state, new_glob, new_regions
    print(f"100 random frames, 10 recurrent steps each: worst absolute difference {worst:.2e}, bound {BOUND:.0e}")
    if worst > BOUND:
        raise SystemExit("parity failed")


if __name__ == "__main__":
    main()
