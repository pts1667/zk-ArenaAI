"""Golden frames for the featurizer: picks frames recorded in live games, and writes what the featurizer makes of them.

  python tools/policy_golden.py frames results/pb-shadow-rec          # the frames and bundle, once, from a recording
  python tools/policy_golden.py expected                              # expected_*.npz, after a deliberate featurizer change
  ~/zkml/bin/python tools/policy_golden.py sim                        # in WSL: frames the sim emits, and a live frame recreated in the sim

Run `expected`, read what it prints, and only commit it if the change was meant.  tests/policy/test_golden.py compares.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bots"))
from zkbot.policy import bundle as bd, featurize as fz, rawframe as rf, tables  # noqa: E402
from zkbot import units  # noqa: E402

DATA = ROOT / "tests" / "policy" / "data"
# Game, and frame number nearest: chosen to show a fight, blips, ghosts, sites changing hands and an idle army.
PICKS = (("game-001", 10968), ("game-002", 18198), ("game-002", 36198), ("game-003", 12090), ("game-003", 23610))
ARRAYS = ("unit_x", "unit_name", "unit_ally", "unit_player", "reg_x", "site_x", "feat_x", "glob_x", "tok_kind", "kind_mask", "ptr_enemy", "ptr_region", "ptr_jump", "ptr_wait", "ptr_produce", "own_beacons", "beacon_regions", "route_bias", "uids", "enemy_tokens", "north", "region_radius")


def pick_frames(recording: pathlib.Path) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    picked, bundle_file = [], None
    for game, wanted in PICKS:
        files = sorted((recording / game).glob("rawframes-team*.npz"))
        frames = rf.load_frames(files[0])
        picked.append(min(frames, key=lambda f: abs(float(f.glob[rf.G["frame"]]) - wanted)))
        bundle_file = sorted((recording / game).glob("bundle-team*.npz"))[0]
    rf.save_frames(DATA / "golden_frames.npz", picked)
    with np.load(bundle_file) as d:
        arrays = {k: d[k] for k in d.files}
    if "spot_list" not in arrays:
        sites = picked[0].sites
        arrays["spot_list"] = np.stack((sites["x"], sites["z"], sites["income"]), axis=1)
    np.savez_compressed(DATA / "golden_bundle.npz", **arrays)
    print("frames at", [int(f.glob[rf.G["frame"]]) for f in picked])


def write_expected() -> None:
    bundle = bd.Bundle.load(DATA / "golden_bundle.npz")
    table = tables.table()
    for prefix, name in (("expected", "golden_frames.npz"), ("expected_sim", "sim_frames.npz")):
        if not (DATA / name).exists():
            continue
        for i, frame in enumerate(rf.load_frames(DATA / name)):
            t = fz.featurize(frame, bundle, table)
            np.savez_compressed(DATA / f"{prefix}_{i}.npz", controllable=t.controllable, **{k: getattr(t, k) for k in ARRAYS})
            _review(i, frame, t, bundle, table)


def sim_frames(decisions: tuple[int, ...] = (1, 50, 100, 200)) -> None:
    """Frames from a real batch game on the same map as the live ones: the fog, memory and blips the sim makes of two armies."""
    import policy_sim_env as se
    env = se.SimEnv(1, 11, ["both"], units=60, threads=1)
    live = bd.Bundle.load(DATA / "golden_bundle.npz")
    titan = next(i for i, (_, b, _) in enumerate(se.pool()) if b.map_w == live.map_w and b.regions == live.regions)
    se._POOL[:] = [se.pool()[titan]]
    env.begin()
    out = []
    for d in range(1, max(decisions) + 1):
        parts = [env.script_orders(0, 0), env.script_orders(0, 1)]
        env.batch.step(np.concatenate(parts), np.array([0, len(parts[0]) + len(parts[1])], np.int32), ticks=se.TICKS_PER_DECISION)
        env.decisions += 1
        if d in decisions:
            f = env.frame(int(d % 2))
            out.append(rf.Frame(f.units.copy(), f.sites.copy(), f.glob.copy(), f.features.copy()))
    rf.save_frames(DATA / "sim_frames.npz", out)
    print("sim frames, records by knowledge:", [np.bincount(f.units["knowledge"], minlength=4).tolist() for f in out])


def recreate_live(index: int = 1) -> dict[str, float]:
    """A live frame's units placed in the sim, and the sim's frame of them read by the same featurizer: what must agree does."""
    import policy_sim_env as se
    import zksim
    frame = rf.load_frames(DATA / "golden_frames.npz")[index]
    bundle = bd.Bundle.load(DATA / "golden_bundle.npz")
    table = tables.table()
    u = frame.units
    cat = se.catalog()
    known = (u["knowledge"] == rf.KNOW_LOS) & (u["def_id"] > 0)
    names = [units.name_of(int(d)) for d in u["def_id"][known]]
    usable = np.array([n in cat.index for n in names])
    live = u[known][usable]
    print(f"live frame {index}: {len(live)} of {int(known.sum())} seen units are in the catalog")
    side = (live["ally_team"] != int(frame.glob[rf.G["ally_team"]])).astype(np.int32)
    world_map = next(m for m, b, _ in se.pool() if b.map_w == bundle.map_w and b.regions == bundle.regions)
    batch = zksim.Batch(cat, 1, threads=1)
    g = frame.glob
    batch.reset(0, world_map, 5, np.array([cat.index[units.name_of(int(d))] for d in live["def_id"]]), side, live["x"], live["z"],
                hp=live["hp_frac"], build=live["build_frac"], base=(float(g[rf.G["base_x"]]), float(g[rf.G["base_z"]]), 0.0, 0.0), sites=bundle.sites())
    batch.step(ticks=0)
    made = batch.frame(0, 0)
    a = fz.featurize(rf.Frame(live, frame.sites, frame.glob), bundle, table)
    b = fz.featurize(made, bundle, table)
    # The uid of a sim unit is its place in the list we gave it.
    lookup = {int(uid): i for i, uid in enumerate(live["uid"])}
    pos_a = {int(uid): k for k, uid in enumerate(a.uids) if uid >= 0}
    pos_b = {int(uid): k for k, uid in enumerate(b.uids) if uid >= 0}
    pairs = [(pos_a[uid], pos_b[lookup[uid]]) for uid in pos_a if lookup[uid] in pos_b]
    i, j = np.array(pairs).T
    columns = {"type": (0, fz._I), "x,z": (fz._I, fz._I + 2), "altitude": (fz._I + 2, fz._I + 3), "hp": (fz._I + 6, fz._I + 7), "build": (fz._I + 8, fz._I + 9),
               "free rays": (fz._R, fz._R + 8), "enemy sum": (fz._R + 8, fz._R + 16), "own sum": (fz._R + 16, fz._R + 24)}
    worst = {}
    for name, (lo, hi) in columns.items():
        d = np.abs(a.unit_x[i, lo:hi] - b.unit_x[j, lo:hi])
        worst[name] = float(d.max())
        print(f"  {name:10s} mean difference {d.mean():.4f}  worst {d.max():.4f}")
    seen_live = {int(x) for x in live["uid"][side == 1]}
    foes = (made.units["ally_team"] == 1) & (made.units["knowledge"] == rf.KNOW_LOS)
    seen_sim = {int(live["uid"][x]) for x in made.units["uid"][foes]}
    print(f"  enemies seen: live {len(seen_live)}, sim {len(seen_sim)}, both {len(seen_live & seen_sim)}")
    print(f"  tokens compared {len(pairs)}, live tokens {len(a.tok_kind)}, sim tokens {len(b.tok_kind)}")
    worst["seen by the sim alone"] = float(len(seen_sim - seen_live))
    worst["seen live"], worst["seen by the sim"] = float(len(seen_live)), float(len(seen_sim))
    air = table.air[table.rows(live["def_id"])]
    ground_tokens = [(p, q) for p, q in zip(i, j) if not air[lookup[int(a.uids[p])]]]
    p, q = np.array(ground_tokens).T
    worst["ground altitude"] = float(np.abs(a.unit_x[p, fz._I + 2] - b.unit_x[q, fz._I + 2]).max())
    return worst


def _review(i: int, frame: rf.Frame, t: fz.Tensors, bundle: bd.Bundle, table: tables.TypeTable) -> None:
    kinds = np.bincount(t.tok_kind, minlength=fz.TOKEN_KINDS).tolist()
    print(f"--- frame {i}: game frame {int(frame.glob[rf.G['frame']])}, {len(frame.units)} records, tokens {len(t.tok_kind)} by kind {kinds}, controllable {t.controllable}")
    by_uid = {int(u["uid"]): u for u in frame.units}
    for k in list(range(min(t.controllable, 2))) + [int(x) for x in t.enemy_tokens[:2]]:
        rec = by_uid[int(t.uids[k])]
        name = units.name_of(int(rec["def_id"])) if rec["def_id"] > 0 else "?"
        x = t.unit_x[k]
        print(f"  token {k} {name} player {rec['player']} ally {rec['ally_team']} knowledge {rec['knowledge']} at ({rec['x']:.0f},{rec['z']:.0f}) hp {rec['hp_frac']:.2f} flags {rec['flags']} order {rec['ord_kind']} age {rec['ord_age']:.0f}"
              f" -> x {x[fz._I]:.3f} z {x[fz._I + 1]:.3f} alt {x[fz._I + 2]:.3f} hp {x[fz._I + 6]:.2f} free {np.round(x[fz._R:fz._R + 8], 2).tolist()} enemy {np.round(x[fz._R + 8:fz._R + 16], 2).tolist()}")
    if t.controllable:
        print("  legal kinds per unit (first 4):", [np.flatnonzero(m).tolist() for m in t.kind_mask[:4]], "regions legal", t.ptr_region.sum(1)[:4].tolist())
        print("  route seconds, first unit, nearest 4:", np.round(np.sort(t.route_bias[0])[:4], 1).tolist())
    print("  north", np.round(t.north, 2).tolist(), "site x", np.round(t.site_x[0], 2).tolist() if len(t.site_x) else [], "glob", np.round(t.glob_x, 2).tolist())


if __name__ == "__main__":
    if sys.argv[1] == "frames":
        pick_frames(pathlib.Path(sys.argv[2]))
    if sys.argv[1] == "sim":
        sys.path.insert(0, str(ROOT / "tools"))
        sim_frames()
    write_expected()
    if sys.argv[1] == "sim":
        for index in range(5):
            recreate_live(index)
