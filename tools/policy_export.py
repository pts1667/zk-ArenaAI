"""Writes the bot's weight file (.npz) from a training checkpoint, or an untrained one for measuring latency.

  python tools/policy_export.py --random D:/zk-learned/weights/random.npz
  ~/zkml/bin/python tools/policy_export.py --checkpoint <run>/latest.pt out.npz    # in WSL
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bots"))
from zkbot.policy import net, tables  # noqa: E402


def write(path: str | pathlib.Path, weights: dict[str, np.ndarray]) -> None:
    table = tables.table()
    meta = {"meta_feature_version": net.FEATURE_VERSION, "meta_action_version": net.ACTION_VERSION, "meta_catalog_hash": table.hash, "meta_names": np.array(table.names)}
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **weights, **meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.random:
        write(args.out, net.random_weights(len(tables.table().names) + 1, args.seed))
    else:
        import torch

        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["actor"]
        write(args.out, {k: v.float().numpy() for k, v in state.items()})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
