"""Train the game value model (ML stage 2) on state-team*.jsonl logs.

  ~/zkml/bin/python tools/train_value.py results/h126* [--epochs 40] [--export]

Input: the last K state snapshots (10 s apart) of one game: per step the
economy/force vector and the 8x8x4 value grid (statelog.state_vector /
state_grid), embedded and summed with a learned time-step embedding, then a
small transformer encoder over time (cf. arXiv 2501.03832: attention over
time and space of a game's feature planes; here the grid is embedded as one
token per step, which suits our small data).

Targets, from the last token:
  win    the game's winner (timeouts / no result excluded from this loss)
  fm     log(our / their metal produced) 5 minutes later (demo team stats)

Prints validation AUC of win by game phase and the error on fm.  --export
writes bots/zkbot/data/valuenet.pt (torch state dict; the tuner reads it
with PyTorch offline, the bot does not need it).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bots"))
sys.path.insert(0, str(ROOT / "tools"))
from zkbot import statelog as sl  # noqa: E402

K = 12
AHEAD = 300


def local_path(p: str) -> pathlib.Path:
    """results.csv holds Windows paths; map them under WSL."""
    if os.name != "nt" and len(p) > 2 and p[1] == ":":
        p = "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")
    return pathlib.Path(p)


def game_labels(gdir: pathlib.Path) -> tuple[float | None, list[tuple[float, float]]]:
    """(won: 1/0/None, [(t, log metal ratio)]) for one game dir."""
    won = None
    demo = None
    res = gdir.parent / "results.csv"
    if res.exists():
        with res.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if local_path(r.get("dir") or "").name == gdir.name:
                    oc = r.get("outcome")
                    if oc == "win":
                        won = 1.0 if "ControllerAI" in (r.get("winner_sides") or "") else 0.0
                    demo = local_path(r.get("demo") or "")
    series = []
    if demo is not None and demo.is_file():
        try:
            from demostats import read_demo_stats
            from report import our_team
            s = read_demo_stats(demo)
            me = our_team(s["script"])
            a = {h["frame"] // 30: h.get("metalProduced", 0.0) for h in s["teams"][me]}
            b = {h["frame"] // 30: h.get("metalProduced", 0.0) for h in s["teams"][1 - me]}
            for t in sorted(set(a) & set(b)):
                if a[t] > 0 and b[t] > 0:
                    series.append((t, math.log(a[t] / b[t])))
        except Exception:
            pass
    return won, series


def build(dirs):
    files = []
    for d in dirs:
        files += glob.glob(os.path.join(d, "**", "state-team*.jsonl"), recursive=True)
    V, G, T, W, FM, GAME = [], [], [], [], [], []
    for f in sorted(set(files)):
        gdir = pathlib.Path(f).parent
        recs = [json.loads(x) for x in open(f, encoding="utf8") if x.strip()]
        if len(recs) < 3:
            continue
        won, series = game_labels(gdir)
        vec = np.array([sl.state_vector(r) for r in recs], np.float32)
        grd = np.array([sl.state_grid(r) for r in recs], np.float32)
        ts = [r["t"] for r in recs]
        for i in range(len(recs)):
            lo = max(0, i - K + 1)
            v = np.zeros((K, sl.STATE_DIM), np.float32)
            g = np.zeros((K, len(grd[0])), np.float32)
            v[K - (i - lo + 1):] = vec[lo:i + 1]
            g[K - (i - lo + 1):] = grd[lo:i + 1]
            fm = float("nan")
            fut = [m for t, m in series if t >= ts[i] + AHEAD]
            if fut:
                fm = fut[0]
            V.append(v)
            G.append(g)
            T.append(ts[i])
            W.append(float("nan") if won is None else won)
            FM.append(fm)
            GAME.append(str(gdir))
    return (np.stack(V), np.stack(G), np.array(T, np.float32), np.array(W, np.float32),
            np.array(FM, np.float32), GAME)


class ValueNet(nn.Module):
    def __init__(self, d=64, layers=3, heads=4):
        super().__init__()
        self.vin = nn.Linear(sl.STATE_DIM, d)
        self.gin = nn.Sequential(nn.Linear(sl.GRID * sl.GRID * 4, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(K, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, v, g):
        h = self.vin(v) + self.gin(g) + self.pos
        h = self.enc(h)
        return self.head(h[:, -1])


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label.sum()
    neg = len(label) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    t0 = time.time()
    V, G, T, W, FM, GAME = build(a.dirs)
    games = sorted(set(GAME))
    random.shuffle(games)
    vg = set(games[: int(len(games) * a.val)]) if a.val > 0 else set()
    is_val = np.array([g in vg for g in GAME])
    print(f"{len(V)} steps from {len(games)} games ({len(vg)} val); win-labelled {np.isfinite(W).mean():.0%}, "
          f"wins {np.nanmean(W):.1%}; fm-labelled {np.isfinite(FM).mean():.0%}; {time.time() - t0:.0f}s")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Vt, Gt = torch.from_numpy(V).to(dev), torch.from_numpy(G).to(dev)
    Wt, Ft = torch.from_numpy(W).to(dev), torch.from_numpy(FM).to(dev)
    model = ValueNet().to(dev)
    fwd = model if a.no_compile else torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    tr, va = np.where(~is_val)[0], (np.where(is_val)[0] if is_val.any() else np.where(~is_val)[0])
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    print(f"params {sum(p.numel() for p in model.parameters()):,} on {dev}")

    def loss_fn(out, w, fm):
        lw = torch.isfinite(w)
        lf = torch.isfinite(fm)
        l1 = F.binary_cross_entropy_with_logits(out[lw, 0], w[lw]) if lw.any() else out.sum() * 0
        l2 = F.smooth_l1_loss(out[lf, 1], fm[lf]) if lf.any() else out.sum() * 0
        return l1 + l2

    for ep in range(a.epochs):
        perm = np.random.permutation(tr)
        tl = nb = 0
        for s in range(0, len(perm), a.batch):
            b = perm[s:s + a.batch]
            if len(b) < a.batch:
                b = np.concatenate([b, perm[: a.batch - len(b)]])
            bt = torch.from_numpy(b).to(dev)
            loss = loss_fn(fwd(Vt[bt], Gt[bt]), Wt[bt], Ft[bt])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
            nb += 1
        if ep % 5 == 4 or ep == a.epochs - 1:
            model.eval()
            with torch.no_grad():
                out = torch.cat([model(Vt[torch.from_numpy(va[s:s + 1024]).to(dev)],
                                       Gt[torch.from_numpy(va[s:s + 1024]).to(dev)])
                                 for s in range(0, len(va), 1024)]).cpu().numpy()
            model.train()
            parts = []
            for lo, hi in ((0, 300), (300, 600), (600, 900), (900, 1e9)):
                m = (T[va] >= lo) & (T[va] < hi) & np.isfinite(W[va])
                if m.sum() > 10:
                    parts.append(f"{int(lo // 60)}m+ AUC {auc(out[m, 0], W[va][m] > 0.5):.2f}")
            fmm = np.isfinite(FM[va])
            mae = float(np.abs(out[fmm, 1] - FM[va][fmm]).mean()) if fmm.any() else float("nan")
            base = float(np.abs(FM[va][fmm] - np.median(FM[tr][np.isfinite(FM[tr])])).mean()) if fmm.any() else float("nan")
            print(f"ep {ep + 1:3d} train {tl / nb:.4f}  win {'  '.join(parts)}  fm MAE {mae:.3f} (median {base:.3f})")
    if a.export:
        out = ROOT / "bots" / "zkbot" / "data" / "valuenet.pt"
        torch.save(model.state_dict(), out)
        print(f"exported {out}")


if __name__ == "__main__":
    main()
