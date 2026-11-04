"""Train the fight evaluator (ML stage 1) on fights-team*.jsonl logs.

  # in WSL, with the venv from docs/ML.md:
  ~/zkml/bin/python tools/train_fightnet.py results/h12[6-9]* [--epochs 30] [--export]

Model: set transformer over the entities of a fight snapshot (both sides),
mean+max pooled per side, predicting for each horizon the fraction of our
and of their snapshot value lost (soft-target BCE).  Split by game (no
snapshot of a validation game is trained on).  Reports validation loss and
how well the model ranks fights won (their loss > ours) against the
power-ratio rule the bot uses today.  --export writes
bots/zkbot/data/fightnet.npz for the numpy forward in bots/zkbot/fightnet.py
and checks that both give the same output.
"""
from __future__ import annotations

import argparse
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
from zkbot import fightnet as fn  # noqa: E402
from zkbot import catalog  # noqa: E402

FEAT = fn.TYPE_DIM + fn.DYN_DIM


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)

    def forward(self, h, mask):
        b, n, d = h.shape
        a = self.ln1(h)
        q, k, v = self.qkv(a).chunk(3, dim=-1)
        sh = lambda t: t.view(b, n, self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
        o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=mask[:, None, None, :])
        h = h + self.out(o.transpose(1, 2).reshape(b, n, d))
        return h + self.ff2(F.gelu(self.ff1(self.ln2(h)), approximate="tanh"))


class Net(nn.Module):
    def __init__(self, d=64, layers=3, heads=4, ff=128, n_out=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(FEAT, d), nn.GELU(approximate="tanh"), nn.Linear(d, d))
        self.emb = nn.Embedding(fn.N_HASH, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.enc = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(approximate="tanh"), nn.Linear(d, n_out))
        self.layers, self.heads = layers, heads

    def forward(self, x, ids, side, mask):
        h = self.inp(x) + self.emb(ids)
        for blk in self.enc:
            h = blk(h, mask)
        h = self.lnf(h)
        pooled = []
        for s in (0, 1):
            m = (mask & (side == s)).unsqueeze(-1)
            cnt = m.sum(1).clamp(min=1)
            pooled.append((h * m).sum(1) / cnt)
            mx = h.masked_fill(~m, -1e4).max(1).values
            pooled.append(torch.where(m.any(1), mx, torch.zeros_like(mx)))
        return self.head(torch.cat(pooled, -1))


def load(dirs: list[str]):
    files = []
    for d in dirs:
        files += glob.glob(os.path.join(d, "**", "fights-team*.jsonl"), recursive=True)
    games, recs = [], []
    for f in sorted(set(files)):
        g = os.path.dirname(f)
        for line in open(f, encoding="utf8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if len(r.get("out") or {}) < len(fn.HORIZONS):
                continue
            if any(e[1] in fn.NOT_UNITS for e in r["ents"]):
                continue          # logs before the terraunit fix: losses unreliable
            recs.append(r)
            games.append(g)
    return recs, games


def tensorize(recs):
    n = len(recs)
    N = fn.MAX_ENTS
    X = np.zeros((n, N, FEAT), np.float32)
    I = np.zeros((n, N), np.int64)
    S = np.full((n, N), 2, np.int64)
    M = np.zeros((n, N), bool)
    Y = np.zeros((n, 2 * len(fn.HORIZONS)), np.float32)
    V = np.zeros((n, 2), np.float32)
    P = np.zeros((n, 2), np.float32)          # power-ratio baseline inputs
    for i, r in enumerate(recs):
        rows, ids, sides, vals = fn.featurize(r["ents"])
        k = len(rows)
        X[i, :k] = rows
        I[i, :k] = ids
        S[i, :k] = sides
        M[i, :k] = True
        for s in (0, 1):
            V[i, s] = sum(v for v, sd in zip(vals, sides) if sd == s)
        for side, name, _dx, _dz, _dh, hp, flags in r["ents"][:N]:
            st = catalog.stats(name)
            if st and not flags & fn.FLAG_BLIP:
                P[i, side] += math.sqrt(max(0.0, st["dps_ground"]) * max(1.0, st["hp"] * hp))
        for j, h in enumerate(fn.HORIZONS):
            ol, tl = r["out"][str(h)]
            Y[i, j] = min(1.0, ol / max(1.0, V[i, 0]))
            Y[i, len(fn.HORIZONS) + j] = min(1.0, tl / max(1.0, V[i, 1]))
    return X, I, S, M, Y, V, P


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label.sum()
    neg = len(label) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def export(model: Net, path: pathlib.Path):
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    sd["meta_layers"] = np.array(model.layers)
    sd["meta_heads"] = np.array(model.heads)
    sd["meta_horizons"] = np.array(fn.HORIZONS)
    np.savez(path, **sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    t0 = time.time()
    recs, games = load(a.dirs)
    ug = sorted(set(games))
    random.shuffle(ug)
    val_games = set(ug[: int(len(ug) * a.val)]) if a.val > 0 else set()
    is_val = np.array([g in val_games for g in games])
    print(f"{len(recs)} snapshots from {len(ug)} games ({is_val.sum()} val); load {time.time() - t0:.0f}s")
    X, I, S, M, Y, V, P = tensorize(recs)
    print(f"tensorized {time.time() - t0:.0f}s; mean targets {Y.mean(0).round(3)}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = lambda arr: torch.from_numpy(arr).to(dev)  # noqa: E731
    Xt, It, St, Mt, Yt = T(X), T(I), T(S), T(M), T(Y)
    tr = np.where(~is_val)[0]
    va = np.where(is_val)[0] if is_val.any() else tr   # --val 0: smoke test only
    model = Net(d=a.d, layers=a.layers, n_out=Y.shape[1]).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()):,} on {dev}")
    fwd = model if a.no_compile else torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    def batch_iter(idx, shuffle):
        idx = np.random.permutation(idx) if shuffle else idx
        for s in range(0, len(idx), a.batch):
            b = idx[s:s + a.batch]
            if len(b) < a.batch:          # fixed shapes: no recompiles
                b = np.concatenate([b, b[: a.batch - len(b)]]) if shuffle else b
            yield torch.from_numpy(b).to(dev)

    def evaluate():
        model.eval()
        outs = []
        with torch.no_grad():
            for b in batch_iter(va, False):
                outs.append(torch.sigmoid(fwd(Xt[b], It[b], St[b], Mt[b])).cpu().numpy())
        model.train()
        p = np.concatenate(outs)
        y = Y[va]
        bce = float(-(y * np.log(p + 1e-6) + (1 - y) * np.log(1 - p + 1e-6)).mean())
        hn = len(fn.HORIZONS)
        our_l = p[:, hn - 1] * V[va, 0]
        their_l = p[:, 2 * hn - 1] * V[va, 1]
        true_win = Y[va, 2 * hn - 1] * V[va, 1] > Y[va, hn - 1] * V[va, 0]
        a_net = auc(np.log((their_l + 50) / (our_l + 50)), true_win)
        a_rule = auc(np.log((P[va, 0] + 1) / (P[va, 1] + 1)), true_win)
        mae = float(np.abs(p - y).mean())
        return bce, mae, a_net, a_rule, true_win.mean()

    for ep in range(a.epochs):
        tl, nb = 0.0, 0
        te = time.time()
        for b in batch_iter(tr, True):
            # Unknown-name dropout: the model must also work from stats alone
            # (new units, balance changes).
            ids = It[b] * (torch.rand_like(It[b], dtype=torch.float32) > 0.3)
            loss = F.binary_cross_entropy_with_logits(fwd(Xt[b], ids, St[b], Mt[b]), Yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
            nb += 1
        if ep % 5 == 4 or ep == a.epochs - 1:
            bce, mae, an, ar, wr = evaluate()
            print(f"ep {ep + 1:3d} train {tl / nb:.4f}  val bce {bce:.4f} mae {mae:.4f}  "
                  f"AUC(fight won) net {an:.3f} vs power-ratio {ar:.3f}  (won {wr:.0%})  {time.time() - te:.1f}s/ep")
    if a.export:
        out = ROOT / "bots" / "zkbot" / "data" / "fightnet.npz"
        export(model, out)
        net = fn.FightNet(str(out))
        i = int(va[0])
        k = int(M[i].sum())
        ref = torch.sigmoid(model(Xt[i:i + 1, :k], It[i:i + 1, :k], St[i:i + 1, :k], Mt[i:i + 1, :k]))[0]
        got = net.forward(X[i, :k], I[i, :k], S[i, :k])
        print(f"exported {out}  numpy/torch max diff {np.abs(ref.detach().cpu().numpy() - got).max():.2e}")


if __name__ == "__main__":
    main()
