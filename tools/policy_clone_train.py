"""Behaviour cloning: the windowed actor on latents (policy_latent.py) learns to do what the script did, from recorded games
(policy_clone_data.py), before RL.

    ~/zkml/bin/python tools/policy_clone_train.py --data /mnt/d/zk-learned/clone/army-2 --maps /mnt/d/zk-learned/maps --run /mnt/d/zk-learned/runs/clone-1

A sample is a stretch of one side of one game: LENGTH decisions in order, with the frames of the actor's reach before them, which
its window reads.  Workers featurize the frames and turn the script's orders into the policy's choices (`policy_clone_data.labels`);
the frozen encoder makes each frame's latents, what the script did at each frame becomes that frame's action tokens, and the actor
reads the whole stretch in one forward pass, as in play: [frame] [what was done] [frame] [what was done] ...  So a decision is
predicted from the frames and from what the script did before it, never from what it did then.  The loss is the negative
log-probability of the script's choices, per unit.

Most labels are keep, so it's weighted down, as the user asked (2026-11-08):

  --looked-only   (the default) only decisions where the script looked count: it looks every SCRIPT_EVERY decisions, and in
                  between, keep is a default, not a choice.  --all-decisions counts them all.
  --keep-weight   what a unit told to keep counts for, against 1 for any other choice; 0 leaves keep out.  A unit whose only legal
                  choice is keep never counts.

Weighting keep by w scales the odds the net learns for it by w: at 0.1, with keep 79% of labels, it learns to keep 27% of the time,
and at first acts everywhere.  So keep's logit gets -log w added back (`keep_shift`) wherever the policy plays: in the held-out
measures, and in the saved actor's kind head.  The gradient still leans on the rest, which is what the weight was for; the policy
plays at the script's rate.  At 0 nothing teaches keep, and nothing can be added back.

The checkpoint holds the actor's weights and this run's settings; `policy_train.py --init-actor` starts RL from it.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pathlib
import signal
import sys
import time
import zlib
from collections import Counter, deque

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import policy_clone_data as cd  # noqa: E402
import policy_latent  # noqa: E402
import policy_model as pm  # noqa: E402
import policy_sim_env as se  # noqa: E402
from zkbot.policy import bundle as bd, featurize as fz, rawframe as rf, tables  # noqa: E402

NAME_DROPOUT = 0.3          # as policy_train.py: some names read as unknown, so that the net can't lean on names alone
HELD_OUT = 50               # one game file in this many is for validation, by a hash of its name


# Workers --------------------------------------------------------------------------------------------------------------------------------------------

_W: dict = {}


def _setup(bundles: dict, regions: int, vocabulary: int, looked_only: bool) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _W.update(bundles=bundles, regions=regions, vocabulary=vocabulary, looked_only=looked_only, table=tables.table(), cache={})


def _game(path: str):
    cache = _W["cache"]
    if path not in cache:
        if len(cache) > 8:
            cache.pop(next(iter(cache)))
        cache[path] = cd.load(pathlib.Path(path))
    return cache[path]


def _sample(spec: tuple[str, int, int, int, int]) -> dict:
    """One stretch of a game, ready to batch: what the encoder reads (history, then the stretch), what the script did at each of
    those frames (the actor reads it as action tokens), what the heads read (the counted decisions), and their labels."""
    path, start, length, reach, seed = spec
    frames, orders, info = _game(path)
    bundle = _W["bundles"][info["map"]]
    table = _W["table"]
    rng = np.random.default_rng(seed)
    names = np.arange(_W["vocabulary"], dtype=np.int32)
    names[rng.random(_W["vocabulary"]) < NAME_DROPOUT] = _W["vocabulary"] - 1
    start = min(start, len(frames) - 1)
    first = max(0, start - reach)
    end = min(len(frames), start + length)
    tensors, done, labels, counted, missed = [], [], [], [], 0
    for k in range(first, end):
        t = fz.featurize(frames[k], bundle, table, enabled_kinds=se.KINDS)
        x, z = cd.actions.world_positions(t, bundle)
        decision, m = cd.actions.encode(se.policy_orders(orders[k]), t, bundle, x, z)
        done.append(decision)
        if k >= start and (not _W["looked_only"] or k % se.SCRIPT_EVERY == 0):
            labels.append(decision)
            counted.append(k - start)
            missed += m
        t.true_name, t.unit_name = t.unit_name, names[t.unit_name]
        tensors.append(t)
    seen = pm.collate(tensors, regions=_W["regions"])
    heads = pm.collate([tensors[(start - first) + i] for i in counted], regions=_W["regions"]) if counted else None
    return {"seen": seen, "history": start - first, "length": end - start, "heads": heads, "counted": counted, "missed": missed,
            "orders": int(sum(len(orders[k]) for k in range(start, end))),
            "kind": [d.kind for d in labels], "enemy": [d.enemy for d in labels], "region": [d.region for d in labels],
            "fine": [d.fine for d in labels], "produce": [d.produce for d in labels],
            "done": {key: [getattr(d, key) for d in done] for key in ("kind", "enemy", "region", "fine", "produce")}}


def bundles(maps: str | None) -> dict[str, bd.Bundle]:
    """What featurizing needs of each map: its bundle, by slug.  Only the bundles: the sim's maps are most of a pool's memory
    (298 maps: 0.17 GB of bundles against about 7.5 GB in all), and the trainer doesn't run the sim."""
    if not maps:
        return {f.name: b for _, b, f in se.pool()}
    return {path.name[:-len(".bundle.npz")]: bd.Bundle.load(path) for path in sorted(pathlib.Path(maps).glob("*.bundle.npz"))}


# Data -------------------------------------------------------------------------------------------------------------------------------------------------

def _files(data: pathlib.Path) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Game files and their lengths, training and held out."""
    train, held = [], []
    for path in sorted(data.glob("game-*.npz")):
        try:
            with np.load(path) as d:
                n = len(d["unit_counts"])
        except (OSError, ValueError, KeyError, EOFError):        # still being written
            continue
        if n:
            (held if zlib.crc32(path.name.encode()) % HELD_OUT == 0 else train).append((str(path), n))
    return train, held


def _specs(files: list[tuple[str, int]], length: int, reach: int, rng: np.random.Generator):
    """Stretches of games, drawn in proportion to their length, without end."""
    lengths = np.array([n for _, n in files], float)
    p = lengths / lengths.sum()
    while True:
        i = int(rng.choice(len(files), p=p))
        path, n = files[i]
        start = int(rng.integers(0, max(1, n - length + 1)))
        yield path, start, length, reach, int(rng.integers(1 << 31))


class Prefetch:
    """Samples from the workers, at most `ahead` of them made or being made at once.  (Pool.imap over an endless stream doesn't
    wait: it hands out work as fast as it can and keeps every result until it's read, which filled 30 GB in a minute.)"""

    def __init__(self, pool, specs, ahead: int):
        self.pool, self.specs = pool, specs
        self.waiting = deque(pool.apply_async(_sample, (next(specs),)) for _ in range(ahead))

    def __next__(self) -> dict:
        out = self.waiting.popleft().get()
        self.waiting.append(self.pool.apply_async(_sample, (next(self.specs),)))
        return out


def _fixed_specs(files: list[tuple[str, int]], count: int, length: int, reach: int) -> list[tuple]:
    rng = np.random.default_rng(12345)
    gen = _specs(files, length, reach, rng)
    return [next(gen) for _ in range(count)]


# The model ----------------------------------------------------------------------------------------------------------------------------------------------

def _pad(arrays: list[np.ndarray], shape: tuple, fill: int, device: str) -> torch.Tensor:
    out = np.full(shape, fill, np.int64)
    for i, a in enumerate(arrays):
        out[i, :len(a)] = a
    return torch.as_tensor(out, device=device)


def forward(actor: policy_latent.LatentActor, samples: list[dict], device: str, autocast, kinds: bool = False, blind: bool = False):
    """The counted decisions, their labels, and the log-probability of each unit's; with kinds, also every kind's logit.  blind:
    the actor isn't shown what was done (its action tokens are all zeros), to measure what they're worth."""
    reach, tokens, latent = actor.reach, actor.encoder.tokens, actor.encoder.latent
    longest = max(s["length"] for s in samples)
    frames = torch.zeros(len(samples), reach + longest, tokens, latent, device=device)
    present = torch.zeros(len(samples), reach + longest, dtype=torch.bool, device=device)
    seen = [s["seen"].to(device) for s in samples]
    with torch.no_grad(), autocast:
        for i, s in enumerate(samples):
            h = s["history"]
            frames[i, reach - h:reach + s["length"]] = actor.encoder(seen[i]).float()
            present[i, reach - h:] = True
    with autocast:
        # What the script did at every frame, as the actor reads it: tokens after each frame's.
        parts = []
        for i, s in enumerate(samples if not blind else ()):
            shape = tuple(seen[i].ctrl_mask.shape)
            d = {key: _pad(s["done"][key], shape, 0 if key == "kind" else -1, device) for key in s["done"]}
            done = pm.Actions(d["kind"], d["enemy"], d["region"], d["fine"], d["produce"], torch.zeros(shape, device=device))
            parts.append(actor.action_tokens(seen[i], done))
        acted = torch.zeros(len(samples), reach + longest, policy_latent.ACTION_TOKENS, policy_latent.D, device=device,
                            dtype=parts[0].dtype if parts else torch.float32)
        for i, part in enumerate(parts):
            acted[i, reach - samples[i]["history"]:reach + samples[i]["length"]] = part
        held = actor.sequence(frames, present, acted)[:, reach:]
        now = torch.cat([held[i, s["counted"]] for i, s in enumerate(samples) if s["counted"]])
        whole = pm.join([s["heads"] for s in samples if s["counted"]]).to(device)
        units, reg_h = actor.read(whole, now)
        shape = tuple(whole.ctrl_mask.shape)
        flat = lambda key, fill: _pad([a for s in samples for a in s[key]], shape, fill, device)  # noqa: E731
        a = pm.Actions(flat("kind", 0), flat("enemy", -1), flat("region", -1), flat("fine", -1), flat("produce", -1), torch.zeros(shape, device=device))
        logp, _ = pm.evaluate_actions(actor, whole, units, reg_h, a)
        kind_logits = actor.heads(whole, units, reg_h)[0].float() if kinds else None
    return whole, a, logp.float(), kind_logits


def sayable(whole: pm.Batch, a: pm.Actions) -> torch.Tensor:
    """Units whose label counts at all: controlled, with more than one legal kind, and the script's among them.  An idle unit
    can't keep (featurize: keep is for a unit doing something), but the script often leaves units idle, new ones by the
    factory say, and there's no choice of the policy's that says that."""
    legal = torch.gather(whole.kind_mask, -1, a.kind.clamp_min(0)[..., None])[..., 0]
    return whole.ctrl_mask & (whole.kind_mask.sum(-1) > 1) & legal


def weights(whole: pm.Batch, a: pm.Actions, keep_weight: float) -> torch.Tensor:
    """What each unit's label counts for."""
    return sayable(whole, a).float() * torch.where(a.kind == rf.KEEP, keep_weight, 1.0)


def keep_shift(keep_weight: float) -> float:
    return -float(np.log(keep_weight)) if keep_weight > 0 else 0.0


def playing(actor: policy_latent.LatentActor, keep_weight: float) -> dict:
    """The actor's weights as it should play: keep's logit shifted back."""
    state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    state["head_kind.bias"][rf.KEEP] += keep_shift(keep_weight)
    return state


def measure(whole: pm.Batch, a: pm.Actions, logp: torch.Tensor, kind_logits: torch.Tensor, shift: float = 0.0) -> dict[str, float]:
    """On units with a choice: how often the most likely kind is the script's, for keep and for the rest, and how often it would
    act where the script kept."""
    choice = sayable(whole, a).cpu().numpy()
    unsayable = (whole.ctrl_mask & (whole.kind_mask.sum(-1) > 1)).cpu().numpy() & ~choice
    label = a.kind.cpu().numpy()
    kind_logits = kind_logits.clone()
    kind_logits[..., rf.KEEP] += shift      # as it would play
    guess = kind_logits.argmax(-1).cpu().numpy()
    keep = choice & (label == rf.KEEP)
    act = choice & (label != rf.KEEP)
    out = {"units_keep": float(keep.sum()), "units_act": float(act.sum()), "units_unsayable": float(unsayable.sum()),
           "kind_right_keep": float((guess == label)[keep].mean()) if keep.any() else float("nan"),
           "kind_right_act": float((guess == label)[act].mean()) if act.any() else float("nan"),
           "acts_where_kept": float((guess != rf.KEEP)[keep].mean()) if keep.any() else float("nan"),
           "nll_act": float(-logp.detach().cpu().numpy()[act].mean()) if act.any() else float("nan"),
           "nll_keep": float(-logp.detach().cpu().numpy()[keep].mean()) if keep.any() else float("nan")}
    for k in np.unique(label[act]):
        here = act & (label == k)
        out[f"right_{cd.rf_name(int(k))}"] = float((guess == label)[here].mean())
        out[f"n_{cd.rf_name(int(k))}"] = float(here.sum())
    return out


# Training -------------------------------------------------------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--maps", help="the map pool the games were recorded on; without it, the five training maps")
    ap.add_argument("--run", required=True)
    ap.add_argument("--encoder", default=policy_latent.ENCODER)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--samples", type=int, default=16, help="stretches of games a step")
    ap.add_argument("--length", type=int, default=48, help="decisions a stretch")
    ap.add_argument("--keep-weight", type=float, default=0.1)
    ap.add_argument("--all-decisions", action="store_true", help="count every decision, not only those where the script looked")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--check-every", type=int, default=500)
    ap.add_argument("--check-samples", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--blind", action="store_true", help="don't show the actor what was done: its action tokens are all zeros")
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=1))
    began = time.perf_counter()

    by_map = bundles(args.maps)
    regions = max(b.regions for b in by_map.values())
    table = tables.table()
    vocabulary = len(table.names) + 1
    train_files, held_files = _files(pathlib.Path(args.data))
    pool = multiprocessing.get_context("fork").Pool(args.workers, _setup, (by_map, regions, vocabulary, not args.all_decisions))
    loaded = time.perf_counter() - began

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")
    actor = policy_latent.make(vocabulary, args.encoder).to(device)
    trained = [p for p in actor.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trained, lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    reach = actor.reach
    check = _fixed_specs(held_files, args.check_samples, args.length, reach)
    stream = Prefetch(pool, _specs(train_files, args.length, reach, rng), ahead=2 * args.samples)
    log = open(run / "log.jsonl", "a")
    print(json.dumps({"train_files": len(train_files), "held_files": len(held_files), "loading_seconds": round(loaded, 1),
                      "trained_parameters": sum(p.numel() for p in trained)}), flush=True)

    totals: Counter = Counter()
    step_began = time.perf_counter()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / 200.0) * 0.5 * (1.0 + np.cos(np.pi * step / args.steps))
        for group in optimizer.param_groups:
            group["lr"] = lr
        samples = [next(stream) for _ in range(args.samples)]
        samples = [s for s in samples if s["counted"]]
        if not samples:
            continue
        whole, a, logp, _ = forward(actor, samples, device, autocast, blind=args.blind)
        w = weights(whole, a, args.keep_weight)
        loss = -(w * logp).sum() / w.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trained, 1.0)
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["steps"] += 1
        totals["decisions"] += sum(len(s["counted"]) for s in samples)
        totals["orders"] += sum(s["orders"] for s in samples)
        totals["missed"] += sum(s["missed"] for s in samples)
        if step % args.check_every == 0 or step == args.steps:
            actor.eval()
            parts: list[dict] = []
            with torch.no_grad():
                for i in range(0, len(check), args.samples):
                    chunk = [s for s in pool.map(_sample, check[i:i + args.samples], chunksize=1) if s["counted"]]
                    if chunk:
                        parts.append(measure(*forward(actor, chunk, device, autocast, kinds=True, blind=args.blind), shift=keep_shift(args.keep_weight)))
            actor.train()
            held = {}
            for key in sorted({k for p in parts for k in p}):
                if key.startswith(("units_", "n_")):
                    held[key] = sum(p.get(key, 0.0) for p in parts)
                else:
                    weight = ("units_keep" if key.endswith("keep") or key == "acts_where_kept" else
                              "n_" + key[6:] if key.startswith("right_") else "units_act")
                    vals = [(p[key], p.get(weight, 1.0)) for p in parts if key in p and not np.isnan(p[key])]
                    held[key] = sum(v * n for v, n in vals) / max(sum(n for _, n in vals), 1e-9)
            seconds = time.perf_counter() - step_began
            row = {"step": step, "loss": totals["loss"] / max(totals["steps"], 1), "lr": lr,
                   "decisions_a_second": totals["decisions"] / max(seconds, 1e-9), "missed_share": totals["missed"] / max(totals["orders"], 1),
                   **{f"held_{k}": round(v, 4) for k, v in held.items()}}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            totals.clear()
            step_began = time.perf_counter()
            temp = run / "actor.pt.tmp"
            torch.save({"actor": playing(actor, args.keep_weight), "step": step, "config": vars(args), "vocabulary": vocabulary,
                        "keep_shift": keep_shift(args.keep_weight)}, temp)
            os.replace(temp, run / "actor.pt")
    pool.terminate()


if __name__ == "__main__":
    main()
