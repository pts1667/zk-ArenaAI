"""Behaviour cloning: the windowed actor on latents (policy_latent.py) learns to do what the script did, from recorded games
(policy_clone_data.py), before RL.

    ~/zkml/bin/python tools/policy_clone_train.py --data /mnt/d/zk-learned/clone/army-2 --maps /mnt/d/zk-learned/maps --run /mnt/d/zk-learned/runs/clone-1

A sample is a stretch of one side of one game: LENGTH decisions in order, with the frames of the actor's reach before them, which
its window reads.  Workers featurize the frames and turn the script's orders into the policy's choices (`policy_clone_data.labels`);
the frozen encoder makes each frame's latents, what the script did at each frame becomes that frame's action tokens, and the actor
reads the whole stretch in one forward pass, as in play: [frame] [what was done] [frame] [what was done] ...  So a decision is
predicted from the frames and from what the script did before it, never from what it did then.

The actor gives a decision's orders as actions, one after another, then done (policy_latent.py).  The script's orders become actions
in a random order, at most actions.SLOTS of them (`actions.to_slots`), and the loss is the negative log-probability of each, done
included, averaged over them.  The script's orders have no order of their own, so an action's anchor scores as any unit the script
told something and no action before it picked (policy_model.evaluate_actions' `labelled`): the actor learns which units to give
orders to, and isn't taught an order among them that isn't there.  A unit the script didn't tell anything is a unit no action picks.

  --looked-only   (the default) only decisions where the script looked count: it looks every SCRIPT_EVERY decisions, and in
                  between it does nothing because it isn't looking, not because it chose to.  --all-decisions counts them all.

The checkpoint holds the actor's weights and this run's settings.
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
import torch.nn.functional as F

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


def _game(path: str, side: int):
    cache = _W["cache"]
    if path not in cache:
        if len(cache) > 8:
            cache.pop(next(iter(cache)))
        cache[path] = cd.read_game(pathlib.Path(path))
    return cache[path][side]


def _sample(spec: tuple[str, int, int, int, int, int]) -> dict:
    """One stretch of a game, ready to batch: what the encoder reads (history, then the stretch), what the script did at each of
    those frames (the actor reads it as action tokens), what the heads read (the counted decisions), and their labels."""
    path, side, start, length, reach, seed = spec
    frames, orders, info = _game(path, side)
    bundle = _W["bundles"][info["map"]]
    table = _W["table"]
    rng = np.random.default_rng(seed)
    names = np.arange(_W["vocabulary"], dtype=np.int32)
    names[rng.random(_W["vocabulary"]) < NAME_DROPOUT] = _W["vocabulary"] - 1
    start = min(start, len(frames) - 1)
    first = max(0, start - reach)
    end = min(len(frames), start + length)
    tensors, done, labels, labelled, counted, missed, cut = [], [], [], [], [], 0, 0
    for k in range(first, end):
        t = fz.featurize(frames[k], bundle, table, enabled_kinds=se.KINDS)
        x, z = cd.actions.world_positions(t, bundle)
        decision, m = cd.actions.encode(se.policy_orders(orders[k]), t, bundle, x, z)
        done.append(cd.actions.to_slots(decision, slots=None))
        if k >= start and (not _W["looked_only"] or k % se.SCRIPT_EVERY == 0):
            labels.append(cd.actions.to_slots(decision, rng.permutation(t.controllable)))
            labelled.append(decision.kind != rf.KEEP)
            counted.append(k - start)
            missed += m
            cut += int(labelled[-1].sum()) - int((labels[-1].anchor >= 0).sum())
        t.true_name, t.unit_name = t.unit_name, names[t.unit_name]
        tensors.append(t)
    seen = pm.collate(tensors, regions=_W["regions"])
    heads = pm.collate([tensors[(start - first) + i] for i in counted], regions=_W["regions"]) if counted else None
    return {"seen": seen, "history": start - first, "length": end - start, "heads": heads, "counted": counted, "missed": missed, "cut": cut,
            "orders": int(sum(len(orders[k]) for k in range(start, end))), "labelled": labelled,
            "labels": {key: [getattr(s, key) for s in labels] for key in cd.actions.SLOT_FIELDS},
            "done": {key: [getattr(s, key) for s in done] for key in cd.actions.SLOT_FIELDS}}


def bundles(maps: str | None) -> dict[str, bd.Bundle]:
    """What featurizing needs of each map: its bundle, by slug.  Only the bundles: the sim's maps are most of a pool's memory
    (298 maps: 0.17 GB of bundles against about 7.5 GB in all), and the trainer doesn't run the sim."""
    if not maps:
        return {f.name: b for _, b, f in se.pool()}
    return {path.name[:-len(".bundle.npz")]: bd.Bundle.load(path) for path in sorted(pathlib.Path(maps).glob("*.bundle.npz"))}


# Data -------------------------------------------------------------------------------------------------------------------------------------------------

def _files(data: pathlib.Path) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """Sides of games, (path, side, length), training and held out.  Both sides of a game go the same way."""
    train, held = [], []
    for path, side, n in cd.sides(data):
        (held if zlib.crc32(pathlib.Path(path).name.encode()) % HELD_OUT == 0 else train).append((path, side, n))
    return train, held


def _specs(files: list[tuple[str, int, int]], length: int, reach: int, rng: np.random.Generator):
    """Stretches of games, drawn in proportion to their length, without end."""
    lengths = np.array([n for _, _, n in files], float)
    p = lengths / lengths.sum()
    while True:
        i = int(rng.choice(len(files), p=p))
        path, side, n = files[i]
        start = int(rng.integers(0, max(1, n - length + 1)))
        yield path, side, start, length, reach, int(rng.integers(1 << 31))


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


def _fixed_specs(files: list[tuple[str, int, int]], count: int, length: int, reach: int) -> list[tuple]:
    rng = np.random.default_rng(12345)
    gen = _specs(files, length, reach, rng)
    return [next(gen) for _ in range(count)]


# The model ----------------------------------------------------------------------------------------------------------------------------------------------

def _pad(arrays: list[np.ndarray], width: int, fill: int) -> np.ndarray:
    out = np.full((len(arrays), max(width, 1)), fill, np.int64)
    for i, a in enumerate(arrays):
        out[i, :len(a)] = a
    return out


def _actions(fields: dict[str, list[np.ndarray]], device: str) -> pm.Actions:
    """Decisions' actions, each a list of one array a decision, padded to the longest with no action."""
    width = max(len(a) for a in fields["anchor"])
    return pm.actions_from({key: _pad(arrays, width, cd.actions.UNSET[key]) for key, arrays in fields.items()}, device)


def forward(actor: policy_latent.LatentActor, samples: list[dict], device: str, autocast, measuring: bool = False, blind: bool = False):
    """The counted decisions, their labels (actions), which units the script told something, and each action's log-probability; with
    measuring, also each action's anchor and kind logits.  blind: the actor isn't shown what was done (its action tokens are all
    zeros), to measure what they're worth."""
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
            parts.append(actor.action_tokens(seen[i], _actions(s["done"], device)))
        acted = torch.zeros(len(samples), reach + longest, policy_latent.ACTION_TOKENS, policy_latent.D, device=device,
                            dtype=parts[0].dtype if parts else torch.float32)
        for i, part in enumerate(parts):
            acted[i, reach - samples[i]["history"]:reach + samples[i]["length"]] = part
        held = actor.sequence(frames, present, acted)[:, reach:]
        now = torch.cat([held[i, s["counted"]] for i, s in enumerate(samples) if s["counted"]])
        whole = pm.join([s["heads"] for s in samples if s["counted"]]).to(device)
        ctx = actor.context(whole, now)
        a = _actions({key: [x for s in samples for x in s["labels"][key]] for key in cd.actions.SLOT_FIELDS}, device)
        labelled = torch.as_tensor(_pad([x for s in samples for x in s["labelled"]], whole.ctrl_mask.shape[1], 0) > 0, device=device)
        logp, _ = pm.evaluate_actions(actor, whole, ctx, a, labelled)
        logits = pm.teacher_logits(actor, whole, ctx, a) if measuring else None
    return whole, a, labelled, logp.float(), logits


def measure(a: pm.Actions, labelled: torch.Tensor, logp: torch.Tensor, logits: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    """Over the actions (n_act) and the dones (n_done) of the labels, each given those before it: how often the most likely anchor
    is a unit the script told something and no action has picked yet (anchor_right), or done where there was more to do
    (stops_early); how often it's done where the script was (done_right); how often the most likely kind is the script's, given its
    anchor (kind_right, and right_KIND for each kind); and the negative log-probabilities."""
    anchor_l, kind_l = logits
    is_act = (a.anchor >= 0).cpu().numpy()
    is_done = pm.chosen(a).cpu().numpy() & ~is_act
    guess = anchor_l.argmax(-1)
    remaining = labelled[:, None] & ~pm._taken_before(a.anchor, labelled.shape[1])
    hit = torch.gather(F.pad(remaining, (1, 0)), -1, guess[..., None])[..., 0].cpu().numpy()
    guess = guess.cpu().numpy()
    label = a.kind.cpu().numpy()
    kind_guess = kind_l.argmax(-1).cpu().numpy()
    nll = -logp.detach().cpu().numpy()
    nan = float("nan")
    out = {"n_act": float(is_act.sum()), "n_done": float(is_done.sum()), "n_grouped": float((is_act & (a.group.cpu().numpy() >= 0)).sum()),
           "anchor_right": float(hit[is_act].mean()) if is_act.any() else nan,
           "stops_early": float((guess == 0)[is_act].mean()) if is_act.any() else nan,
           "done_right": float((guess == 0)[is_done].mean()) if is_done.any() else nan,
           "kind_right": float((kind_guess == label)[is_act].mean()) if is_act.any() else nan,
           "nll_act": float(nll[is_act].mean()) if is_act.any() else nan,
           "nll_done": float(nll[is_done].mean()) if is_done.any() else nan}
    for k in np.unique(label[is_act]):
        here = is_act & (label == k)
        out[f"right_{cd.rf_name(int(k))}"] = float((kind_guess == label)[here].mean())
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
        _, a, _, logp, _ = forward(actor, samples, device, autocast, blind=args.blind)
        w = pm.chosen(a).float()
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
        totals["cut"] += sum(s["cut"] for s in samples)
        if step % args.check_every == 0 or step == args.steps:
            actor.eval()
            parts: list[dict] = []
            with torch.no_grad():
                for i in range(0, len(check), args.samples):
                    chunk = [s for s in pool.map(_sample, check[i:i + args.samples], chunksize=1) if s["counted"]]
                    if chunk:
                        parts.append(measure(*forward(actor, chunk, device, autocast, measuring=True, blind=args.blind)[1:]))
            actor.train()
            held = {}
            for key in sorted({k for p in parts for k in p}):
                if key.startswith("n_"):
                    held[key] = sum(p.get(key, 0.0) for p in parts)
                else:
                    weight = ("n_done" if key in ("done_right", "nll_done") else "n_" + key[6:] if key.startswith("right_") else "n_act")
                    vals = [(p[key], p.get(weight, 1.0)) for p in parts if key in p and not np.isnan(p[key])]
                    held[key] = sum(v * n for v, n in vals) / max(sum(n for _, n in vals), 1e-9)
            seconds = time.perf_counter() - step_began
            row = {"step": step, "loss": totals["loss"] / max(totals["steps"], 1), "lr": lr,
                   "decisions_a_second": totals["decisions"] / max(seconds, 1e-9), "missed_share": totals["missed"] / max(totals["orders"], 1),
                   "cut_share": totals["cut"] / max(totals["orders"], 1),
                   **{f"held_{k}": round(v, 4) for k, v in held.items()}}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            totals.clear()
            step_began = time.perf_counter()
            temp = run / "actor.pt.tmp"
            torch.save({"actor": actor.state_dict(), "step": step, "config": vars(args), "vocabulary": vocabulary}, temp)
            os.replace(temp, run / "actor.pt")
    pool.terminate()


if __name__ == "__main__":
    main()
