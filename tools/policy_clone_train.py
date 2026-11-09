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
  --decide-every  recorded decisions between the actor's (SCRIPT_EVERY by default: 3 s): it reads only those frames, as it
                  would play at that rate, so a stretch of 48 is 48 looks.  At 1 it read every frame, 6 for each look it learnt
                  from, and cloned about 50 looks a second (03-training.md 11.20).

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
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import frame_vae  # noqa: E402
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
    _W.update(bundles=bundles, regions=regions, vocabulary=vocabulary, looked_only=looked_only, table=tables.table())


def _samples(spec: tuple[str, int, int, int, int, int, int, int]) -> list[dict]:
    """`count` stretches of one side of a game (`_cut`), one after another from `start`, the game decoded once for them all and
    each frame featurized once: a record is changes along the game, so a stretch late in it decoded the whole game before it, a
    third of a worker's time with one stretch a read; and a stretch's history is the one before's end (2026-11-09)."""
    path, side, start, count, length, reach, seed, every, *rest = spec
    z = rest[0] if rest else None       # the side's strategy (policy_clone_data.load_z), or None
    # Only this side, and only as far as the stretches: decoding was most of a sample's time, as the whole game for a few minutes of it.
    frames, orders, info = cd.read_side(pathlib.Path(path), side, until=(start + count * length - 1) * every + 1)
    decisions = (len(frames) - 1) // every + 1
    starts = [min(start, decisions - 1)] + [s for s in range(start + length, start + count * length, length) if s < decisions]
    bundle = _W["bundles"][info["map"]]
    made: dict[int, tuple] = {}

    def frame(j: int) -> tuple:
        if j not in made:
            k = j * every
            t = fz.featurize(frames[k], bundle, _W["table"], enabled_kinds=se.KINDS, economy=True)
            x, z = cd.actions.world_positions(t, bundle)
            made[j] = (t, *cd.actions.encode(se.policy_orders(orders[k]), t, bundle, x, z))
        return made[j]

    return [{**_cut(frame, orders, decisions, s, length, reach, seed + i, every), "z": z} for i, s in enumerate(starts)]


def _sample(spec: tuple) -> dict:
    """The first stretch of a spec alone (the held-out check's)."""
    path, side, start, _, *rest = spec
    return _samples((path, side, start, 1, *rest))[0]


def _cut(frame, orders: list, decisions: int, start: int, length: int, reach: int, seed: int, every: int) -> dict:
    """One stretch of a game, ready to batch: what the encoder reads (history, then the stretch), what the script did at each of
    those frames (the actor reads it as action tokens), what the heads read (the counted decisions), and their labels.  Positions
    are the actor's decisions, `every` recorded decisions apart: its decision j is recorded decision j * every.  frame(j): decision
    j featurized, and the script's orders there encoded (`_samples`)."""
    rng = np.random.default_rng(seed)
    names = np.arange(_W["vocabulary"], dtype=np.int32)
    names[rng.random(_W["vocabulary"]) < NAME_DROPOUT] = _W["vocabulary"] - 1
    first = max(0, start - reach)
    end = min(decisions, start + length)
    tensors, done, labels, labelled, counted, missed, cut = [], [], [], [], [], 0, 0
    for j in range(first, end):
        t, decision, m = frame(j)
        done.append(cd.actions.to_slots(decision, slots=None))
        if j >= start and (not _W["looked_only"] or j * every % se.SCRIPT_EVERY == 0):
            labels.append(cd.actions.to_slots(decision, rng.permutation(t.controllable)))
            labelled.append(decision.kind != rf.KEEP)
            counted.append(j - start)
            missed += m
            cut += int(labelled[-1].sum()) - int((labels[-1].anchor >= 0).sum())
        # This stretch's names, on a copy: a frame may be another stretch's too.
        tensors.append(replace(t, true_name=t.unit_name, unit_name=names[t.unit_name]))
    # Every frame as the encoder and the action tokens read it; and only for the counted decisions, what the heads alone read.  The
    # heads' batch is those rows of the frames', picked out on the GPU (05-literature.md S2): sending both whole was twice the bytes.
    seen, heads = pm.collate(tensors, regions=_W["regions"]).half().split([start - first + j for j in counted])
    return {"seen": seen, "heads": heads, "history": start - first, "length": end - start, "counted": counted, "missed": missed, "cut": cut,
            "orders": int(sum(len(orders[j * every]) for j in range(start, end))), "labelled": labelled,
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


def _specs(files: list[tuple[str, int, int]], length: int, reach: int, rng: np.random.Generator, every: int, per_read: int = 1,
           zs: dict | None = None):
    """Runs of `per_read` stretches of games, one after another, drawn in proportion to the games' length, without end; in the
    actor's decisions, `every` recorded ones apart.  zs: each side's z by (file name, side) (policy_clone_data.load_z), to send."""
    lengths = np.array([n for _, _, n in files], float)
    p = lengths / lengths.sum()
    while True:
        i = int(rng.choice(len(files), p=p))
        path, side, n = files[i]
        decisions = (n - 1) // every + 1
        start = int(rng.integers(0, max(1, decisions - per_read * length + 1)))
        spec = (path, side, start, per_read, length, reach, int(rng.integers(1 << 31)), every)
        yield spec + (_z_of(zs, path, side),) if zs is not None else spec


def _z_of(zs: dict, path: str, side: int) -> np.ndarray | None:
    found = zs.get((pathlib.Path(path).name, side))
    return None if found is None else found[0]


def _mixed(a, b, share: float, rng: np.random.Generator):
    """From b with probability share, else from a."""
    while True:
        yield next(b) if rng.random() < share else next(a)


class Prefetch:
    """Samples from the workers, at most `ahead` reads of them made or being made at once, handed out at random from at least
    `mixed` made, so that a batch isn't one game's stretches.  (Pool.imap over an endless stream doesn't wait: it hands out work as
    fast as it can and keeps every result until it's read, which filled 30 GB in a minute.)"""

    def __init__(self, pool, specs, ahead: int, mixed: int = 1, seed: int = 0):
        self.pool, self.specs, self.mixed = pool, specs, mixed
        self.rng = np.random.default_rng(seed)
        self.ready: list[dict] = []
        self.waiting = deque(pool.apply_async(_samples, (next(specs),)) for _ in range(ahead))

    def __next__(self) -> dict:
        while len(self.ready) < self.mixed:
            self.ready.extend(self.waiting.popleft().get())
            self.waiting.append(self.pool.apply_async(_samples, (next(self.specs),)))
        i = int(self.rng.integers(len(self.ready)))
        self.ready[i], self.ready[-1] = self.ready[-1], self.ready[i]
        return self.ready.pop()


def _fixed_specs(files: list[tuple[str, int, int]], count: int, length: int, reach: int, every: int, zs: dict | None = None) -> list[tuple]:
    rng = np.random.default_rng(12345)
    gen = _specs(files, length, reach, rng, every, zs=zs)
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


def heads_rows(samples: list[dict]) -> np.ndarray:
    """Where the counted decisions are among the samples' frames, joined: each sample's history, then its stretch."""
    rows, at = [], 0
    for s in samples:
        rows += [at + s["history"] + j for j in s["counted"]]
        at += s["history"] + s["length"]
    return np.array(rows, np.int64)


def z_tensor(samples: list[dict], device: str, rng: np.random.Generator | None = None, dropout: float = 0.0) -> torch.Tensor:
    """B x Z_DIM: each sample's z, zeros where it has none.  With dropout, a sample's z is unknown (all zeros) with that
    probability, and with the same again only its counts are (what play knows: the knobs it's told, not what it'll have built)."""
    z = np.zeros((len(samples), policy_latent.Z_DIM), np.float32)
    for i, s in enumerate(samples):
        if s.get("z") is not None:
            z[i] = s["z"]
    if rng is not None and dropout > 0.0:
        u = rng.random(len(samples))
        z[u < dropout] = 0.0
        counts = (u >= dropout) & (u < 2 * dropout)
        z[np.ix_(counts, np.r_[cd.Z_COUNTED, np.arange(cd.Z_COUNTS.start, cd.Z_COUNTS.stop)])] = 0.0
    return torch.as_tensor(z, device=device)


def forward(actor: policy_latent.LatentActor, samples: list[dict], device: str, autocast, measuring: bool = False, blind: bool = False,
            z: torch.Tensor | None = None, act_dropout: float = 0.0):
    """The counted decisions, their labels (actions), which units the script told something, and each action's log-probability; with
    measuring, also each action's anchor and kind logits.  blind: the actor isn't shown what was done (its action tokens are all
    zeros), to measure what they're worth.  z: B x Z_DIM, each sample's strategy (`z_tensor`), or None.  act_dropout: the share of
    frames whose action tokens are zeros, as if nothing had been done (05-literature.md P4)."""
    reach, tokens, latent = actor.reach, actor.encoder.tokens, actor.encoder.latent
    longest = max(s["length"] for s in samples)
    frames = torch.zeros(len(samples), reach + longest, tokens, latent, device=device)
    present = torch.zeros(len(samples), reach + longest, dtype=torch.bool, device=device)
    # Every sample's frames at once, through the encoder and into action tokens: one call each, not one a sample, as the main
    # process making a step's thousands of small calls was what held it up (03-training 11.20).
    sizes = [s["history"] + s["length"] for s in samples]
    phase = torch.profiler.record_function      # each part's time, under the profiler
    with phase("clone.join"):
        # Everything onto the GPU before anything is computed: a copy from pageable memory waits for the work queued before it,
        # so copies between the parts of the forward pass made the host and the GPU take turns.
        seen = pm.join([s["seen"] for s in samples], device).to(device)
        heads = pm.join([s["heads"] for s in samples if s["counted"]], device).to(device)
        done = None if blind else _actions({key: [x for s in samples for x in s["done"][key]] for key in cd.actions.SLOT_FIELDS}, device)
        a = _actions({key: [x for s in samples for x in s["labels"][key]] for key in cd.actions.SLOT_FIELDS}, device)
        rows, labelled, counted = pm.stage([heads_rows(samples), _pad([x for s in samples for x in s["labelled"]], heads.ctrl_mask.shape[1], 0) > 0,
                                            np.array([i * longest + j for i, s in enumerate(samples) for j in s["counted"]], np.int64)], device)
    with torch.no_grad(), autocast, phase("clone.encoder"):
        latents = actor.encoder(seen).float().split(sizes)
    for i, s in enumerate(samples):
        h = s["history"]
        frames[i, reach - h:reach + s["length"]] = latents[i]
        present[i, reach - h:] = True
    with autocast:
        # What the script did at every frame, as the actor reads it: tokens after each frame's.
        with phase("clone.action_tokens"):
            parts = () if blind else actor.action_tokens(seen, done).split(sizes)
            acted = torch.zeros(len(samples), reach + longest, policy_latent.ACTION_TOKENS, actor.d, device=device,
                                dtype=parts[0].dtype if parts else torch.float32)
            for i, part in enumerate(parts):
                acted[i, reach - samples[i]["history"]:reach + samples[i]["length"]] = part
            if act_dropout > 0.0:
                acted = acted * (torch.rand(acted.shape[:2], device=device) >= act_dropout)[..., None, None].to(acted.dtype)
        with phase("clone.sequence"):
            held = actor.sequence(frames, present, acted, z=z)[:, reach:]
        with phase("clone.context"):
            now = held.flatten(0, 1).index_select(0, counted)
            whole = seen.rows(rows).with_heads(heads)
            ctx = actor.context(whole, now)
        with phase("clone.actions"):
            logp, _ = pm.evaluate_actions(actor, whole, ctx, a, labelled)
            logits = pm.teacher_logits(actor, whole, ctx, a) if measuring else None
    return whole, a, labelled, logp.float(), logits, ctx.ordered


def ordered_loss(whole: pm.Batch, labelled: torch.Tensor, ordered: torch.Tensor) -> torch.Tensor:
    """05-literature.md P2: for every unit an action could pick, whether the script told it anything this decision; a label for
    every unit at every decision, where the anchor's softmax gives one an action.  Mean over those units."""
    legal = pm.anchors_legal(whole)
    bce = F.binary_cross_entropy_with_logits(ordered, labelled.float(), reduction="none")
    return (bce * legal).sum() / legal.sum().clamp_min(1)


def measure(whole: pm.Batch, a: pm.Actions, labelled: torch.Tensor, logp: torch.Tensor, logits: tuple[torch.Tensor, torch.Tensor],
            ordered: torch.Tensor) -> dict[str, float]:
    """Over the actions (n_act) and the dones (n_done) of the labels, each given those before it: how often the most likely anchor
    is a unit the script told something and no action has picked yet (anchor_right), or done where there was more to do
    (stops_early); how often it's done where the script was (done_right); how often the most likely kind is the script's, given its
    anchor (kind_right, and right_KIND for each kind); and the negative log-probabilities, also for each kind (nll_KIND).
    Over the units an action could pick (n_unit): the side loss of whether each is told anything (ordered_bce), what it would be
    guessing the share told for all (ordered_base), and how often the more likely answer is right (ordered_right)."""
    anchor_l, kind_l = logits
    legal = pm.anchors_legal(whole)
    told = labelled[legal].float()
    rate = float(told.mean()) if len(told) else 0.0
    base = F.binary_cross_entropy(torch.full_like(told, min(max(rate, 1e-6), 1 - 1e-6)), told) if len(told) else torch.tensor(float("nan"))
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
           "nll_done": float(nll[is_done].mean()) if is_done.any() else nan,
           "n_unit": float(len(told)), "ordered_bce": float(ordered_loss(whole, labelled, ordered)), "ordered_base": float(base),
           "ordered_right": float(((ordered[legal] > 0) == (told > 0.5)).float().mean()) if len(told) else nan}
    for k in np.unique(label[is_act]):
        here = is_act & (label == k)
        out[f"right_{cd.rf_name(int(k))}"] = float((kind_guess == label)[here].mean())
        out[f"n_{cd.rf_name(int(k))}"] = float(here.sum())
        # The whole action's: for a build, its type, place and offset together.
        out[f"nll_{cd.rf_name(int(k))}"] = float(nll[here].mean())
    return out


# Training -------------------------------------------------------------------------------------------------------------------------------------------------

ECONOMY_KINDS = (rf.BUILD, rf.ASSIST, rf.RECLAIM)


def action_weights(a: pm.Actions, economy_weight: float) -> torch.Tensor:
    """Each label's weight in the loss: 1, or economy_weight for an economy action (05-literature.md C4).  Done is 1."""
    w = pm.chosen(a).float()
    if economy_weight != 1.0:
        eco = (a.anchor >= 0) & torch.isin(a.kind, torch.tensor(ECONOMY_KINDS, device=a.kind.device))
        w = w * torch.where(eco, economy_weight, 1.0)
    return w


def held_out(actor, pool, check: list[tuple], samples: int, device: str, autocast, blind: bool = False, z_known: bool = True) -> dict:
    """The measures (`measure`) over the held-out stretches, each weighed by how many it's over.  z_known: the actor is told each
    side's z, if it has one; else not (as playing with none)."""
    actor.eval()
    parts: list[dict] = []
    with torch.no_grad():
        for i in range(0, len(check), samples):
            chunk = [s for s in pool.map(_sample, check[i:i + samples], chunksize=1) if s["counted"]]
            if chunk:
                z = z_tensor(chunk, device) if z_known and any(s.get("z") is not None for s in chunk) else None
                parts.append(measure(*forward(actor, chunk, device, autocast, measuring=True, blind=blind, z=z)))
    actor.train()
    held = {}
    for key in sorted({k for p in parts for k in p}):
        if key.startswith("n_"):
            held[key] = sum(p.get(key, 0.0) for p in parts)
        else:
            weight = ("n_done" if key in ("done_right", "nll_done") else "n_act" if key == "nll_act" else
                      "n_unit" if key.startswith("ordered_") else
                      "n_" + key[6:] if key.startswith("right_") else "n_" + key[4:] if key.startswith("nll_") else "n_act")
            vals = [(p[key], p.get(weight, 1.0)) for p in parts if key in p and not np.isnan(p[key])]
            held[key] = sum(v * n for v, n in vals) / max(sum(n for _, n in vals), 1e-9)
    return held


def load_actor(path: str | pathlib.Path, vocabulary: int, encoder: str, decide_every: int, device: str,
               size: policy_latent.Size | None = None) -> policy_latent.LatentActor:
    """An actor from a cloning checkpoint, at its own size; or from PPO's (policy_train), its moving average, which plays its tests.
    A checkpoint from before z has no z_in: it starts at zero, as new."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    actor = policy_latent.make(vocabulary, encoder, decide_every, size or policy_latent.Size(**state["size"]))
    missing, unexpected = actor.load_state_dict(state.get("ema_actor", state["actor"]), strict=False)
    assert not unexpected and all(k.startswith("z_in.") for k in missing), (missing, unexpected)
    return actor.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--maps", help="the map pool the games were recorded on; without it, the five training maps")
    ap.add_argument("--run", required=True)
    ap.add_argument("--encoder", default=policy_latent.ENCODER)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--samples", type=int, default=16, help="stretches of games a forward pass")
    ap.add_argument("--accumulate", type=int, default=1, help="forward passes a step: a step learns from samples x accumulate")
    ap.add_argument("--muon-lr", type=float, default=0.0, help="above 0, Muon trains the actor's hidden matrices at this rate (S4)")
    ap.add_argument("--length", type=int, default=48, help="the actor's decisions a stretch")
    ap.add_argument("--decide-every", type=int, default=se.SCRIPT_EVERY,
                    help="recorded decisions between the actor's: it reads only the frames it decides at.  The script looks every "
                         "SCRIPT_EVERY, so by default the actor decides when it looks; PPO must play at the same rate")
    ap.add_argument("--all-decisions", action="store_true", help="count every decision, not only those where the script looked")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--check-every", type=int, default=500)
    ap.add_argument("--check-samples", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--blind", action="store_true", help="don't show the actor what was done: its action tokens are all zeros")
    ap.add_argument("--compile", choices=("none", "default", "reduce-overhead"), default="none",
                    help="torch.compile the time transformer; reduce-overhead also replays it as CUDA graphs")
    ap.add_argument("--per-read", type=int, default=4, help="stretches a worker cuts, one after another, from each side of a game it decodes")
    ap.add_argument("--attention", choices=("flex", "dense"), default="flex",
                    help="the time transformer's attention on CUDA: flex skips the keys out of the window (a step 32%% faster, 2026-11-09)")
    ap.add_argument("--gpu-memory", type=float, default=0.85, help="share of the GPU's memory this may use before it fails")
    ap.add_argument("--ordered-weight", type=float, default=1.0, help="the side loss of whether each unit is told anything (P2)")
    ap.add_argument("--size", choices=("default", "small"), default="default", help="the actor's: policy_latent.SIZE or SMALL (P5)")
    ap.add_argument("--init", help="start from this cloning checkpoint's weights (its size), as for DAgger's fine-tuning")
    ap.add_argument("--eval-only", action="store_true",
                    help="with --init: only the held-out measures, as trained, blind, and with z unknown if there's z (P4)")
    ap.add_argument("--economy-weight", type=float, default=1.0, help="an economy action's weight in the loss, BUILD, ASSIST and RECLAIM (C4)")
    ap.add_argument("--finish-winners", type=float, default=0.0,
                    help="the last share of steps learn only from sides that won (P6); needs the data's z.npz (policy_clone_data zindex)")
    ap.add_argument("--act-dropout", type=float, default=0.0, help="the share of frames whose action tokens are zeros (P4)")
    ap.add_argument("--z", action="store_true", help="tell the actor each side's strategy, z, from the data's z.npz (P8, C5)")
    ap.add_argument("--z-dropout", type=float, default=0.2,
                    help="with --z: the share of samples with z unknown, and the same again with only its counts unknown")
    ap.add_argument("--mix", help="another dataset to draw from as well, such as DAgger's (P3)")
    ap.add_argument("--mix-share", type=float, default=0.2, help="the share of samples from --mix")
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        (run / "config.json").write_text(json.dumps(vars(args), indent=1))
    began = time.perf_counter()

    by_map = bundles(args.maps)
    regions = max(b.regions for b in by_map.values())
    table = tables.table()
    vocabulary = len(table.names) + 1
    data = pathlib.Path(args.data)
    train_files, held_files = _files(data)
    zs = None
    if args.z or args.finish_winners > 0.0:
        zs = cd.load_z(data)
        if args.mix and (pathlib.Path(args.mix) / "z.npz").exists():
            zs.update(cd.load_z(pathlib.Path(args.mix)))
    send_z = zs if args.z else None
    mix_files = _files(pathlib.Path(args.mix))[0] if args.mix else None
    pool = multiprocessing.get_context("fork").Pool(args.workers, _setup, (by_map, regions, vocabulary, not args.all_decisions))
    loaded = time.perf_counter() - began

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Past the card's memory, the Windows driver spills into system RAM (30 GB on 2026-11-09) rather than failing: fail instead.
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory)
    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")
    assert se.SCRIPT_EVERY % args.decide_every == 0, "the actor's decisions must include each one where the script looks"
    size = policy_latent.SMALL if args.size == "small" else None
    if args.init:
        actor = load_actor(args.init, vocabulary, args.encoder, args.decide_every, device, size)
    else:
        actor = policy_latent.make(vocabulary, args.encoder, args.decide_every, size or policy_latent.SIZE).to(device)
    policy_latent.compile_for_cloning(actor, args.compile, flex=device == "cuda" and args.attention == "flex")
    reach = actor.reach
    check = _fixed_specs(held_files, args.check_samples, args.length, reach, args.decide_every, zs=send_z)

    if args.eval_only:
        out = {"init": args.init, "check_samples": args.check_samples}
        for name, kw in (("normal", {}), ("blind", {"blind": True})) + ((("z_unknown", {"z_known": False}),) if send_z else ()):
            out[name] = {k: round(v, 4) for k, v in held_out(actor, pool, check, args.samples, device, autocast, **kw).items()}
            print(json.dumps({name: out[name]}), flush=True)
        (run / "eval.json").write_text(json.dumps(out, indent=1))
        pool.terminate()
        os._exit(0)

    trained = [p for p in actor.parameters() if p.requires_grad]
    optimizer = frame_vae.Optimisers(actor, SimpleNamespace(lr=args.lr, muon_lr=args.muon_lr, weight_decay=0.01), policy_latent.answers)
    rng = np.random.default_rng(args.seed)

    def stream_of(files):
        specs = _specs(files, args.length, reach, rng, args.decide_every, args.per_read, zs=send_z)
        if mix_files:
            specs = _mixed(specs, _specs(mix_files, args.length, reach, rng, args.decide_every, args.per_read, zs=send_z), args.mix_share, rng)
        # Each worker read gives per_read stretches; the trainer draws from 4 forward passes' worth made, so a pass mixes many games.
        return Prefetch(pool, specs, ahead=max(2 * args.samples, 3 * args.workers), mixed=4 * args.samples, seed=args.seed)

    stream = stream_of(train_files)
    winners_from, winners = args.steps + 1, []
    if args.finish_winners > 0.0:
        winners_from = int(args.steps * (1.0 - args.finish_winners)) + 1
        winners = [f for f in train_files if zs.get((pathlib.Path(f[0]).name, f[1]), (None, 0.0))[1] > 0.0]
    log = open(run / "log.jsonl", "a")
    print(json.dumps({"train_files": len(train_files), "held_files": len(held_files), "loading_seconds": round(loaded, 1),
                      "trained_parameters": sum(p.numel() for p in trained), "mix_files": len(mix_files) if mix_files else 0,
                      "winners_from": winners_from if args.finish_winners else None, "winning_sides": len(winners)}), flush=True)
    z_rng = np.random.default_rng(args.seed + 1)

    totals: Counter = Counter()
    step_began = time.perf_counter()
    for step in range(1, args.steps + 1):
        if step == winners_from:
            stream = stream_of(winners)     # what the old stream had in hand is dropped
        lr = args.lr * min(1.0, step / 200.0) * 0.5 * (1.0 + np.cos(np.pi * step / args.steps))
        optimizer.set_rate(lr)
        optimizer.zero_grad(set_to_none=True)
        # A step's samples in `accumulate` parts: the step's loss is the mean of theirs.  Each part's are taken as it's needed, so the
        # workers make the next while the GPU runs this one: taking a whole step's first left each waiting on the other, the GPU
        # busy a third of the time (2026-11-09).
        for _ in range(args.accumulate):
            samples = [s for s in (next(stream) for _ in range(args.samples)) if s["counted"]]
            if not samples:
                continue
            z = z_tensor(samples, device, z_rng, args.z_dropout) if args.z else None
            whole, a, labelled, logp, _, ordered = forward(actor, samples, device, autocast, blind=args.blind, z=z, act_dropout=args.act_dropout)
            # Over the labels' count, not the weights' sum, so that economy_weight adds weight rather than moving it.
            w = action_weights(a, args.economy_weight)
            loss = -(w * logp).sum() / pm.chosen(a).float().sum().clamp_min(1.0) / args.accumulate
            side = ordered_loss(whole, labelled, ordered) / args.accumulate
            (loss + args.ordered_weight * side).backward()
            totals["loss"] += float(loss.detach())
            totals["ordered"] += float(side.detach())
            totals["decisions"] += sum(len(s["counted"]) for s in samples)
            totals["orders"] += sum(s["orders"] for s in samples)
            totals["missed"] += sum(s["missed"] for s in samples)
            totals["cut"] += sum(s["cut"] for s in samples)
        torch.nn.utils.clip_grad_norm_(trained, 1.0)
        optimizer.step()
        totals["steps"] += 1
        if step % args.check_every == 0 or step == args.steps:
            held = held_out(actor, pool, check, args.samples, device, autocast, blind=args.blind)
            if send_z and step == args.steps:
                held.update({f"zu_{k}": v for k, v in held_out(actor, pool, check, args.samples, device, autocast, z_known=False).items()})
            seconds = time.perf_counter() - step_began
            row = {"step": step, "loss": totals["loss"] / max(totals["steps"], 1), "ordered_bce": totals["ordered"] / max(totals["steps"], 1), "lr": lr,
                   "decisions_a_second": totals["decisions"] / max(seconds, 1e-9), "missed_share": totals["missed"] / max(totals["orders"], 1),
                   "cut_share": totals["cut"] / max(totals["orders"], 1),
                   **{f"held_{k}": round(v, 4) for k, v in held.items()}}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            totals.clear()
            step_began = time.perf_counter()
            temp = run / "actor.pt.tmp"
            torch.save({"actor": actor.state_dict(), "step": step, "config": vars(args), "vocabulary": vocabulary,
                        "size": asdict(actor.size)}, temp)
            os.replace(temp, run / "actor.pt")
    pool.terminate()
    os._exit(0)     # Pool.terminate can hang on its workers' pipes


if __name__ == "__main__":
    main()
