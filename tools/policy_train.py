"""The PPO trainer (docs/learned-policy/03-training.md section 1), for the actor on latents (tools/policy_latent.py) in the sim.  In WSL:

  ~/zkml/bin/python tools/policy_train.py --run /mnt/d/zk-learned/runs/NAME --decisions 2000000
  ~/zkml/bin/python tools/policy_train.py --run ... --check-repro       # one iteration twice, compared
  ~/zkml/bin/python tools/policy_train.py --run ... --check-kill        # killed while saving, the checkpoint still loads

Rollouts keep raw frames, not tensors, and featurize per minibatch.  A segment is 32 decisions.  Each unit gets its own clipped
ratio, averaged within a decision, with one advantage per decision.  The actor reads frames in order, so it takes each segment in
one forward pass, from the latents kept at collection and those of the frames before it, which the segment stores as its start.
(The first actor, recurrent over every token, was trained step by step; it went on 2026-11-08 with the toy game it was tested on.)
"""
from __future__ import annotations

import argparse
import copy
import functools
import json
import math
import multiprocessing
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field

# Before torch starts: reserved memory grew past what was used, as batch shapes changed, until the card spilled into system memory.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bots"))
import policy_latent  # noqa: E402
import policy_model as pm  # noqa: E402
from zkbot.policy import actions as act, bundle as bd, featurize as fz, rawframe as rf, tables  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
HEADS = ("kind", "region", "enemy", "fine", "produce")
NAME_DROPOUT = 0.3
ENTROPY_FLOOR = 0.3         # of a head's maximum
ENTROPY_BASE = 0.01
ENTROPY_MAX = 0.3


@dataclass
class Config:
    seed: int = 1
    decisions: int = 2_000_000
    iteration: int = 8192           # decisions collected per iteration
    games: int = 64
    passes: int = 2
    segment: int = 32
    # Decisions x padded tokens in a minibatch, which is what its memory follows.
    token_budget: int = 165_000
    gpu_share: float = 0.85         # of the card all runs on it may hold.  Past the card it spills into shared memory, at a fifth of the speed.
    gpu_runs: int = 1               # trainings sharing the card, each with its part of gpu_share.  The limit is per process, so it must be told.
    empty_cache: bool = True        # hand back what the allocator holds after each iteration
    lr_start: float = 3e-4
    lr_end: float = 1e-4
    clip_start: float = 0.15
    clip_end: float = 0.1
    gae: float = 0.95
    value_weight: float = 0.5
    grad_clip: float = 0.5
    adam_eps: float = 1e-5
    checkpoint_every: int = 20
    eval_every: int = 20
    eval_games: int = 64
    enabled_kinds: tuple[int, ...] = fz.STAGE2_KINDS
    data_workers: int = 6
    concurrent: bool = True         # collect the next iteration during this one's training
    decide_every: int = 1           # in 15-frame slots: 1 is 2 Hz, 2 is 1 Hz.  Scripted opponents keep their own rate.
    hours: float = 0.0              # stop after this long as well as at `decisions`, if not 0
    self_play_at: float = 0.8       # wins against the script over the last two tests that start the move to self-play
    self_play_holds_below: float = 0.7      # and over the same two, what pauses it
    self_play_per_iteration: float = 0.01   # share of new games added each iteration: 100 iterations from none to all
    latent_encoder: str = policy_latent.ENCODER     # the trained frame encoder whose latents the actor reads


# A big frame costs more memory per token than a small one.  Fitted on the first actor, whose attention was over every pair of
# tokens, at 40, 128 and 256 units a side: this many tokens is where a token's pairs cost what the token itself does.  Kept as a
# cautious bound until it's measured on this actor.
PAIR_TOKENS = 1200
ACT_PAIRS = 24_000_000      # batch x tokens^2 of one forward pass while playing: its pair table is about 20 bytes a pair
FLAT_CHUNK = 64             # decisions a data worker featurizes at a time
# Records past the caps become no tokens.
UNIT_TOKENS = fz.CAP_UNITS + fz.CAP_HIDDEN
# A process's GPU memory that its allocator doesn't count, such as CUDA's own.
PROCESS_OVERHEAD = 400 * 2**20


def gpu_fraction(share: float, runs: int, card_bytes: int) -> float:
    """One run's limit, as a fraction of the card, when `runs` trainings share `share` of it."""
    fraction = (share - (runs - 1) * PROCESS_OVERHEAD / card_bytes) / runs
    if fraction <= 0.0:
        raise ValueError(f"{runs} runs don't fit in {share} of a {card_bytes / 2**30:.1f} GB card")
    return fraction


def memory_tokens(tokens: int) -> float:
    """What a padded frame of this many tokens costs in training's memory, in tokens of a small one."""
    return tokens * (1.0 + tokens / PAIR_TOKENS)


def beta(progress: float) -> float:
    """How much of the reward is the final result: 03-training.md section 1.3."""
    if progress < 0.1:
        return 0.0
    return min(0.9, 0.9 * (progress - 0.1) / 0.5)


@dataclass
class SelfPlay:
    """How much of training is against itself: none until it beats the script, then more each iteration, pausing if it slips."""
    start_at: float
    hold_below: float
    per_iteration: float
    share: float = 0.0
    started_at: int = -1            # iteration
    tests: list[float] = field(default_factory=list)

    def _recent(self) -> float | None:
        """Two tests, since one of 64 games has a standard error of about 5 points."""
        return sum(self.tests[-2:]) / 2.0 if len(self.tests) >= 2 else None

    def note_test(self, wins: float, iteration: int) -> None:
        self.tests.append(wins)
        recent = self._recent()
        if self.started_at < 0 and recent is not None and recent >= self.start_at:
            self.started_at = iteration

    def advance(self) -> float:
        if self.started_at >= 0 and self._recent() >= self.hold_below:
            self.share = min(1.0, self.share + self.per_iteration)
        return self.share


@dataclass
class Step:
    frame: rf.Frame
    kind: np.ndarray
    enemy: np.ndarray
    region: np.ndarray
    fine: np.ndarray
    produce: np.ndarray
    logp: np.ndarray
    value: float = 0.0
    reward: float = 0.0
    advantage: float = 0.0
    ret: float = 0.0
    latents: np.ndarray | None = None   # tokens x latent: this frame's, as the actor read it


@dataclass
class Segment:
    steps: list[Step]
    history: np.ndarray | None      # the frames before its first step, as the actor keeps them (policy_latent.LatentActor.initial_state)
    seen: np.ndarray | None
    bundle: bd.Bundle
    names: np.ndarray


@dataclass
class Trajectory:
    bundle: bd.Bundle
    names: np.ndarray
    steps: list[Step] = field(default_factory=list)
    history: np.ndarray | None = None   # at the first step not yet in a segment
    seen: np.ndarray | None = None
    segment_start: int = 0
    done: bool = False
    bootstrap: float = 0.0


def name_map(rng: np.random.Generator, vocabulary: int) -> np.ndarray:
    """For one seat of one game, some names read as unknown throughout, so the net can't lean on names alone."""
    out = np.arange(vocabulary, dtype=np.int32)
    out[rng.random(vocabulary) < NAME_DROPOUT] = vocabulary - 1
    return out


_WORKER: dict = {}


def _worker_state(bundles: dict, kinds: tuple[int, ...]) -> dict:
    return {"bundles": bundles, "kinds": kinds, "table": tables.table()}


def _worker_setup(bundles: dict, kinds: tuple[int, ...]) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER.update(_worker_state(bundles, kinds))


def _drop_names(t: fz.Tensors, names: np.ndarray) -> None:
    t.true_name, t.unit_name = t.unit_name, names[t.unit_name]


def _worker_chunk(chunk: list[tuple[rf.Frame, int]], state: dict | None = None) -> list[fz.Tensors]:
    """In a worker process, `state` is the process's own; in this one, each DataWorkers passes its own."""
    w = _WORKER if state is None else state
    return [fz.featurize(f, w["bundles"][key], w["table"], enabled_kinds=w["kinds"], privileged=True) for f, key in chunk]


def _worker_batch(work: tuple[list[tuple[rf.Frame, int, np.ndarray]], int], state: dict | None = None) -> pm.Batch:
    """A chunk of a minibatch's decisions, ready for the GPU."""
    items, regions = work
    tensors = _worker_chunk([(f, key) for f, key, _ in items], state)
    for t, (_, _, names) in zip(tensors, items):
        _drop_names(t, names)
    return pm.collate(tensors, regions=regions)


class DataWorkers:
    """Featurizing in processes: it is the one part of a decision the GIL would otherwise hold to a single thread.  They are
    forked with every bundle in memory, so a frame only sends a key; fork before the GPU is touched."""

    def __init__(self, count: int, bundles: list[bd.Bundle], kinds: tuple[int, ...]):
        self.count = count
        self.keys = {id(b) for b in bundles}
        by_key = {id(b): b for b in bundles}
        self.pool = multiprocessing.get_context("fork").Pool(count, _worker_setup, (by_key, kinds)) if count else None
        # Without processes, its own: two learners in one process (as in the tests) would otherwise share a module's.
        self.state = _worker_state(by_key, kinds) if self.pool is None else None

    def map(self, items: list[tuple[rf.Frame, bd.Bundle]]) -> list[fz.Tensors]:
        pairs = [(f, id(b)) for f, b in items]
        if self.pool is None:
            return _worker_chunk(pairs, self.state)
        size = max(1, (len(pairs) + self.count - 1) // self.count)
        chunks = [pairs[i:i + size] for i in range(0, len(pairs), size)]
        return [t for part in self.pool.map(_worker_chunk, chunks, chunksize=1) for t in part]

    def batches(self, steps: list[list[tuple[rf.Frame, bd.Bundle, np.ndarray]]], regions: int):
        """A minibatch's chunks, in order.  The workers run ahead of the GPU, which never waits on numpy unless they fall behind."""
        work = [([(f, id(b), names) for f, b, names in step], regions) for step in steps]
        if self.pool is None:
            return (_worker_batch(w, self.state) for w in work)
        return self.pool.imap(_worker_batch, work, chunksize=1)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()


def _act_slices(items: list[fz.Tensors]) -> list[slice]:
    """Seats in order, cut where the next would take a forward pass's padded pairs over the limit."""
    out, start, most = [], 0, 0
    for i, t in enumerate(items):
        tokens = len(t.unit_x) + len(t.reg_x) + len(t.site_x) + len(t.feat_x) + 2
        if i > start and (i - start + 1) * max(most, tokens) ** 2 > ACT_PAIRS:
            out.append(slice(start, i))
            start, most = i, 0
        most = max(most, tokens)
    out.append(slice(start, len(items)))
    return out


def _stack_padded(parts: list[np.ndarray]) -> np.ndarray:
    """Per-unit arrays of forward passes that padded to different widths.  A reader takes only each seat's own units."""
    out = np.zeros((sum(len(p) for p in parts), max(p.shape[1] for p in parts), *parts[0].shape[2:]), parts[0].dtype)
    at = 0
    for p in parts:
        out[at:at + len(p), :p.shape[1]] = p
        at += len(p)
    return out


class Player:
    """Chooses actions: the learner's own nets, or a frozen copy while the learner goes on training."""

    def __init__(self, actor: policy_latent.LatentActor, critic: pm.Critic, workers: DataWorkers, device: str):
        self.actor, self.critic, self.workers = actor, critic, workers
        # Its own: an autocast object keeps state between enter and exit, and two threads may each be inside one.
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
        self.device = device

    def _features(self, items: list[tuple[rf.Frame, bd.Bundle, np.ndarray]]) -> list[fz.Tensors]:
        out = self.workers.map([(f, b) for f, b, _ in items])
        for t, (_, _, names) in zip(out, items):
            _drop_names(t, names)
        return out

    @torch.no_grad()
    def _forward(self, items: list[fz.Tensors], history: torch.Tensor, seen: torch.Tensor):
        b = pm.collate(items).to(self.device)
        with self.autocast:
            units, reg_h, new_history, new_seen, keep = self.actor.decide(b, history, seen)
            a = pm.sample_actions(self.actor, b, units, reg_h)
            new_history = self.actor.acted(b, a, new_history)       # what it did, for the decisions after
            value = self.critic(b)
        out = {k: getattr(a, k).cpu().numpy() for k in ("kind", "enemy", "region", "fine", "produce", "logp")}
        out.update({k: v.cpu().numpy() for k, v in keep.items()})
        return out, value.float().cpu().numpy(), new_history.float(), new_seen.float()

    def act(self, frames: list[tuple[rf.Frame, bd.Bundle, np.ndarray]], history: torch.Tensor, seen: torch.Tensor):
        """Forward passes for many games, as few as fit the memory.  Returns tensors, numpy actions, values and the new state."""
        items = self._features(frames)
        parts = [self._forward(items[s], history[s], seen[s]) for s in _act_slices(items)]
        if len(parts) == 1:
            return (items, *parts[0])
        out = {k: _stack_padded([p[0][k] for p in parts]) for k in parts[0][0]}
        return items, out, np.concatenate([p[1] for p in parts]), torch.cat([p[2] for p in parts]), torch.cat([p[3] for p in parts])


class Learner:
    def __init__(self, cfg: Config, run: pathlib.Path, workers: DataWorkers, device: str = "cuda"):
        self.cfg, self.run, self.device, self.workers = cfg, run, device, workers
        torch.manual_seed(cfg.seed)
        self.table = tables.table()
        self.vocabulary = len(self.table.names) + 1
        self.actor = policy_latent.make(self.vocabulary, cfg.latent_encoder, cfg.decide_every).to(device)
        self.critic = pm.Critic(self.vocabulary).to(device)
        self.trained = [p for p in list(self.actor.parameters()) + list(self.critic.parameters()) if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.trained, lr=cfg.lr_start, eps=cfg.adam_eps)
        self.decisions = 0
        self.iteration = 0
        self.entropy_coef = {h: ENTROPY_BASE for h in HEADS}
        self.self_play = SelfPlay(cfg.self_play_at, cfg.self_play_holds_below, cfg.self_play_per_iteration)
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
        self.player = Player(self.actor, self.critic, workers, device)
        if device == "cuda":
            # It empties its cache at this limit, and fails rather than spills.
            torch.cuda.set_per_process_memory_fraction(gpu_fraction(cfg.gpu_share, cfg.gpu_runs, torch.cuda.mem_get_info()[1]))

    @property
    def progress(self) -> float:
        return min(1.0, self.decisions / self.cfg.decisions)

    def act(self, frames: list[tuple[rf.Frame, bd.Bundle, np.ndarray]], history: torch.Tensor, seen: torch.Tensor):
        return self.player.act(frames, history, seen)

    def frozen_player(self, workers: DataWorkers) -> Player:
        actor, critic = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        for p in list(actor.parameters()) + list(critic.parameters()):
            p.requires_grad_(False)
        return Player(actor, critic, workers, self.device)

    @torch.no_grad()
    def copy_weights_to(self, player: Player) -> None:
        for mine, theirs in ((self.actor, player.actor), (self.critic, player.critic)):
            for a, b in zip(theirs.parameters(), mine.parameters()):
                a.copy_(b)

    def _minibatches(self, segments: list[Segment], rng: np.random.Generator) -> list[list[Segment]]:
        """Whole segments up to the token budget, near-equal sizes together so padding stays small.  Sizes are jittered, so a
        minibatch isn't all from one stage of the game, and the minibatches then go in random order.  The budget counts padded
        tokens, weighed by `memory_tokens`: a small frame batched with a big map's costs as much as the big map's.  Token groups
        pad separately, so each is tracked on its own."""
        units = np.array([max(min(len(s.frame.units), UNIT_TOKENS) + min(len(s.frame.sites), fz.CAP_SITES) + min(len(s.frame.features), fz.CAP_FEATURES) for s in seg.steps) for seg in segments])
        regions = np.array([seg.bundle.regions for seg in segments])
        lengths = np.array([len(seg.steps) for seg in segments])
        order = np.argsort((units + regions) * rng.uniform(0.85, 1.15, len(segments)))
        batches, current = [], []
        decisions = most_units = most_regions = 0
        for i in order:
            grown = (decisions + lengths[i]) * memory_tokens(max(most_units, units[i]) + max(most_regions, regions[i]) + 2)
            if current and grown > self.cfg.token_budget:
                batches.append(current)
                current = []
                decisions = most_units = most_regions = 0
            current.append(segments[int(i)])
            decisions += lengths[i]
            most_units, most_regions = max(most_units, units[i]), max(most_regions, regions[i])
        if current:
            batches.append(current)
        return [batches[int(k)] for k in rng.permutation(len(batches))]

    def _loss(self, batch: list[Segment], clip: float, note) -> torch.Tensor:
        """Every decision of each segment in one forward pass, from the latents it was played with."""
        device = self.device
        steps = [step for seg in batch for step in seg.steps]
        work = [(step.frame, seg.bundle, seg.names) for seg in batch for step in seg.steps]
        chunks = [work[i:i + FLAT_CHUNK] for i in range(0, len(work), FLAT_CHUNK)]
        whole = pm.join(list(self.workers.batches(chunks, max(seg.bundle.regions for seg in batch)))).to(device)
        lengths = [len(seg.steps) for seg in batch]
        latents = np.zeros((len(batch), max(lengths), *steps[0].latents.shape), np.float32)
        for i, seg in enumerate(batch):
            latents[i, :len(seg.steps)] = [s.latents for s in seg.steps]
        history = torch.as_tensor(np.stack([seg.history for seg in batch]), device=device)
        seen = torch.as_tensor(np.stack([seg.seen for seg in batch]), device=device)
        with self.autocast:
            a = _stored_actions(steps, whole, device)
            acted = self.actor.segment_actions(whole, a, lengths)
            now = self.actor.segments(history, seen, torch.as_tensor(latents, device=device), lengths, acted)
            units, reg_h = self.actor.read(whole, now)
            logp, entropy = pm.evaluate_actions(self.actor, whole, units, reg_h, a)
            value = self.critic(whole)
        return self._ppo_loss(steps, whole, logp, entropy, value, clip, note)

    def _ppo_loss(self, steps: list[Step], whole: pm.Batch, logp: torch.Tensor, entropy: dict, value: torch.Tensor, clip: float, note) -> torch.Tensor:
        cfg, device = self.cfg, self.device
        decisions = len(steps)
        advantages = np.array([s.advantage for s in steps], np.float32)
        adv = torch.as_tensor((advantages - advantages.mean()) / (float(advantages.std()) + 1e-6), device=device)
        ret = torch.as_tensor([s.ret for s in steps], device=device, dtype=torch.float32)
        old = _pad([s.logp for s in steps], whole.ctrl_mask.shape, 0.0, device, torch.float32)
        weights = (whole.ctrl_mask & (whole.kind_mask.sum(-1) > 1)).float()
        ratio = (logp - old).clamp(-20.0, 20.0).exp()
        per_unit = -torch.min(ratio * adv[:, None], ratio.clamp(1.0 - clip, 1.0 + clip) * adv[:, None])
        count = weights.sum(-1)
        has = count > 0
        pg = torch.zeros((), device=device)
        if has.any():
            pg = ((per_unit * weights).sum(-1)[has] / count[has]).sum() / decisions
            with torch.no_grad():
                note("kl", float(((old - logp) * weights).sum() / weights.sum()), float(weights.sum()))
                note("clip_fraction", float((((ratio - 1.0).abs() > clip).float() * weights).sum() / weights.sum()), float(weights.sum()))
        value_loss = (value - ret).pow(2).sum() / decisions
        entropy_loss = torch.zeros((), device=device)
        for h in HEADS:
            e, mask = entropy[h]
            total, n = (e * mask).sum(), float(mask.sum())
            entropy_loss = entropy_loss + self.entropy_coef[h] * total / max(n, 1.0)
            if n:
                note(f"entropy_{h}", float(total.detach()) / n, n)
        note("loss_pg", float(pg.detach()))
        note("loss_value", float(value_loss.detach()))
        note("minibatch_decisions", decisions)
        return pg + cfg.value_weight * value_loss - entropy_loss

    def train(self, segments: list[Segment], rng: np.random.Generator, between=None) -> dict[str, float]:
        """between: called after each optimizer step, the moments when the weights are whole."""
        cfg = self.cfg
        p = self.progress
        clip = cfg.clip_start + (cfg.clip_end - cfg.clip_start) * p
        lr = cfg.lr_start + (cfg.lr_end - cfg.lr_start) * p
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        totals: dict[str, float] = {}
        counts: dict[str, float] = {}

        def note(name: str, value: float, weight: float = 1.0) -> None:
            totals[name] = totals.get(name, 0.0) + value * weight
            counts[name] = counts.get(name, 0.0) + weight

        for _ in range(cfg.passes):
            for batch in self._minibatches(segments, rng):
                self.optimizer.zero_grad(set_to_none=True)
                self._loss(batch, clip, note).backward()
                grad = torch.nn.utils.clip_grad_norm_(self.trained, cfg.grad_clip)
                self.optimizer.step()
                note("grad_norm", float(grad))
                if between is not None:
                    between()
        return {k: totals[k] / max(counts[k], 1e-9) for k in totals}

    def adjust_entropy(self, metrics: dict[str, float]) -> None:
        """Keeps each head above a share of its maximum, so none collapses."""
        maxima = {"kind": math.log(4.0), "region": math.log(4.0), "enemy": math.log(8.0), "fine": math.log(fz.FINE_OPTIONS), "produce": math.log(4.0)}
        for h in HEADS:
            e = metrics.get(f"entropy_{h}")
            if e is None:
                continue
            if e < ENTROPY_FLOOR * maxima[h]:
                self.entropy_coef[h] = min(ENTROPY_MAX, self.entropy_coef[h] * 1.1)
            else:
                self.entropy_coef[h] = max(ENTROPY_BASE, self.entropy_coef[h] * 0.97)

    def save(self) -> None:
        """A kill at any moment leaves latest.json naming a whole checkpoint."""
        directory = self.run / "ckpt"
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*.tmp"):       # from a process killed while saving
            stale.unlink()
        name = f"it-{self.iteration:06d}.pt"
        temp = directory / (name + ".tmp")
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "iteration": self.iteration, "decisions": self.decisions, "entropy_coef": self.entropy_coef, "self_play": asdict(self.self_play)}, temp)
        os.replace(temp, directory / name)
        marker = self.run / "latest.json.tmp"
        marker.write_text(json.dumps({"checkpoint": f"ckpt/{name}", "iteration": self.iteration, "decisions": self.decisions}))
        os.replace(marker, self.run / "latest.json")

    def resume(self) -> bool:
        marker = self.run / "latest.json"
        if not marker.exists():
            return False
        state = torch.load(self.run / json.loads(marker.read_text())["checkpoint"], map_location=self.device, weights_only=True)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.iteration, self.decisions = state["iteration"], state["decisions"]
        self.entropy_coef = state["entropy_coef"]
        if "self_play" in state:
            self.self_play = SelfPlay(**state["self_play"])
        return True


def _pad(arrays: list[np.ndarray], shape: tuple, fill: float, device: str, dtype: torch.dtype) -> torch.Tensor:
    out = np.full((shape[0], shape[1]), fill, np.float32 if dtype == torch.float32 else np.int64)
    for i, a in enumerate(arrays):
        out[i, :len(a)] = a
    return torch.as_tensor(out, device=device, dtype=dtype)


def _stored_actions(steps: list[Step], b: pm.Batch, device: str) -> pm.Actions:
    shape = b.ctrl_mask.shape
    kind = _pad([s.kind for s in steps], shape, 0, device, torch.int64)
    enemy = _pad([s.enemy for s in steps], shape, -1, device, torch.int64)
    region = _pad([s.region for s in steps], shape, -1, device, torch.int64)
    fine = _pad([s.fine for s in steps], shape, -1, device, torch.int64)
    produce = _pad([s.produce for s in steps], shape, -1, device, torch.int64)
    return pm.Actions(kind, enemy, region, fine, produce, torch.zeros(shape, device=device))


def finish(trajectories: list[Trajectory], gae: float) -> tuple[list[Segment], list[float], list[float]]:
    """Advantages over each seat's own decisions; gamma is 1."""
    segments: list[Segment] = []
    values, returns = [], []
    for tr in trajectories:
        steps = tr.steps
        running = 0.0
        next_value = 0.0 if tr.done else tr.bootstrap
        for s in reversed(steps):
            delta = s.reward + next_value - s.value
            running = delta + gae * running
            s.advantage = running
            s.ret = running + s.value
            next_value = s.value
            values.append(s.value)
            returns.append(s.ret)
    return segments, values, returns


class Seats:
    """What the actor keeps of every seat being played, on the GPU."""

    def __init__(self, seats: int, device: str, actor: policy_latent.LatentActor):
        self.history, self.seen = actor.initial_state(seats, 0, device)

    def clear(self, seat: int) -> None:
        self.history[seat] = 0.0
        self.seen[seat] = 0.0


class Collector:
    """Plays games for as long as the run lasts.  An iteration stops them nowhere in particular: each seat's rollout is cut, the
    critic stands in for the rest, and the next iteration carries on from there with what the actor kept."""

    def __init__(self, learner: Learner, env, rng: np.random.Generator):
        self.learner, self.env, self.rng = learner, env, rng
        self.player = learner.player
        env.begin()
        self.seats = Seats(env.seat_count, learner.device, learner.actor)
        self.current: dict[int, Trajectory] = {}
        for seat in env.policy_seats():
            self._open(seat)

    def _open(self, seat: int) -> None:
        """A new game for this seat: clear what it kept, draw which names it can read."""
        self.seats.clear(seat)
        tr = Trajectory(self.env.bundle_of(seat), name_map(self.rng, self.learner.vocabulary))
        tr.history, tr.seen = _kept(self.seats.history[seat], self.seats.seen[seat])
        self.current[seat] = tr

    def _carry_on(self, seat: int) -> None:
        old = self.current[seat]
        tr = Trajectory(old.bundle, old.names)
        tr.history, tr.seen = _kept(self.seats.history[seat], self.seats.seen[seat])
        self.current[seat] = tr

    def collect(self, want: int) -> tuple[list[Segment], dict[str, float]]:
        """Until `want` decisions of the learning policy are stored."""
        learner, env, cfg, seats, current = self.learner, self.env, self.learner.cfg, self.seats, self.current
        stored = 0
        finished: list[Trajectory] = []
        segments: list[Segment] = []
        wins: dict[str, list[float]] = {}
        masked_out = 0
        kind_counts = np.zeros(rf.KIND_COUNT, np.int64)
        progress = learner.progress
        while stored < want:
            acting = env.policy_seats_to_act()
            if not acting:
                env.step_scripted_only()
                continue
            for seat in acting:
                # It sat out as a script, and meanwhile its game may have moved to another map.
                if not current[seat].steps and current[seat].bundle is not env.bundle_of(seat):
                    self._open(seat)
            frames = [(env.frame(seat), env.bundle_of(seat), current[seat].names) for seat in acting]
            index = torch.as_tensor(acting, device=learner.device)
            items, out, value, new_history, new_seen = self.player.act(frames, seats.history[index], seats.seen[index])
            for k, seat in enumerate(acting):
                t = items[k]
                c = t.controllable
                kind = out["kind"][k, :c].copy()
                if c:
                    masked_out += int((~t.kind_mask[np.arange(c), kind]).sum())
                kind_counts += np.bincount(kind, minlength=rf.KIND_COUNT)
                raw = frames[k][0]
                step = Step(rf.Frame(raw.units.copy(), raw.sites.copy(), raw.glob.copy(), raw.features.copy()), kind, out["enemy"][k, :c].copy(),
                            out["region"][k, :c].copy(), out["fine"][k, :c].copy(), out["produce"][k, :c].copy(), out["logp"][k, :c].copy(), float(value[k]))
                if "latents" in out:
                    step.latents = out["latents"][k].copy()
                tr = current[seat]
                tr.steps.append(step)
                stored += 1
                x, z = act.world_positions(t, tr.bundle)
                env.give_orders(seat, act.decode(act.Decision(step.kind, step.enemy, step.region, step.fine, step.produce), t, tr.bundle, x, z))
                # A segment ends here: the next one starts from what the actor keeps now.
                if len(tr.steps) - tr.segment_start == cfg.segment:
                    _cut(tr, segments, new_history[k], new_seen[k])
            seats.history[index] = new_history
            seats.seen[index] = new_seen
            env.step(progress)
            for seat in acting:
                current[seat].steps[-1].reward = env.reward(seat, beta(progress))
            for seat, mode, won in env.finished_seats():
                wins.setdefault(mode, []).append(won)
                tr = current[seat]
                tr.done = True
                _close(tr, segments)
                finished.append(tr)
                self._open(seat)
        # Whatever is still running stops here for now.
        acting = [s for s, tr in current.items() if tr.steps]
        if acting:
            frames = [(env.frame(seat), env.bundle_of(seat), current[seat].names) for seat in acting]
            index = torch.as_tensor(acting, device=learner.device)
            _, _, value, _, _ = self.player.act(frames, seats.history[index], seats.seen[index])
            for k, seat in enumerate(acting):
                tr = current[seat]
                tr.bootstrap = float(value[k])
                _close(tr, segments)
                finished.append(tr)
                self._carry_on(seat)
        # The same horizon in game time whatever the decision rate.
        _, values, returns = finish(finished, cfg.gae ** cfg.decide_every)
        v, r = np.array(values), np.array(returns)
        explained = 1.0 - float(np.var(r - v)) / max(float(np.var(r)), 1e-8)
        metrics = {"explained_variance": explained, "masked_out": masked_out, "decisions": stored, "return_min": float(r.min()), "return_max": float(r.max())}
        total = max(int(kind_counts.sum()), 1)
        metrics.update({f"share_kind{k}": float(kind_counts[k]) / total for k in range(rf.KIND_COUNT) if kind_counts[k]})
        for mode, results in wins.items():
            metrics[f"wins_{mode}"] = float(np.mean(results))
            metrics[f"games_{mode}"] = len(results)
        return segments, metrics


def _kept(history: torch.Tensor, seen: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """What a seat kept, as a segment starting now stores it: the latents and action tokens of the frames before, as they were read."""
    return history.cpu().numpy().copy(), seen.cpu().numpy().copy()


def _cut(tr: Trajectory, segments: list[Segment], history: torch.Tensor, seen: torch.Tensor) -> None:
    steps = tr.steps[tr.segment_start:]
    segments.append(Segment(steps, tr.history, tr.seen, tr.bundle, tr.names))
    tr.segment_start = len(tr.steps)
    tr.history, tr.seen = _kept(history, seen)


def _close(tr: Trajectory, segments: list[Segment]) -> None:
    if tr.segment_start < len(tr.steps):
        segments.append(Segment(tr.steps[tr.segment_start:], tr.history, tr.seen, tr.bundle, tr.names))
        tr.segment_start = len(tr.steps)


@torch.no_grad()
def evaluate(learner: Learner, make_env, opponent: str, games: int, seed: int) -> dict[str, float]:
    """The current policy against a script, sampled as in training.  Each game slot counts its first game only: the games that
    end soonest would otherwise count most, and those are not a fair sample."""
    env = make_env(games, seed, modes=[opponent])
    env.begin()
    seats = Seats(env.seat_count, learner.device, learner.actor)
    names = np.arange(learner.vocabulary, dtype=np.int32)
    results: dict[int, float] = {}
    records: dict[int, dict] = {}
    while len(results) < games:
        acting = env.policy_seats_to_act()
        if not acting:
            env.step_scripted_only()
        else:
            frames = [(env.frame(seat), env.bundle_of(seat), names) for seat in acting]
            index = torch.as_tensor(acting, device=learner.device)
            items, out, _, new_history, new_seen = learner.act(frames, seats.history[index], seats.seen[index])
            for k, seat in enumerate(acting):
                t = items[k]
                c = t.controllable
                d = act.Decision(out["kind"][k, :c], out["enemy"][k, :c], out["region"][k, :c], out["fine"][k, :c], out["produce"][k, :c])
                b = env.bundle_of(seat)
                x, z = act.world_positions(t, b)
                env.give_orders(seat, act.decode(d, t, b, x, z))
            seats.history[index] = new_history
            seats.seen[index] = new_seen
            env.step(1.0)
        for seat, _, won in env.finished_seats():
            if seat not in results and seat in getattr(env, "ended", {}):
                records[seat] = env.ended[seat]
            results.setdefault(seat, won)
            seats.clear(seat)
    r = np.array(list(results.values()), np.float32)
    return {"win_rate": float((r == 1).mean()), "draw_rate": float((r == 0.5).mean()), "loss_rate": float((r == 0).mean()), "games": len(r), "records": list(records.values())}


def make_sim(games: int, seed: int, modes: list[str] | None = None, decide_every: int = 1):
    import policy_sim_env

    return policy_sim_env.SimEnv(games, seed, modes or ["fight"], decide_every=decide_every)


class Ahead:
    """Collects the next iteration while this one trains, with a copy of the weights.  It starts late enough to end about when
    training does, so that the copy is as recent as it can be without making training wait."""

    def __init__(self, collector: Collector, player: Player, want: int, delay: float):
        self.collector, self.player, self.want, self.delay = collector, player, want, delay
        self.t0 = time.perf_counter()
        self.thread: threading.Thread | None = None
        self.out = None
        self.error: BaseException | None = None
        self.seconds = 0.0

    def _run(self) -> None:
        began = time.perf_counter()
        try:
            self.out = self.collector.collect(self.want)
        except BaseException as e:      # raised again on the training thread, at join
            self.error = e
        self.seconds = time.perf_counter() - began

    def tick(self, force: bool = False) -> None:
        if self.thread is None and (force or time.perf_counter() - self.t0 >= self.delay):
            self.collector.learner.copy_weights_to(self.player)
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def join(self) -> tuple[list[Segment], dict[str, float], float]:
        self.tick(force=True)
        self.thread.join()
        if self.error is not None:
            raise self.error
        return (*self.out, self.seconds)


def train(cfg: Config, run: pathlib.Path, make_env, device: str = "cuda") -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    env = make_env(cfg.games, cfg.seed)
    # Separate pools: training queues a minibatch's chunks at once, and collection mustn't wait behind them.  Both before the GPU.
    workers = DataWorkers(cfg.data_workers, env.bundles(), cfg.enabled_kinds)
    act_workers = DataWorkers(cfg.data_workers, env.bundles(), cfg.enabled_kinds) if cfg.concurrent else None
    learner = Learner(cfg, run, workers, device)
    if learner.resume():
        print(f"resumed at iteration {learner.iteration}, {learner.decisions} decisions", flush=True)
        env = make_env(cfg.games, cfg.seed + learner.iteration)
    rng = np.random.default_rng(cfg.seed + learner.iteration)
    collector = Collector(learner, env, np.random.default_rng((cfg.seed, learner.iteration)))
    frozen = learner.frozen_player(act_workers) if cfg.concurrent else None
    ahead: Ahead | None = None
    collect_s = train_s = 0.0
    log = (run / "metrics.jsonl").open("a")
    began = time.perf_counter()
    env.self_play_share = learner.self_play.share
    while learner.decisions < cfg.decisions:
        t0 = time.perf_counter()
        if ahead is None:
            segments, metrics = collector.collect(cfg.iteration)
            collect_s = time.perf_counter() - t0
        else:
            segments, metrics, collect_s = ahead.join()
        t1 = time.perf_counter()
        if cfg.concurrent:
            collector.player = frozen
            ahead = Ahead(collector, frozen, cfg.iteration, max(0.0, train_s - collect_s))
        metrics.update(learner.train(segments, rng, between=ahead.tick if ahead else None))
        if ahead:
            ahead.tick(force=True)
        t2 = time.perf_counter()
        train_s = t2 - t1
        if cfg.empty_cache and device == "cuda":
            torch.cuda.empty_cache()
        learner.decisions += metrics["decisions"]
        learner.iteration += 1
        learner.adjust_entropy(metrics)
        metrics.update(iteration=learner.iteration, total_decisions=learner.decisions, collect_s=collect_s, wait_s=t1 - t0, train_s=train_s,
                       collect_rate=metrics["decisions"] / max(collect_s, 1e-9), train_rate=metrics["decisions"] * cfg.passes / train_s,
                       overall_rate=metrics["decisions"] / (t2 - t0), game_minutes=learner.decisions * cfg.decide_every / 120.0,
                       beta=beta(learner.progress), entropy_coef=dict(learner.entropy_coef))
        if metrics["masked_out"]:
            raise RuntimeError(f"{metrics['masked_out']} chosen actions were outside their mask")
        if learner.iteration % cfg.eval_every == 0:
            for opponent in env.eval_opponents:
                tested = time.perf_counter()
                result = evaluate(learner, make_env, opponent, cfg.eval_games, cfg.seed * 1000 + learner.iteration)
                metrics[f"eval_{opponent}"] = result["win_rate"]
                metrics[f"eval_{opponent}_draws"], metrics[f"eval_{opponent}_losses"] = result["draw_rate"], result["loss_rate"]
                metrics[f"eval_{opponent}_seconds"] = time.perf_counter() - tested
                with (run / "tests.jsonl").open("a") as tests:
                    tests.write(json.dumps({"iteration": learner.iteration, "decisions": learner.decisions, "opponent": opponent, "games": result["records"]}) + "\n")
                if opponent == "fight":
                    learner.self_play.note_test(result["win_rate"], learner.iteration)
        env.self_play_share = metrics["self_play_share"] = learner.self_play.advance()
        if learner.iteration % cfg.checkpoint_every == 0:
            learner.save()
        log.write(json.dumps(metrics) + "\n")
        log.flush()
        show = {k: round(v, 3) for k, v in metrics.items() if isinstance(v, float) and (k.startswith(("wins_", "eval_", "entropy_", "share_kind")) or k in ("explained_variance", "kl", "clip_fraction", "collect_rate", "train_rate", "overall_rate", "minibatch_decisions"))}
        print(f"it {learner.iteration} decisions {learner.decisions} {show}", flush=True)
        if (run / "stop").exists() or (cfg.hours and time.perf_counter() - began >= cfg.hours * 3600.0):
            break
    if ahead:
        ahead.join()
    learner.save()
    workers.close()
    if act_workers:
        act_workers.close()


def check_repro(cfg: Config, run: pathlib.Path, make_env) -> None:
    """The same iteration twice from the same seed: 03-training.md section 7."""
    # Plain GPU scatter adds accumulate in no fixed order, and Adam magnifies the rounding on near-zero gradients.
    torch.use_deterministic_algorithms(True, warn_only=True)
    sums = []
    env = make_env(cfg.games, cfg.seed)
    workers = DataWorkers(cfg.data_workers, env.bundles(), cfg.enabled_kinds)
    for attempt in range(2):
        torch.manual_seed(cfg.seed)
        learner = Learner(cfg, run / f"repro{attempt}", workers)
        rng = np.random.default_rng(cfg.seed)
        segments, metrics = Collector(learner, make_env(cfg.games, cfg.seed), rng).collect(cfg.iteration)
        metrics.update(learner.train(segments, rng))
        sums.append(({k: v.detach().cpu().clone() for k, v in learner.actor.state_dict().items()}, metrics))
    workers.close()
    worst = max(float((sums[0][0][k] - sums[1][0][k]).abs().max()) for k in sums[0][0])
    worst_name = max(sums[0][0], key=lambda k: float((sums[0][0][k] - sums[1][0][k]).abs().max()))
    print(f"one iteration twice: worst parameter difference {worst:.2e} in {worst_name}; losses {sums[0][1]['loss_pg']:.6f} {sums[1][1]['loss_pg']:.6f}")
    if worst > 1e-4:
        raise SystemExit("not reproducible within 1e-4")


def _save_forever(cfg: Config, run: pathlib.Path) -> None:
    learner = Learner(cfg, run, DataWorkers(0, [], cfg.enabled_kinds))
    while True:
        learner.iteration += 1
        learner.save()
        print("saved", flush=True)


def check_kill(cfg: Config, run: pathlib.Path, tries: int = 12) -> None:
    """Kills a process in the middle of saving, again and again.  Whatever latest.json names must load, every time."""
    rng = np.random.default_rng(cfg.seed)
    args = [sys.executable, __file__, "--run", str(run), "--latent-encoder", cfg.latent_encoder, "--save-forever"]
    seen = set()
    for attempt in range(tries):
        child = subprocess.Popen(args, stdout=subprocess.PIPE, text=True)
        assert child.stdout is not None
        child.stdout.readline()         # one whole save, so the marker exists and the GPU is warm
        time.sleep(float(rng.uniform(0.0, 0.4)))
        child.kill()
        child.wait()
        learner = Learner(cfg, run, DataWorkers(0, [], cfg.enabled_kinds))
        assert learner.resume(), "latest.json is missing"
        seen.add(learner.iteration)
        print(f"kill {attempt}: resumed at iteration {learner.iteration}", flush=True)
        for leftover in (run / "ckpt").glob("*.pt"):
            leftover.unlink(missing_ok=True) if leftover.name != json.loads((run / "latest.json").read_text())["checkpoint"].split("/")[-1] else None
    print(f"every kill left a loadable checkpoint; {len(seen)} different iterations seen")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--decisions", type=int, default=Config.decisions)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--games", type=int, default=Config.games)
    ap.add_argument("--iteration", type=int, default=Config.iteration)
    ap.add_argument("--passes", type=int, default=Config.passes)
    ap.add_argument("--segment", type=int, default=Config.segment)
    ap.add_argument("--token-budget", type=int, default=Config.token_budget)
    ap.add_argument("--concurrent", action=argparse.BooleanOptionalAction, default=Config.concurrent, help="collect the next iteration during this one's training")
    ap.add_argument("--decide-every", type=int, default=Config.decide_every, help="in 15-frame slots: 1 is 2 Hz, 2 is 1 Hz")
    ap.add_argument("--workers", type=int, default=Config.data_workers)
    ap.add_argument("--hours", type=float, default=Config.hours, help="stop after this long as well as at --decisions")
    ap.add_argument("--eval-every", type=int, default=Config.eval_every, help="iterations between tests against the script")
    ap.add_argument("--latent-encoder", default=Config.latent_encoder, help="the trained frame encoder whose latents the actor reads")
    ap.add_argument("--kinds", help="action kinds to allow, such as 0,3: a test of one head with nothing else to fall back on")
    ap.add_argument("--gpu-runs", type=int, default=Config.gpu_runs, help="how many trainings share the card, this one included")
    ap.add_argument("--empty-cache", action=argparse.BooleanOptionalAction, default=Config.empty_cache, help="hand back what the allocator holds after each iteration")
    ap.add_argument("--check-repro", action="store_true")
    ap.add_argument("--check-kill", action="store_true")
    ap.add_argument("--save-forever", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    import policy_sim_env

    cfg = Config(seed=args.seed, decisions=args.decisions, games=args.games, iteration=args.iteration, passes=args.passes,
                 segment=args.segment, token_budget=args.token_budget, data_workers=args.workers, concurrent=args.concurrent,
                 decide_every=args.decide_every, hours=args.hours, eval_every=args.eval_every, gpu_runs=args.gpu_runs, empty_cache=args.empty_cache,
                 latent_encoder=args.latent_encoder, enabled_kinds=policy_sim_env.KINDS)     # what the sim obeys
    if args.kinds:
        cfg.enabled_kinds = tuple(int(k) for k in args.kinds.split(","))
    run = pathlib.Path(args.run)
    make_env = functools.partial(make_sim, decide_every=cfg.decide_every)
    if args.save_forever:
        _save_forever(cfg, run)
    elif args.check_kill:
        check_kill(cfg, run)
    elif args.check_repro:
        check_repro(cfg, run, make_env)
    else:
        train(cfg, run, make_env)


if __name__ == "__main__":
    main()
