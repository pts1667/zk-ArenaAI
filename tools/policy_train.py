"""The PPO trainer (docs/learned-policy/03-training.md, "PPO"), for the actor on latents (tools/policy_latent.py) in the sim.  In WSL:

  ~/zkml/bin/python tools/policy_train.py --run /mnt/d/zk-learned/runs/NAME --decisions 2000000
  ~/zkml/bin/python tools/policy_train.py --run ... --check-repro       # one iteration twice, compared
  ~/zkml/bin/python tools/policy_train.py --run ... --check-kill        # killed while saving, the checkpoint still loads

Rollouts keep raw frames, not tensors, and featurize per minibatch.  A segment is 32 decisions.  Each action gets its own clipped
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
HEADS = ("anchor", "kind", "region", "enemy", "fine", "produce", "place", "offset", "wreck", "group", "roles")
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
    # Each sample learnt from once: OpenAI Five found reuse of 2-3 can halve learning, and kept it near 1 (04-literature.md S5).
    passes: int = 1
    # Evaluation games are played by an exponential moving average of the weights, updated each optimizer step (0: the weights
    # themselves); it plays better than the weights it follows (Generals.io, 2606.23348).
    ema: float = 0.99
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
    diversity: float = 0.05         # a finished unit of a type the policy rarely makes is worth up to this (policy_sim_env._diversity;
                                    # 03-training.md, "Reward"): about a 300-metal unit's loss mid-game
    # PPO from a cloned actor (04-literature.md P7, C6, C7).
    init: str = ""                  # a cloning checkpoint (policy_clone_train's actor.pt): the actor starts as it, at its size
    maps: str = ""                  # a map pool (tools/map_gen.py), its held-out maps left out; "" for the five training maps
    variants: bool = False          # each side's army script draws its knobs (policy_variants.py)
    critic_bins: int = 0            # above 0, the critic's value is a distribution over this many, trained by HL-Gauss
    critic_span: float = 1.5        # from -span to span: the result is 1 at most, and the shaping a little on top
    entropy_floor: float = 0.3      # ENTROPY_FLOOR: a head below this share of its maximum entropy gets more bonus.  A clone's heads
                                    # all are, so from a clone it's 0, and the KL to the clone keeps its spread instead
    critic_warmup: int = 0          # iterations at the start where only the critic learns: the actor stays as it was
    kl_start: float = 0.0           # the weight of the KL to the frozen clone (`init`), at the start
    kl_end: float = 0.0             # and at the end, linear between: annealed but kept
    clone_weight: float = 0.0       # the cloning loss on what the scripts advise on its own frames (policy_sim_env.advise)
    dual_clip: float = 0.0          # dual-clip PPO's c (3 in Ye et al. 2020): a bad action's loss at most c x |advantage|; 0 off
    economy_start: float = 0.0      # the share of games where the policy runs its own economy, at the start
    economy_end: float = 0.0        # and at the end, linear between
    income_weight: float = 0.0      # a zero-sum income reward (policy_sim_env.reward), faded by beta as the rest of the shaping
    fine_sim: bool = False          # the sim steps every slot and the policy decides every decide_every of them, as the cloning
                                    # data was recorded: a clone played so had 6 extractors at 10 minutes, and 1 at decide_every 6
    z: str = "none"                 # "variant": the actor is told its side's army-script knobs as z (the counts unknown), as
                                    # policy_clone_eval --z variant; "none": z unknown


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
    """How much of the reward is the final result: 03-training.md, "Reward"."""
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
    """A decision as it was played.  Its actions, one entry each, in the order they were chosen (actions.Slots)."""
    frame: rf.Frame
    anchor: np.ndarray
    kind: np.ndarray
    enemy: np.ndarray
    region: np.ndarray
    fine: np.ndarray
    produce: np.ndarray
    group: np.ndarray
    roles: np.ndarray
    place: np.ndarray
    offset: np.ndarray
    logp: np.ndarray
    value: float = 0.0
    reward: float = 0.0
    advantage: float = 0.0
    ret: float = 0.0
    latents: np.ndarray | None = None   # tokens x latent: this frame's, as the actor read it
    label: dict | None = None           # what the scripts advised here, as slots (actions.SLOT_FIELDS), when they advise
    labelled: np.ndarray | None = None  # which controllable units the advice told something
    z: np.ndarray | None = None         # the side's strategy as the actor was told it (Config.z), the same all game


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
    return [fz.featurize(f, w["bundles"][key], w["table"], enabled_kinds=w["kinds"], privileged=True, economy=True) for f, key in chunk]


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
    def _forward(self, items: list[fz.Tensors], history: torch.Tensor, seen: torch.Tensor, z: torch.Tensor | None = None):
        b = pm.collate(items).to(self.device)
        with self.autocast:
            ctx, new_history, new_seen, keep = self.actor.decide(b, history, seen, z)
            a = pm.sample_actions(self.actor, b, ctx)
            new_history = self.actor.acted(b, a, new_history)       # what it did, for the decisions after
            value = self.critic(b) if self.critic is not None else torch.zeros(len(items), device=self.device)
        out = {k: getattr(a, k).cpu().numpy() for k in act.SLOT_FIELDS + ("logp",)}
        out.update({k: v.cpu().numpy() for k, v in keep.items()})
        return out, value.float().cpu().numpy(), new_history.float(), new_seen.float()

    def act(self, frames: list[tuple[rf.Frame, bd.Bundle, np.ndarray]], history: torch.Tensor, seen: torch.Tensor, z: torch.Tensor | None = None):
        """Forward passes for many games, as few as fit the memory.  Returns tensors, numpy actions, values and the new state.
        z: each seat's strategy (policy_latent.Z_DIM), or None."""
        items = self._features(frames)
        parts = [self._forward(items[s], history[s], seen[s], None if z is None else z[s]) for s in _act_slices(items)]
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
        size, state = policy_latent.SIZE, None
        if cfg.init:
            state = torch.load(cfg.init, map_location="cpu", weights_only=False)
            size = policy_latent.Size(**state["size"])
        self.actor = policy_latent.make(self.vocabulary, cfg.latent_encoder, cfg.decide_every, size).to(device)
        self.reference = None           # the clone as it was, frozen, for the KL to it
        if state is not None:
            missing, unexpected = self.actor.load_state_dict(state["actor"], strict=False)
            assert not unexpected and all(k.startswith("z_in.") for k in missing), (missing, unexpected)
            if cfg.kl_start or cfg.kl_end:
                self.reference = copy.deepcopy(self.actor).eval()
                for p in self.reference.parameters():
                    p.requires_grad_(False)
        self.critic = pm.Critic(self.vocabulary, bins=cfg.critic_bins, span=cfg.critic_span).to(device)
        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        critic_params = [p for p in self.critic.parameters() if p.requires_grad]
        self.trained = actor_params + critic_params
        # Two groups, so that the critic can learn alone at first (critic_warmup).
        self.optimizer = torch.optim.Adam([{"params": actor_params}, {"params": critic_params}], lr=cfg.lr_start, eps=cfg.adam_eps)
        self.economy_share = cfg.economy_start
        self.decisions = 0
        self.iteration = 0
        self.entropy_coef = {h: ENTROPY_BASE for h in HEADS}
        self.self_play = SelfPlay(cfg.self_play_at, cfg.self_play_holds_below, cfg.self_play_per_iteration)
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
        self.player = Player(self.actor, self.critic, workers, device)
        self.ema = self.frozen_player(workers) if cfg.ema > 0.0 else None
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

    @torch.no_grad()
    def ema_step(self) -> None:
        """The moving average a step nearer the weights, in what's trained."""
        if self.ema is None:
            return
        pairs = [(e, w) for mine, theirs in ((self.actor, self.ema.actor), (self.critic, self.ema.critic))
                 for e, w in zip(theirs.parameters(), mine.parameters()) if w.requires_grad]
        torch._foreach_lerp_([e for e, _ in pairs], [w for _, w in pairs], 1.0 - self.cfg.ema)

    @property
    def evaluator(self) -> Player:
        return self.ema if self.ema is not None else self.player

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
        latents = torch.as_tensor(latents, device=device)
        # A segment is one game's, so its first step's z is all of its steps'.
        z = None if steps[0].z is None else torch.as_tensor(np.stack([seg.steps[0].z for seg in batch]), device=device)
        cfg = self.cfg
        with self.autocast:
            a = _stored_actions(steps, device)
            acted = self.actor.segment_actions(whole, a, lengths)
            now = self.actor.segments(history, seen, latents, lengths, acted, z)
            ctx = self.actor.context(whole, now)
            logp, entropy = pm.evaluate_actions(self.actor, whole, ctx, a)
            value = self.critic(whole, logits=True)
            extra = torch.zeros((), device=device)
            if self.reference is not None:
                # KL(policy || clone) on the actions taken, as k3 = r - 1 - log r with r = clone / policy: never below 0.
                with torch.no_grad():
                    ref = self.reference
                    ref_now = ref.segments(history, seen, latents, lengths, ref.segment_actions(whole, a, lengths), z)
                    ref_logp, _ = pm.evaluate_actions(ref, whole, ref.context(whole, ref_now), a)
                w = pm.chosen(a).float()
                log_r = (ref_logp.float() - logp).clamp(-20.0, 20.0)
                kl = ((log_r.exp() - 1.0 - log_r) * w).sum() / w.sum().clamp_min(1.0)
                coef = cfg.kl_start + (cfg.kl_end - cfg.kl_start) * self.progress
                extra = extra + coef * kl
                note("kl_clone", float(kl.detach()))
            if cfg.clone_weight and any(s.label is not None for s in steps):
                lab, labelled, has = _label_actions(steps, whole.ctrl_mask.shape[1], device)
                lab_logp, _ = pm.evaluate_actions(self.actor, whole, ctx, lab, labelled)
                w = pm.chosen(lab).float() * has[:, None]
                clone = -(w * lab_logp).sum() / w.sum().clamp_min(1.0)
                extra = extra + cfg.clone_weight * clone
                note("loss_clone", float(clone.detach()))
        return self._ppo_loss(steps, a, logp, entropy, value, clip, note) + extra

    def _ppo_loss(self, steps: list[Step], a: pm.Actions, logp: torch.Tensor, entropy: dict, value: torch.Tensor, clip: float, note) -> torch.Tensor:
        """Each action's own clipped ratio, averaged over the decision's actions, done included; one advantage a decision."""
        cfg, device = self.cfg, self.device
        decisions = len(steps)
        advantages = np.array([s.advantage for s in steps], np.float32)
        adv = torch.as_tensor((advantages - advantages.mean()) / (float(advantages.std()) + 1e-6), device=device)
        ret = torch.as_tensor([s.ret for s in steps], device=device, dtype=torch.float32)
        old = a.logp
        weights = pm.chosen(a).float()
        ratio = (logp - old).clamp(-20.0, 20.0).exp()
        per_action = -torch.min(ratio * adv[:, None], ratio.clamp(1.0 - clip, 1.0 + clip) * adv[:, None])
        if cfg.dual_clip:
            # Dual-clip: where the advantage is negative, a ratio far above 1 costs at most dual_clip x |advantage|.
            per_action = torch.where(adv[:, None] < 0, torch.min(per_action, -cfg.dual_clip * adv[:, None]), per_action)
        count = weights.sum(-1)
        has = count > 0
        pg = torch.zeros((), device=device)
        if has.any():
            pg = ((per_action * weights).sum(-1)[has] / count[has]).sum() / decisions
            with torch.no_grad():
                note("kl", float(((old - logp) * weights).sum() / weights.sum()), float(weights.sum()))
                note("clip_fraction", float((((ratio - 1.0).abs() > clip).float() * weights).sum() / weights.sum()), float(weights.sum()))
        if self.critic.bins:
            value_loss = self.critic.hl_gauss(value, ret).sum() / decisions
        else:
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
        if self.iteration < cfg.critic_warmup:
            self.optimizer.param_groups[0]["lr"] = 0.0      # the actor's
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
                self.ema_step()
                note("grad_norm", float(grad))
                if between is not None:
                    between()
        return {k: totals[k] / max(counts[k], 1e-9) for k in totals}

    def adjust_entropy(self, metrics: dict[str, float]) -> None:
        """Keeps each head above a share of its maximum, so none collapses."""
        maxima = {"anchor": math.log(4.0), "kind": math.log(4.0), "region": math.log(4.0), "enemy": math.log(8.0), "fine": math.log(fz.FINE_OPTIONS), "produce": math.log(4.0),
                  "place": math.log(8.0), "offset": math.log(fz.PLACE_OFFSETS), "wreck": math.log(4.0), "group": math.log(4.0), "roles": math.log(2.0)}
        for h in HEADS:
            e = metrics.get(f"entropy_{h}")
            if e is None:
                continue
            if e < self.cfg.entropy_floor * maxima[h]:
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
        ema = {"ema_actor": self.ema.actor.state_dict(), "ema_critic": self.ema.critic.state_dict()} if self.ema is not None else {}
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "optimizer": self.optimizer.state_dict(), **ema,
                    "size": asdict(self.actor.size), "critic_size": self.critic.size,
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
        if self.ema is not None:
            self.ema.actor.load_state_dict(state.get("ema_actor", state["actor"]))
            self.ema.critic.load_state_dict(state.get("ema_critic", state["critic"]))
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


def advice_label(env, seat: int, t: fz.Tensors, bundle: bd.Bundle, rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    """What the scripts advised the seat at the step just taken (policy_sim_env.advise), as slots in a random order and which
    units it told something: as policy_clone_train labels a recorded decision."""
    import policy_sim_env as se
    (army, ao), (eco, eo) = env.army_orders, env.eco_orders
    orders = np.concatenate((army[ao[seat]:ao[seat + 1]], eco[eo[seat]:eo[seat + 1]]))
    x, z = act.world_positions(t, bundle)
    decision, _ = act.encode(se.policy_orders(orders), t, bundle, x, z)
    slots = act.to_slots(decision, rng.permutation(t.controllable))
    return {f: getattr(slots, f) for f in act.SLOT_FIELDS}, decision.kind != rf.KEEP


def _label_actions(steps: list[Step], units: int, device: str) -> tuple[pm.Actions, torch.Tensor, torch.Tensor]:
    """The steps' advice as actions, which units it told something (B x units), and which steps have any (B)."""
    has = np.array([s.label is not None for s in steps])
    width = max([len(s.label["anchor"]) for s in steps if s.label is not None] + [1])
    fields = {}
    for f in act.SLOT_FIELDS:
        out = np.full((len(steps), width), act.UNSET[f], np.int64)
        for i, s in enumerate(steps):
            if s.label is not None:
                out[i, :len(s.label[f])] = s.label[f]
        fields[f] = out
    labelled = np.zeros((len(steps), units), bool)
    for i, s in enumerate(steps):
        if s.labelled is not None:
            labelled[i, :len(s.labelled)] = s.labelled
    return pm.actions_from(fields, device), torch.as_tensor(labelled, device=device), torch.as_tensor(has, device=device)


def seat_z(env, seats: list[int], how: str) -> np.ndarray | None:
    """seats x Z_DIM, what each seat's actor is told of its strategy (Config.z), or None for nothing."""
    if how == "none" or not seats:
        return None
    import policy_clone_data as cd
    return np.stack([cd.variant_z(None if env.styles[s // 2] is None else env.styles[s // 2][s % 2]) for s in seats])


def _z_tensor(z: np.ndarray | None, device: str) -> torch.Tensor | None:
    return None if z is None else torch.as_tensor(z, device=device)


def _stored_actions(steps: list[Step], device: str) -> pm.Actions:
    """The actions the decisions were played with, and their log-probabilities then."""
    return pm.actions_from({f: np.stack([getattr(s, f) for s in steps]) for f in act.SLOT_FIELDS + ("logp",)}, device)


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
        env.diversity = learner.cfg.diversity
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
        masked_out = actions = 0
        bonus = 0.0
        kind_counts = np.zeros(rf.KIND_COUNT, np.int64)
        progress = learner.progress
        while stored < want:
            # With act_every (policy_sim_env), the sim steps between the policy's decisions: a decision's reward is all of them.
            acting = env.policy_seats_to_act()
            if acting:
                for seat in acting:
                    # It sat out as a script, and meanwhile its game may have moved to another map.
                    if not current[seat].steps and current[seat].bundle is not env.bundle_of(seat):
                        self._open(seat)
                frames = [(env.frame(seat), env.bundle_of(seat), current[seat].names) for seat in acting]
                index = torch.as_tensor(acting, device=learner.device)
                told = seat_z(env, acting, cfg.z)
                items, out, value, new_history, new_seen = self.player.act(frames, seats.history[index], seats.seen[index], _z_tensor(told, learner.device))
                for k, seat in enumerate(acting):
                    t = items[k]
                    raw = frames[k][0]
                    step = Step(rf.Frame(raw.units.copy(), raw.sites.copy(), raw.glob.copy(), raw.features.copy()),
                                *(out[h][k].copy() for h in act.SLOT_FIELDS + ("logp",)), float(value[k]))
                    step.z = None if told is None else told[k]
                    picked = step.anchor >= 0
                    masked_out += int((~t.kind_mask[step.anchor[picked], step.kind[picked]]).sum())
                    kind_counts += np.bincount(step.kind[picked], minlength=rf.KIND_COUNT)
                    actions += int(picked.sum())
                    if "latents" in out:
                        step.latents = out["latents"][k].copy()
                    tr = current[seat]
                    tr.steps.append(step)
                    stored += 1
                    x, z = act.world_positions(t, tr.bundle)
                    env.give_orders(seat, act.decode(act.to_decision(act.Slots(*(getattr(step, h) for h in act.SLOT_FIELDS)), t.controllable), t, tr.bundle, x, z))
                    # A segment ends here: the next one starts from what the actor keeps now.
                    if len(tr.steps) - tr.segment_start == cfg.segment:
                        _cut(tr, segments, new_history[k], new_seen[k])
                seats.history[index] = new_history
                seats.seen[index] = new_seen
            env.step(progress)
            if getattr(env, "advise", False):
                for k, seat in enumerate(acting):
                    step = current[seat].steps[-1]
                    step.label, step.labelled = advice_label(env, seat, items[k], current[seat].bundle, self.rng)
            for seat, tr in current.items():
                if tr.steps:
                    tr.steps[-1].reward += env.reward(seat, beta(progress))
            for seat in acting:
                bonus += float(env.bonus[seat // 2, seat % 2])
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
            _, _, value, _, _ = self.player.act(frames, seats.history[index], seats.seen[index], _z_tensor(seat_z(env, acting, cfg.z), learner.device))
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
        metrics = {"explained_variance": explained, "masked_out": masked_out, "decisions": stored, "actions_a_decision": actions / max(stored, 1),
                   "diversity_a_decision": bonus / max(stored, 1), "return_min": float(r.min()), "return_max": float(r.max())}
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
    """The current policy (the moving average of its weights, `Config.ema`) against a script, sampled as in training.  Each game
    slot counts its first game only: the games that end soonest would otherwise count most, and those are not a fair sample."""
    player = learner.evaluator
    env = make_env(games, seed, modes=[opponent])
    env.policy_economy_share = learner.economy_share
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
            told = _z_tensor(seat_z(env, acting, learner.cfg.z), learner.device)
            items, out, _, new_history, new_seen = player.act(frames, seats.history[index], seats.seen[index], told)
            for k, seat in enumerate(acting):
                t = items[k]
                d = act.to_decision(act.Slots(*(out[h][k] for h in act.SLOT_FIELDS)), t.controllable)
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


_MAP_POOLS: dict = {}


def make_sim(games: int, seed: int, modes: list[str] | None = None, decide_every: int = 1, maps: str = "", variants: bool = False,
             advise: bool = False, income_weight: float = 0.0, fine_sim: bool = False):
    """maps: a map pool, loaded once and shared, its held-out maps left out; "" for the five training maps."""
    import policy_sim_env

    pool = weights = None
    if maps:
        if maps not in _MAP_POOLS:
            from frame_world import held_out_maps, pool_weights
            loaded = policy_sim_env.load_pool(maps)
            w = pool_weights(loaded)
            w[held_out_maps(loaded)] = 0.0
            policy_sim_env.share_maps(loaded)
            _MAP_POOLS[maps] = (loaded, w / w.sum())
        pool, weights = _MAP_POOLS[maps]
    env = policy_sim_env.SimEnv(games, seed, modes or ["fight"], decide_every=1 if fine_sim else decide_every, maps=pool, weights=weights,
                                variants=variants)
    env.advise, env.income_weight = advise, income_weight
    env.act_every = decide_every if fine_sim else 1
    return env


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
        learner.economy_share = cfg.economy_start + (cfg.economy_end - cfg.economy_start) * learner.progress
        env.policy_economy_share = learner.economy_share
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
    """The same iteration twice from the same seed: 03-training.md, "Checkpoints and stopping"."""
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
    ap.add_argument("--ema", type=float, default=Config.ema, help="decay of the weights' moving average that plays the tests; 0: the weights")
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
    ap.add_argument("--init", default="", help="a cloning checkpoint to start the actor from (P7)")
    ap.add_argument("--maps", default="", help="a map pool, its held-out maps left out; without it, the five training maps")
    ap.add_argument("--variants", action="store_true", help="each side's army script draws its knobs (policy_variants.py)")
    ap.add_argument("--lr-start", type=float, default=Config.lr_start)
    ap.add_argument("--lr-end", type=float, default=Config.lr_end)
    ap.add_argument("--critic-bins", type=int, default=0, help="HL-Gauss bins for the critic (C6); 0 for one number")
    ap.add_argument("--critic-span", type=float, default=Config.critic_span)
    ap.add_argument("--entropy-floor", type=float, default=Config.entropy_floor, help="0 from a clone: no head's bonus rises")
    ap.add_argument("--critic-warmup", type=int, default=0, help="iterations where only the critic learns")
    ap.add_argument("--kl-start", type=float, default=0.0, help="the KL to the frozen clone's weight at the start")
    ap.add_argument("--kl-end", type=float, default=0.0, help="and at the end")
    ap.add_argument("--clone-weight", type=float, default=0.0, help="the cloning loss on the scripts' advice (they then advise)")
    ap.add_argument("--dual-clip", type=float, default=0.0, help="dual-clip PPO's c; 0 off")
    ap.add_argument("--economy-start", type=float, default=0.0, help="share of games where the policy runs its own economy, at the start")
    ap.add_argument("--economy-end", type=float, default=0.0, help="and at the end")
    ap.add_argument("--income-weight", type=float, default=0.0, help="the zero-sum income reward's weight")
    ap.add_argument("--diversity", type=float, default=Config.diversity)
    ap.add_argument("--fine-sim", action="store_true", help="the sim steps every slot, the policy decides every --decide-every")
    ap.add_argument("--z", choices=("none", "variant"), default=Config.z,
                    help="what the actor is told of its strategy: nothing, or its side's army-script knobs (needs --variants)")
    ap.add_argument("--check-repro", action="store_true")
    ap.add_argument("--check-kill", action="store_true")
    ap.add_argument("--save-forever", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.z == "variant" and not args.variants:
        ap.error("--z variant needs --variants: without them there are no knobs to tell")
    import policy_sim_env

    cfg = Config(seed=args.seed, decisions=args.decisions, games=args.games, iteration=args.iteration, passes=args.passes, ema=args.ema,
                 segment=args.segment, token_budget=args.token_budget, data_workers=args.workers, concurrent=args.concurrent,
                 decide_every=args.decide_every, hours=args.hours, eval_every=args.eval_every, gpu_runs=args.gpu_runs, empty_cache=args.empty_cache,
                 latent_encoder=args.latent_encoder, enabled_kinds=policy_sim_env.KINDS,     # what the sim obeys
                 init=args.init, maps=args.maps, variants=args.variants, lr_start=args.lr_start, lr_end=args.lr_end,
                 critic_bins=args.critic_bins, critic_span=args.critic_span, entropy_floor=args.entropy_floor, critic_warmup=args.critic_warmup, kl_start=args.kl_start, kl_end=args.kl_end,
                 clone_weight=args.clone_weight, dual_clip=args.dual_clip, economy_start=args.economy_start, economy_end=args.economy_end,
                 income_weight=args.income_weight, diversity=args.diversity, fine_sim=args.fine_sim, z=args.z)
    if args.kinds:
        cfg.enabled_kinds = tuple(int(k) for k in args.kinds.split(","))
    run = pathlib.Path(args.run)
    make_env = functools.partial(make_sim, decide_every=cfg.decide_every, maps=cfg.maps, variants=cfg.variants, advise=cfg.clone_weight > 0,
                                 income_weight=cfg.income_weight, fine_sim=cfg.fine_sim)
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
