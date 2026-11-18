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
import collections
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
sys.path.insert(0, str(ROOT / "sim" / "py"))
import policy_graphs  # noqa: E402
import policy_latent  # noqa: E402
import policy_model as pm  # noqa: E402
import zkpolicy  # noqa: E402  (the library itself is loaded when first used)
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
    # Decisions x (padded tokens + DECISION_TOKENS) in a minibatch, which is what its memory follows: at 4.1 GB a 100,000 at
    # worst, 100,000 fits the learner's 4.6 beside 4 collectors (full-2, 2026-11-12), where it settled at 85,000-95,000 from
    # 130,000, which split 12 minibatches in the first iteration.  A step took about 0.5 s whether of 32 decisions or 384, so the
    # bigger the faster.  The most it may be: SPLIT_SHRINK and BUDGET_GROW keep it to what fits.
    token_budget: int = 100_000
    gpu_share: float = 0.85         # of the card all runs on it may hold.  Past the card it spills into shared memory, at a fifth of the speed.
    gpu_runs: int = 1               # trainings sharing the card, each with its part of gpu_share.  The limit is per process, so it must be told.
    empty_cache: bool = True        # hand back what the allocator holds after each iteration
    lr_start: float = 3e-4
    lr_end: float = 1e-4
    # The actor's rate follows the iteration's KL as well as the schedule (the user, 2026-11-12): KL_SLOW slower after an
    # iteration past twice this, back toward the schedule (never past it) after one under half of it; 0: the schedule alone.
    # In full-2, as the KL to the clone faded from 0.48 to 0.094, PPO's KL an iteration rose from 0.009 to 0.052.
    kl_target: float = 0.01
    # The share of its schedule the actor's rate starts at, for that guard to raise or lower: under 1 where the schedule
    # starts above what a new actor stands (rdnn-1: at the whole of 2e-3 the KL was 1.2 to 10.6 an iteration).
    lr_scale_start: float = 1.0
    # The rate of an RDNN's recurrent weights, in their own Adam (Learner.recurrent), at the schedule's start, falling as it
    # does and whatever the KL guard leaves the actor of it; 0: the schedule's own.  The paper's best was 5e-3.
    recurrent_lr: float = 0.0
    # An RDNN's steps through a segment, in training, through torch.compile (policy_latent.compile_memory): a sixth of the
    # loop's time at segments of 60, for about 20 s of compiling when the learner starts.  Not the collectors', which step
    # the memory once a decision inside their graphs.
    compile_memory: bool = False
    # The critic's rate at the start, falling as the actor's does; 0: the actor's.  A clone's actor wants a small rate, and a new
    # critic at it learns nothing: at 1e-5 its explained variance stayed at 0.0000 for 19 iterations (full-1, 2026-11-10).
    critic_lr: float = 0.0
    # Above 0, Muon trains the critic's hidden matrices at this rate, falling as the critic's other rate does, and Adam the rest of
    # it (the user, 2026-11-12).  The critic learns from nothing here, as the frame encoder did, where Muon beat AdamW (S4).
    critic_muon_lr: float = 0.0
    # The same for the actor's hidden matrices (policy_latent.answers keeps its heads' layers on Adam), as cloning and DAgger trained
    # them, falling as its Adam rate does and 0 while the critic warms up; 0: Adam for all of it (2026-11-12, against Adam in full-2).
    actor_muon_lr: float = 0.0
    # SPO's eps, the ratio's trust region 1 +- eps (Learner._update_loss), at the start and at the end, linear between.  PPO's
    # clip had these until 2026-11-10.
    epsilon_start: float = 0.15
    epsilon_end: float = 0.1
    gae: float = 0.95
    value_weight: float = 0.5
    grad_clip: float = 0.5
    adam_eps: float = 1e-5
    checkpoint_every: int = 20
    eval_every: int = 20
    # Every test plays the same games, from the seed cfg.seed x 1000, a run's first test (iteration 0) among them, so that tests pair
    # game for game (`score_test`); 128 of them, and scored on more than the result (game_score; the user, 2026-11-14).
    eval_games: int = 128
    enabled_kinds: tuple[int, ...] = fz.STAGE2_KINDS
    data_workers: int = 6
    # Featurizing in C++ (sim/py/zkpolicy.py: libzkpolicy, bit for bit with featurize's Python, tools/policy_parity.py), on
    # this many threads in each collector, which featurizes a round's frames in its own process, and on one in each data
    # worker; 0: featurize's Python (numba), a frame at a time.
    featurize_threads: int = 0
    concurrent: bool = True         # collect the next iteration during this one's training
    decide_every: int = 1           # in 15-frame slots: 1 is 2 Hz, 2 is 1 Hz.  Scripted opponents keep their own rate.
    hours: float = 0.0              # stop after this long as well as at `decisions`, if not 0
    # The share of training games the policy plays against itself, every player's team of them (policy_sim_env's mode "mirror"),
    # each side's decisions learnt from; the rest it plays against the scripts, as it does every test.
    self_play: float = 0.0
    latent_encoder: str = policy_latent.ENCODER     # the trained frame encoder whose latents the actor reads
    diversity: float = 0.05         # a finished unit of a type the policy rarely makes is worth up to this (policy_sim_env._diversity;
                                    # 03-training.md, "Reward"): about a 300-metal unit's loss mid-game
    # PPO from a cloned actor (04-literature.md P7, C6, C7).
    init: str = ""                  # a cloning checkpoint (policy_clone_train's actor.pt): the actor starts as it, at its size
    size: str = "default"           # without one, a new actor's (policy_latent.SIZES): "gru" has a GRU for memory, "rdnn" an RDNN in its place
    maps: str = ""                  # a map pool (tools/map_gen.py), its held-out maps left out; "" for the five training maps
    variants: bool = False          # each side's army script draws its knobs (policy_variants.py)
    critic_bins: int = 0            # above 0, the critic's value is a distribution over this many, trained by HL-Gauss
    critic_span: float = 1.5        # from -span to span: the result is 1 at most, and the shaping a little on top
    # The reward's terms (policy_sim_env.TERMS), each with a head of the critic and the weight of its advantage (`weigh`;
    # 03-training.md, "Reward"): ("blend", 1.0) alone is the one reward, the shaping fading into the result by beta.
    terms: tuple[tuple[str, float], ...] = (("blend", 1.0),)
    critic_rewarm: int = 10         # after a resume that changed the terms, iterations where only the critic learns: its heads start afresh
    entropy_floor: float = 0.3      # ENTROPY_FLOOR: a head below this share of its maximum entropy gets more bonus.  A clone's heads
                                    # all are, so from a clone it's 0, and the KL to the clone keeps its spread instead
    critic_warmup: int = 0          # iterations at the start where only the critic learns: the actor stays as it was
    critic_size: str = "default"    # policy_model.CRITIC_SIZES: "small" beside a small actor
    kl_start: float = 0.0           # the weight of the KL to the frozen clone (`init`), at iteration 0
    # Its share of that by iteration, linear between and the last after: from 100% to 20% by iteration 100 and gone by 500.  Linear
    # in the run's decisions to a kept 0.2 of it, as before (2026-11-10), it was still 0.48 at iteration 88 of 1.85M decisions:
    # weighed, about 0.14 to the policy gradient's 0.003 (the user).
    kl_fade: tuple[tuple[int, float], ...] = ((0, 1.0), (100, 0.2), (500, 0.0))
    clone_weight: float = 0.0       # the cloning loss on what the scripts advise on its own frames (policy_sim_env.advise), at iteration 0
    # Its share of that by iteration, as the KL's (the user, 2026-11-10).  Kept whole, it was 0.5 x about 2.4 of `full-1`'s loss.
    clone_fade: tuple[tuple[int, float], ...] = ((0, 1.0), (100, 0.2), (500, 0.0))
    economy_start: float = 0.0      # the share of games where the policy runs its own economy, at the start
    economy_end: float = 0.0        # and at the end, linear between
    income_weight: float = 0.0      # a zero-sum income reward (policy_sim_env.reward), faded by beta as the rest of the shaping
    # At a decided game's end, this x the result x the share of the time limit left (policy_sim_env._time_left), not faded by beta: a
    # win sooner is worth a little more (the user, 2026-11-10).
    speed_weight: float = 0.0
    # The share of training games, by iteration, that a lead can end (policy_sim_env.lead_share), linear between and the last
    # after: 90% at first, none from iteration 100, so the policy learns to finish games (the user, 2026-11-10).  Tests keep it.
    lead_fade: tuple[tuple[int, float], ...] = ((0, 0.9), (100, 0.0))
    # The kinds of game played and how likely each is (policy_sim_env.GAME_KINDS), and the slots a game: 0 for what the biggest of
    # them needs, 2 for 1v1s alone.  The headline test stays a 1v1 against the script, so that runs stay comparable; `test_kinds`
    # adds a test of each of those kinds, the policy on team 0 and the scripts on every other (the user, 2026-11-10).
    kinds: tuple[tuple[str, float], ...] = (("1v1", 1.0),)
    # The share of training games that aren't 1v1s, by iteration, linear between and the last after, as kl_fade: split among
    # `kinds`' other kinds as they're weighed there, the rest 1v1s (policy_sim_env.other_kinds_share).  (): `kinds`' own weights
    # throughout.  The pipeline's: none till iteration 500, half from 2,500 (the user, 2026-11-16).
    kinds_ramp: tuple[tuple[int, float], ...] = ()
    slots: int = 0
    test_kinds: tuple[str, ...] = ()
    # How likely each chicken difficulty is, by name (zksim.CHICKEN_DIFFICULTIES): () for policy_sim_env.CHICKEN_DIFFICULTY_WEIGHTS.
    chicken_difficulties: tuple[tuple[str, float], ...] = ()
    fine_sim: bool = False          # the sim steps every slot and the policy decides every decide_every of them, as the cloning
                                    # data was recorded: a clone played so had 6 extractors at 10 minutes, and 1 at decide_every 6
    z: str = "none"                 # "variant": the actor is told its side's army-script knobs as z (the counts unknown), as
                                    # policy_clone_eval --z variant; "none": z unknown; "objective": its own weights of the
                                    # reward's terms this game (below)
    # A seat's own objective (the user, 2026-11-17, should the policy stop moving; z "objective", GRPO): in each new game each
    # seat's weights of the terms are the run's, each times exp(objective_sigma x a standard normal draw), and with
    # probability explore_share one of the exploratory terms among `terms` (policy_sim_env.MADE_TERMS, there at weight 0) is
    # weighed explore_weight for it (`draw_objective`).  Its actor is told the draws as z; told nothing, as in a test, it
    # plays the run's own weights.  Untold, the draws would average out to the run's weights and noise.
    objective_sigma: float = 0.0
    explore_share: float = 0.0
    explore_weight: float = 0.5
    replay_hours: float = 0.0       # every this many hours, replay_games games against the script, kept for the dashboard's viewer
    replay_games: int = 4           # (policy_replay.py) in `replays`, or the run's replays/; 0 hours for none
    replays: str = ""
    graphs: bool = True             # play each forward pass as a CUDA graph (policy_graphs.py); not while collecting concurrently in
                                    # this process, where a second thread would use the GPU during a capture
    collectors: int = 0             # processes that play the games, each its share of them, and the tests (policy_collectors.py);
                                    # 0: this process plays
    collector_gb: float = 1.0       # the GPU memory each collector may hold, out of gpu_share.  The learner's training needs about
                                    # 4.5 GB of the rest (at 4 x 1.5 it had 2.6 GB, and ran out); a collector of 16 games ran out at
                                    # 0.5 (2026-11-10), and at 0.75 in full-2's first iterations of mixed games, as its frames grew
                                    # (2026-11-12).  At the caps, 16 games' two kept shapes (policy_graphs) reserve 0.83 GB.
    pass_seats: int = 1             # how many seats a game a collector's pass takes: its rows are its games times this.  Against
                                    # itself the policy has 2 seats a 1v1, so at 1 a round is two passes; at 2 it's one, of twice
                                    # the rows and so about twice the memory (the actor with a GRU has room for it)
    # GRPO from branched games in place of the critic (03-training.md, "GRPO"; the user, 2026-11-13): above 0, the games go in
    # groups of this many, each group's games copies of one game played on apart, for a stretch of game time drawn evenly from
    # `horizon` (GroupCollector).  Then each seat's stretch is scored by its terms (`terms`) against the same seat's in the
    # group's other games, and one game still going is copied into the rest for the next.  0: PPO, with its critic.
    group: int = 0
    horizon: tuple[float, float] = (60.0, 180.0)    # seconds
    # A branch's advantages (GroupCollector._judge, Learner.weigh_groups; the user, 2026-11-13, after the trial's swing to MOVE).
    # group_scale "terms": each term less its group's mean, over that term's mean spread in the iteration's groups where it
    # varied, then weighed (Multi-GRPO's scaling by reward, at the iteration's level so that a term that seldom varies isn't
    # blown up); "group": the weighed score less its group's mean, over its group's spread (GRPO's own); "none": neither (Dr.
    # GRPO's).  reward_to_go: each decision judged on its branch's terms from it on, against its group's from the same moment
    # (DeepSeekMath's process supervision), not on the whole stretch's.  group_step: decisions an optimizer step, in whole
    # groups, their minibatches' gradients added up, so that what a group's branches all did alike cancels out; 0 for a step a
    # minibatch.  1,024 is about 3 groups of 8, and 16 steps an iteration of 16,384, as DAPO takes 16 a rollout batch: the
    # GRPO trial took about 35 an iteration of 4,768, at 0.001-0.01 of KL an iteration.  Under GRPO the advantages aren't
    # normalised again over each minibatch, and a decision's actions are summed.
    group_scale: str = "terms"
    reward_to_go: bool = True
    group_step: int = 1024
    # With reward_to_go, the economy term judges a decision on the income balance's change from it on, (1/span) x the sum over
    # the steps after it of (income - income when it was taken), not on the level, which a branch ahead when the decision was
    # taken would otherwise credit it with (the user, 2026-11-14, after a review of arXiv 2508.08221 here).  And so every
    # term that is a level's average over time (policy_sim_env.LEVEL_TERMS: the leads too).
    economy_change: bool = True
    # A decision's weighed advantage is clipped to +- adv_clip (0: not), and SPO's pull back on a ratio grows no more past
    # spo_bound x eps from 1, as Huber's loss (0: SPO's square throughout), so that no one action moves the policy far (the
    # same).  Under GRPO only.  full-3's GRPO had a loss_pg of 2.585 at iteration 130, against about 0.06.
    adv_clip: float = 3.0
    spo_bound: float = 2.0
    # The least entropy bonus a head has (ENTROPY_BASE's); below 0, ENTROPY_BASE under PPO and none under GRPO, where it
    # outweighed the advantages: full-3's offset head went from 1.84 to 3.06 of its most, ln 25 = 3.22 (the same).
    entropy_base: float = -1.0
    # Bootstrapped GRPO (the user, 2026-11-14): a decision's return is its branch's terms from it to its stretch's end,
    # discounted as the critic's are (policy_sim_env.term_gamma), plus the critic's value of where the branch ended (nothing if
    # its game or its alliance did), and its advantages that less its group's mean from the same moment (Learner.bootstrap).
    # So a decision is judged past its stretch, which only the shaping terms reached before, and what the critic gets wrong
    # alike for a group's ends cancels.  The critic learns those returns (n-step ones), in the learner only: it plays nothing,
    # and the actor waits critic_warmup iterations for it.
    bootstrap: bool = False
    # Resignation, as AlphaGo Zero's (the user, 2026-11-14), with bootstrap: at a stretch's end the game a group goes on from is
    # given up, and a new one started, when the critic's result for the policy's side is under the threshold (Learner.resign_at).
    # resign_free of games are never given up but played out, each saying the least value it reached and whether it was then
    # lost; the threshold is the highest of RESIGN_THRESHOLDS under which at least resign_least played-out games fell and at
    # most resign_false of them weren't lost.  None till then, and nothing measured while the critic warms up.
    resign: bool = False
    resign_free: float = 0.1
    resign_false: float = 0.05
    resign_least: int = 20


MUON_DECAY = 0.01           # the critic's Muon's weight decay (--critic-muon-lr), as cloning's
RESIGN_THRESHOLDS = (-0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.6, -0.5)    # Config.resign's candidates, of the critic's result
RESIGN_WINDOW = 500         # the played-out games it's chosen from, the latest
HOPELESS = (-0.8, -0.9)     # the critic's result under which a branch's end is logged as hopeless (Learner.bootstrap)
# What a decision costs in training's memory beside its tokens, in tokens.  On full-2's first segments (2026-11-12), a decision
# of 165-237 tokens took 14.1-16.7 MiB whether 32 or 384 were trained together, the critic about 14 KiB a token of it; but in
# its PPO, 241 decisions of about 280 tokens fit where 151 of about 590 and 103 of about 860 didn't, so the rest grows with
# the tokens too (the actor's reading of the units).  Fitted on the 2,042 minibatches of full-2's 50 iterations with Adam
# (minibatches.jsonl), memory was 0.52 GB + 9.3 MiB a decision + 30.9 KiB a token, give or take 0.36 GB; the worst (95th
# percentile) per unit of cost was level between small frames and big at 200: 4.0 and 4.1 GB a 100,000.
DECISION_TOKENS = 200
# The token budget follows the learner's memory: a tenth less after an iteration where a minibatch didn't fit (once, as the
# iteration's minibatches were all planned before it), and a twentieth more, up to --token-budget, after one whose training
# peaked under GROW_BELOW of the learner's limit, in what its tensors held: what the allocator reserves stays near the limit.
SPLIT_SHRINK = 0.9
KL_SLOW = 1.5               # Config.kl_target's step
RECURRENT_EPS = 1e-8        # Adam's eps for an RDNN's recurrent weights (Learner.recurrent): their gradient is far under the policy's
BUDGET_GROW = 1.05
GROW_BELOW = 0.85
ACT_PAIRS = 24_000_000      # batch x tokens^2 of one forward pass while playing: its pair table is about 20 bytes a pair
FLAT_CHUNK = 64             # decisions a data worker featurizes at a time
# Records past the caps become no tokens.
UNIT_TOKENS = fz.CAP_UNITS + fz.CAP_HIDDEN
# A process's GPU memory that its allocator doesn't count, such as CUDA's own.
PROCESS_OVERHEAD = 400 * 2**20


def gpu_fraction(share: float, runs: int, card_bytes: int, others: float = 0.0) -> float:
    """One run's limit, as a fraction of the card, when `runs` trainings share `share` of it.  others: bytes the run's own collector
    processes hold, their overhead included (`collector_bytes`)."""
    fraction = (share - (runs - 1) * PROCESS_OVERHEAD / card_bytes) / runs - others / card_bytes
    if fraction <= 0.0:
        raise ValueError(f"{runs} runs don't fit in {share} of a {card_bytes / 2**30:.1f} GB card")
    return fraction


def collector_bytes(cfg: "Config") -> float:
    """What a run's collector processes hold of the card in all."""
    return cfg.collectors * (cfg.collector_gb * 2**30 + PROCESS_OVERHEAD)


def padded_tokens(batch: list["Segment"]) -> int:
    """A minibatch's padded tokens a decision, as `Learner._minibatches` counts them (DECISION_TOKENS aside)."""
    units = max(min(len(s.frame.units), UNIT_TOKENS) + min(len(s.frame.sites), fz.CAP_SITES) + min(len(s.frame.features), fz.CAP_FEATURES)
                for seg in batch for s in seg.steps)
    return units + max(seg.bundle.regions for seg in batch) + 2


def spo_objective(ratio: torch.Tensor, adv: torch.Tensor, eps: float, bound: float = 0.0) -> torch.Tensor:
    """Simple Policy Optimization's, to maximise for each sample: r A - |A| (r - 1)^2 / (2 eps).  With a bound above 0, the
    penalty goes on in a straight line past |r - 1| = bound x eps, as Huber's loss, so that its pull back stays bound x |A|
    (Config.spo_bound)."""
    dev = ratio - 1.0
    if bound > 0.0:
        edge, far = bound * eps, dev.abs()
        penalty = torch.where(far <= edge, dev.square() / (2.0 * eps), edge * (far - 0.5 * edge) / eps)
    else:
        penalty = dev.square() / (2.0 * eps)
    return ratio * adv - adv.abs() * penalty


def fade_share(iteration: int, fade: tuple[tuple[int, float], ...]) -> float:
    """The share of a weight left at this iteration (Config.kl_fade, clone_fade): linear between fade's (iteration, share)
    points, and the last point's after it."""
    at, share = zip(*fade)
    return float(np.interp(iteration, at, share))


def set_games(env, games: dict[str, float | None]) -> None:
    """Learner.games, for the env's games starting from now on."""
    for name, value in games.items():
        setattr(env, name, value)


def beta(progress: float) -> float:
    """How much of the reward is the final result: 03-training.md, "Reward"."""
    if progress < 0.1:
        return 0.0
    return min(0.9, 0.9 * (progress - 0.1) / 0.5)


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
    value: np.ndarray | float = 0.0     # by term (Config.terms), as are reward, advantages and ret
    reward: np.ndarray | float = 0.0
    advantages: np.ndarray | float = 0.0
    ret: np.ndarray | float = 0.0
    advantage: float = 0.0              # the terms' advantages weighed together (Learner.weigh)
    latents: np.ndarray | None = None   # tokens x latent: this frame's, as the actor read it
    label: dict | None = None           # what the scripts advised here, as slots (actions.SLOT_FIELDS), when they advise
    labelled: np.ndarray | None = None  # which controllable units the advice told something
    z: np.ndarray | None = None         # the side's strategy as the actor was told it (Config.z), the same all game
    slot: int = -1                      # GRPO: the env's slot it was taken at
    mark: np.ndarray | None = None      # GRPO: its branch's terms summed up to it, by term (GroupCollector)
    counted: int = 0                    # GRPO: and the steps they were summed over
    level: float | np.ndarray = 0.0     # GRPO: by term, each level term's level when it was taken (Config.economy_change)
    rest: np.ndarray | None = None      # bootstrapped GRPO: its branch's terms from it to its stretch's end, discounted to it
    discount: np.ndarray | None = None  # and each term's discount from it to that end


@dataclass
class Segment:
    steps: list[Step]
    history: np.ndarray | None      # the frames before its first step, as the actor keeps them (policy_latent.LatentActor.initial_state)
    seen: np.ndarray | None
    bundle: bd.Bundle
    names: np.ndarray
    group: int = -1                 # GRPO: the group it was judged in, an id a seat's team in a group's stretch (GroupCollector._judge)
    spreads: np.ndarray | None = None   # GRPO: that group's spread of each term
    branch: int = -1                    # bootstrapped GRPO: its branch's id
    members: int = 0                    # and how many branches its group's seat had
    end: rf.Frame | None = None         # where its branch ended, on the branch's last segment; None if its game or alliance did
    end_history: np.ndarray | None = None   # and what its seat kept there, as `history`: a critic of the actor's memory reads it
    objective: np.ndarray | None = None     # GRPO, z "objective": its seat's own weights of the terms this game; None: the run's


@dataclass
class Trajectory:
    bundle: bd.Bundle
    names: np.ndarray
    steps: list[Step] = field(default_factory=list)
    history: np.ndarray | None = None   # at the first step not yet in a segment
    seen: np.ndarray | None = None
    segment_start: int = 0
    done: bool = False
    bootstrap: np.ndarray | float = 0.0     # by term


def name_map(rng: np.random.Generator, vocabulary: int) -> np.ndarray:
    """For one seat of one game, some names read as unknown throughout, so the net can't lean on names alone."""
    out = np.arange(vocabulary, dtype=np.int32)
    out[rng.random(vocabulary) < NAME_DROPOUT] = vocabulary - 1
    return out


_WORKER: dict = {}


def _worker_state(bundles: dict, kinds: tuple[int, ...], threads: int = 0) -> dict:
    return {"bundles": bundles, "kinds": kinds, "table": tables.table(), "threads": threads}


def _worker_setup(bundles: dict, kinds: tuple[int, ...], threads: int = 0) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _WORKER.update(_worker_state(bundles, kinds, threads))


def _drop_names(t: fz.Tensors, names: np.ndarray) -> None:
    t.true_name, t.unit_name = t.unit_name, names[t.unit_name]


def _worker_chunk(chunk: list[tuple[rf.Frame, int]], state: dict | None = None) -> list[fz.Tensors]:
    """In a worker process, `state` is the process's own; in this one, each DataWorkers passes its own.  With threads, in C++
    on that many (Config.featurize_threads): the same tensors, each frame's as views of a buffer of its own."""
    w = _WORKER if state is None else state
    if w["threads"]:
        return zkpolicy.featurize_many([(f, w["bundles"][key]) for f, key in chunk], w["table"], w["threads"], enabled_kinds=w["kinds"],
                                       privileged=True, economy=True)
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

    def __init__(self, count: int, bundles: list[bd.Bundle], kinds: tuple[int, ...], threads: int = 0):
        """threads: the C++ featurizer's, in each process that featurizes (`_worker_chunk`); 0: featurize's Python."""
        self.count = count
        self.keys = {id(b) for b in bundles}
        by_key = {id(b): b for b in bundles}
        self.pool = multiprocessing.get_context("fork").Pool(count, _worker_setup, (by_key, kinds, threads)) if count else None
        # Without processes, its own: two learners in one process (as in the tests) would otherwise share a module's.
        self.state = _worker_state(by_key, kinds, threads) if self.pool is None else None

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
    """Seats in order, cut where the next would take a forward pass's padded pairs over the limit.  (As graphs the seats go by
    level, not in order: policy_graphs.Graphs.passes.)"""
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
    """Chooses actions: the learner's own nets, or a frozen copy while the learner goes on training.  graphs: each forward pass a
    CUDA graph's replay (policy_graphs.py), on CUDA; one thread only."""

    def __init__(self, actor: policy_latent.LatentActor, critic: pm.Critic, workers: DataWorkers, device: str, graphs: bool = False):
        self.actor, self.critic, self.workers = actor, critic, workers
        # Its own: an autocast object keeps state between enter and exit, and two threads may each be inside one.
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
        self.device = device
        self.graphs = policy_graphs.Graphs(actor, critic) if graphs and str(device).startswith("cuda") else None

    def _features(self, items: list[tuple[rf.Frame, bd.Bundle, np.ndarray]]) -> list[fz.Tensors]:
        out = self.workers.map([(f, b) for f, b, _ in items])
        for t, (_, _, names) in zip(out, items):
            _drop_names(t, names)
        return out

    @torch.no_grad()
    def _forward(self, items: list[fz.Tensors], history: torch.Tensor, seen: torch.Tensor, z: torch.Tensor | None = None):
        if self.graphs is not None:
            return self.graphs.forward(items, history, seen, z)
        b = pm.collate(items).to(self.device)
        with self.autocast:
            ctx, new_history, new_seen, keep = self.actor.decide(b, history, seen, z)
            a = pm.sample_actions(self.actor, b, ctx)
            new_history = self.actor.acted(b, a, new_history)       # what it did, for the decisions after
            value = self.critic(b) if self.critic is not None else torch.zeros(len(items), 1, device=self.device)
        out = {k: getattr(a, k).cpu().numpy() for k in act.SLOT_FIELDS + ("logp",)}
        out.update({k: v.cpu().numpy() for k, v in keep.items()})
        return out, value.float().cpu().numpy(), new_history.float(), new_seen.float()

    def act(self, frames: list[tuple[rf.Frame, bd.Bundle, np.ndarray]], history: torch.Tensor, seen: torch.Tensor, z: torch.Tensor | None = None,
            rows: int = 0):
        """Forward passes for many games, as few as fit the memory.  Returns tensors, numpy actions, values and the new state.
        z: each seat's strategy (policy_latent.Z_DIM), or None.  rows: the most seats a graph's pass takes (0: all of them).
        As graphs, the seats of a level of frame size go together (policy_graphs.Graphs.passes), so small frames don't pad to
        the widest; what comes back is in the seats' order all the same."""
        items = self._features(frames)
        if self.graphs is None:
            passes = [list(range(len(items))[s]) for s in _act_slices(items)]
        else:
            passes = self.graphs.passes(items, rows) or [[]]
        parts = []
        for seats in passes:
            at = slice(seats[0], seats[-1] + 1) if seats and seats[-1] - seats[0] + 1 == len(seats) else torch.as_tensor(seats, dtype=torch.long, device=history.device)
            parts.append(self._forward([items[i] for i in seats], history[at], seen[at], None if z is None else z[at]))
        if len(parts) == 1:
            return (items, *parts[0])
        back = np.argsort(np.concatenate(passes), kind="stable")        # each seat's place among the passes' rows
        there = torch.as_tensor(back, device=history.device)
        out = {k: _stack_padded([p[0][k] for p in parts])[back] for k in parts[0][0]}
        return items, out, np.concatenate([p[1] for p in parts])[back], torch.cat([p[2] for p in parts])[there], torch.cat([p[3] for p in parts])[there]


class Learner:
    def __init__(self, cfg: Config, run: pathlib.Path, workers: DataWorkers, device: str = "cuda"):
        self.cfg, self.run, self.device, self.workers = cfg, run, device, workers
        torch.manual_seed(cfg.seed)
        self.table = tables.table()
        self.vocabulary = len(self.table.names) + 1
        size, state = policy_latent.SIZES[cfg.size], None
        if cfg.init:
            state = torch.load(cfg.init, map_location="cpu", weights_only=False)
            size = policy_latent.Size(**state["size"])
        self.actor = policy_latent.make(self.vocabulary, cfg.latent_encoder, cfg.decide_every, size).to(device)
        self.reference = None           # the clone as it was, frozen, for the KL to it
        if state is not None:
            missing, unexpected = self.actor.load_state_dict(state["actor"], strict=False)
            assert not unexpected and all(k.startswith("z_in.") for k in missing), (missing, unexpected)
            if cfg.kl_start:
                self.reference = copy.deepcopy(self.actor).eval()
                for p in self.reference.parameters():
                    p.requires_grad_(False)
        # GRPO has none, a branch's group standing in for it (GroupCollector), but bootstrapped (Config.bootstrap).
        if cfg.critic_size == "memory":
            # A small critic a term, reading the actor's own latents and its GRU's state (policy_model.MemoryCritic).
            if not (self.actor.gru and cfg.group and cfg.bootstrap) or cfg.resign:
                raise ValueError("--critic-size memory reads the actor's GRU (--size gru, or rdnn), in bootstrapped GRPO (--group, --bootstrap), without --resign")
            self.critic = pm.MemoryCritic(self.actor.encoder.tokens * self.actor.encoder.latent + self.actor.gru, bins=cfg.critic_bins,
                                          span=cfg.critic_span, terms=len(cfg.terms)).to(device)
        else:
            self.critic = None if cfg.group and not cfg.bootstrap else pm.Critic(self.vocabulary, bins=cfg.critic_bins, span=cfg.critic_span, terms=len(cfg.terms),
                                                                                 **pm.CRITIC_SIZES[cfg.critic_size]).to(device)
        self.term_names = tuple(name for name, _ in cfg.terms)
        self.term_weights = np.array([w for _, w in cfg.terms])
        if cfg.resign and not (cfg.group and cfg.bootstrap and "result" in self.term_names):
            raise ValueError("--resign reads the critic's result: it needs --group, --bootstrap and the result term")
        self.lineages: collections.deque = collections.deque(maxlen=RESIGN_WINDOW)     # played-out games: (least value, not lost)
        self.resign_at: float | None = None
        self.end_stats: dict[str, float] = {}       # bootstrapped GRPO's ends, this iteration's
        # The iteration from which the actor learns too (Config.critic_rewarm).
        self.warm_until = cfg.critic_warmup if self.critic is not None else 0
        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        critic_params = [p for p in self.critic.parameters() if p.requires_grad] if self.critic is not None else []
        self.trained = actor_params + critic_params
        self.actor_trained, self.critic_trained = actor_params, critic_params
        # Muon's, as frame_vae.Optimisers chooses: the weights of the actor's linear layers but those giving answers
        # (policy_latent.answers), as cloning has them, and of the critic's but the one giving the value.
        actor_hidden = ({id(m.weight) for name, m in self.actor.named_modules()
                         if isinstance(m, torch.nn.Linear) and m.weight.requires_grad and not policy_latent.answers(name, m)}
                        if cfg.actor_muon_lr > 0.0 else set())
        critic_hidden = ({id(m.weight) for name, m in self.critic.named_modules() if isinstance(m, torch.nn.Linear) and name.split(".")[0] != "value"}
                         if cfg.critic_muon_lr > 0.0 and self.critic is not None else set())
        # An RDNN's recurrent weights (policy_latent.RDNNCell.recurrent) have an Adam of their own, with a small eps: their
        # gradient is far under this one's.  At the schedule's rate, whatever the KL guard leaves the actor of it.
        memory = getattr(self.actor, "memory", None)
        recurrent = memory.recurrent() if isinstance(memory, policy_latent.RDNNCell) else []
        self.recurrent = torch.optim.Adam(recurrent, lr=cfg.recurrent_lr or cfg.lr_start, eps=RECURRENT_EPS) if recurrent else None
        if cfg.compile_memory and recurrent:
            policy_latent.compile_memory()
        hidden = actor_hidden | critic_hidden | {id(p) for p in recurrent}
        # Two groups each, so that the critic can learn alone at first (critic_warmup).
        self.optimizer = torch.optim.Adam([{"params": [p for p in actor_params if id(p) not in hidden]},
                                           {"params": [p for p in critic_params if id(p) not in hidden], "lr": cfg.critic_lr or cfg.lr_start}],
                                          lr=cfg.lr_start, eps=cfg.adam_eps)
        groups = [{"params": [p for p in params if id(p) in chosen], "role": role, "lr": rate}
                  for role, params, chosen, rate in (("actor", actor_params, actor_hidden, cfg.actor_muon_lr),
                                                     ("critic", critic_params, critic_hidden, cfg.critic_muon_lr)) if chosen]
        self.muon = torch.optim.Muon(groups, weight_decay=MUON_DECAY) if groups else None
        self.economy_share = cfg.economy_start
        self.decisions = 0
        self.iteration = 0
        self.entropy_coef = {h: self.entropy_base for h in HEADS}
        self.token_budget = cfg.token_budget        # SPLIT_SHRINK, BUDGET_GROW
        self.lr_scale = cfg.lr_scale_start          # the actor's rate's share of its schedule's (Config.kl_target)
        self.actions_scale = 1.0                    # GRPO: the iteration's mean actions a decision, done included (_update_loss)
        self.splits = 0
        self.shrunk = False                         # this iteration
        self.ahead: dict[tuple, object] = {}        # the next minibatch's decisions, asked for already (_fetch_ahead)
        self.memory_limit = 0.0                     # what the learner may reserve on the card, if it's on one
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
        self.graphs = cfg.graphs and not cfg.concurrent
        self.player = Player(self.actor, None if cfg.group else self.critic, workers, device, self.graphs)     # GRPO plays no critic
        self.ema = self.frozen_player(workers, self.graphs) if cfg.ema > 0.0 else None
        if device == "cuda":
            # It empties its cache at this limit, and fails rather than spills.
            card = torch.cuda.mem_get_info()[1]
            fraction = gpu_fraction(cfg.gpu_share, cfg.gpu_runs, card, collector_bytes(cfg))
            torch.cuda.set_per_process_memory_fraction(fraction)
            self.memory_limit = fraction * card

    @property
    def progress(self) -> float:
        return min(1.0, self.decisions / self.cfg.decisions)

    @property
    def entropy_base(self) -> float:
        """A head's least entropy bonus (Config.entropy_base)."""
        base = self.cfg.entropy_base
        return base if base >= 0.0 else 0.0 if self.cfg.group else ENTROPY_BASE

    def act(self, frames: list[tuple[rf.Frame, bd.Bundle, np.ndarray]], history: torch.Tensor, seen: torch.Tensor):
        return self.player.act(frames, history, seen)

    def frozen_player(self, workers: DataWorkers, graphs: bool = False) -> Player:
        actor, critic = copy.deepcopy(self.actor), None if self.cfg.group else copy.deepcopy(self.critic)
        for p in list(actor.parameters()) + (list(critic.parameters()) if critic is not None else []):
            p.requires_grad_(False)
        return Player(actor, critic, workers, self.device, graphs)

    def _pairs(self, player: Player) -> list[tuple[torch.nn.Module, torch.nn.Module]]:
        """Each of this learner's nets with a player's."""
        return [(self.actor, player.actor)] + ([(self.critic, player.critic)] if self.critic is not None and player.critic is not None else [])

    @torch.no_grad()
    def copy_weights_to(self, player: Player) -> None:
        for mine, theirs in self._pairs(player):
            for a, b in zip(theirs.parameters(), mine.parameters()):
                a.copy_(b)

    @torch.no_grad()
    def ema_step(self) -> None:
        """The moving average a step nearer the weights, in what's trained."""
        if self.ema is None:
            return
        pairs = [(e, w) for mine, theirs in self._pairs(self.ema) for e, w in zip(theirs.parameters(), mine.parameters()) if w.requires_grad]
        torch._foreach_lerp_([e for e, _ in pairs], [w for _, w in pairs], 1.0 - self.cfg.ema)

    @property
    def evaluator(self) -> Player:
        return self.ema if self.ema is not None else self.player

    def _minibatches(self, segments: list[Segment], rng: np.random.Generator) -> list[list[Segment]]:
        """Whole segments up to the token budget, near-equal sizes together so padding stays small.  Sizes are jittered, so a
        minibatch isn't all from one stage of the game, and the minibatches then go in random order.  The budget counts a
        decision's padded tokens and `DECISION_TOKENS`: a small frame batched with a big map's costs as much as the big map's.
        Token groups pad separately, so each is tracked on its own."""
        units = np.array([max(min(len(s.frame.units), UNIT_TOKENS) + min(len(s.frame.sites), fz.CAP_SITES) + min(len(s.frame.features), fz.CAP_FEATURES) for s in seg.steps) for seg in segments])
        regions = np.array([seg.bundle.regions for seg in segments])
        lengths = np.array([len(seg.steps) for seg in segments])
        order = np.argsort((units + regions) * rng.uniform(0.85, 1.15, len(segments)))
        batches, current = [], []
        decisions = most_units = most_regions = 0
        for i in order:
            grown = (decisions + lengths[i]) * (max(most_units, units[i]) + max(most_regions, regions[i]) + 2 + DECISION_TOKENS)
            if current and grown > self.token_budget:
                batches.append(current)
                current = []
                decisions = most_units = most_regions = 0
            current.append(segments[int(i)])
            decisions += lengths[i]
            most_units, most_regions = max(most_units, units[i]), max(most_regions, regions[i])
        if current:
            batches.append(current)
        return [batches[int(k)] for k in rng.permutation(len(batches))]

    def _featurize(self, batch: list[Segment]):
        """A minibatch's decisions as the data workers make them ready for the GPU, in chunks: with processes, begun now."""
        work = [(step.frame, seg.bundle, seg.names) for seg in batch for step in seg.steps]
        chunks = [work[i:i + FLAT_CHUNK] for i in range(0, len(work), FLAT_CHUNK)]
        return self.workers.batches(chunks, max(seg.bundle.regions for seg in batch))

    def _fetch_ahead(self, batch: list[Segment]) -> None:
        """The next minibatch's decisions asked of the data workers while the GPU is on this one's: `_loss` waited on them for a
        fifth of a minibatch's time, the GPU with nothing of the learner's to do (py-spy, 2026-11-16).  `_loss` takes them when
        it's handed these same segments; a minibatch done again in parts is asked for then."""
        if self.workers.pool is not None:
            self.ahead[tuple(map(id, batch))] = self._featurize(batch)

    def _loss(self, batch: list[Segment], eps: float, note) -> torch.Tensor:
        """Every decision of each segment in one forward pass, from the latents it was played with."""
        device = self.device
        steps = [step for seg in batch for step in seg.steps]
        ahead = self.ahead.pop(tuple(map(id, batch)), None)
        whole = pm.join(list(ahead if ahead is not None else self._featurize(batch))).to(device)
        lengths = [len(seg.steps) for seg in batch]
        latents = np.zeros((len(batch), max(lengths), *steps[0].latents.shape), np.float32)
        for i, seg in enumerate(batch):
            latents[i, :len(seg.steps)] = [s.latents for s in seg.steps]
        history = torch.as_tensor(np.stack([seg.history for seg in batch]), device=device)
        seen = torch.as_tensor(np.stack([seg.seen for seg in batch]), device=device)
        latents = torch.as_tensor(latents, device=device)
        # A segment is one game's, so its first step's z is all of its steps'.
        z = None if steps[0].z is None else torch.as_tensor(np.stack([seg.steps[0].z for seg in batch]), device=device)
        with self.autocast:
            a = _stored_actions(steps, device)
            acted = self.actor.segment_actions(whole, a, lengths)
            kept: dict = {}
            now = self.actor.segments(history, seen, latents, lengths, acted, z, keep=kept)
            ctx = self.actor.context(whole, now)
            logp, entropy = pm.evaluate_actions(self.actor, whole, ctx, a)
            if "memory" in kept:
                # The memory's state after each decision: one that grows without end, or goes to nothing, shows here first.
                state = kept["memory"][1].detach().abs()
                note("state_abs", float(state.mean()))
                note("state_max", float(state.max()))
            if isinstance(self.critic, pm.MemoryCritic):
                value = self.critic(*(t.detach() for t in kept["memory"]), logits=True)
            else:
                value = self.critic(whole, logits=True) if self.critic is not None else None
            extra = torch.zeros((), device=device)
            coef = self.kl_weight
            if self.reference is not None and coef > 0.0:
                # KL(policy || clone) on the actions taken, as k3 = r - 1 - log r with r = clone / policy: never below 0.
                with torch.no_grad():
                    ref = self.reference
                    ref_now = ref.segments(history, seen, latents, lengths, ref.segment_actions(whole, a, lengths), z)
                    ref_logp, _ = pm.evaluate_actions(ref, whole, ref.context(whole, ref_now), a)
                w = pm.chosen(a).float()
                log_r = (ref_logp.float() - logp).clamp(-20.0, 20.0)
                kl = ((log_r.exp() - 1.0 - log_r) * w).sum() / w.sum().clamp_min(1.0)
                extra = extra + coef * kl
                note("kl_clone", float(kl.detach()))
            clone_weight = self.clone_weight
            if clone_weight > 0.0 and any(s.label is not None for s in steps):
                lab, labelled, has = _label_actions(steps, whole.ctrl_mask.shape[1], device)
                lab_logp, _ = pm.evaluate_actions(self.actor, whole, ctx, lab, labelled)
                w = pm.chosen(lab).float() * has[:, None]
                clone = -(w * lab_logp).sum() / w.sum().clamp_min(1.0)
                extra = extra + clone_weight * clone
                note("loss_clone", float(clone.detach()))
        return self._update_loss(steps, a, logp, entropy, value, eps, note) + extra

    def _update_loss(self, steps: list[Step], a: pm.Actions, logp: torch.Tensor, entropy: dict, value: torch.Tensor | None, eps: float, note) -> torch.Tensor:
        """Simple Policy Optimization's policy loss (Xie et al., docs/research/simple-policy-optimization.md), with the value and
        entropy terms.  Each action's own ratio r scores r A - |A| (r - 1)^2 / (2 eps): greatest at r = 1 + sign(A) eps, and
        pulling every ratio back toward that, where PPO's clip left those past it with no gradient.  Averaged over the
        decision's actions, done included; one advantage a decision."""
        cfg, device = self.cfg, self.device
        decisions = len(steps)
        advantages = np.array([s.advantage for s in steps], np.float32)
        if cfg.group:       # GRPO: as judged against their groups (GroupCollector._judge, weigh_groups), not again over the minibatch
            note("adv_abs_max", float(np.abs(advantages).max()))
            if cfg.adv_clip > 0.0:
                note("adv_clipped", float((np.abs(advantages) > cfg.adv_clip).mean()), decisions)
                advantages = np.clip(advantages, -cfg.adv_clip, cfg.adv_clip)
            adv = torch.as_tensor(advantages, device=device)
        else:
            adv = torch.as_tensor((advantages - advantages.mean()) / (float(advantages.std()) + 1e-6), device=device)
        old = a.logp
        weights = pm.chosen(a).float()
        ratio = (logp - old).clamp(-20.0, 20.0).exp()
        per_action = -spo_objective(ratio, adv[:, None], eps, cfg.spo_bound if cfg.group else 0.0)
        count = weights.sum(-1)
        has = count > 0
        pg = torch.zeros((), device=device)
        if has.any():
            if cfg.group:
                # Summed over a decision's actions, over the iteration's mean count of them (Dr. GRPO): each order counts alike
                # however many its decision has, where an average weighed a losing decision's orders the less, the more it had.
                pg = (per_action * weights).sum() / (decisions * self.actions_scale)
            else:
                pg = ((per_action * weights).sum(-1)[has] / count[has]).sum() / decisions
            with torch.no_grad():
                total = float(weights.sum())
                deviation = (ratio - 1.0).abs()
                note("kl", float(((old - logp) * weights).sum()) / total, total)
                note("clip_fraction", float(((deviation > eps).float() * weights).sum()) / total, total)     # outside 1 +- eps
                note("ratio_deviation", float((deviation * weights).sum()) / total, total)
                note("ratio_deviation_max", float((deviation * weights).max()))
        value_loss = torch.zeros((), device=device)
        if self.critic is not None:
            # Each term's head as if it were the only one: their losses add up.
            ret = torch.as_tensor(np.stack([np.broadcast_to(s.ret, (self.critic.terms,)) for s in steps]), device=device, dtype=torch.float32)
            per_term = (self.critic.hl_gauss(value, ret) if self.critic.bins else (value - ret).pow(2)).sum(0) / decisions
            value_loss = per_term.sum()
            note("loss_value", float(value_loss.detach()))
            if cfg.group:       # for each term's explained variance (`train`), as the collector gives it under PPO
                with torch.no_grad():
                    guess = (value.float().softmax(-1) * self.critic.centres).sum(-1) if self.critic.bins else value.float()
                    for name, x in zip(self.term_names, (guess - ret).square().mean(0).tolist()):
                        note(f"value_error_{name}", x, decisions)
            if len(per_term) > 1:
                for name, x in zip(self.term_names, per_term.detach().tolist()):
                    note(f"loss_value_{name}", x)
        entropy_loss = torch.zeros((), device=device)
        for h in HEADS:
            e, mask = entropy[h]
            total, n = (e * mask).sum(), float(mask.sum())
            entropy_loss = entropy_loss + self.entropy_coef[h] * total / max(n, 1.0)
            if n:
                note(f"entropy_{h}", float(total.detach()) / n, n)
        note("loss_pg", float(pg.detach()))
        return pg + cfg.value_weight * value_loss - entropy_loss

    @property
    def kl_weight(self) -> float:
        """The KL to the clone's weight this iteration (Config.kl_fade)."""
        return self.cfg.kl_start * fade_share(self.iteration, self.cfg.kl_fade) if self.reference is not None else 0.0

    @property
    def clone_weight(self) -> float:
        """The cloning loss's weight this iteration (Config.clone_fade)."""
        return self.cfg.clone_weight * fade_share(self.iteration, self.cfg.clone_fade)

    @property
    def lead_share(self) -> float:
        """The share of the training games starting now that a lead can end (Config.lead_fade)."""
        return fade_share(self.iteration, self.cfg.lead_fade)

    @property
    def other_kinds_share(self) -> float | None:
        """The share of the training games starting now that aren't 1v1s (Config.kinds_ramp); None without a ramp."""
        return fade_share(self.iteration, self.cfg.kinds_ramp) if self.cfg.kinds_ramp else None

    @property
    def games(self) -> dict[str, float | None]:
        """What the training games starting now are, as policy_sim_env.SimEnv's attributes (`set_games`): who runs the policy's
        economy in them, who plays its enemies, whether a lead can end them, and their kinds.  A test's are its own (`evaluate`)."""
        return {"policy_economy_share": self.economy_share, "self_play_share": self.cfg.self_play, "lead_share": self.lead_share,
                "other_kinds_share": self.other_kinds_share}

    @property
    def advising(self) -> bool:
        """Whether the scripts should advise in what's collected now: while the cloning loss has weight this iteration or the next,
        which is when what's collected concurrently trains."""
        return self.cfg.clone_weight > 0.0 and max(fade_share(self.iteration + k, self.cfg.clone_fade) for k in (0, 1)) > 0.0

    def set_rates(self) -> None:
        """The actor's rate on its schedule, or 0 while the critic warms up; the critic's scaled as the actor's falls."""
        cfg = self.cfg
        lr = cfg.lr_start + (cfg.lr_end - cfg.lr_start) * self.progress
        actor = 0.0 if self.iteration < self.warm_until else lr * self.lr_scale
        self.optimizer.param_groups[0]["lr"] = actor
        self.optimizer.param_groups[1]["lr"] = cfg.critic_lr * lr / cfg.lr_start if cfg.critic_lr else lr
        for group in self.muon.param_groups if self.muon is not None else ():
            group["lr"] = (cfg.actor_muon_lr * actor if group["role"] == "actor" else cfg.critic_muon_lr * lr) / cfg.lr_start
        if self.recurrent is not None:
            own = cfg.recurrent_lr * lr / cfg.lr_start if cfg.recurrent_lr else lr
            self.recurrent.param_groups[0]["lr"] = 0.0 if self.iteration < self.warm_until else own

    def _memory_weights(self) -> torch.Tensor | None:
        """An RDNN's recurrent weights as they are now (`RDNNCell.weights`' J), or None with another memory."""
        memory = getattr(self.actor, "memory", None)
        return memory.weights()[0].detach() if isinstance(memory, policy_latent.RDNNCell) else None

    def _clear_grads(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=True)
        if self.recurrent is not None:
            self.recurrent.zero_grad(set_to_none=True)

    def _backward(self, batch: list[Segment], eps: float, note, share: float = 1.0) -> None:
        """The minibatch's gradient, `share` of it, added to what the parameters hold (`train` clears them at each step).  It's
        taken apart from theirs and added once it's whole, so one that doesn't fit the card leaves theirs as it was (copying
        theirs first, to put back, was an eighth of training's time: py-spy, 2026-11-16).  That one is done again in 2, then
        4... runs of its segments, each loss weighed by its share of the decisions, and the budget shrinks for the rest of the
        run."""
        decisions = sum(len(seg.steps) for seg in batch)
        parts = 1
        while True:
            notes = []
            grads = part = None
            try:
                for piece in np.array_split(np.arange(len(batch)), parts):
                    run = [batch[int(k)] for k in piece]
                    weight = share * sum(len(seg.steps) for seg in run) / decisions
                    part = torch.autograd.grad(self._loss(run, eps, lambda *a: notes.append(a)) * weight, self.trained, allow_unused=True)
                    grads = list(part) if grads is None else [b if a is None else a if b is None else a + b for a, b in zip(grads, part)]
                break
            except torch.OutOfMemoryError:
                if parts >= len(batch):
                    raise
            # Out of the handler, so the failed attempt's tensors are freed.
            grads = part = None
            torch.cuda.empty_cache()
            parts = min(2 * parts, len(batch))
            self.splits += 1
            if not self.shrunk:
                self.shrunk = True
                self.token_budget = int(self.token_budget * SPLIT_SHRINK)
            print(json.dumps({"split": parts, "decisions": decisions, "token_budget": self.token_budget}), flush=True)
        more = [(p, g) for p, g in zip(self.trained, grads) if g is not None and p.grad is not None]
        for p, g in zip(self.trained, grads):
            if g is not None and p.grad is None:
                p.grad = g.contiguous()         # its own memory, to add the next ones in place: a sum's gradient is one number, expanded
        if more:
            torch._foreach_add_([p.grad for p, _ in more], [g for _, g in more])
        for a in notes:
            note(*a)
        note("minibatch_decisions", decisions)

    def train(self, segments: list[Segment], rng: np.random.Generator, between=None) -> dict[str, float]:
        """between: called after each optimizer step, the moments when the weights are whole."""
        cfg = self.cfg
        p = self.progress
        eps = cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * p
        self.set_rates()
        self.shrunk = False
        totals: dict[str, float] = {}
        counts: dict[str, float] = {}
        scales = self.prepare(segments)

        def note(name: str, value: float, weight: float = 1.0) -> None:
            if name.endswith("_max"):       # the iteration's most
                totals[name], counts[name] = max(totals.get(name, value), value), 1.0
                return
            totals[name] = totals.get(name, 0.0) + value * weight
            counts[name] = counts.get(name, 0.0) + weight

        peaks = [0, 0]
        log = (self.run / "minibatches.jsonl").open("a") if self.memory_limit else None
        for _ in range(cfg.passes):
            plan = self._plan(segments, rng)
            # One minibatch ahead of the GPU (_fetch_ahead).
            self.ahead = {}
            coming = iter([batch for batches in plan for batch in batches][1:])
            for batches in plan:
                total = sum(len(seg.steps) for batch in batches for seg in batch)
                self._clear_grads()
                for batch in batches:
                    if log is not None:
                        torch.cuda.reset_peak_memory_stats()
                    splits = self.splits
                    following = next(coming, None)
                    if following is not None:
                        self._fetch_ahead(following)
                    self._backward(batch, eps, note, sum(len(seg.steps) for seg in batch) / total)
                    if log is not None:
                        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
                        peaks = [max(peaks[0], allocated), max(peaks[1], reserved)]
                        log.write(json.dumps({"iteration": self.iteration, "decisions": sum(len(seg.steps) for seg in batch),
                                              "tokens": padded_tokens(batch), "allocated_gb": round(allocated / 2**30, 3),
                                              "reserved_gb": round(reserved / 2**30, 3), "split": self.splits > splits}) + "\n")
                # The actor's and the critic's apart: together, a young critic's gradient shrank the actor's, and under Adam's
                # eps a small gradient makes a smaller step, not the same one.
                grad = torch.nn.utils.clip_grad_norm_(self.actor_trained, cfg.grad_clip)
                if self.critic_trained:
                    note("grad_norm_critic", float(torch.nn.utils.clip_grad_norm_(self.critic_trained, cfg.grad_clip)))
                before = self._memory_weights()
                self.optimizer.step()
                if self.muon is not None:
                    self.muon.step()
                if self.recurrent is not None:
                    self.recurrent.step()
                self.ema_step()
                if before is not None:
                    # Whether an RDNN's recurrent weights learn: their gradient is small beside Adam's eps (RDNNCell's notes).
                    note("memory_step", float((self._memory_weights() - before).abs().mean() / before.mean()))
                note("grad_norm", float(grad))
                note("step_decisions", total)
                if between is not None:
                    between()
        out = {k: totals[k] / max(counts[k], 1e-9) for k in totals}
        out.update(token_budget=self.token_budget, splits=self.splits, lr_scale=self.lr_scale)
        if len(scales) > 1:
            out.update({f"{'term' if cfg.group else 'return'}_scale_{name}": float(x) for name, x in zip(self.term_names, scales)})
        out.update(self.end_stats)
        for i, name in enumerate(self.term_names if f"value_error_{self.term_names[0]}" in out else ()):
            out[f"explained_variance{'_' + name if i else ''}"] = 1.0 - out[f"value_error_{name}"] / max(float(self.return_var[i]), 1e-8)
        self.follow_kl(out.get("kl"))
        if log is not None:
            log.close()
            out.update(learner_allocated_gb=peaks[0] / 2**30, learner_reserved_gb=peaks[1] / 2**30)     # training's peaks
            if not self.shrunk and peaks[0] < GROW_BELOW * self.memory_limit:
                self.token_budget = min(cfg.token_budget, int(self.token_budget * BUDGET_GROW))
        if self.reference is not None:
            out["kl_weight"] = self.kl_weight
        if self.cfg.clone_weight:
            out["clone_weight"] = self.clone_weight
        return out

    def prepare(self, segments: list[Segment]) -> np.ndarray:
        """Each decision's advantage, its terms weighed, and the mean count of actions a decision (`_update_loss`).  Returns
        the terms' scales."""
        steps = [s for seg in segments for s in seg.steps]
        if self.cfg.group and self.cfg.bootstrap:
            self.bootstrap(segments)
        if not self.cfg.group:
            scales = self.weigh(steps)
        elif self.cfg.group_scale == "terms":
            scales = self.weigh_groups(segments)
        else:           # the collector gave each step its advantage (GroupCollector._judge)
            scales = np.ones(1)
        self.actions_scale = float(np.mean([1 + int((s.anchor[:-1] >= 0).sum()) for s in steps])) if steps else 1.0
        return scales

    def bootstrap(self, segments: list[Segment]) -> None:
        """Bootstrapped GRPO (Config.bootstrap): each decision's return, the critic's target, is its branch's terms from it to its
        stretch's end (`Step.rest`, GroupCollector._judge) and the critic's value of where the branch ended, discounted to it.
        Its advantages are that, the economy less the income balance when it was taken (Config.economy_change), less the
        group's mean of the same from the same moment, a branch with no decision then counting 0, as the collector judges them
        unbootstrapped; each group's spreads (weigh_groups) are of the same at its stretch's start."""
        ends = {seg.branch: (seg.end, seg.bundle, seg.names, seg.end_history) for seg in segments if seg.end is not None}
        values = dict(zip(ends, self.values_of([e[:3] for e in ends.values()], [e[3] for e in ends.values()])))
        self.end_stats = {}
        if values and "result" in self.term_names:
            # How much play is spent where the critic already calls the game lost: each valued end, by its branch's decisions.
            at = self.term_names.index("result")
            counts: dict[int, int] = {}
            for seg in segments:
                if seg.branch in values:
                    counts[seg.branch] = counts.get(seg.branch, 0) + len(seg.steps)
            result = np.array([values[b][at] for b in counts])
            weight = np.array(list(counts.values()), np.float64)
            self.end_stats["end_value_result"] = float(result.mean())
            for line in HOPELESS:
                self.end_stats[f"hopeless{round(-100 * line)}"] = float(weight[result < line].sum() / weight.sum())
        from policy_sim_env import LEVEL_TERMS
        change = np.array([self.cfg.economy_change and name in LEVEL_TERMS for name in self.term_names])
        means: dict[tuple[int, int], np.ndarray] = {}
        starts: dict[int, dict[int, tuple[int, np.ndarray]]] = {}       # group: branch: its first decision's slot and its terms
        for seg in segments:
            end = values.get(seg.branch, 0.0)
            first = starts.setdefault(seg.group, {})
            for step in seg.steps:
                step.ret = step.rest + step.discount * end
                own = step.ret - change * step.level
                means[seg.group, step.slot] = means.get((seg.group, step.slot), 0.0) + own / seg.members
                if seg.branch not in first or step.slot < first[seg.branch][0]:
                    first[seg.branch] = (step.slot, own)
        for seg in segments:
            seg.spreads = np.stack([own for _, own in starts[seg.group].values()]).std(0)
            for step in seg.steps:
                step.advantages = step.ret - change * step.level - means[seg.group, step.slot]
        returns = np.stack([step.ret for seg in segments for step in seg.steps]) if segments else np.zeros((0, len(self.term_names)))
        self.return_var = returns.var(0) if len(returns) else np.ones(len(self.term_names))

    @property
    def resign_state(self) -> tuple[bool, float | None]:
        """For the collectors (Config.resign): whether to measure the critic's result where groups go on, and the threshold."""
        measuring = self.cfg.resign and self.iteration >= self.warm_until
        return measuring, self.resign_at if measuring else None

    def follow_resignation(self, records: list[tuple[float, float]]) -> dict[str, float]:
        """The played-out games of a collection, each (the least result the critic gave it where it went on, 1 if it then wasn't
        lost), into the window, and the threshold from it (Config.resign).  Returns its metrics."""
        cfg = self.cfg
        self.lineages.extend(records)
        out = {"played_out": len(records), "played_out_window": len(self.lineages)}
        self.resign_at = None
        for line in RESIGN_THRESHOLDS:
            kept = [k for low, k in self.lineages if low < line]
            if len(kept) >= cfg.resign_least:
                out[f"resign_false{round(-100 * line)}"] = float(np.mean(kept))
                if np.mean(kept) <= cfg.resign_false:
                    self.resign_at = line
        out["resign_at"] = self.resign_at if self.resign_at is not None else 0.0
        out["resigning"] = float(self.resign_at is not None)
        return out

    @torch.no_grad()
    def values_of(self, frames: list[tuple[rf.Frame, bd.Bundle, np.ndarray]], kept: list[np.ndarray | None] | None = None) -> list[np.ndarray]:
        """The critic's value of each (frame, bundle, names), by term.  kept: what each frame's seat kept before it
        (Segment.end_history), which a critic of the actor's memory reads the frame from."""
        if not frames:
            return []
        out = []
        regions = max(b.regions for _, b, _ in frames)
        starts = range(0, len(frames), FLAT_CHUNK)
        for i, batch in zip(starts, self.workers.batches([frames[i:i + FLAT_CHUNK] for i in starts], regions)):
            batch = batch.to(self.device)
            with self.autocast:
                if isinstance(self.critic, pm.MemoryCritic):
                    history = torch.as_tensor(np.stack(kept[i:i + FLAT_CHUNK]), device=self.device)
                    values = self.critic(*self.actor.memory_at(batch, history))
                else:
                    values = self.critic(batch)
            out.extend(values.float().cpu().numpy())
        return out

    def gradient(self, segments: list[Segment], rng: np.random.Generator) -> torch.Tensor:
        """The actor's gradient of the loss over these segments as one optimizer step would take it, before its clipping,
        flattened; their advantages as `prepare` left them.  A critic's (--bootstrap) is left out: it doesn't learn from the
        advantages, and would make any two halves alike.  The parameters' gradients are cleared after."""
        cfg = self.cfg
        eps = cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * self.progress
        total = sum(len(seg.steps) for seg in segments)
        self._clear_grads()
        for batch in self._minibatches(segments, rng):
            self._backward(batch, eps, lambda *a: None, sum(len(seg.steps) for seg in batch) / total)
        out = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten().float() for p in self.actor_trained])
        self._clear_grads()
        return out

    def _plan(self, segments: list[Segment], rng: np.random.Generator) -> list[list[list[Segment]]]:
        """The iteration's optimizer steps, each as its minibatches (`_minibatches`).  Under GRPO with Config.group_step, whole
        groups in a random order, each step's up to that many decisions or just past (the last joins the one before when it would
        be under half of it); else a step a minibatch."""
        cfg = self.cfg
        if not (cfg.group and cfg.group_step):
            return [[batch] for batch in self._minibatches(segments, rng)]
        by: dict[int, list[Segment]] = {}
        for seg in segments:
            by.setdefault(seg.group, []).append(seg)
        parts: list[list[Segment]] = []
        current: list[Segment] = []
        n = 0
        for g in rng.permutation(sorted(by)):
            current += by[int(g)]
            n += sum(len(seg.steps) for seg in by[int(g)])
            if n >= cfg.group_step:
                parts.append(current)
                current, n = [], 0
        if current:
            if parts and n < cfg.group_step / 2:
                parts[-1] += current
            else:
                parts.append(current)
        return [self._minibatches(part, rng) for part in parts]

    def weigh_groups(self, segments: list[Segment]) -> np.ndarray:
        """GRPO's advantages with Config.group_scale "terms": each decision's terms less its group's (GroupCollector._judge), each
        over that term's mean spread in this iteration's groups where it varied, weighed (Config.terms; or as its seat's own
        objective weighs them, `Segment.objective`).  Returns the scales."""
        spreads = {seg.group: seg.spreads for seg in segments}
        s = np.stack(list(spreads.values())) if spreads else np.zeros((0, len(self.term_weights)))
        varied = s > 1e-9
        scales = np.where(varied.any(0), (s * varied).sum(0) / np.maximum(varied.sum(0), 1), 1.0)
        for seg in segments:
            weights = (self.term_weights if seg.objective is None else seg.objective) / scales
            for step in seg.steps:
                step.advantage = float(np.dot(step.advantages, weights))
        return scales

    def weigh(self, steps: list[Step]) -> np.ndarray:
        """Each step's advantage, its terms' weighed (Config.terms), each first over its returns' spread this iteration, from the
        5th percentile to the 95th, when that's over 1, as DreamerV3 scales its returns (Hafner et al. 2023): so a term whose
        returns spread wide doesn't drown the rest, and one that's nearly constant isn't blown up.  Returns the scales."""
        if not steps:
            return np.ones(len(self.term_weights))
        returns = np.stack([np.broadcast_to(s.ret, self.term_weights.shape) for s in steps])
        scales = np.maximum(1.0, np.percentile(returns, 95, 0) - np.percentile(returns, 5, 0))
        weights = self.term_weights / scales
        for s in steps:
            s.advantage = float(np.dot(np.broadcast_to(s.advantages, weights.shape), weights))
        return scales

    def follow_kl(self, kl: float | None) -> None:
        """The actor's rate for the next iteration, from this one's KL (Config.kl_target)."""
        target = self.cfg.kl_target
        if not target or kl is None or self.iteration < self.warm_until:
            return
        if kl > 2.0 * target:
            self.lr_scale /= KL_SLOW
        elif kl < 0.5 * target:
            self.lr_scale = min(1.0, self.lr_scale * KL_SLOW)

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
                self.entropy_coef[h] = max(self.entropy_base, self.entropy_coef[h] * 0.97)

    def save(self) -> None:
        """A kill at any moment leaves latest.json naming a whole checkpoint."""
        directory = self.run / "ckpt"
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*.tmp"):       # from a process killed while saving
            stale.unlink()
        name = f"it-{self.iteration:06d}.pt"
        temp = directory / (name + ".tmp")
        critic = {"critic": self.critic.state_dict(), "critic_size": self.critic.size} if self.critic is not None else {}
        ema = {}
        if self.ema is not None:
            ema = {"ema_actor": self.ema.actor.state_dict(), **({"ema_critic": self.ema.critic.state_dict()} if self.ema.critic is not None else {})}
        muon = {"muon": self.muon.state_dict()} if self.muon is not None else {}
        if self.recurrent is not None:
            muon["recurrent"] = self.recurrent.state_dict()
        torch.save({"actor": self.actor.state_dict(), **critic, "optimizer": self.optimizer.state_dict(), **ema, **muon,
                    "size": asdict(self.actor.size),
                    "iteration": self.iteration, "decisions": self.decisions, "entropy_coef": self.entropy_coef, "lr_scale": self.lr_scale,
                    "terms": list(self.term_names), "warm_until": self.warm_until,
                    "lineages": [list(r) for r in self.lineages], "resign_at": self.resign_at}, temp)
        os.replace(temp, directory / name)
        marker = self.run / "latest.json.tmp"
        marker.write_text(json.dumps({"checkpoint": f"ckpt/{name}", "iteration": self.iteration, "decisions": self.decisions}))
        os.replace(marker, self.run / "latest.json")

    def resume(self) -> bool:
        marker = self.run / "latest.json"
        if not marker.exists():
            return False
        state = torch.load(self.run / json.loads(marker.read_text())["checkpoint"], map_location=self.device, weights_only=True)
        if (self.critic is None) != ("critic" not in state):
            raise SystemExit(f"{self.run} was trained {'with' if 'critic' in state else 'without'} a critic: a run can't go between PPO "
                             "or bootstrapped GRPO (--bootstrap) and GRPO without, but another can start from its actor (--init)")
        # Other terms than the checkpoint's: the critic keeps what it reads, and its value heads start afresh, Adam's moments of
        # them too, while it learns them alone for a while (Config.critic_rewarm).
        was = list(state.get("terms", ["blend"]))
        fresh = self.critic is not None and was != list(self.term_names)
        self.actor.load_state_dict(state["actor"])
        if self.critic is not None:
            _load_critic(self.critic, state["critic"], fresh, was, self.term_names)
        moments = state["optimizer"]
        if fresh and isinstance(self.critic, pm.MemoryCritic):
            # A net a term: the critic's parameters aren't those the moments were kept for.  Adam keeps the actor's, its first
            # group's, and starts on the critic's anew.
            kept = len(moments["param_groups"][0]["params"])
            mine = {**moments["param_groups"][1], "params": list(range(kept, kept + len(self.optimizer.param_groups[1]["params"])))}
            moments = {"state": {k: v for k, v in moments["state"].items() if k < kept}, "param_groups": [moments["param_groups"][0], mine]}
        self.optimizer.load_state_dict(moments)
        if fresh:
            for p in self.critic.value.parameters():
                self.optimizer.state.pop(p, None)
        if self.muon is not None:
            self.muon.load_state_dict(state["muon"])
        if self.recurrent is not None and "recurrent" in state:
            self.recurrent.load_state_dict(state["recurrent"])
        if self.ema is not None:
            self.ema.actor.load_state_dict(state.get("ema_actor", state["actor"]))
            if self.ema.critic is not None:
                _load_critic(self.ema.critic, state.get("ema_critic", state["critic"]), fresh, was, self.term_names)
        self.iteration, self.decisions = state["iteration"], state["decisions"]
        self.warm_until = self.iteration + self.cfg.critic_rewarm if fresh else state.get("warm_until", self.cfg.critic_warmup)
        # With no floor (Config.entropy_floor 0) no bonus rises above the base, which may have changed since.
        self.entropy_coef = state["entropy_coef"] if self.cfg.entropy_floor > 0.0 else {h: self.entropy_base for h in HEADS}
        self.lr_scale = state.get("lr_scale", 1.0)
        self.lineages.extend(tuple(r) for r in state.get("lineages", ()))
        self.resign_at = state.get("resign_at")
        return True


def parse_terms(given: list[str]) -> tuple[tuple[str, float], ...]:
    """--terms' TERM:WEIGHT pairs, each a term of policy_sim_env.TERMS once."""
    import policy_sim_env
    out = tuple((name, float(w)) for name, w in (t.split(":") for t in given))
    names = [name for name, _ in out]
    if not out or len(set(names)) < len(names) or not set(names) <= set(policy_sim_env.TERMS):
        raise ValueError(f"--terms: each of {policy_sim_env.TERMS} once at most, as TERM:WEIGHT")
    return out


def _load_critic(critic: pm.Critic, state: dict, fresh: bool, was: tuple[str, ...] = (), now: tuple[str, ...] = ()) -> None:
    """A critic's weights; with fresh (other terms than it had, `was`, in `now`'s order), all but its value heads', which stay
    as a new critic's (zero).  A critic of the actor's memory has a net a term: the terms it had keep theirs whole, and the
    rest stay as new."""
    if not fresh:
        critic.load_state_dict(state)
        return
    if isinstance(critic, pm.MemoryCritic):
        own = critic.state_dict()
        for i, name in enumerate(now):
            if name in was:
                for key in [k for k in own if k.split(".")[1] == str(i)]:
                    own[key] = state[key.replace(f".{i}.", f".{list(was).index(name)}.", 1)]
        critic.load_state_dict(own)
        return
    with torch.no_grad():
        for p in critic.value.parameters():
            p.zero_()
    missing, unexpected = critic.load_state_dict({k: v for k, v in state.items() if not k.startswith("value.")}, strict=False)
    assert not unexpected and all(k.startswith("value.") for k in missing), (missing, unexpected)


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
    if how == "objective":      # the run's own weights, till a collector says a seat's own (GroupCollector._told)
        return np.zeros((len(seats), policy_latent.Z_DIM), np.float32)
    import policy_clone_data as cd
    styles = [env.styles[env.game_of(s)] for s in seats]
    return np.stack([cd.variant_z(None if st is None else st[env.team_of(s)]) for s, st in zip(seats, styles)])


def draw_objective(rng: np.random.Generator, names: tuple[str, ...], weights: np.ndarray, sigma: float, share: float,
                   bonus: float) -> tuple[np.ndarray, np.ndarray]:
    """A seat's own weights of the reward's terms for a game, and what its actor is told of them (Config.objective_sigma): each
    of the run's weights times exp(sigma x a standard normal draw), and with probability `share` one exploratory term
    (policy_sim_env.MADE_TERMS) among `names` at `bonus`.  z has a number a term, in the terms' order: the draw (so in sigmas)
    for a term the run weighs, 1 for the exploratory term that is on, 0 for the rest: all zeros is the run's own weights."""
    import policy_sim_env
    assert len(names) <= policy_latent.Z_DIM, "a number of z a term"
    draws = rng.standard_normal(len(names)) * (np.asarray(weights) > 0) * (sigma > 0)
    own = np.asarray(weights, np.float64) * np.exp(sigma * draws)
    z = np.zeros(policy_latent.Z_DIM, np.float32)
    z[:len(names)] = draws
    made = [i for i, name in enumerate(names) if name in policy_sim_env.MADE_TERMS]
    if made and rng.random() < share:
        i = made[int(rng.integers(len(made)))]
        own[i], z[i] = bonus, 1.0
    return own, z


def _z_tensor(z: np.ndarray | None, device: str) -> torch.Tensor | None:
    return None if z is None else torch.as_tensor(z, device=device)


def _stored_actions(steps: list[Step], device: str) -> pm.Actions:
    """The actions the decisions were played with, and their log-probabilities then."""
    return pm.actions_from({f: np.stack([getattr(s, f) for s in steps]) for f in act.SLOT_FIELDS + ("logp",)}, device)


def finish(trajectories: list[Trajectory], gae: float, gammas: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Each term's advantages over each seat's own decisions, by GAE at lambda `gae`, with the term's gamma a decision (1 but for
    the discounted averages: policy_sim_env.term_gamma).  Returns the values and returns, decisions x terms."""
    gammas = np.ones(1) if gammas is None else np.asarray(gammas, np.float64)
    values, returns = [], []
    for tr in trajectories:
        running = np.zeros(len(gammas))
        next_value = running if tr.done else tr.bootstrap
        for s in reversed(tr.steps):
            delta = s.reward + gammas * next_value - s.value
            running = delta + gae * gammas * running
            s.advantages = running
            s.ret = running + s.value
            next_value = s.value
            values.append(s.value)
            returns.append(s.ret)
    shape = (-1, len(gammas))
    return np.array(values, np.float64).reshape(shape), np.array(returns, np.float64).reshape(shape)


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
        self.won: dict[int, float] = {}     # each seat that ended: 1 won, 0.5 drew, 0 lost
        env.diversity = learner.cfg.diversity
        from policy_sim_env import LEAD_TERMS
        env.count_leads = any(name in LEAD_TERMS for name in learner.term_names)
        env.begin()
        self.seats = Seats(env.seat_count, learner.device, learner.actor)
        self.current: dict[int, Trajectory] = {}
        self.out: list[Segment] = []                    # this collection's segments
        self.tally = Tally.empty(learner.term_names)    # and what it counted
        for seat in env.policy_seats():
            self._open(seat)

    def _open(self, seat: int) -> None:
        """A new game for this seat: clear what it kept, draw which names it can read."""
        self.seats.clear(seat)
        tr = Trajectory(self.env.bundle_of(seat), name_map(self.rng, self.learner.vocabulary))
        tr.history, tr.seen = _kept(self.seats.history[seat], self.seats.seen[seat])
        self.current[seat] = tr

    def _told(self, seats: list[int], told: np.ndarray | None) -> np.ndarray | None:
        """What each of these seats' actors is told (z), from what the env says of them (`seat_z`)."""
        return told

    def _carry_on(self, seat: int) -> None:
        old = self.current[seat]
        tr = Trajectory(old.bundle, old.names)
        tr.history, tr.seen = _kept(self.seats.history[seat], self.seats.seen[seat])
        self.current[seat] = tr

    def _segments_of(self, seat: int) -> list[Segment]:
        """Where a seat's segments go as they're cut: into this collection."""
        return self.out

    def collect(self, want: int) -> tuple[list[Segment], dict[str, float]]:
        """Until `want` decisions of the learning policy are stored."""
        segments, tally = self.collect_tally(want)
        return segments, tally.metrics()

    def _act(self, acting: list[int]) -> list[fz.Tensors]:
        """The policy's decisions for these seats, each a step of its seat's trajectory, and their orders given.  Their tensors."""
        learner, env, cfg, seats, current, tally = self.learner, self.env, self.learner.cfg, self.seats, self.current, self.tally
        for seat in acting:
            # A slot unused till its game had more teams; or it sat out as a script, and meanwhile its game may have moved to another
            # map.
            if seat not in current or (not current[seat].steps and current[seat].bundle is not env.bundle_of(seat)):
                self._open(seat)
        frames = [(env.frame(seat), env.bundle_of(seat), current[seat].names) for seat in acting]
        index = torch.as_tensor(acting, device=learner.device)
        told = self._told(acting, seat_z(env, acting, cfg.z))
        # A pass of a seat a game at most (Config.pass_seats: or more), which a collector's part of the card is sized for: against
        # itself the policy has every player's seat of a game, 2 to 8 of them.
        items, out, value, new_history, new_seen = self.player.act(frames, seats.history[index], seats.seen[index], _z_tensor(told, learner.device),
                                                                   rows=env.count * cfg.pass_seats)
        for k, seat in enumerate(acting):
            t = items[k]
            raw = frames[k][0]
            step = Step(rf.Frame(raw.units.copy(), raw.sites.copy(), raw.glob.copy(), raw.features.copy()),
                        *(out[h][k].copy() for h in act.SLOT_FIELDS + ("logp",)), value[k].copy())
            step.z = None if told is None else told[k]
            picked = step.anchor >= 0
            tally.masked_out += int((~t.kind_mask[step.anchor[picked], step.kind[picked]]).sum())
            tally.kind_counts += np.bincount(step.kind[picked], minlength=rf.KIND_COUNT)
            tally.actions += int(picked.sum())
            if "latents" in out:
                step.latents = out["latents"][k].copy()
            tr = current[seat]
            tr.steps.append(step)
            x, z = act.world_positions(t, tr.bundle)
            env.give_orders(seat, act.decode(act.to_decision(act.Slots(*(getattr(step, h) for h in act.SLOT_FIELDS)), t.controllable), t, tr.bundle, x, z))
            # A segment ends here: the next one starts from what the actor keeps now.
            if len(tr.steps) - tr.segment_start == cfg.segment:
                _cut(tr, self._segments_of(seat), new_history[k], new_seen[k])
        seats.history[index] = new_history
        seats.seen[index] = new_seen
        return items

    def _advise(self, acting: list[int], items: list[fz.Tensors]) -> None:
        """After a step: what the scripts advised on the decisions just taken, when they advise, and their diversity bonus counted."""
        env = self.env
        if getattr(env, "advise", False):
            for k, seat in enumerate(acting):
                step = self.current[seat].steps[-1]
                step.label, step.labelled = advice_label(env, seat, items[k], self.current[seat].bundle, self.rng)
        for seat in acting:
            self.tally.bonus += float(env.bonus[env.game_of(seat), env.team_of(seat)])

    def _finished(self) -> list[int]:
        """The policy's seats whose game ended, or whose alliance went out, at the step just taken, their results counted.  Read the
        rewards first: the env starts the ended games afresh now (policy_sim_env.SimEnv.finished_seats)."""
        env, tally, out = self.env, self.tally, []
        for seat, mode, won in env.finished_seats():
            facts = getattr(env, "ended", {}).get(seat, {})
            # 1v1s under the mode alone, as they always were; other kinds apart (wins_fight_ffa).
            kind = facts.get("game_kind", "1v1")
            tally.wins.setdefault(mode if kind == "1v1" else f"{mode}_{kind}", []).append(won)
            if "lead_rule" in facts:
                tally.lead_rule.append(float(facts["lead_rule"]))
            self.won[seat] = won
            out.append(seat)
        return out

    def collect_tally(self, want: int) -> tuple[list[Segment], "Tally"]:
        """`collect`, with what it counted as a `Tally`, which several collectors' add up."""
        learner, env, cfg, seats, current = self.learner, self.env, self.learner.cfg, self.seats, self.current
        env.advise = learner.advising       # the army script's advice, for the cloning loss while it has any weight
        self.out, self.tally = [], Tally.empty(learner.term_names)
        segments, tally = self.out, self.tally
        finished: list[Trajectory] = []
        progress = learner.progress
        while tally.stored < want:
            # With act_every (policy_sim_env), the sim steps between the policy's decisions: a decision's reward is all of them.
            acting = env.policy_seats_to_act()
            items = self._act(acting) if acting else []
            tally.stored += len(acting)
            env.step(progress)
            self._advise(acting, items)
            for seat, tr in current.items():
                if tr.steps:
                    tr.steps[-1].reward = tr.steps[-1].reward + env.reward_terms(seat, learner.term_names, beta(progress))
            for seat in self._finished():
                tr = current.get(seat)
                if tr is None:      # a slot of a bigger game than any before, over before it decided
                    self._open(seat)
                    continue
                tr.done = True
                _close(tr, segments)
                finished.append(tr)
                self._open(seat)
        # Whatever is still running stops here for now.
        acting = [s for s, tr in current.items() if tr.steps]
        if acting:
            frames = [(env.frame(seat), env.bundle_of(seat), current[seat].names) for seat in acting]
            index = torch.as_tensor(acting, device=learner.device)
            _, _, value, _, _ = self.player.act(frames, seats.history[index], seats.seen[index], _z_tensor(seat_z(env, acting, cfg.z), learner.device),
                                                rows=env.count * cfg.pass_seats)
            for k, seat in enumerate(acting):
                tr = current[seat]
                tr.bootstrap = value[k].copy()
                _close(tr, segments)
                finished.append(tr)
                self._carry_on(seat)
        # The same horizon in game time whatever the decision rate.
        gammas = np.array([env.term_gamma(name) ** env.act_every for name in learner.term_names])
        tally.values, tally.returns = finish(finished, cfg.gae ** cfg.decide_every, gammas)
        tally.graphs = self.player.graphs.stats() if self.player.graphs is not None else {}
        return segments, tally


class GroupCollector(Collector):
    """GRPO's play (Config.group; 03-training.md, "GRPO").  The games go in groups, each group's games copies of one game
    (policy_sim_env.SimEnv.copy_game), with what the actor kept of each seat, played on apart by the policy for a stretch of
    game time drawn from Config.horizon.  Each seat's stretch in each game, a branch, is then scored by its terms against the same
    seat's in the group's other games, and each decision of it judged by that (`_judge`).  Then one of
    the group's games still going, drawn at random, is copied into the rest for the next stretch; when none is, a new game is.
    So every decision played is learnt from, and the games go on as the policy plays them.  A branch still being played when a
    collection ends goes to a later one: its decisions were played by the weights of their time, as concurrent collection's are."""

    def __init__(self, learner: Learner, env, rng: np.random.Generator):
        cfg = learner.cfg
        if cfg.group < 2 or env.count % cfg.group:
            raise ValueError(f"{env.count} games don't make groups of {cfg.group}")
        if cfg.bootstrap and cfg.group_scale != "terms":
            raise ValueError("--bootstrap weighs the terms as --group-scale terms does")
        self.learner, self.env, self.rng = learner, env, rng
        self.player = learner.player
        env.diversity = cfg.diversity
        import policy_sim_env
        env.count_made = any(name in policy_sim_env.MADE_TERMS for name in learner.term_names)
        env.count_leads = any(name in policy_sim_env.LEAD_TERMS for name in learner.term_names)
        env.hold_ended = True       # a copy that ends waits for the rest of its group
        env.begin()
        self.seats = Seats(env.seat_count, learner.device, learner.actor)
        self.size, groups = cfg.group, env.count // cfg.group
        self.current: dict[int, Trajectory] = {}        # each seat's branch, while it's played
        self.pending: dict[int, list[Segment]] = {}     # its segments, cut as they're played, till it's judged
        self.score: dict[int, np.ndarray] = {}          # its terms' sums so far
        self.names: list[list[np.ndarray]] = [[] for _ in range(groups)]     # each group's game's name maps, by team
        # Each group's game's objectives, by team: its own weights of the terms and what its actor is told (draw_objective),
        # the same in every copy of the game, so a seat's branches are judged by one objective.  None: the run's weights.
        self.objectives: list[list[tuple[np.ndarray, np.ndarray]]] | None = [[] for _ in range(groups)] if cfg.z == "objective" else None
        self.left = np.zeros(groups, np.int64)          # the steps each group's stretch has left: from 0, one starts now
        self.span = np.ones(groups, np.int64)           # and its steps in all
        self.terms = learner.term_names
        # A branch's economy is the income balance summed over it, over the stretch's length: its mean, if in play throughout.
        # Bootstrapped (Config.bootstrap), each term is as the critic values it, the economy a discounted average of the
        # balance, and each is summed discounted (policy_sim_env.term_gamma), so that the critic's value of the end adds on.
        self.boot = cfg.bootstrap
        # What's said of the economy there goes for each term that is a level's average (policy_sim_env.LEVEL_TERMS).
        levels = policy_sim_env.LEVEL_TERMS
        self.asked = self.terms if self.boot else tuple(levels.get(name, name) for name in self.terms)
        self.means = np.array([name in levels and not self.boot for name in self.terms])
        self.gammas = np.array([env.term_gamma(name) if self.boot else 1.0 for name in self.terms])
        self.level_at = [i for i, name in enumerate(self.terms) if name in levels]       # where Step.level has a level
        self.level_names = tuple(levels[self.terms[i]] for i in self.level_at)
        self.change = cfg.economy_change and bool(self.level_at) and (cfg.reward_to_go or self.boot)
        self.counted: dict[int, int] = {}               # each seat's steps summed in its score
        self.won: dict[int, float] = {}
        # Resignation (Config.resign): the critic (the learner's, or the collector's copy of it), whether to measure, the
        # threshold, and each group's game: played out, and the least the critic gave it.  Its own generator, so that the
        # games' draws are as they'd be without it.
        self.critic = getattr(learner, "critic", None) if cfg.resign else None
        self.measuring, self.resign_at = False, None
        self.free = np.ones(groups, bool)
        self.low = np.full(groups, np.inf)
        self.resign_rng = np.random.default_rng((cfg.seed, 7)) if cfg.resign else None
        self.out = []
        self.tally = Tally.empty(self.terms)

    def _open(self, seat: int) -> None:
        """A branch for this seat: from what the actor kept of it, as copied with its game, reading its game's names."""
        g, team = divmod(seat, self.env.slots)
        tr = Trajectory(self.env.bundle_of(seat), self.names[g // self.size][team])
        tr.history, tr.seen = _kept(self.seats.history[seat], self.seats.seen[seat])
        self.current[seat], self.pending[seat] = tr, []
        self.score[seat], self.counted[seat] = np.zeros(len(self.terms)), 0

    def _segments_of(self, seat: int) -> list[Segment]:
        return self.pending[seat]

    def _told(self, seats: list[int], told: np.ndarray | None) -> np.ndarray | None:
        if self.objectives is not None:
            for i, seat in enumerate(seats):
                g, team = divmod(seat, self.env.slots)
                told[i] = self.objectives[g // self.size][team][1]
        return told

    def collect_tally(self, want: int) -> tuple[list[Segment], "Tally"]:
        """Until `want` decisions of judged branches are stored."""
        learner, env = self.learner, self.env
        env.advise = learner.advising
        self.out, self.tally = [], Tally.empty(self.terms)
        progress = learner.progress
        b = beta(progress)
        while self.tally.stored < want:
            if env.slot % env.act_every == 0:       # stretches end and start where every seat decides
                for k in np.flatnonzero(self.left <= 0):
                    self._branch(int(k))
            acting = env.policy_seats_to_act()
            items = self._act(acting) if acting else []
            for seat in acting:         # where each decision's branch stood: what comes after judges it (Config.reward_to_go)
                step = self.current[seat].steps[-1]
                step.slot, step.mark, step.counted = env.slot, self.score[seat].copy(), self.counted[seat]
                if self.change:
                    step.level = np.zeros(len(self.terms))
                    step.level[self.level_at] = env.reward_terms(seat, self.level_names, b)
            env.step(progress)
            self.left -= 1
            self._advise(acting, items)
            for seat, tr in self.current.items():
                if not tr.done:
                    self.score[seat] += self.gammas ** self.counted[seat] * env.reward_terms(seat, self.asked, b)
                    self.counted[seat] += 1
            for seat in self._finished():
                if seat in self.current:
                    self.current[seat].done = True
        self.tally.graphs = self.player.graphs.stats() if self.player.graphs is not None else {}
        return self.out, self.tally

    def _branch(self, k: int) -> None:
        """Group k's stretch ends: its branches are judged, and the next starts from one of its games still going for the policy,
        drawn at random, copied into the others; or from a new game, when none is."""
        env, size, slots = self.env, self.size, self.env.slots
        games = list(range(k * size, (k + 1) * size))
        alive = [s for s, tr in self.current.items() if not tr.done and env.game_of(s) in games]
        going = sorted({env.game_of(s) for s in alive if not env.waiting[env.game_of(s)]})
        self._judge(k)
        results = [self.won.pop(s) for s in [s for s in self.won if env.game_of(s) in games]]
        src = int(self.rng.choice(going)) if going else -1
        if src >= 0 and self.measuring:
            # Resignation (Config.resign): the critic's result for the policy's side of the game the group would go on from.
            value = self._value(k, [s for s in alive if env.game_of(s) == src])
            self.low[k] = min(self.low[k], value)
            if not self.free[k] and self.resign_at is not None and value < self.resign_at:
                self.tally.add("resigned", 1)
                src = -1
        elif src < 0 and self.free[k] and np.isfinite(self.low[k]) and results:
            # A game played out to its end: the least it was given, and whether it wasn't lost (of its copies' ends, the share).
            self.tally.lineages.append((float(self.low[k]), float(np.mean([w >= 0.5 for w in results]))))
        if src < 0:
            src = games[0]
            if self.critic is not None:
                self.free[k] = self.resign_at is None or self.resign_rng.random() < self.learner.cfg.resign_free
                self.low[k] = np.inf
            env.restart(src)
            for t in range(slots):
                self.seats.clear(src * slots + t)
            self.names[k] = [name_map(self.rng, self.learner.vocabulary) for _ in range(slots)]
            if self.objectives is not None:
                cfg = self.learner.cfg
                self.objectives[k] = [draw_objective(self.rng, self.terms, self.learner.term_weights, cfg.objective_sigma, cfg.explore_share,
                                                     cfg.explore_weight) for _ in range(slots)]
            self.tally.add("new_games", 1)
        mine = slice(src * slots, (src + 1) * slots)
        for dst in games:
            if dst != src:
                env.copy_game(src, dst)
                theirs = slice(dst * slots, (dst + 1) * slots)
                self.seats.history[theirs] = self.seats.history[mine]
                self.seats.seen[theirs] = self.seats.seen[mine]
        decision = env.seconds_a_step * env.act_every
        low, high = (max(1, round(s / decision)) for s in self.learner.cfg.horizon)
        self.left[k] = self.span[k] = env.act_every * int(self.rng.integers(low, max(low, high) + 1))

    @torch.no_grad()
    def _value(self, k: int, seats: list[int]) -> float:
        """The critic's result for the least of these seats of group k's game, where it stands now."""
        env = self.env
        items = self.player._features([(env.frame(s), env.bundle_of(s), self.names[k][s % env.slots]) for s in seats])
        with self.player.autocast:
            values = self.critic(pm.collate(items).to(self.learner.device))
        return float(values[:, self.terms.index("result")].float().min())

    def _judge(self, k: int) -> None:
        """Group k's branches, each seat's against the same seat's in the group's other games.  A branch's terms are each one's
        sum over it, the economy's over the stretch's length.  With Config.reward_to_go a decision is judged on its branch's terms
        from it on, less the group's mean of theirs from the same moment (a branch whose game had ended by then has nothing more);
        else on the whole stretch's, less the group's mean; the economy from it on as the change from the income balance when
        it was taken (Config.economy_change).  Those are its `advantages`, by term; its `advantage` is them weighed
        (Config.terms): by the learner over each term's spread in the iteration (Config.group_scale "terms",
        Learner.weigh_groups), or here, over the group's spread of weighed scores ("group") or not ("none").  A seat with one
        branch, or whose branches all scored alike, teaches nothing: its branches are dropped.  Bootstrapped (Config.bootstrap),
        each decision's terms to the stretch's end and their discount to it are kept, and where each branch ended, for the
        learner to add the critic's value of that and judge them (Learner.bootstrap); only a seat with one branch is dropped."""
        tally, slots, cfg = self.tally, self.env.slots, self.learner.cfg
        per = np.where(self.means, 1.0 / max(int(self.span[k]), 1), 1.0)
        teams: dict[int, list[int]] = {}
        ends: dict[int, rf.Frame] = {}
        kept: dict[int, np.ndarray] = {}
        for seat in [s for s in self.current if s // slots // self.size == k]:
            tr = self.current.pop(seat)
            if self.boot and not tr.done:
                raw = self.env.frame(seat)
                ends[seat] = rf.Frame(raw.units.copy(), raw.sites.copy(), raw.glob.copy(), raw.features.copy())
                if cfg.critic_size == "memory":
                    kept[seat] = self.seats.history[seat].cpu().numpy()
            _close(tr, self.pending[seat])
            teams.setdefault(seat % slots, []).append(seat)
        for members in teams.values():
            # The team's own objective this game (draw_objective), the same in each copy: its branches are judged by it.
            own = None if self.objectives is None else self.objectives[k][members[0] % slots][0]
            weights = self.learner.term_weights if own is None else own
            finals = np.stack([self.score.pop(s) for s in members]) * per       # branches x terms
            counts = [self.counted.pop(s) for s in members]
            segments = [self.pending.pop(s) for s in members]
            spreads, spread = finals.std(0), float((finals @ weights).std())
            if len(members) < 2 or (not self.boot and (not (spreads > 1e-9).any() or (cfg.group_scale == "group" and spread < 1e-9))):
                tally.add("dropped", len(members))
                continue
            steps = [[step for seg in segs for step in seg.steps] for segs in segments]
            if self.boot:
                for seat, final, count, segs, branch in zip(members, finals, counts, segments, steps):
                    for step in branch:
                        step.rest = (final - step.mark) / self.gammas ** step.counted
                        step.discount = self.gammas ** (count - step.counted)
                    mark = int(self.rng.integers(1 << 62))
                    for seg in segs:
                        seg.branch, seg.members = mark, len(members)
                    if segs:
                        segs[-1].end, segs[-1].end_history = ends.get(seat), kept.get(seat)
            elif cfg.reward_to_go:
                ahead = [[final - (step.mark + self.means * step.level * (count - step.counted)) * per for step in branch]
                         for final, count, branch in zip(finals, counts, steps)]
                mean: dict[int, np.ndarray] = {}
                for branch, rests in zip(steps, ahead):
                    for step, rest in zip(branch, rests):
                        mean[step.slot] = mean.get(step.slot, 0.0) + rest / len(members)
                for branch, rests in zip(steps, ahead):
                    for step, rest in zip(branch, rests):
                        step.advantages = rest - mean[step.slot]
            else:
                for final, branch in zip(finals, steps):
                    for step in branch:
                        step.advantages = final - finals.mean(0)
            scale = spread if cfg.group_scale == "group" else 1.0
            group = int(self.rng.integers(1 << 62))
            if own is not None:     # the branches judged with an exploratory term on (Config.explore_share)
                tally.add("explore_branches", len(members) * float((own[self.learner.term_weights == 0] > 0).any()))
            for segs, branch in zip(segments, steps):
                for step in branch if not self.boot else ():
                    step.advantage = float(step.advantages @ weights) / scale      # "terms": the learner weighs them again
                for seg in segs:
                    seg.group, seg.spreads, seg.objective = group, spreads, own
                    self.out.append(seg)
                    tally.stored += len(seg.steps)
            tally.add("groups", 1)
            tally.add("branches", len(members))
            tally.add("branch_decisions", sum(len(seg.steps) for segs in segments for seg in segs))
            tally.add("spread", spread)
            for name, x in zip(self.terms, spreads):
                tally.add(f"spread_{name}", float(x))


@dataclass
class Tally:
    """What a collection counted, before it's metrics: several collectors' add up (`merge`)."""
    stored: int
    masked_out: int
    actions: int
    bonus: float
    kind_counts: np.ndarray
    wins: dict[str, list[float]]
    values: np.ndarray              # decisions x terms
    returns: np.ndarray
    graphs: dict[str, float]        # policy_graphs.Graphs.stats, so far
    lead_rule: list[float] = field(default_factory=list)    # 1 for each game ended that a lead could have ended, else 0
    terms: tuple[str, ...] = ("blend",)
    groups: dict[str, float] = field(default_factory=dict)  # GRPO's sums (GroupCollector._judge)
    lineages: list = field(default_factory=list)            # games played out (GroupCollector._branch, Config.resign)

    @staticmethod
    def empty(terms: tuple[str, ...]) -> "Tally":
        none = np.zeros((0, len(terms)))
        return Tally(0, 0, 0, 0.0, np.zeros(rf.KIND_COUNT, np.int64), {}, none, none, {}, [], tuple(terms))

    def add(self, name: str, value: float) -> None:
        self.groups[name] = self.groups.get(name, 0.0) + value

    @staticmethod
    def merge(tallies: list["Tally"]) -> "Tally":
        wins: dict[str, list[float]] = {}
        graphs: dict[str, float] = {}
        groups: dict[str, float] = {}
        for t in tallies:
            for mode, results in t.wins.items():
                wins.setdefault(mode, []).extend(results)
            for into, part in ((graphs, t.graphs), (groups, t.groups)):
                for k, v in part.items():
                    into[k] = into.get(k, 0) + v
        return Tally(sum(t.stored for t in tallies), sum(t.masked_out for t in tallies), sum(t.actions for t in tallies),
                     sum(t.bonus for t in tallies), np.sum([t.kind_counts for t in tallies], 0), wins,
                     np.concatenate([t.values for t in tallies]), np.concatenate([t.returns for t in tallies]), graphs,
                     [x for t in tallies for x in t.lead_rule], tallies[0].terms, groups, [x for t in tallies for x in t.lineages])

    def metrics(self) -> dict[str, float]:
        """The first term's explained variance and returns' range without a suffix, as with one term; with more, each term's too.
        With GRPO, none of those: the branches judged and dropped, the games started, and each group's spread of scores on
        average, the weighed score's (group_spread) and each term's."""
        stored = self.stored
        metrics = {"masked_out": self.masked_out, "decisions": stored, "actions_a_decision": self.actions / max(stored, 1),
                   "diversity_a_decision": self.bonus / max(stored, 1)}
        for i, name in enumerate(self.terms if len(self.values) else ()):
            v, r = self.values[:, i], self.returns[:, i]
            suffix = f"_{name}" if i else ""
            metrics.update({f"explained_variance{suffix}": 1.0 - float(np.var(r - v)) / max(float(np.var(r)), 1e-8),
                            f"return_min{suffix}": float(r.min()), f"return_max{suffix}": float(r.max())})
        g = self.groups
        if g:
            metrics.update(branches=g.get("branches", 0), branches_dropped=g.get("dropped", 0), new_games=g.get("new_games", 0),
                           resigned=g.get("resigned", 0))
            if "explore_branches" in g:     # the share of them judged with an exploratory term on (Config.z "objective")
                metrics["explore_branches"] = g["explore_branches"] / max(g.get("branches", 0), 1)
            if g.get("groups"):
                metrics["group_spread"] = g["spread"] / g["groups"]
                metrics["branch_decisions"] = g["branch_decisions"] / g["branches"]
                metrics.update({f"group_spread_{name}": g[f"spread_{name}"] / g["groups"] for name in self.terms})
        total = max(int(self.kind_counts.sum()), 1)
        metrics.update({f"share_kind{k}": float(self.kind_counts[k]) / total for k in range(rf.KIND_COUNT) if self.kind_counts[k]})
        for mode, results in self.wins.items():
            metrics[f"wins_{mode}"] = float(np.mean(results))
            metrics[f"games_{mode}"] = len(results)
        if self.lineages:       # not a number: the learner takes it out (Learner.follow_resignation)
            metrics["lineages"] = list(self.lineages)
        if self.lead_rule:      # of the training games that ended, the share a lead could have ended: lead_share, as drawn
            metrics["lead_rule_games"] = float(np.mean(self.lead_rule))
        metrics.update(self.graphs)
        if self.graphs.get("graph_replays"):
            metrics["graph_steps_a_round"] = self.graphs["graph_steps"] / self.graphs["graph_replays"]
        return metrics


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
def evaluate(learner: Learner, make_env, opponent: str, games: int, seed: int, replays: dict | None = None, kind: str = "1v1") -> dict[str, float]:
    """The current policy (the moving average of its weights, `Config.ema`) against a script, sampled as in training, in games of
    one kind (policy_sim_env.GAME_KINDS), the policy on team 0: a win is its alliance's result 1, as a loss is its alliance out.
    Each game slot counts its first game only: the games that end soonest would otherwise count most, and those are not a fair
    sample.  replays: policy_replay.Capture's arguments, to keep every game for the dashboard.  Each game's record says its slot
    and the seed, so that tests of one seed pair game for game (`score_test`); its alliances' metal income summed over the game,
    income_own and income_enemy; and the economy at a few minutes (policy_clone_eval.Notes)."""
    from policy_clone_eval import Notes        # which imports this
    player = learner.evaluator
    # A 1v1 test in a batch of 1v1s, whatever slots the training's games have.
    env = make_env(games, seed, modes=[opponent], kinds={kind: 1.0}, **({"slots": 0} if kind == "1v1" else {}))
    env.policy_economy_share = learner.economy_share
    env.advise = False          # no cloning loss learns from a test
    env.begin()
    if replays is not None:
        import policy_replay
        env.capture = policy_replay.Capture(env, games=games, **replays)
    seats = Seats(env.seat_count, learner.device, learner.actor)
    names = np.arange(learner.vocabulary, dtype=np.int32)
    results: dict[int, float] = {}
    records: dict[int, dict] = {}
    notes = Notes(env)
    income = np.zeros((env.count, 2))
    metal = rf.G["metal_income"]
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
        notes.after_step()
        for g in range(env.count):
            if g * env.slots not in results:        # the policy's seat, team 0's
                n = int(env.n_teams[g])
                ally, flow = env.ally[g][:n], env.batch.globs[g, :n, metal]
                income[g] += (float(flow[ally == ally[0]].sum()), float(flow[ally != ally[0]].sum()))
        for seat, _, won in env.finished_seats():
            if seat not in results and seat in getattr(env, "ended", {}):
                g = env.game_of(seat)
                records[seat] = {**env.ended[seat], "slot": seat, "seed": seed, "income_own": float(income[g, 0]),
                                 "income_enemy": float(income[g, 1]), "minutes": dict(notes.minutes[g])}
                notes.finished([(seat, "", won)])
            results.setdefault(seat, won)
            seats.clear(seat)
    r = np.array(list(results.values()), np.float32)
    return {"win_rate": float((r == 1).mean()), "draw_rate": float((r == 0.5).mean()), "loss_rate": float((r == 0).mean()), "games": len(r), "records": list(records.values())}


# A test game's score (the user, 2026-11-14): its result, and its alliance's shares of the value at the end, of the metal
# income over the game and of the damage done, each 0 to 1 and 0.5 when even, weighed as GRPO's terms are (result, score,
# economy, combat).
SCORE_PARTS = (("result", 1.0), ("value_share", 0.5), ("income_share", 0.3), ("damage_share", 0.3))


def _share(own: float, other: float) -> float:
    return own / (own + other) if own + other > 0.0 else 0.5


def game_score(record: dict) -> dict[str, float]:
    """A test game's parts (SCORE_PARTS) and their weighed mean, `score`."""
    parts = {"result": {"won": 1.0, "drew": 0.5}.get(record["result"], 0.0),
             "value_share": _share(record["own_value"], record["enemy_value"]),
             "income_share": _share(record.get("income_own", 0.0), record.get("income_enemy", 0.0)),
             "damage_share": _share(record.get("damage_dealt", 0.0), record.get("damage_taken", 0.0))}
    parts["score"] = sum(w * parts[k] for k, w in SCORE_PARTS) / sum(w for _, w in SCORE_PARTS)
    return parts


def score_test(name: str, records: list[dict], first: list[dict] | None) -> dict[str, float]:
    """A test's metrics beside its rates: its games' score and parts on average (`game_score`), the score's standard error,
    the economy at policy_clone_eval.MINUTES, and, against the run's first test (`first`), the score's change game for game
    where the two played the same game (seed and slot), with its standard error: what a map or a start makes of a game's
    score cancels out."""
    from policy_clone_eval import MINUTES
    if not records:
        return {}
    scores = [game_score(r) for r in records]
    out = {f"{name}_{k}": float(np.mean([s[k] for s in scores])) for k in scores[0]}
    out[f"{name}_score_se"] = float(np.std([s["score"] for s in scores], ddof=1) / math.sqrt(len(scores))) if len(scores) > 1 else 0.0
    for m in MINUTES:
        for who, tag in (("actor", ""), ("script", "_script")):
            rows = [r["minutes"][m][who] for r in records if m in r.get("minutes", {})]
            for key in ("income", "extractors") if rows else ():
                out[f"{name}{tag}_{key}_at{m}"] = float(np.mean([x[key] for x in rows]))
    before = {(r["seed"], r["slot"]): game_score(r)["score"] for r in first or () if "seed" in r}
    change = [s["score"] - before[r["seed"], r["slot"]] for r, s in zip(records, scores) if (r.get("seed"), r.get("slot")) in before]
    if len(change) > 1:
        out[f"{name}_score_change"] = float(np.mean(change))
        out[f"{name}_score_change_se"] = float(np.std(change, ddof=1) / math.sqrt(len(change)))
        out[f"{name}_paired"] = len(change)
    return out


def first_test(run: pathlib.Path, opponent: str, kind: str | None) -> list[dict] | None:
    """The games of the run's first test against this opponent, of this kind (None: the headline 1v1's), that says its seeds."""
    path = run / "tests.jsonl"
    for line in path.read_text().splitlines() if path.exists() else ():
        row = json.loads(line)
        if row["opponent"] == opponent and row.get("kind") == kind and any("seed" in g for g in row["games"]):
            return row["games"]
    return None


_MAP_POOLS: dict = {}


def make_sim(games: int, seed: int, modes: list[str] | None = None, decide_every: int = 1, maps: str = "", variants: bool = False,
             advise: bool = False, income_weight: float = 0.0, fine_sim: bool = False, threads: int = 16, speed_weight: float = 0.0,
             kinds: dict[str, float] | None = None, slots: int = 0, difficulties: dict[str, float] | None = None):
    """maps: a map pool, loaded once and shared, its held-out maps left out; "" for the five training maps.  threads: the sim's
    and the C++ scripts' each.  The lead rule is on in every game (lead_share 1): the trainer lowers it for its own games.
    kinds, slots, difficulties: policy_sim_env.SimEnv's kinds, teams (0: what the kinds need) and chicken difficulties; 1v1s and
    the default difficulties if None."""
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
    env = policy_sim_env.SimEnv(games, seed, modes or ["fight"], threads=threads, decide_every=1 if fine_sim else decide_every, maps=pool,
                                weights=weights, variants=variants, teams=slots or None, kinds=kinds, difficulties=difficulties or None)
    env.advise, env.income_weight, env.speed_weight = advise, income_weight, speed_weight
    env.act_every = decide_every if fine_sim else 1
    env.align = env.act_every > 1           # every seat decides on the same slots: one forward pass a decision (SimEnv.align)
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
    # Every env the run makes plays its kinds of game, the collectors' too; a test says its own (`evaluate`).
    make_env = functools.partial(make_env, kinds=dict(cfg.kinds), slots=cfg.slots, difficulties=dict(cfg.chicken_difficulties))
    # With collector processes this one plays nothing: one game's env is enough for the map pool's bundles, and with no threads of
    # its own it has none running when they fork.
    env = make_env(1, cfg.seed, threads=1) if cfg.collectors else make_env(cfg.games, cfg.seed)
    # Separate pools: training queues a minibatch's chunks at once, and collection mustn't wait behind them.  Both before the GPU.
    # A thread each where there are processes: a minibatch's chunks are spread over them already.
    threads = min(cfg.featurize_threads, 1) if cfg.data_workers else cfg.featurize_threads
    workers = DataWorkers(cfg.data_workers, env.bundles(), cfg.enabled_kinds, threads)
    act_workers = DataWorkers(cfg.data_workers, env.bundles(), cfg.enabled_kinds, threads) if cfg.concurrent and not cfg.collectors else None
    remote = None
    if cfg.collectors:
        import policy_collectors
        remote = policy_collectors.Collectors(cfg, make_env, env.bundles())     # forked before the GPU, as the workers are
    learner = Learner(cfg, run, workers, device)
    if learner.resume():
        print(f"resumed at iteration {learner.iteration}, {learner.decisions} decisions", flush=True)
        if remote is None:
            env = make_env(cfg.games, cfg.seed + learner.iteration)
    rng = np.random.default_rng(cfg.seed + learner.iteration)
    learner.economy_share = cfg.economy_start + (cfg.economy_end - cfg.economy_start) * learner.progress
    if remote is None:
        set_games(env, learner.games)       # from the first games, which the collector starts
        collector = (GroupCollector if cfg.group else Collector)(learner, env, np.random.default_rng((cfg.seed, learner.iteration)))
    else:
        remote.start(learner, cfg.seed + learner.iteration, device)
    frozen = learner.frozen_player(act_workers) if act_workers else None
    ahead: Ahead | None = None
    collect_s = train_s = 0.0
    log = (run / "metrics.jsonl").open("a")
    began = last_replays = time.perf_counter()
    test = remote.evaluate if remote is not None else functools.partial(evaluate, make_env=make_env)

    def tests() -> dict[str, float]:
        """The headline 1v1s against each opponent and a test of each of Config.test_kinds, of the same games every time."""
        out = {}
        cases = [(f"eval_{o}", o, None) for o in env.eval_opponents] + [(f"eval_fight_{k}", "fight", k) for k in cfg.test_kinds]
        for name, opponent, kind in cases:
            first = first_test(run, opponent, kind)
            tested = time.perf_counter()
            result = test(learner, opponent=opponent, games=cfg.eval_games, seed=cfg.seed * 1000, **({"kind": kind} if kind else {}))
            out[name] = result["win_rate"]
            out[f"{name}_draws"], out[f"{name}_losses"] = result["draw_rate"], result["loss_rate"]
            out.update(score_test(name, result["records"], first))
            out[f"{name}_seconds"] = time.perf_counter() - tested
            with (run / "tests.jsonl").open("a") as file:
                file.write(json.dumps({"iteration": learner.iteration, "decisions": learner.decisions, "opponent": opponent,
                                       **({"kind": kind} if kind else {}), "games": result["records"]}) + "\n")
        return out

    if learner.iteration == 0 and cfg.eval_every * cfg.iteration <= cfg.decisions:
        # A run that tests at all tests what it starts from too: the games its later tests are paired with.
        first = {"iteration": 0, "total_decisions": 0, **tests()}
        log.write(json.dumps(first) + "\n")
        log.flush()
        print(f"it 0 {({k: round(v, 3) for k, v in first.items() if k.startswith('eval_') and isinstance(v, float)})}", flush=True)
    while learner.decisions < cfg.decisions:
        learner.economy_share = cfg.economy_start + (cfg.economy_end - cfg.economy_start) * learner.progress
        # The training games' only: a test makes its own, 1v1s against the script with the lead rule in every game.  Collectors are
        # sent them with each collection.
        now = learner.games
        set_games(env, now)
        if remote is None and cfg.group:
            collector.measuring, collector.resign_at = learner.resign_state
        t0 = time.perf_counter()
        if remote is not None:
            if not remote.collecting:
                remote.collect(learner, cfg.iteration)
            segments, metrics, collect_s = remote.result()
            if cfg.concurrent:      # the next iteration's, by the weights before this one's update, while it trains
                remote.collect(learner, cfg.iteration)
        elif ahead is None:
            segments, metrics = collector.collect(cfg.iteration)
            collect_s = time.perf_counter() - t0
        else:
            segments, metrics, collect_s = ahead.join()
        t1 = time.perf_counter()
        if cfg.concurrent and remote is None:
            collector.player = frozen
            ahead = Ahead(collector, frozen, cfg.iteration, max(0.0, train_s - collect_s))
        lineages = metrics.pop("lineages", [])
        if cfg.resign:
            metrics.update(learner.follow_resignation(lineages))
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
                       beta=beta(learner.progress), entropy_coef=dict(learner.entropy_coef), lead_share=now["lead_share"],
                       self_play_share=now["self_play_share"])
        if now["other_kinds_share"] is not None:
            metrics["other_kinds_share"] = now["other_kinds_share"]
        if device == "cuda":        # the most this process's allocator held while training, for sizing gpu_share's parts
            metrics.setdefault("learner_reserved_gb", torch.cuda.max_memory_reserved() / 2**30)
            metrics.setdefault("learner_allocated_gb", torch.cuda.max_memory_allocated() / 2**30)
        if metrics["masked_out"]:
            raise RuntimeError(f"{metrics['masked_out']} chosen actions were outside their mask")
        if learner.iteration % cfg.eval_every == 0:
            metrics.update(tests())
        if cfg.replay_hours and time.perf_counter() - last_replays >= cfg.replay_hours * 3600.0:
            tested = time.perf_counter()
            hours = (tested - began) / 3600.0
            result = test(learner, opponent="fight", games=cfg.replay_games, seed=cfg.seed * 1000 + learner.iteration + 500, replays={
                "out": cfg.replays or run / "replays", "stage": "ppo", "label": "PPO against the army script",
                "prefix": f"it{learner.iteration:05d}-", "info": {"iteration": learner.iteration, "decisions": learner.decisions, "hours": round(hours, 2)}})
            metrics["replays_win_rate"], metrics["replays_seconds"] = result["win_rate"], time.perf_counter() - tested
            last_replays = time.perf_counter()
        if learner.iteration % cfg.checkpoint_every == 0:
            learner.save()
        log.write(json.dumps(metrics) + "\n")
        log.flush()
        show = {k: round(v, 3) for k, v in metrics.items() if isinstance(v, float) and (k.startswith(("wins_", "eval_", "entropy_", "share_kind")) or k in ("explained_variance", "group_spread", "kl", "clip_fraction", "collect_rate", "train_rate", "overall_rate", "minibatch_decisions", "adv_abs_max", "ratio_deviation_max"))}
        print(f"it {learner.iteration} decisions {learner.decisions} {show}", flush=True)
        if (run / "stop").exists() or (cfg.hours and time.perf_counter() - began >= cfg.hours * 3600.0):
            break
    if ahead:
        ahead.join()
    learner.save()
    workers.close()
    if act_workers:
        act_workers.close()
    if remote is not None:
        remote.close()


def check_repro(cfg: Config, run: pathlib.Path, make_env) -> None:
    """The same iteration twice from the same seed: 03-training.md, "Checkpoints and stopping"."""
    # Plain GPU scatter adds accumulate in no fixed order, and Adam magnifies the rounding on near-zero gradients.
    torch.use_deterministic_algorithms(True, warn_only=True)
    make_env = functools.partial(make_env, kinds=dict(cfg.kinds), slots=cfg.slots, difficulties=dict(cfg.chicken_difficulties))
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


def grad_check(cfg: Config, run: pathlib.Path, make_env, decisions: int, splits: int = 3, device: str = "cuda") -> None:
    """Whether GRPO's gradient is more signal than noise (03-training.md, "Signal or noise"): `decisions` played by the run's
    checkpoint, or by Config.init (with --bootstrap, a checkpoint's critic bootstraps the advantages), their groups split in
    two at random `splits` times, and the halves' gradients (`Learner.gradient`, the actor's) compared by their cosine: with
    the advantages as trained (all), with each term's alone (its share of them), and with them shuffled among the decisions,
    a floor for what the halves share whatever the advantages.  No entropy bonus.  To run/grad-check.json."""
    if not cfg.group:
        raise SystemExit("--grad-check is GRPO's: --group")
    make_env = functools.partial(make_env, kinds=dict(cfg.kinds), slots=cfg.slots, difficulties=dict(cfg.chicken_difficulties))
    env = make_env(cfg.games, cfg.seed)
    workers = DataWorkers(cfg.data_workers, env.bundles(), cfg.enabled_kinds)
    learner = Learner(cfg, run, workers, device)
    learner.resume()
    learner.entropy_coef = {h: 0.0 for h in HEADS}
    learner.economy_share = cfg.economy_start + (cfg.economy_end - cfg.economy_start) * learner.progress
    set_games(env, learner.games)
    rng = np.random.default_rng(cfg.seed)
    segments, metrics = GroupCollector(learner, env, np.random.default_rng((cfg.seed, learner.iteration))).collect(decisions)
    scales = learner.prepare(segments)
    steps = [s for seg in segments for s in seg.steps]
    terms = np.stack([np.broadcast_to(s.advantages, scales.shape) for s in steps]) * (learner.term_weights / scales)
    cases = {"all": terms.sum(1), **{name: terms[:, i] for i, name in enumerate(learner.term_names)}, "shuffled": rng.permutation(terms.sum(1))}
    groups = np.array(sorted({seg.group for seg in segments}))
    rows = []
    for k in range(splits):
        picked = set(rng.permutation(groups)[:len(groups) // 2].tolist())
        halves = [[seg for seg in segments if (seg.group in picked) == side] for side in (True, False)]
        row = {}
        for case, advantages in cases.items():
            for s, a in zip(steps, advantages):
                s.advantage = float(a)
            a, b = (learner.gradient(half, rng) for half in halves)
            row[case] = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
            row[f"{case}_norms"] = [float(a.norm()), float(b.norm())]
        rows.append(row)
        print(json.dumps({"split": k, **{c: round(row[c], 4) for c in cases}}), flush=True)
    out = {"iteration": learner.iteration, "decisions": len(steps), "groups": len(groups), "scales": dict(zip(learner.term_names, scales.tolist())),
           "mean": {c: float(np.mean([r[c] for r in rows])) for c in cases}, "splits": rows,
           "collected": {k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
    run.mkdir(parents=True, exist_ok=True)
    (run / "grad-check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["mean"]), flush=True)
    workers.close()


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
    ap.add_argument("--featurize-threads", type=int, default=Config.featurize_threads,
                    help="featurize in C++ (libzkpolicy) on this many threads in each collector and one in each data worker; 0: in Python")
    ap.add_argument("--hours", type=float, default=Config.hours, help="stop after this long as well as at --decisions")
    ap.add_argument("--eval-every", type=int, default=Config.eval_every, help="iterations between tests against the script")
    ap.add_argument("--latent-encoder", default=Config.latent_encoder, help="the trained frame encoder whose latents the actor reads")
    ap.add_argument("--action-kinds", help="action kinds to allow, such as 0,3: a test of one head with nothing else to fall back on")
    ap.add_argument("--kinds", nargs="+", default=[f"{k}:{w:g}" for k, w in Config.kinds], metavar="KIND:WEIGHT",
                    help="kinds of game and how likely each, among 1v1, teams, ffa and chicken, as 1v1:0.5 teams:0.2 ffa:0.2 chicken:0.1 "
                         "(default: %(default)s)")
    ap.add_argument("--kinds-ramp", nargs="*", default=[f"{i}:{s:g}" for i, s in Config.kinds_ramp], metavar="ITERATION:SHARE",
                    help="the share of training games that aren't 1v1s, by iteration, as --kl-fade: split among --kinds' other kinds "
                         "as they're weighed there, the rest 1v1s (500:0 2500:0.5: none till iteration 500, half from 2500); "
                         "without it, --kinds' own weights throughout")
    ap.add_argument("--self-play", type=float, default=Config.self_play,
                    help="the share of training games the policy plays against itself, every player's team of them; the rest are "
                         "against the scripts, as every test is")
    ap.add_argument("--slots", type=int, default=Config.slots, help="teams a game has room for; 0: what the biggest of --kinds needs")
    ap.add_argument("--test-kinds", nargs="*", default=list(Config.test_kinds),
                    help="kinds of game to test the policy at too, on team 0, besides the 1v1 test that's the headline")
    ap.add_argument("--chicken-difficulties", nargs="+", metavar="NAME:WEIGHT",
                    help="how likely each chicken difficulty is, as very-easy:1 easy:1 normal:1 (default policy_sim_env.CHICKEN_DIFFICULTY_WEIGHTS)")
    ap.add_argument("--gpu-runs", type=int, default=Config.gpu_runs, help="how many trainings share the card, this one included")
    ap.add_argument("--gpu-share", type=float, default=Config.gpu_share,
                    help="of the card the run may hold, its collectors' part included: the learner's limit is what they leave of it")
    ap.add_argument("--empty-cache", action=argparse.BooleanOptionalAction, default=Config.empty_cache, help="hand back what the allocator holds after each iteration")
    ap.add_argument("--init", default="", help="a cloning checkpoint to start the actor from (P7)")
    ap.add_argument("--size", choices=tuple(policy_latent.SIZES), default=Config.size,
                    help="a new actor's size, without --init: gru has a GRU for memory in place of the time transformer, "
                         "rdnn a Recurrent Divisive Normalization Network (arXiv 2608.01947) in the GRU's place")
    ap.add_argument("--maps", default="", help="a map pool, its held-out maps left out; without it, the five training maps")
    ap.add_argument("--variants", action="store_true", help="each side's army script draws its knobs (policy_variants.py)")
    ap.add_argument("--lr-start", type=float, default=Config.lr_start)
    ap.add_argument("--lr-end", type=float, default=Config.lr_end)
    ap.add_argument("--kl-target", type=float, default=Config.kl_target, help="the actor's rate slows when an iteration's KL passes twice this; 0: off")
    ap.add_argument("--compile-memory", action="store_true",
                    help="an RDNN's steps through a segment, in training, through torch.compile")
    ap.add_argument("--recurrent-lr", type=float, default=Config.recurrent_lr,
                    help="the rate of an RDNN's recurrent weights, in their own Adam; 0: the schedule's (--lr-start)")
    ap.add_argument("--lr-scale-start", type=float, default=Config.lr_scale_start,
                    help="the share of its schedule the actor's rate starts at, for the KL guard to raise or lower (a new run's; a resumed one keeps its own)")
    ap.add_argument("--critic-lr", type=float, default=Config.critic_lr, help="the critic's rate at the start, scaled as the actor's falls; 0: the actor's")
    ap.add_argument("--critic-muon-lr", type=float, default=Config.critic_muon_lr,
                    help="above 0, Muon trains the critic's hidden matrices at this rate at the start, scaled as the actor's falls")
    ap.add_argument("--actor-muon-lr", type=float, default=Config.actor_muon_lr,
                    help="above 0, Muon trains the actor's hidden matrices at this rate at the start, on the actor's schedule")
    ap.add_argument("--critic-bins", type=int, default=0, help="HL-Gauss bins for the critic (C6); 0 for one number")
    ap.add_argument("--critic-span", type=float, default=Config.critic_span)
    ap.add_argument("--entropy-floor", type=float, default=Config.entropy_floor, help="0 from a clone: no head's bonus rises")
    ap.add_argument("--critic-warmup", type=int, default=0, help="iterations where only the critic learns")
    ap.add_argument("--critic-size", choices=tuple(pm.CRITIC_SIZES), default=Config.critic_size, help="small: beside a small actor (--size gru)")
    ap.add_argument("--terms", nargs="+", default=[f"{k}:{w:g}" for k, w in Config.terms], metavar="TERM:WEIGHT",
                    help="the reward's terms, each a head of the critic, and their advantages' weights, among blend (the one reward, "
                         "faded by beta), result, score, economy and combat, as result:1 score:0.5 economy:0.3 combat:0.3 (default: %(default)s)")
    ap.add_argument("--critic-rewarm", type=int, default=Config.critic_rewarm,
                    help="after resuming with other --terms than the checkpoint's, iterations where only the critic learns")
    ap.add_argument("--group", type=int, default=Config.group,
                    help="GRPO from branched games in place of the critic, in groups of this many games; 0: PPO with its critic")
    ap.add_argument("--horizon", type=float, nargs=2, default=list(Config.horizon), metavar=("LOW", "HIGH"),
                    help="the seconds of game a branch plays, drawn evenly between (default: %(default)s)")
    ap.add_argument("--group-scale", choices=("terms", "group", "none"), default=Config.group_scale,
                    help="GRPO's advantages: each term over its spread in the iteration's groups, then weighed; the weighed score "
                         "over its group's spread (GRPO's own); or neither (Dr. GRPO's)")
    ap.add_argument("--reward-to-go", action=argparse.BooleanOptionalAction, default=Config.reward_to_go,
                    help="judge each decision on its branch's terms from it on, not on its whole stretch's")
    ap.add_argument("--group-step", type=int, default=Config.group_step,
                    help="decisions an optimizer step under GRPO, in whole groups; 0 for a step a minibatch")
    ap.add_argument("--economy-change", action=argparse.BooleanOptionalAction, default=Config.economy_change,
                    help="GRPO's economy term from each decision on as the income balance's change from then, not its level")
    ap.add_argument("--adv-clip", type=float, default=Config.adv_clip, help="GRPO: a decision's weighed advantage clipped to +- this; 0: not")
    ap.add_argument("--spo-bound", type=float, default=Config.spo_bound,
                    help="GRPO: SPO's penalty straight past |r - 1| = this x eps, so its pull stays bounded; 0: the square throughout")
    ap.add_argument("--entropy-base", type=float, default=Config.entropy_base,
                    help="a head's least entropy bonus; below 0, %g under PPO and 0 under GRPO" % ENTROPY_BASE)
    ap.add_argument("--eval-games", type=int, default=Config.eval_games, help="games a test, the same ones each time")
    ap.add_argument("--bootstrap", action=argparse.BooleanOptionalAction, default=Config.bootstrap,
                    help="GRPO: each branch's return to its stretch's end plus the critic's value of where it ended, which it learns")
    ap.add_argument("--resign", action=argparse.BooleanOptionalAction, default=Config.resign,
                    help="bootstrapped GRPO: give up a group's game where the critic calls it lost, as AlphaGo Zero, the threshold "
                         "kept to --resign-false of the --resign-free share played out")
    ap.add_argument("--resign-free", type=float, default=Config.resign_free, help="share of games never given up, but played out to measure")
    ap.add_argument("--resign-false", type=float, default=Config.resign_false, help="most share of games under the threshold not lost")
    ap.add_argument("--resign-least", type=int, default=Config.resign_least, help="played-out games under a threshold before it's used")
    ap.add_argument("--grad-check", type=int, default=0, metavar="DECISIONS",
                    help="GRPO: collect this many decisions from the run's checkpoint (or --init) and write how alike two halves' "
                         "gradients are (grad_check), then stop")
    ap.add_argument("--grad-splits", type=int, default=3, help="--grad-check: how many times the groups are split in two at random")
    ap.add_argument("--kl-start", type=float, default=0.0, help="the KL to the frozen clone's weight at iteration 0")
    ap.add_argument("--kl-fade", nargs="+", default=[f"{i}:{s:g}" for i, s in Config.kl_fade], metavar="ITERATION:SHARE",
                    help="its share of that by iteration, linear between, the last after (default: %(default)s)")
    ap.add_argument("--clone-weight", type=float, default=0.0, help="the cloning loss on the scripts' advice (they then advise), at iteration 0")
    ap.add_argument("--clone-fade", nargs="+", default=[f"{i}:{s:g}" for i, s in Config.clone_fade], metavar="ITERATION:SHARE",
                    help="its share of that by iteration, as --kl-fade (default: %(default)s)")
    ap.add_argument("--economy-start", type=float, default=0.0, help="share of games where the policy runs its own economy, at the start")
    ap.add_argument("--economy-end", type=float, default=0.0, help="and at the end")
    ap.add_argument("--income-weight", type=float, default=0.0, help="the zero-sum income reward's weight")
    ap.add_argument("--speed-weight", type=float, default=Config.speed_weight,
                    help="at a decided game's end, this x the result x the share of the time limit left, not faded by beta")
    ap.add_argument("--lead-fade", nargs="+", default=[f"{i}:{s:g}" for i, s in Config.lead_fade], metavar="ITERATION:SHARE",
                    help="the share of training games a lead can end, by iteration, as --kl-fade; tests keep it in every game "
                         "(default: %(default)s)")
    ap.add_argument("--diversity", type=float, default=Config.diversity)
    ap.add_argument("--fine-sim", action="store_true", help="the sim steps every slot, the policy decides every --decide-every")
    ap.add_argument("--objective-sigma", type=float, default=Config.objective_sigma,
                    help="with --z objective: each seat's weights of the terms are the run's times exp(this x a standard normal draw), new each game")
    ap.add_argument("--explore-share", type=float, default=Config.explore_share,
                    help="with --z objective: the share of a seat's games where one of --terms' exploratory terms (made_*, given at weight 0) is on")
    ap.add_argument("--explore-weight", type=float, default=Config.explore_weight, help="the weight of an exploratory term that is on")
    ap.add_argument("--z", choices=("none", "variant", "objective"), default=Config.z,
                    help="what the actor is told of its strategy: nothing, or its side's army-script knobs (needs --variants)")
    ap.add_argument("--replay-hours", type=float, default=Config.replay_hours,
                    help="every this many hours, --replay-games games against the script kept for the dashboard; 0 for none")
    ap.add_argument("--replay-games", type=int, default=Config.replay_games)
    ap.add_argument("--replays", default="", help="where they go (default RUN/replays)")
    ap.add_argument("--graphs", action=argparse.BooleanOptionalAction, default=Config.graphs, help="play each forward pass as a CUDA graph")
    ap.add_argument("--collectors", type=int, default=Config.collectors, help="processes that play the games and the tests; 0: this one")
    ap.add_argument("--collector-gb", type=float, default=Config.collector_gb, help="the GPU memory each collector may hold")
    ap.add_argument("--pass-seats", type=int, default=Config.pass_seats,
                    help="how many seats a game a collector's forward pass takes: 2 decides both sides of a 1v1 against itself at once, in twice the memory")
    ap.add_argument("--check-repro", action="store_true")
    ap.add_argument("--check-kill", action="store_true")
    ap.add_argument("--save-forever", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.z == "variant" and not args.variants:
        ap.error("--z variant needs --variants: without them there are no knobs to tell")
    if args.z == "objective" and not args.group:
        ap.error("--z objective draws each seat's weights of the terms for each game of a group: it needs --group")
    if args.group and (args.group < 2 or args.games % (args.group * max(args.collectors, 1))):
        ap.error(f"--group: {args.games} games don't make groups of {args.group} in each of {max(args.collectors, 1)} collectors")
    if not 0.0 <= args.self_play <= 1.0:
        ap.error("--self-play is a share of the games, 0 to 1")
    fades = {}
    for name in ("kl_fade", "clone_fade", "lead_fade", "kinds_ramp"):
        fade = fades[name] = tuple((int(i), float(s)) for i, s in (point.split(":") for point in getattr(args, name)))
        if any(a[0] >= b[0] for a, b in zip(fade, fade[1:])):
            ap.error(f"--{name.replace('_', '-')}'s iterations must rise")
    import policy_sim_env
    try:
        kinds = policy_sim_env.parse_kinds(args.kinds)
        if fades["kinds_ramp"] and not (kinds.get("1v1", 0.0) > 0.0 and any(w > 0.0 for k, w in kinds.items() if k != "1v1")):
            raise ValueError("--kinds-ramp is between 1v1s and --kinds' other kinds: --kinds needs both")
        if any(not 0.0 <= s <= 1.0 for _, s in fades["kinds_ramp"]):
            raise ValueError("--kinds-ramp's shares are of the games, 0 to 1")
        policy_sim_env.parse_kinds(args.test_kinds)
        difficulties = {policy_sim_env.zksim.CHICKEN_DIFFICULTIES[d]: w for d, w in
                        policy_sim_env.difficulty_weights(args.chicken_difficulties).items()} if args.chicken_difficulties else {}
        terms = parse_terms(args.terms)
    except ValueError as e:
        ap.error(str(e))

    cfg = Config(terms=terms, critic_rewarm=args.critic_rewarm, seed=args.seed, decisions=args.decisions, games=args.games, iteration=args.iteration, passes=args.passes, ema=args.ema,
                 segment=args.segment, token_budget=args.token_budget, data_workers=args.workers, featurize_threads=args.featurize_threads, concurrent=args.concurrent,
                 decide_every=args.decide_every, hours=args.hours, eval_every=args.eval_every, gpu_runs=args.gpu_runs, gpu_share=args.gpu_share, empty_cache=args.empty_cache,
                 latent_encoder=args.latent_encoder, enabled_kinds=policy_sim_env.KINDS,     # what the sim obeys
                 init=args.init, size=args.size, maps=args.maps, variants=args.variants, lr_start=args.lr_start, lr_end=args.lr_end, kl_target=args.kl_target, lr_scale_start=args.lr_scale_start, recurrent_lr=args.recurrent_lr, compile_memory=args.compile_memory, critic_lr=args.critic_lr, critic_muon_lr=args.critic_muon_lr, actor_muon_lr=args.actor_muon_lr,
                 critic_bins=args.critic_bins, critic_span=args.critic_span, entropy_floor=args.entropy_floor, critic_warmup=args.critic_warmup, critic_size=args.critic_size, kl_start=args.kl_start, **fades,
                 clone_weight=args.clone_weight, economy_start=args.economy_start, economy_end=args.economy_end,
                 income_weight=args.income_weight, speed_weight=args.speed_weight, diversity=args.diversity, fine_sim=args.fine_sim, z=args.z,
                 objective_sigma=args.objective_sigma, explore_share=args.explore_share, explore_weight=args.explore_weight,
                 replay_hours=args.replay_hours, replay_games=args.replay_games, replays=args.replays, graphs=args.graphs,
                 collectors=args.collectors, collector_gb=args.collector_gb, pass_seats=args.pass_seats, kinds=tuple(kinds.items()), slots=args.slots,
                 test_kinds=tuple(args.test_kinds), chicken_difficulties=tuple(difficulties.items()),
                 group=args.group, horizon=tuple(args.horizon), group_scale=args.group_scale, reward_to_go=args.reward_to_go,
                 group_step=args.group_step, economy_change=args.economy_change, adv_clip=args.adv_clip, spo_bound=args.spo_bound,
                 entropy_base=args.entropy_base, eval_games=args.eval_games, bootstrap=args.bootstrap, resign=args.resign,
                 resign_free=args.resign_free, resign_false=args.resign_false, resign_least=args.resign_least, self_play=args.self_play)
    if args.action_kinds:
        cfg.enabled_kinds = tuple(int(k) for k in args.action_kinds.split(","))
    run = pathlib.Path(args.run)
    make_env = functools.partial(make_sim, decide_every=cfg.decide_every, maps=cfg.maps, variants=cfg.variants, advise=cfg.clone_weight > 0,
                                 income_weight=cfg.income_weight, fine_sim=cfg.fine_sim, speed_weight=cfg.speed_weight)
    if args.save_forever:
        _save_forever(cfg, run)
    elif args.check_kill:
        check_kill(cfg, run)
    elif args.check_repro:
        check_repro(cfg, run, make_env)
    elif args.grad_check:
        grad_check(cfg, run, make_env, args.grad_check, args.grad_splits)
    else:
        train(cfg, run, make_env)


if __name__ == "__main__":
    main()
