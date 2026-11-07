"""A frame VAE: compresses what a seat sees into a few small latent tokens, for the actor to read in place of its tokens.

LeJEPA (tools/frame_jepa.py) asks a frame's latent to match the same frame with units dropped, so it's taught to leave out which units are where.  This asks it to give
that back instead, through a small decoder:

    python tools/frame_vae.py train --data /mnt/d/zk-learned/frames/NAME --run /mnt/d/zk-learned/runs/NAME

**The decoder is asked about regions, not units.**  Units are an unordered list of any length, and rebuilding one needs matching, which isn't small.  So each region's
static columns (place, terrain, spots) make a query that reads the latents, and it answers with what's in that region now.  The map is known to the actor, so the latent
spends nothing on terrain.  It sees only what the seat sees: asked for what's hidden, it could only guess.

**The loss** is the negative ELBO of a Gaussian with variance `beta` over targets scaled to variance 1, so beta says about how much of a column may stay unexplained.
With `--autoencoder` there's no noise and no KL, so nothing prices what a latent carries: the loss is the error alone, and each latent token goes through a LayerNorm with no
weights of its own, which is what keeps its numbers on one scale for the actor.

With `--two-part`, each cell answers twice: is anything here, and if so how much.  Most cells are empty and those that aren't hardly differ from each other, which leaves a
single squared error mostly paying for where things are and blurring both.  The first answer is a Bernoulli's and the second a Gaussian's on filled cells only.  Together they
still give one number to compare, the expected value.

With `--region-pool`, the encoder sums what each region's units say into that region's token before its queries read anything (`FrameEncoder.pooled`).

With `--questions`, regions aren't all it's asked: rebuilding every unit would need matching, so it's asked about a random few from each frame, which keeps the cost fixed however
many there are.  "Where's a unit like this?" gets a description that gives no place away, and answers with several places and a weight for each, since it might describe
five units alike.  "What's at this exact place?" answers whether a unit or a wreck is, and if so what it is, with places just beside things to teach it the difference.  What's
in a description needn't be kept to answer the first, which is why the second asks for it.  How many there are at a place comes from the regions.

With `--muon-lr`, Muon trains the hidden layers (`Optimisers`).

Data: tools/frame_data.py.  Encoder, shards and probes: tools/frame_jepa.py.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bots"))
from frame_jepa import (CrossBlock, EncoderConfig, FrameBatch, FrameEncoder, Shards, TrainConfig, _collection, _learning_rate, _save_state, collapse_report, collate_frames,  # noqa: E402
                        encode_shards, nearest_region, probe_report)
from zkbot.policy import bundle as bd, featurize as fz, net as npnet, tables  # noqa: E402

# reg_x's columns: to this one the map's, from it what's there now (featurize._regions).
REGION_STATIC = 27
REGION_NOW = ("own_value", "seen_enemy_value", "radar_enemy_value", "own_dps", "seen_enemy_dps", "remembered", "wreck_metal")
CLASSES = ("structure", "builder", "air", "ship", "ground")
SIDES = ("own", "enemy")
REGION_TARGETS = (*REGION_NOW, *(f"{s}_{c}" for s in SIDES for c in CLASSES))
ALLY_SELF = 0               # featurize.relative_id numbers alliances from the viewer's
VALUE_SCALE = 10.0          # as featurize._regions scales value
STD_FLOOR = 0.01            # a column that hardly varies isn't worth blowing up
FILLED_ABOVE = 1e-3         # how far from a column's scaled 0 a cell counts as holding something
NORM_BATCHES = 16

# Questions for each frame.
ASK_WHERE_UNITS, ASK_WHERE_WRECKS = 16, 4
ASK_AT_UNITS, ASK_AT_WRECKS, ASK_BESIDE, ASK_ANYWHERE = 16, 4, 16, 8
AT_RADIUS = 0.02            # of the map's -1..1: this close to a thing's place is at it, about 60 elmos on a mid-sized map
BESIDE_FURTHEST = 8.0       # in radii: where the places asked about beside a unit go up to
PLACE_SPREAD = (0.005, 2.0) # of one of an answer's places
# Of a unit's token, what can go in a description: velocity, health, build, flags, a blip's speed, a memory's age.  Not place, altitude, orders, which can aim at places,
# or rays, which show the ground around it.  Of a wreck's only metal: its others are how far it is from the base and the army.
UNIT_STATE = (fz._I + 3, fz._I + 19)
WRECK_METAL = 2


def unit_classes(table: tables.TypeTable) -> torch.Tensor:
    """Each row's index in CLASSES.  The unknown row costs nothing, so summaries and unknown names add nothing whatever their class."""
    out = np.full(len(table.mobile), CLASSES.index("ground"), np.int64)
    out[table.move_class == bd.MOVE_CLASSES.index("ship")] = CLASSES.index("ship")
    out[table.air] = CLASSES.index("air")
    out[table.builder] = CLASSES.index("builder")
    out[~table.mobile] = CLASSES.index("structure")
    return torch.from_numpy(out)


@dataclass(frozen=True)
class VaeConfig:
    latent: int = 32        # numbers in each latent token, of which there's one per encoder query
    decoder_d: int = 96
    decoder_heads: int = 4
    decoder_rounds: int = 2
    beta: float = 0.1
    beta_warmup: int = 2000 # steps for beta to rise from 0, so the decoder gets something to read first
    variational: bool = True
    two_part: bool = False  # each region cell answers whether anything's there and how much, instead of one number
    questions: bool = False # also ask about a few units and wrecks from each frame
    question_weight: float = 1.0    # at 1 a question counts as much as a region cell's answer, both being in nats
    places: int = 8         # how many places a "where" answer may give


class Questions(NamedTuple):
    """What's asked of a frame, Q first.  Each `_ok` says which are real, as a frame may have too few to ask about."""
    name: torch.Tensor          # B x ASK_WHERE_UNITS: a unit's description
    side: torch.Tensor
    kind: torch.Tensor
    state: torch.Tensor
    place: torch.Tensor         # ... x 2: the answer
    unit_ok: torch.Tensor
    wreck_metal: torch.Tensor   # B x ASK_WHERE_WRECKS x 1
    wreck_place: torch.Tensor
    wreck_ok: torch.Tensor
    at: torch.Tensor            # B x A x 2: the places asked about
    at_ok: torch.Tensor
    near_unit: torch.Tensor     # B x A: whether one's at the place, and if so what it is
    at_name: torch.Tensor
    at_side: torch.Tensor
    at_state: torch.Tensor
    near_wreck: torch.Tensor
    at_metal: torch.Tensor      # B x A x 1


class Answers(NamedTuple):
    place_weights: torch.Tensor # B x W x places, as logits: units' questions first, then wrecks'
    place_means: torch.Tensor   # B x W x places x 2
    place_spreads: torch.Tensor # B x W x places
    unit_there: torch.Tensor    # B x A: logits
    wreck_there: torch.Tensor
    name: torch.Tensor          # B x A x vocabulary
    side: torch.Tensor          # B x A x 2
    state: torch.Tensor
    metal: torch.Tensor


def _pick(mask: torch.Tensor, count: int, generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
    """B x count: random entries of each row that mask allows, and which of those are real."""
    top = (torch.rand(mask.shape, device=mask.device, generator=generator) * mask).topk(min(count, mask.shape[1]), dim=1)
    return top.indices, top.values > 0.0


def _take(x: torch.Tensor, at: torch.Tensor) -> torch.Tensor:
    return x.gather(1, at) if x.dim() == 2 else x.gather(1, at[..., None].expand(-1, -1, x.shape[-1]))


def _nearest(places: torch.Tensor, things: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each place: the nearest of the real things, and how far it is."""
    apart = (places[:, :, None] - things[:, None]).square().sum(-1).masked_fill(~mask[:, None], float("inf"))
    closest = apart.min(-1)
    return closest.indices, closest.values.sqrt()


@torch.no_grad()
def draw_questions(b, generator: torch.Generator | None = None) -> Questions:
    x, w = b.unit_x.float(), b.feat_x.float()
    place, wreck_place = x[..., fz._I:fz._I + 2], w[..., :2]
    side = (b.unit_ally != ALLY_SELF).long()
    state = x[..., UNIT_STATE[0]:UNIT_STATE[1]]
    metal = w[..., WRECK_METAL:WRECK_METAL + 1]
    units = b.unit_mask & (b.tok_kind != fz.TK_SUMMARY)     # a summary isn't anywhere in particular
    who, unit_ok = _pick(units, ASK_WHERE_UNITS, generator)
    which, wreck_ok = _pick(b.feat_mask, ASK_WHERE_WRECKS, generator)

    at_unit, at_unit_ok = _pick(units, ASK_AT_UNITS, generator)
    at_wreck, at_wreck_ok = _pick(b.feat_mask, ASK_AT_WRECKS, generator)
    beside, beside_ok = _pick(units, ASK_BESIDE, generator)
    random = lambda *shape: torch.rand(*shape, device=x.device, generator=generator)  # noqa: E731
    angle = random(*beside.shape) * 2.0 * torch.pi
    how_far = AT_RADIUS * BESIDE_FURTHEST ** random(*beside.shape)      # evenly at each scale, from just outside a unit's radius
    offset = torch.stack((angle.cos(), angle.sin()), -1) * how_far[..., None]
    anywhere = random(len(x), ASK_ANYWHERE, 2) * 2.0 - 1.0
    at = torch.cat((_take(place, at_unit), _take(wreck_place, at_wreck), (_take(place, beside) + offset).clamp(-1.0, 1.0), anywhere), 1)
    at_ok = torch.cat((at_unit_ok, at_wreck_ok, beside_ok, torch.ones_like(anywhere[..., 0], dtype=torch.bool)), 1)

    close_unit, unit_far = _nearest(at, place, units)
    close_wreck, wreck_far = _nearest(at, wreck_place, b.feat_mask)
    return Questions(_take(b.unit_name, who), _take(side, who), _take(b.tok_kind, who), _take(state, who), _take(place, who), unit_ok,
                     _take(metal, which), _take(wreck_place, which), wreck_ok,
                     at, at_ok, unit_far < AT_RADIUS, _take(b.unit_name, close_unit), _take(side, close_unit), _take(state, close_unit),
                     wreck_far < AT_RADIUS, _take(metal, close_wreck))


def question_cost(q: Questions, a: Answers) -> tuple[torch.Tensor, torch.Tensor]:
    """For each frame, in nats, summed over its questions: of "where", and of "what's here"."""
    wanted = torch.cat((q.place, q.wreck_place), 1)[:, :, None]
    spread = a.place_spreads
    each = (-0.5 * (wanted - a.place_means).square().sum(-1) / spread.square() - 2.0 * spread.log() - np.log(2.0 * np.pi)).float()
    density = torch.logsumexp(F.log_softmax(a.place_weights.float(), -1) + each, -1)
    where = -(density * torch.cat((q.unit_ok, q.wreck_ok), 1)).sum(-1)
    there = F.binary_cross_entropy_with_logits(a.unit_there.float(), q.near_unit.float(), reduction="none") + F.binary_cross_entropy_with_logits(a.wreck_there.float(), q.near_wreck.float(), reduction="none")
    unit = (F.cross_entropy(a.name.float().flatten(0, 1), q.at_name.flatten(), reduction="none").view_as(q.at_name)
            + F.cross_entropy(a.side.float().flatten(0, 1), q.at_side.flatten(), reduction="none").view_as(q.at_side)
            + 0.5 * (a.state.float() - q.at_state).square().sum(-1))
    wreck = 0.5 * (a.metal.float() - q.at_metal).square().sum(-1)
    return where, ((there + q.near_unit * unit + q.near_wreck * wreck) * q.at_ok).sum(-1)


class Optimisers:
    """AdamW, and Muon for the weight matrices of hidden layers if `muon_lr` is set.  Muon makes a matrix's momentum orthogonal, which means nothing for other shapes, and as
    its authors do, it leaves alone embeddings, learned queries and the layers giving answers."""

    def __init__(self, model: nn.Module, cfg: TrainConfig):
        hidden = {id(m.weight) for name, m in model.named_modules() if isinstance(m, nn.Linear) and not name.endswith("_out")} if cfg.muon_lr > 0.0 else set()
        decay = [p for p in model.parameters() if p.dim() > 1 and id(p) not in hidden]
        plain = [p for p in model.parameters() if p.dim() <= 1]
        self.parts = {"adam": torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": plain, "weight_decay": 0.0}], lr=cfg.lr)}
        for group in self.parts["adam"].param_groups:
            group["scale"] = 1.0
        if hidden:
            self.parts["muon"] = torch.optim.Muon([p for p in model.parameters() if id(p) in hidden], lr=cfg.muon_lr, weight_decay=cfg.weight_decay)
            for group in self.parts["muon"].param_groups:
                group["scale"] = cfg.muon_lr / cfg.lr

    @property
    def param_groups(self) -> list[dict]:
        return [group for part in self.parts.values() for group in part.param_groups]

    def set_rate(self, rate: float) -> None:
        for group in self.param_groups:
            group["lr"] = rate * group["scale"]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for part in self.parts.values():
            part.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for part in self.parts.values():
            part.step()

    def state_dict(self) -> dict:
        return {name: part.state_dict() for name, part in self.parts.items()}

    def load_state_dict(self, state: dict) -> None:
        for name, part in self.parts.items():
            part.load_state_dict(state[name])


def _encoder_config() -> EncoderConfig:
    """As the LeJEPA run's, but its queries are the output."""
    return EncoderConfig(z=0)


class FrameVae(nn.Module):
    def __init__(self, cfg: VaeConfig, encoder_cfg: EncoderConfig, table: tables.TypeTable):
        super().__init__()
        self.cfg, self.encoder_cfg = cfg, encoder_cfg
        vocabulary = len(table.names) + 1
        self.encoder = FrameEncoder(encoder_cfg, vocabulary)
        self.to_mean = nn.Linear(encoder_cfg.d, cfg.latent)
        if cfg.variational:
            self.to_log_var = nn.Linear(encoder_cfg.d, cfg.latent)
            nn.init.zeros_(self.to_log_var.weight)
            nn.init.constant_(self.to_log_var.bias, -2.0)
        else:
            self.latent_norm = nn.LayerNorm(cfg.latent, elementwise_affine=False)
        d = cfg.decoder_d
        self.latent_in = nn.Linear(cfg.latent, d)
        self.region_in = nn.Linear(REGION_STATIC + 4 * encoder_cfg.position_bands, d)
        self.global_query = nn.Parameter(torch.randn(1, d) / d ** 0.5)
        self.reads = nn.ModuleList(CrossBlock(d, cfg.decoder_heads) for _ in range(cfg.decoder_rounds))
        self.lnf = nn.LayerNorm(d)
        self.region_out = nn.Linear(d, len(REGION_TARGETS) * (2 if cfg.two_part else 1))
        self.global_out = nn.Linear(d, fz.GLOBAL_DIM)
        if cfg.questions:
            self.name_in = nn.Embedding(vocabulary, npnet.NAME_DIM)
            self.side_in = nn.Embedding(len(SIDES), npnet.ID_DIM)
            self.kind_in = nn.Embedding(fz.TOKEN_KINDS, npnet.ID_DIM)
            self.unit_in = nn.Linear(npnet.NAME_DIM + 2 * npnet.ID_DIM + UNIT_STATE[1] - UNIT_STATE[0], d)
            self.wreck_in = nn.Linear(1, d)
            self.at_in = nn.Linear(2 + 4 * encoder_cfg.position_bands, d)
            self.where_out = nn.Linear(d, 4 * cfg.places)
            self.state_size = UNIT_STATE[1] - UNIT_STATE[0]
            self.at_out = nn.Linear(d, 3 + vocabulary + len(SIDES) + self.state_size)
        self.register_buffer("cost", torch.from_numpy(table.cost.astype(np.float32)), persistent=False)
        self.register_buffer("unit_class", unit_classes(table), persistent=False)
        for name, size in (("region", len(REGION_TARGETS)), ("global", fz.GLOBAL_DIM)):
            self.register_buffer(f"{name}_mean", torch.zeros(size))
            self.register_buffer(f"{name}_std", torch.ones(size))

    # What goes in ---------------------------------------------------------------------------------------------------------------------------------------

    def encode(self, b) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Mean and log variance, each B x queries x latent.  An autoencoder's latent has no variance."""
        h = self.encoder.read(b)
        if not self.cfg.variational:
            return self.latent_norm(self.to_mean(h)), None
        return self.to_mean(h), self.to_log_var(h).clamp(-12.0, 6.0)

    def latent_tokens(self, b) -> torch.Tensor:
        """What the actor would read: B x queries x latent, with no noise."""
        return self.encode(b)[0]

    def latent_vector(self, b) -> torch.Tensor:
        return self.latent_tokens(b).flatten(1)

    def decode(self, latents: torch.Tensor, b) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw region answers, B x R x REGION_TARGETS or twice that (see region_parts), and scaled globals, B x GLOBAL_DIM."""
        z = self.latent_in(latents)
        q = torch.cat((self.region_in(self.encoder._with_position(b.reg_x[..., :REGION_STATIC], 0)), self.global_query.expand(len(z), -1, -1)), 1)
        everyone = torch.ones_like(z[:, :, 0], dtype=torch.bool)
        for read in self.reads:
            q = read(q, z, everyone)
        q = self.lnf(q)
        return self.region_out(q[:, :-1]), self.global_out(q[:, -1])

    def region_parts(self, region: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """What decode gave as scores for whether each cell is filled, and how much is there if so.  One number does for both when it isn't two-part."""
        return region.chunk(2, -1) if self.cfg.two_part else (region, region)

    def empty_cell(self) -> torch.Tensor:
        """What a cell holding 0 scales to, for each column."""
        return -self.region_mean / self.region_std

    def filled(self, scaled: torch.Tensor) -> torch.Tensor:
        return (scaled - self.empty_cell()).abs() > FILLED_ABOVE

    def region_answer(self, region: torch.Tensor) -> torch.Tensor:
        """The scaled value to expect in each cell, whichever way it answers."""
        if not self.cfg.two_part:
            return region
        score, size = self.region_parts(region)
        return self.empty_cell() + score.sigmoid() * (size - self.empty_cell())

    def answer_questions(self, latents: torch.Tensor, q: Questions) -> Answers:
        z = self.latent_in(latents)
        units = self.unit_in(torch.cat((self.name_in(q.name), self.side_in(q.side), self.kind_in(q.kind), q.state), -1))
        h = torch.cat((units, self.wreck_in(q.wreck_metal), self.at_in(self.encoder._with_position(q.at, 0))), 1)
        everyone = torch.ones_like(z[:, :, 0], dtype=torch.bool)
        for read in self.reads:
            h = read(h, z, everyone)
        h = self.lnf(h)
        asked = q.name.shape[1] + q.wreck_metal.shape[1]
        weights, means, spreads = self.where_out(h[:, :asked]).split((self.cfg.places, 2 * self.cfg.places, self.cfg.places), -1)
        unit_there, wreck_there, metal, name, side, state = self.at_out(h[:, asked:]).split((1, 1, 1, self.encoder.vocabulary, len(SIDES), self.state_size), -1)
        low, high = np.log(PLACE_SPREAD[0]), np.log(PLACE_SPREAD[1])
        return Answers(weights, means.unflatten(-1, (self.cfg.places, 2)), spreads.clamp(low, high).exp(), unit_there[..., 0], wreck_there[..., 0], name, side, state, metal)

    @staticmethod
    def sampled(mean: torch.Tensor, log_var: torch.Tensor | None) -> torch.Tensor:
        return mean if log_var is None else mean + torch.randn_like(mean) * (0.5 * log_var).exp()

    def forward(self, b, sample: bool = True):
        mean, log_var = self.encode(b)
        return (*self.decode(self.sampled(mean, log_var) if sample else mean, b), mean, log_var)

    # What should come out -------------------------------------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def region_targets(self, b) -> torch.Tensor:
        """B x R x REGION_TARGETS, unscaled: reg_x's own columns, then value by side and class, with a unit counting for the nearest region."""
        x = b.unit_x.float()
        count, regions = x.shape[0], b.reg_x.shape[1]
        hp = torch.where(x[..., fz._I + 7] > 0.5, torch.ones_like(x[..., 0]), x[..., fz._I + 6])
        value = self.cost[b.unit_name] * hp * b.unit_mask
        side = (b.unit_ally != ALLY_SELF).long()
        cell = nearest_region(b) * len(SIDES) * len(CLASSES) + side * len(CLASSES) + self.unit_class[b.unit_name]
        grid = torch.zeros(count, regions * len(SIDES) * len(CLASSES), device=x.device).scatter_add(1, cell, value)
        grid = (grid.view(count, regions, -1).log1p() / VALUE_SCALE)
        return torch.cat((b.reg_x[..., REGION_STATIC:].float(), grid), -1)

    @torch.no_grad()
    def fit_scales(self, batches: list[FrameBatch]) -> None:
        """Each target column's mean and spread, over real regions."""
        regions = torch.cat([self.region_targets(b)[b.reg_mask] for b in batches])
        glob = torch.cat([b.glob_x.float() for b in batches])
        self.region_mean.copy_(regions.mean(0))
        self.region_std.copy_(regions.std(0).clamp_min(STD_FLOOR))
        self.global_mean.copy_(glob.mean(0))
        self.global_std.copy_(glob.std(0).clamp_min(STD_FLOOR))

    def scaled_targets(self, b) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.region_targets(b) - self.region_mean) / self.region_std, (b.glob_x.float() - self.global_mean) / self.global_std


def kl_per_token_number(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """KL from N(0, 1) to N(mean, var), in nats, for each number."""
    return 0.5 * (mean.square() + log_var.exp() - 1.0 - log_var)


def vae_loss(model: FrameVae, b, beta: float) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    """Gaussian ELBO with variance beta, summed over a frame's cells, then divided by the mean cells in a frame, which leaves it at the scale of one cell whatever the map.
    Two-part, a cell costs its Bernoulli and, if filled, its Gaussian instead.  `recon` is the expected value's squared error either way, so runs compare.
    Questions are in nats too, so they add to the cells' as they are, and `recon` leaves them out.
    Returns the loss, what to log, and the means."""
    q = draw_questions(b) if model.cfg.questions else None
    with torch.autocast(b.reg_x.device.type, dtype=torch.bfloat16):
        mean, log_var = model.encode(b)
        latents = model.sampled(mean, log_var)
        region, glob = model.decode(latents, b)
        said = model.answer_questions(latents, q) if q else None
    region, glob, mean = region.float(), glob.float(), mean.float()
    want_region, want_glob = model.scaled_targets(b)
    region_error = ((model.region_answer(region) - want_region).square().sum(-1) * b.reg_mask).sum(-1)
    global_error = (glob - want_glob).square().sum(-1)
    region_cost = 0.5 * region_error
    if model.cfg.two_part:
        score, size = model.region_parts(region)
        filled = model.filled(want_region).float()
        each = F.binary_cross_entropy_with_logits(score, filled, reduction="none") + 0.5 * filled * (size - want_region).square()
        region_cost = (each.sum(-1) * b.reg_mask).sum(-1)
    cells = b.reg_mask.sum(-1).float() * len(REGION_TARGETS) + fz.GLOBAL_DIM
    recon = float(((region_error + global_error).sum() / cells.sum()).detach())
    asked = {}
    frame_cost = region_cost + 0.5 * global_error
    if q:
        where, what = question_cost(q, said)
        frame_cost = frame_cost + model.cfg.question_weight * (where + what)
        asked = {"where_nats": float(where.sum().detach() / (q.unit_ok.sum() + q.wreck_ok.sum()).clamp_min(1)), "what_nats": float(what.sum().detach() / q.at_ok.sum().clamp_min(1))}
    if log_var is None:
        loss = frame_cost.mean() / cells.mean()
        return loss, {"recon": recon, "kl_nats": 0.0, "active_numbers": float((mean.flatten(0, 1).std(0) > 0.01).float().sum().detach()) * mean.shape[1], **asked}, mean.detach()
    numbers = kl_per_token_number(mean, log_var.float())
    kl = numbers.flatten(1).sum(-1)
    loss = (frame_cost + beta * kl).mean() / cells.mean()
    return loss, {"recon": recon, "kl_nats": float(kl.mean().detach()),
                  "active_numbers": float((numbers.mean(0) > 0.01).float().sum().detach()), **asked}, mean.detach()


# Is it any good? ----------------------------------------------------------------------------------------------------------------------------------------


@torch.no_grad()
def reconstruction_report(model: FrameVae, paths: list[pathlib.Path], held_out, device: str, frames: int, batch: int = 128) -> dict[str, float]:
    """R^2 of what it gives back, on maps it never saw, against the best anyone could do knowing only the map: each region's mean on that map, over these same frames, so
    a generous mark to beat.  The `found_` figures are over cells where the column isn't 0, where most of what matters is, and which a model that says nothing gets wrong.
    They're harsh: filled cells hardly differ from each other, so a few wrong ones send them far below 0 however well the rest does.  So `where_` says where things are apart
    from how much: of as many cells as are filled, those it rates highest, the share that are.  And `how_much_` is the R^2 over those it got right."""
    answers, scores, sizes_said, truths, keys, globals_out, globals_true = [], [], [], [], [], [], []
    left = frames
    for path in paths:
        s = Shards._read(path)
        chosen = np.flatnonzero(np.isin(s["map"], held_out))[:left]
        for start in range(0, len(chosen), batch):
            indices = chosen[start:start + batch]
            b = collate_frames([Shards.frame(s, int(i)) for i in indices]).to(device)
            region, glob, _, _ = model(b, sample=False)
            want_region, want_glob = model.scaled_targets(b)
            score, size = model.region_parts(region)
            answers.append(model.region_answer(region)[b.reg_mask].float().cpu().numpy())
            scores.append(score[b.reg_mask].float().cpu().numpy())
            sizes_said.append(size[b.reg_mask].float().cpu().numpy())
            truths.append(want_region[b.reg_mask].float().cpu().numpy())
            which = torch.arange(b.reg_x.shape[1], device=device).expand(len(b.reg_mask), -1)
            map_ids = torch.as_tensor(s["map"][indices], device=device)[:, None].expand_as(which)
            keys.append((map_ids[b.reg_mask] * 1000 + which[b.reg_mask]).cpu().numpy())
            globals_out.append(glob.float().cpu().numpy())
            globals_true.append(want_glob.float().cpu().numpy())
        left -= len(chosen)
        if not left:
            break
    if not answers:
        return {}
    answer, score, said, truth, key = np.concatenate(answers), np.concatenate(scores), np.concatenate(sizes_said), np.concatenate(truths), np.concatenate(keys)
    _, group = np.unique(key, return_inverse=True)
    sizes = np.bincount(group)
    baseline = np.stack([np.bincount(group, truth[:, c]) / sizes for c in range(truth.shape[1])], 1)[group]
    total = np.maximum(((truth - truth.mean(0)) ** 2).sum(0), 1e-9)
    r2, map_r2 = 1.0 - ((truth - answer) ** 2).sum(0) / total, 1.0 - ((truth - baseline) ** 2).sum(0) / total
    out = {"recon_r2": float(r2.mean()), "recon_map_r2": float(map_r2.mean())}
    filled = model.filled(torch.from_numpy(truth).to(device)).cpu().numpy()
    for c, name in enumerate(REGION_TARGETS):
        out[f"recon_r2_{name}"], out[f"recon_map_r2_{name}"] = float(r2[c]), float(map_r2[c])
        there = filled[:, c]
        if there.sum() > 16:
            held = truth[there, c]
            spread = max(float(((held - held.mean()) ** 2).sum()), 1e-9)
            out[f"found_r2_{name}"] = float(1.0 - ((held - answer[there, c]) ** 2).sum() / spread)
            out[f"found_map_r2_{name}"] = float(1.0 - ((held - baseline[there, c]) ** 2).sum() / spread)
            top = np.argsort(-score[:, c])[:int(there.sum())]
            right = top[there[top]]
            out[f"where_{name}"] = len(right) / len(top)
            if len(right) > 16:
                got = truth[right, c]
                out[f"how_much_{name}"] = float(1.0 - ((got - said[right, c]) ** 2).sum() / max(float(((got - got.mean()) ** 2).sum()), 1e-9))
    glob_out, glob_true = np.concatenate(globals_out), np.concatenate(globals_true)
    varied = glob_true.std(0) > 1e-3
    out["recon_r2_global"] = float((1.0 - ((glob_true - glob_out) ** 2).sum(0)[varied] / ((glob_true - glob_true.mean(0)) ** 2).sum(0)[varied]).mean())
    return out


@torch.no_grad()
def question_report(model: FrameVae, paths: list[pathlib.Path], held_out, device: str, frames: int, batch: int = 128, seed: int = 4) -> dict[str, float]:
    """How well it answers on maps it never saw.  A "where" answer has several places, so `where_top_` is the share whose likeliest place is within that distance, which
    identical units hold down, and `where_any_` the share with any place that close.  Knowing nothing gives 1.39 nats, a uniform place on the map.  By frame size too, as
    that's where a latent of a fixed size should run out."""
    generator = torch.Generator(device=device).manual_seed(seed)
    tops, anys, nats, crowds, wreck_tops, wreck_anys, wreck_nats = [], [], [], [], [], [], []
    unit_scores, unit_truth, wreck_scores, wreck_truth = [], [], [], []
    names, name_wanted, sides, side_wanted, states, state_wanted, metals, metal_wanted = [], [], [], [], [], [], [], []
    left = frames
    for path in paths:
        s = Shards._read(path)
        chosen = np.flatnonzero(np.isin(s["map"], held_out))[:left]
        for start in range(0, len(chosen), batch):
            b = collate_frames([Shards.frame(s, int(i)) for i in chosen[start:start + batch]]).to(device)
            q = draw_questions(b, generator)
            a = model.answer_questions(model.encode(b)[0], q)
            wanted = torch.cat((q.place, q.wreck_place), 1)
            apart = (a.place_means - wanted[:, :, None]).norm(dim=-1)
            top = apart.gather(-1, a.place_weights.argmax(-1, keepdim=True))[..., 0]
            nearest = apart.amin(-1)
            spread = a.place_spreads
            each = -0.5 * apart.square() / spread.square() - 2.0 * spread.log() - np.log(2.0 * np.pi)
            density = torch.logsumexp(F.log_softmax(a.place_weights, -1) + each, -1)
            count = q.name.shape[1]
            crowd = (b.unit_mask & (b.tok_kind != fz.TK_SUMMARY)).sum(-1, keepdim=True).expand(-1, count)
            tops.append(top[:, :count][q.unit_ok].cpu()), anys.append(nearest[:, :count][q.unit_ok].cpu()), nats.append(-density[:, :count][q.unit_ok].cpu()), crowds.append(crowd[q.unit_ok].cpu())
            wreck_tops.append(top[:, count:][q.wreck_ok].cpu()), wreck_anys.append(nearest[:, count:][q.wreck_ok].cpu()), wreck_nats.append(-density[:, count:][q.wreck_ok].cpu())
            unit_scores.append(a.unit_there[q.at_ok].cpu()), unit_truth.append(q.near_unit[q.at_ok].cpu())
            wreck_scores.append(a.wreck_there[q.at_ok].cpu()), wreck_truth.append(q.near_wreck[q.at_ok].cpu())
            found, found_wreck = q.at_ok & q.near_unit, q.at_ok & q.near_wreck
            names.append(a.name[found].argmax(-1).cpu()), name_wanted.append(q.at_name[found].cpu())
            sides.append(a.side[found].argmax(-1).cpu()), side_wanted.append(q.at_side[found].cpu())
            states.append(a.state[found].cpu()), state_wanted.append(q.at_state[found].cpu())
            metals.append(a.metal[found_wreck].cpu()), metal_wanted.append(q.at_metal[found_wreck].cpu())
        left -= len(chosen)
        if not left:
            break
    if not tops:
        return {}
    top, near, crowd = torch.cat(tops), torch.cat(anys), torch.cat(crowds)
    out = {"where_unit_nats": float(torch.cat(nats).mean()), "where_unit_median": float(top.median()), "where_unit_any_median": float(near.median())}
    for distance in (0.02, 0.05, 0.15):
        out[f"where_top_{distance}"] = float((top < distance).float().mean())
        out[f"where_any_{distance}"] = float((near < distance).float().mean())
    for label, low, high in (("few", 0, 30), ("some", 31, 70), ("many", 71, 100000)):
        here = (crowd >= low) & (crowd <= high)
        if here.sum() > 16:
            out[f"where_any_median_{label}"] = float(near[here].median())
            out[f"where_top_median_{label}"] = float(top[here].median())
    wreck = torch.cat(wreck_anys)
    if len(wreck) > 16:
        out.update(where_wreck_nats=float(torch.cat(wreck_nats).mean()), where_wreck_any_median=float(wreck.median()), where_wreck_top_median=float(torch.cat(wreck_tops).median()))
    for label, scores, truth in (("unit", unit_scores, unit_truth), ("wreck", wreck_scores, wreck_truth)):
        said, real = torch.cat(scores) > 0.0, torch.cat(truth)
        if real.sum() > 16 and (~real).sum() > 16:
            out[f"what_{label}_found"], out[f"what_{label}_false"] = float(said[real].float().mean()), float(said[~real].float().mean())
    if name_wanted and len(torch.cat(name_wanted)) > 16:
        out["what_type"] = float((torch.cat(names) == torch.cat(name_wanted)).float().mean())
        out["what_side"] = float((torch.cat(sides) == torch.cat(side_wanted)).float().mean())
        said, real = torch.cat(states), torch.cat(state_wanted)
        for label, column in (("vx", 0), ("vz", 1), ("hp", 3), ("build", 5)):
            out[f"what_{label}_r2"] = float(1.0 - (said[:, column] - real[:, column]).square().sum() / (real[:, column] - real[:, column].mean()).square().sum().clamp_min(1e-9))
    if metal_wanted and len(torch.cat(metal_wanted)) > 16:
        said, real = torch.cat(metals), torch.cat(metal_wanted)
        out["what_metal_r2"] = float(1.0 - (said - real).square().sum() / (real - real.mean()).square().sum().clamp_min(1e-9))
    return out


# Training -----------------------------------------------------------------------------------------------------------------------------------------------


def save_vae(path: pathlib.Path, model: FrameVae, step: int) -> None:
    torch.save({"cfg": asdict(model.cfg), "encoder_cfg": asdict(model.encoder_cfg), "model": model.state_dict(), "step": step, "feature_version": npnet.FEATURE_VERSION}, path)


def load_vae(path: pathlib.Path, device: str = "cpu") -> FrameVae:
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["feature_version"] != npnet.FEATURE_VERSION:
        raise ValueError(f"{path}: feature version {saved['feature_version']}, this code uses {npnet.FEATURE_VERSION}")
    model = FrameVae(VaeConfig(**saved["cfg"]), EncoderConfig(**saved["encoder_cfg"]), tables.table())
    model.load_state_dict(saved["model"])
    return model.to(device).eval()


def beta_at(cfg: VaeConfig, step: int) -> float:
    return cfg.beta * min(1.0, step / cfg.beta_warmup) if cfg.beta_warmup else cfg.beta


def train(data: pathlib.Path, run: pathlib.Path, cfg: TrainConfig = TrainConfig(), vae_cfg: VaeConfig = VaeConfig(), encoder_cfg: EncoderConfig | None = None, device: str | None = None) -> FrameVae:
    """A `stop` file in the run's directory ends it at the next report.  A run's directory that holds a state goes on from there, with this run's random numbers."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder_cfg = encoder_cfg or _encoder_config()
    run.mkdir(parents=True, exist_ok=True)
    manifest, paths, held_out, groups, outcomes = _collection(data)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    shards = Shards(paths, rng, maps=tuple(m for m in range(len(manifest["maps"])) if m not in held_out))
    model = FrameVae(vae_cfg, encoder_cfg, tables.table()).to(device)
    (run / "config.json").write_text(json.dumps({"train": asdict(cfg), "vae": asdict(vae_cfg), "encoder": asdict(encoder_cfg), "data": str(data)}, indent=1), encoding="utf8")
    optimiser = Optimisers(model, cfg)
    started = time.time()
    first = 0
    if (run / "state.pt").exists():
        state = torch.load(run / "state.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        first = state["step"] + 1
    else:
        model.fit_scales([shards.sample_probes(cfg.batch)[0].to(device) for _ in range(NORM_BATCHES)])
    step = first - 1
    with open(run / "metrics.jsonl", "a", encoding="utf8") as log:
        for step in range(first, cfg.steps):
            optimiser.set_rate(_learning_rate(cfg, step))
            frames, _ = shards.sample_probes(cfg.batch)
            frames = frames.to(device)
            loss, parts, mean = vae_loss(model, frames, beta_at(vae_cfg, step))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            optimiser.step()
            if step % cfg.report_every == 0 or step == cfg.steps - 1:
                row = {"step": step, "loss": float(loss.detach()), "grad_norm": grad, "seconds": time.time() - started, "lr": optimiser.param_groups[0]["lr"], "beta": beta_at(vae_cfg, step), **parts, **collapse_report(mean.flatten(1))}
                if step % cfg.probe_every == 0 or step == cfg.steps - 1:     # from the start, for what an untrained encoder gets
                    model.eval()
                    order = [paths[int(i)] for i in rng.permutation(len(paths))]
                    sample = encode_shards(model.latent_vector, order, device, cfg.probe_frames)
                    row.update(probe_report(sample, tuple(manifest["probes"]), outcomes, held_out, groups))
                    row.update(reconstruction_report(model, order, held_out, device, cfg.probe_frames // 4))
                    if vae_cfg.questions:
                        row.update(question_report(model, order, held_out, device, cfg.probe_frames // 4))
                    model.train()
                log.write(json.dumps(row) + "\n")
                log.flush()
                print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items() if k in ("step", "loss", "recon", "kl_nats", "where_nats", "what_nats", "seconds", "recon_r2", "where_any_0.05", "where_top_0.05", "what_type")}, flush=True)
                if (run / "stop").exists() or (cfg.hours and time.time() - started > cfg.hours * 3600.0):
                    break
            if step and step % cfg.checkpoint_every == 0:
                save_vae(run / f"vae-{step:07d}.pt", model, step)
                _save_state(run / "state.pt", model, optimiser, step)
    save_vae(run / "vae-last.pt", model, step)
    _save_state(run / "state.pt", model, optimiser, step)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=pathlib.Path, required=True)
    t.add_argument("--run", type=pathlib.Path, required=True)
    t.add_argument("--steps", type=int, default=TrainConfig.steps)
    t.add_argument("--batch", type=int, default=TrainConfig.batch)
    t.add_argument("--hours", type=float, default=0.0)
    t.add_argument("--probe-every", type=int, default=TrainConfig.probe_every)
    t.add_argument("--latent", type=int, default=VaeConfig.latent)
    t.add_argument("--beta", type=float, default=VaeConfig.beta)
    t.add_argument("--autoencoder", action="store_true", help="no noise and no KL; --beta is unused")
    t.add_argument("--two-part", action="store_true", help="each cell answers whether anything's there and how much")
    t.add_argument("--region-pool", type=int, default=_encoder_config().region_pool, help="numbers each region token gets from summing its units; 0 for none")
    t.add_argument("--questions", action="store_true", help="also ask about a few units and wrecks from each frame")
    t.add_argument("--question-weight", type=float, default=VaeConfig.question_weight)
    t.add_argument("--muon-lr", type=float, default=TrainConfig.muon_lr, help="above 0, Muon trains the hidden layers at this rate")
    t.add_argument("--seed", type=int, default=TrainConfig.seed)
    t.add_argument("--decoder-d", type=int, default=VaeConfig.decoder_d)
    t.add_argument("--queries", type=int, default=_encoder_config().queries, help="latent tokens")
    t.add_argument("--d", type=int, default=_encoder_config().d, help="the encoder's width")
    t.add_argument("--rounds", type=int, default=_encoder_config().rounds, help="the encoder's")
    t.add_argument("--device", default=None)
    args = parser.parse_args()
    train(args.data, args.run, TrainConfig(steps=args.steps, batch=args.batch, hours=args.hours, probe_every=args.probe_every, seed=args.seed, muon_lr=args.muon_lr),
          VaeConfig(latent=args.latent, beta=args.beta, decoder_d=args.decoder_d, variational=not args.autoencoder, two_part=args.two_part, questions=args.questions, question_weight=args.question_weight),
          replace(_encoder_config(), queries=args.queries, d=args.d, rounds=args.rounds, region_pool=args.region_pool), device=args.device)


if __name__ == "__main__":
    main()
