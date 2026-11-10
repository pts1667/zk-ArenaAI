"""What the frame encoders share: frames as padded token batches, the encoder that reads them (`FrameEncoder`, which
frame_vae.py trains), shards of collected frames, and linear probes of what a latent knows.

The encoder reads tokens as the policy's are padded, so `FrameEncoder` takes a `policy_model.Batch` as it stands, or a
`FrameBatch` from `collate_frames`.  Data: tools/frame_data.py.  (A LeJEPA-trained encoder, frame_jepa.py, was tried first and
removed on 2026-11-08: 03-training.md, "Dead ends".)
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bots"))
from policy_model import ID_DIM, NAME_DIM, Block  # noqa: E402
from zkbot.policy import featurize as fz  # noqa: E402

# The first two columns of each token group's features are its place on the map, -1 to 1.
POSITION_COLUMN = {"unit": fz._I, "reg": 0, "site": 0, "feat": 0}
FEATURE_LIMIT = 8.0         # a few type stats run to 72 (a jammer's); nothing else passes 4
GROUPS = (("unit", fz.UNIT_DIM), ("reg", fz.REGION_DIM), ("site", fz.SITE_DIM), ("feat", fz.FEATURE_DIM))
GROUP_KINDS = {"reg": fz.TK_REGION, "site": fz.TK_SITE, "feat": fz.TK_FEATURE}

@dataclass(frozen=True)
class EncoderConfig:
    d: int = 128
    heads: int = 4
    queries: int = 24       # how many learned queries read the tokens.  Attention costs tokens x queries, so hundreds of tokens stay cheap.
    rounds: int = 2         # reads of the tokens, each followed by the queries talking among themselves
    token_layers: int = 0   # self-attention over the tokens first.  Costs tokens squared, so it's off by default.
    region_pool: int = 0    # numbers each region token gets from summing its units, as attention averages and so can't count.  0: none.
    z: int = 256            # the latent: what the main model reads
    position_bands: int = 6
    projector: int = 1024   # unused: LeJEPA's projector, kept so that saved configs load
    projected: int = 128    # unused, likewise



@dataclass
class FrameBatch:
    """Padded frames, with the names of `policy_model.Batch`."""
    unit_x: torch.Tensor
    unit_name: torch.Tensor
    unit_ally: torch.Tensor
    unit_player: torch.Tensor
    unit_mask: torch.Tensor
    tok_kind: torch.Tensor      # B x U
    reg_x: torch.Tensor
    reg_mask: torch.Tensor
    site_x: torch.Tensor
    site_mask: torch.Tensor
    feat_x: torch.Tensor
    feat_mask: torch.Tensor
    glob_x: torch.Tensor

    def to(self, device: str) -> "FrameBatch":
        return FrameBatch(**{k: v.to(device, non_blocking=True) for k, v in self.__dict__.items()})


def collate_frames(items: list[dict[str, np.ndarray]]) -> FrameBatch:
    """Frames as `Shards.frame` gives them."""
    b = len(items)
    arrays: dict[str, torch.Tensor] = {}
    for name, dim in GROUPS:
        longest = max(max(len(t[f"{name}_x"]) for t in items), 1)
        x = np.zeros((b, longest, dim), np.float32)
        mask = np.zeros((b, longest), bool)
        for i, t in enumerate(items):
            n = len(t[f"{name}_x"])
            x[i, :n], mask[i, :n] = t[f"{name}_x"], True
        arrays[f"{name}_x"], arrays[f"{name}_mask"] = torch.from_numpy(x), torch.from_numpy(mask)
    u = arrays["unit_x"].shape[1]
    for name in ("unit_name", "unit_ally", "unit_player", "tok_kind"):
        out = np.zeros((b, u), np.int64)
        for i, t in enumerate(items):
            out[i, :len(t[name])] = t[name]
        arrays[name] = torch.from_numpy(out)
    arrays["glob_x"] = torch.from_numpy(np.stack([t["glob_x"] for t in items]).astype(np.float32))
    return FrameBatch(**arrays)


class CrossBlock(nn.Module):
    """Queries reading tokens."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln_q = nn.LayerNorm(d)
        self.ln_t = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        b, q_count, d = queries.shape
        split = lambda t: t.view(b, t.shape[1], self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
        k, v = self.kv(self.ln_t(tokens)).chunk(2, -1)
        o = F.scaled_dot_product_attention(split(self.q(self.ln_q(queries))), split(k), split(v), attn_mask=key_mask[:, None, None, :])
        queries = queries + self.out(o.transpose(1, 2).reshape(b, q_count, d))
        return queries + self.ff2(F.gelu(self.ff1(self.ln2(queries)), approximate="tanh"))


@torch.no_grad()
def nearest_region(b) -> torch.Tensor:
    """B x U: the region with the nearest centre to each unit.  Close enough for a sum over a region, since the region's outline isn't in the frame."""
    units = b.unit_x[:, :, None, POSITION_COLUMN["unit"]:POSITION_COLUMN["unit"] + 2].float()
    centres = b.reg_x[:, None, :, POSITION_COLUMN["reg"]:POSITION_COLUMN["reg"] + 2].float()
    return (units - centres).square().sum(-1).masked_fill(~b.reg_mask[:, None], float("inf")).argmin(-1)


class FrameEncoder(nn.Module):
    """Tokens in, one vector out.  What it gives doesn't depend on the tokens' order, or on padding."""

    def __init__(self, cfg: EncoderConfig, vocabulary: int):
        super().__init__()
        self.cfg, self.vocabulary = cfg, vocabulary
        d = cfg.d
        extra = 4 * cfg.position_bands
        self.name_embed = nn.Embedding(vocabulary, NAME_DIM)
        self.ally_embed = nn.Embedding(fz.MAX_ALLY_TEAMS + 1, ID_DIM)
        self.player_embed = nn.Embedding(fz.MAX_PLAYERS + 1, ID_DIM)
        self.kind_embed = nn.Embedding(fz.TOKEN_KINDS, d)
        self.unit_in0 = nn.Linear(fz.UNIT_DIM + extra + NAME_DIM + 2 * ID_DIM, d)
        self.unit_in1 = nn.Linear(d, d)
        self.reg_in = nn.Linear(fz.REGION_DIM + extra + cfg.region_pool, d)
        if cfg.region_pool:
            self.pool_in = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(approximate="tanh"), nn.Linear(2 * d, cfg.region_pool))
        self.site_in = nn.Linear(fz.SITE_DIM + extra, d)
        self.feat_in = nn.Linear(fz.FEATURE_DIM + extra, d)
        self.glob_in = nn.Linear(fz.GLOBAL_DIM, d)
        self.token_blocks = nn.ModuleList(Block(d, cfg.heads) for _ in range(cfg.token_layers))
        self.queries = nn.Parameter(torch.randn(cfg.queries, d) / math.sqrt(d))
        self.cross = nn.ModuleList(CrossBlock(d, cfg.heads) for _ in range(cfg.rounds))
        self.latent_blocks = nn.ModuleList(Block(d, cfg.heads) for _ in range(cfg.rounds))
        self.lnf = nn.LayerNorm(d)
        self.to_z = nn.Linear(cfg.queries * d, cfg.z) if cfg.z else nn.Identity()      # z 0: no vector, for a model that reads the queries themselves
        self.register_buffer("bands", math.pi * 2.0 ** torch.arange(cfg.position_bands, dtype=torch.float32), persistent=False)

    def _with_position(self, x: torch.Tensor, column: int) -> torch.Tensor:
        """Sines of a token's place, since a linear layer on -1..1 alone can't tell neighbours apart on a large map."""
        at = x[..., column:column + 2].unsqueeze(-1) * self.bands
        return torch.cat((x.clamp(-FEATURE_LIMIT, FEATURE_LIMIT), at.sin().flatten(-2), at.cos().flatten(-2)), -1)

    def pooled(self, unit_tokens: torch.Tensor, b) -> torch.Tensor:
        """B x R x region_pool: what each region's units add up to, each number squashed by asinh, so a region of 100 units is on the scale of the rest of its token."""
        each = (self.pool_in(unit_tokens) * b.unit_mask[..., None]).float()
        at = nearest_region(b)[..., None].expand(-1, -1, self.cfg.region_pool)
        return torch.zeros(len(each), b.reg_x.shape[1], self.cfg.region_pool, device=each.device).scatter_add(1, at, each).asinh()

    def tokens(self, b) -> tuple[torch.Tensor, torch.Tensor]:
        """All tokens and which are real: units, regions, sites, features, global."""
        units = torch.cat((self._with_position(b.unit_x, POSITION_COLUMN["unit"]), self.name_embed(b.unit_name), self.ally_embed(b.unit_ally), self.player_embed(b.unit_player)), -1)
        hu = self.unit_in1(F.gelu(self.unit_in0(units), approximate="tanh")) + self.kind_embed(b.tok_kind[:, :b.unit_x.shape[1]])
        regions = self._with_position(b.reg_x, POSITION_COLUMN["reg"])
        if self.cfg.region_pool:
            regions = torch.cat((regions, self.pooled(hu, b)), -1)
        hr = self.reg_in(regions) + self.kind_embed.weight[GROUP_KINDS["reg"]]
        hs = self.site_in(self._with_position(b.site_x, POSITION_COLUMN["site"])) + self.kind_embed.weight[GROUP_KINDS["site"]]
        hf = self.feat_in(self._with_position(b.feat_x, POSITION_COLUMN["feat"])) + self.kind_embed.weight[GROUP_KINDS["feat"]]
        hg = (self.glob_in(b.glob_x) + self.kind_embed.weight[fz.TK_GLOBAL])[:, None]
        everything = torch.cat((hu, hr, hs, hf, hg), 1)
        mask = torch.cat((b.unit_mask, b.reg_mask, b.site_mask, b.feat_mask, torch.ones_like(b.unit_mask[:, :1])), 1)
        return everything, mask

    def forward(self, b) -> torch.Tensor:
        """B x z.  b: a FrameBatch or a policy_model.Batch."""
        return self.to_z(self.read(b).flatten(1))

    def read(self, b) -> torch.Tensor:
        """B x queries x d: what the queries hold after reading the tokens."""
        h, mask = self.tokens(b)
        for block in self.token_blocks:
            h = block(h, None, mask)
        q = self.queries.expand(len(h), -1, -1)
        everyone = torch.ones_like(q[:, :, 0], dtype=torch.bool)
        for cross, latent in zip(self.cross, self.latent_blocks):
            q = latent(cross(q, h, mask), None, everyone)
        return self.lnf(q)



# The data --------------------------------------------------------------------------------------------------------------------------------------------


class Shards:
    """Frames from shards on disk, a few shards in memory at a time and one swapped out every so often, so a pass needn't hold them all.  The defaults hold about 1.5 GB."""

    def __init__(self, paths: list[pathlib.Path], rng: np.random.Generator, in_memory: int = 64, swap_every: int = 25, maps: tuple[int, ...] | None = None):
        """maps: only frames of these."""
        self.paths, self.rng, self.swap_every, self.maps = paths, rng, swap_every, maps
        self.loaded = [self._load(paths[int(i)]) for i in rng.permutation(len(paths))[:in_memory]]
        self.draws = 0

    @staticmethod
    def _read(path: pathlib.Path) -> dict[str, np.ndarray]:
        with np.load(path) as d:
            return {k: d[k] for k in d.files}

    def _load(self, path: pathlib.Path) -> dict:
        s = self._read(path)
        anchors = s["later"] >= 0
        if self.maps is not None:
            anchors &= np.isin(s["map"], self.maps)
        s["anchors"] = np.flatnonzero(anchors)
        return s

    @staticmethod
    def frame(s: dict, i: int) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, _ in GROUPS:
            a, b = s[f"{name}_at"][i], s[f"{name}_at"][i + 1]
            out[f"{name}_x"] = s[f"{name}_x"][a:b].astype(np.float32)
        a, b = s["unit_at"][i], s["unit_at"][i + 1]
        out["unit_name"], out["unit_ally"], out["unit_player"], out["tok_kind"] = s["unit_name"][a:b], s["unit_ally"][a:b], s["unit_player"][a:b], s["unit_kind"][a:b]
        out["glob_x"] = s["glob_x"][i]
        return out

    def _draw(self, count: int) -> list[tuple[dict, int]]:
        self.draws += 1
        if self.draws % self.swap_every == 0:
            self.loaded[int(self.rng.integers(len(self.loaded)))] = self._load(self.paths[int(self.rng.integers(len(self.paths)))])
        usable = [s for s in self.loaded if len(s["anchors"])]
        chosen = [usable[int(self.rng.integers(len(usable)))] for _ in range(count)]
        return [(s, int(s["anchors"][int(self.rng.integers(len(s["anchors"])))])) for s in chosen]

    def sample(self, count: int) -> tuple[FrameBatch, FrameBatch]:
        """Frames and each one's moment-later frame."""
        drawn = self._draw(count)
        return collate_frames([self.frame(s, i) for s, i in drawn]), collate_frames([self.frame(s, int(s["later"][i])) for s, i in drawn])

    def sample_probes(self, count: int) -> tuple[FrameBatch, np.ndarray]:
        """Frames and what's known about each."""
        drawn = self._draw(count)
        return collate_frames([self.frame(s, i) for s, i in drawn]), np.stack([s["probes"][i] for s, i in drawn])



# Is the latent any good? ----------------------------------------------------------------------------------------------------------------------------------


def collapse_report(z: torch.Tensor) -> dict[str, float]:
    """N x K.  Whether the batch's latents still differ, and along how many directions.  Effective rank is the exponential of the singular values' entropy."""
    z = z.float()
    std = z.std(0)
    values = torch.linalg.svdvals(z - z.mean(0))
    p = values / values.sum().clamp_min(1e-9)
    return {"std_mean": float(std.mean()), "std_min": float(std.min()), "effective_rank": float(torch.exp(-(p * (p + 1e-12).log()).sum()))}


def ridge_r2(train_z: np.ndarray, train_y: np.ndarray, test_z: np.ndarray, test_y: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    """A linear probe: fit on one set, R^2 on the other, per column of y, or NaN for a column that hardly varies there (no ships on a dry map).  Standardised first, so that the ridge means the same for every latent."""
    mean, std = train_z.mean(0), train_z.std(0) + 1e-6
    a, b = (train_z - mean) / std, (test_z - mean) / std
    offset = train_y.mean(0)
    w = np.linalg.solve(a.T @ a + ridge * len(a) * np.eye(a.shape[1]), a.T @ (train_y - offset))
    total = ((test_y - test_y.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ((test_y - offset - b @ w) ** 2).sum(0) / np.maximum(total, 1e-9)
    return np.where(test_y.std(0) > 1e-2, r2, np.nan)


@torch.no_grad()
def encode_shards(encoder, paths: list[pathlib.Path], device: str, frames: int, batch: int = 128, each: int = 200) -> dict[str, np.ndarray]:
    """Latents of a sample, with what's known about each frame.  At most `each` frames, evenly spaced, from a shard, since one shard is only a few games.
    encoder: anything that turns a batch into B x size."""
    latents, probes, maps, games, seats, sources = [], [], [], [], [], []
    left = frames
    for path in paths:
        s = Shards._read(path)
        total = len(s["map"])
        n = min(total, left, each)
        chosen = np.linspace(0, total - 1, n).astype(np.int64)
        for start in range(0, n, batch):
            latents.append(encoder(collate_frames([Shards.frame(s, int(i)) for i in chosen[start:start + batch]]).to(device)).float().cpu().numpy())
        probes.append(s["probes"][chosen])
        maps.append(s["map"][chosen])
        games.append(s["game"][chosen])
        seats.append(s["seat"][chosen])
        sources.append(s["source"][chosen])
        left -= n
        if not left:
            break
    return {"z": np.concatenate(latents), "probes": np.concatenate(probes), "map": np.concatenate(maps), "game": np.concatenate(games), "seat": np.concatenate(seats), "source": np.concatenate(sources)}


def probe_report(sample: dict[str, np.ndarray], names: tuple[str, ...], outcomes: tuple[np.ndarray, np.ndarray] | None, held_out: tuple[int, ...] | list[int], groups: dict[str, list[int]] | None = None) -> dict[str, float]:
    """R^2 on maps the probes didn't fit on, and on each of `groups` of them.  Outcome is side-relative: +1 if this frame's seat went on to win.  Only finished games count, and only script games, since a scenario is cut off."""
    test = np.isin(sample["map"], held_out)
    if not test.any() or test.all():
        return {}
    r2 = ridge_r2(sample["z"][~test], sample["probes"][~test], sample["z"][test], sample["probes"][test])
    out = {f"r2_{n}": float(v) for n, v in zip(names, r2) if np.isfinite(v)}
    for group, maps in (groups or {}).items():
        chosen = test & np.isin(sample["map"], maps)
        if chosen.sum() > 64:
            r2 = ridge_r2(sample["z"][~test], sample["probes"][~test], sample["z"][chosen], sample["probes"][chosen])
            out.update({f"r2_{group}_{n}": float(v) for n, v in zip(names, r2) if np.isfinite(v)})
    if outcomes is not None and len(outcomes[0]):
        at = np.searchsorted(outcomes[0], sample["game"])
        at = np.minimum(at, len(outcomes[0]) - 1)
        known = (outcomes[0][at] == sample["game"]) & (sample["source"] == 0)
        result = outcomes[1][at] * np.where(sample["seat"] == 0, 1.0, -1.0)
        fit, check = known & ~test, known & test
        if fit.sum() > 8 * sample["z"].shape[1] and check.sum() > 16:
            out["r2_outcome"] = float(ridge_r2(sample["z"][fit], result[fit, None], sample["z"][check], result[check, None])[0])
    return out


# Training ------------------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    batch: int = 256        # a step costs much the same from 64 to 256, and frames carry SIGReg's test
    steps: int = 100000
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup: int = 2000
    grad_clip: float = 1.0
    report_every: int = 100
    probe_every: int = 5000
    probe_frames: int = 16000
    checkpoint_every: int = 10000
    hours: float = 0.0
    seed: int = 1
    muon_lr: float = 0.0    # above 0, Muon takes the hidden layers' weights at this rate, on the same schedule, and AdamW the rest



def learning_rate(cfg: TrainConfig, step: int) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)))


def collection(data: pathlib.Path) -> tuple[dict, list[pathlib.Path], list[int], dict[str, list[int]], tuple[np.ndarray, np.ndarray]]:
    """What tools/frame_data.py collect wrote: its manifest, shards, maps nothing fits on (tools/frame_world.py held_out_maps), groups of maps to report on, and how games ended."""
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("feature_version") not in fz.ENCODER_FEATURE_VERSIONS:
        raise ValueError(f"{data}: frames of feature version {manifest.get('feature_version')}, this code's encoder reads {fz.ENCODER_FEATURE_VERSIONS}: collect them again")
    groups = {"real": [m for m, k in enumerate(manifest["kinds"]) if k == "real"], "made": [m for m, k in enumerate(manifest["kinds"]) if k == "made"], "wet": [m for m, w in enumerate(manifest["water"]) if w >= 0.03]}
    with np.load(data / "outcomes.npz") as ends:
        outcomes = (ends["game"], ends["result"])
    return manifest, sorted(data.glob("*-*.npz")), manifest["held_out"], groups, outcomes


def save_state(path: pathlib.Path, model: nn.Module, optimiser: torch.optim.Optimizer, step: int) -> None:
    """All a run needs to go on from here, replacing the last.  Interrupted, it leaves the last as it was."""
    partial = path.with_suffix(".partial")
    torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(), "step": step}, partial)
    partial.replace(path)
