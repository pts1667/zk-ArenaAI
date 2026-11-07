"""A frame model: compresses what a seat sees, from up to several hundred tokens, into one vector the main model can read.  Built after LeJEPA.

There's no decoder and no teacher network, and no stop-gradient (arXiv 2511.08544).  Several views of a frame go through one encoder.  Two things are asked of the result:

    invariance  every view's embedding sits near the mean embedding of the global views
    SIGReg      over a batch, embeddings look like a standard normal, tested along random directions

The second is what stops every frame getting the same vector, the trivial way to pass the first.

    python tools/frame_jepa.py train --data /mnt/d/zk-learned/frames/NAME --run /mnt/d/zk-learned/runs/NAME

The encoder reads tokens as the main model's are padded, so `FrameEncoder` takes a `policy_model.Batch` as it stands.  Data: tools/frame_data.py.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bots"))
from policy_model import Block  # noqa: E402
from zkbot.policy import featurize as fz, net as npnet, tables  # noqa: E402

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
    z: int = 256            # the latent: what the main model reads
    position_bands: int = 6
    projector: int = 1024   # width of the projector's hidden layers
    projected: int = 128    # what the losses see


@dataclass(frozen=True)
class View:
    """What one view keeps of a frame.  Each token is kept or not on its own, and the global token always is."""
    keep_units: float = 1.0
    keep_regions: float = 1.0
    keep_sites: float = 1.0
    keep_features: float = 1.0
    unknown_names: float = 0.0  # share of unit names read as unknown, as in training the policy
    position_jitter: float = 0.0    # of positions, in -1..1 of the map
    later: bool = False         # the same seat's next frame, if there is one


# The whole frame and a moment later are the global views, and the rest must match them from less.
GLOBAL_VIEWS = (View(), View(later=True))
LOCAL_VIEWS = (View(0.7, 0.8, 0.8, 0.7, 0.1, 0.005), View(0.7, 0.8, 0.8, 0.7, 0.1, 0.005), View(0.4, 0.5, 0.5, 0.4, 0.1, 0.01), View(0.4, 0.5, 0.5, 0.4, 0.1, 0.01))


@dataclass(frozen=True)
class LossConfig:
    weight: float = 0.02    # lambda: (1 - lambda) invariance + lambda SIGReg, as in the paper's own code
    slices: int = 1024
    points: int = 17
    t_max: float = 3.0


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


def apply_view(b: FrameBatch, view: View, unknown: int, generator: torch.Generator | None = None) -> FrameBatch:
    """Drops tokens by their masks, on whatever device the batch is on."""
    def kept(mask: torch.Tensor, rate: float) -> torch.Tensor:
        if rate >= 1.0:
            return mask
        return mask & (torch.rand(mask.shape, device=mask.device, generator=generator) < rate)

    def jittered(x: torch.Tensor, column: int) -> torch.Tensor:
        if view.position_jitter <= 0.0:
            return x
        x = x.clone()
        x[..., column:column + 2] += view.position_jitter * torch.randn(*x.shape[:-1], 2, device=x.device, generator=generator)
        return x

    names = b.unit_name
    if view.unknown_names > 0.0:
        names = torch.where(torch.rand(names.shape, device=names.device, generator=generator) < view.unknown_names, torch.full_like(names, unknown), names)
    return replace(b, unit_mask=kept(b.unit_mask, view.keep_units), reg_mask=kept(b.reg_mask, view.keep_regions), site_mask=kept(b.site_mask, view.keep_sites),
                   feat_mask=kept(b.feat_mask, view.keep_features), unit_name=names, unit_x=jittered(b.unit_x, POSITION_COLUMN["unit"]),
                   reg_x=jittered(b.reg_x, POSITION_COLUMN["reg"]), site_x=jittered(b.site_x, POSITION_COLUMN["site"]), feat_x=jittered(b.feat_x, POSITION_COLUMN["feat"]))


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


class FrameEncoder(nn.Module):
    """Tokens in, one vector out.  What it gives doesn't depend on the tokens' order, or on padding."""

    def __init__(self, cfg: EncoderConfig, vocabulary: int):
        super().__init__()
        self.cfg, self.vocabulary = cfg, vocabulary
        d = cfg.d
        extra = 4 * cfg.position_bands
        self.name_embed = nn.Embedding(vocabulary, npnet.NAME_DIM)
        self.ally_embed = nn.Embedding(fz.MAX_ALLY_TEAMS + 1, npnet.ID_DIM)
        self.player_embed = nn.Embedding(fz.MAX_PLAYERS + 1, npnet.ID_DIM)
        self.kind_embed = nn.Embedding(fz.TOKEN_KINDS, d)
        self.unit_in0 = nn.Linear(fz.UNIT_DIM + extra + npnet.NAME_DIM + 2 * npnet.ID_DIM, d)
        self.unit_in1 = nn.Linear(d, d)
        self.reg_in = nn.Linear(fz.REGION_DIM + extra, d)
        self.site_in = nn.Linear(fz.SITE_DIM + extra, d)
        self.feat_in = nn.Linear(fz.FEATURE_DIM + extra, d)
        self.glob_in = nn.Linear(fz.GLOBAL_DIM, d)
        self.token_blocks = nn.ModuleList(Block(d, cfg.heads) for _ in range(cfg.token_layers))
        self.queries = nn.Parameter(torch.randn(cfg.queries, d) / math.sqrt(d))
        self.cross = nn.ModuleList(CrossBlock(d, cfg.heads) for _ in range(cfg.rounds))
        self.latent_blocks = nn.ModuleList(Block(d, cfg.heads) for _ in range(cfg.rounds))
        self.lnf = nn.LayerNorm(d)
        self.to_z = nn.Linear(cfg.queries * d, cfg.z)
        self.register_buffer("bands", math.pi * 2.0 ** torch.arange(cfg.position_bands, dtype=torch.float32), persistent=False)

    def _with_position(self, x: torch.Tensor, column: int) -> torch.Tensor:
        """Sines of a token's place, since a linear layer on -1..1 alone can't tell neighbours apart on a large map."""
        at = x[..., column:column + 2].unsqueeze(-1) * self.bands
        return torch.cat((x.clamp(-FEATURE_LIMIT, FEATURE_LIMIT), at.sin().flatten(-2), at.cos().flatten(-2)), -1)

    def tokens(self, b) -> tuple[torch.Tensor, torch.Tensor]:
        """All tokens and which are real: units, regions, sites, features, global."""
        units = torch.cat((self._with_position(b.unit_x, POSITION_COLUMN["unit"]), self.name_embed(b.unit_name), self.ally_embed(b.unit_ally), self.player_embed(b.unit_player)), -1)
        hu = self.unit_in1(F.gelu(self.unit_in0(units), approximate="tanh")) + self.kind_embed(b.tok_kind[:, :b.unit_x.shape[1]])
        hr = self.reg_in(self._with_position(b.reg_x, POSITION_COLUMN["reg"])) + self.kind_embed.weight[GROUP_KINDS["reg"]]
        hs = self.site_in(self._with_position(b.site_x, POSITION_COLUMN["site"])) + self.kind_embed.weight[GROUP_KINDS["site"]]
        hf = self.feat_in(self._with_position(b.feat_x, POSITION_COLUMN["feat"])) + self.kind_embed.weight[GROUP_KINDS["feat"]]
        hg = (self.glob_in(b.glob_x) + self.kind_embed.weight[fz.TK_GLOBAL])[:, None]
        everything = torch.cat((hu, hr, hs, hf, hg), 1)
        mask = torch.cat((b.unit_mask, b.reg_mask, b.site_mask, b.feat_mask, torch.ones_like(b.unit_mask[:, :1])), 1)
        return everything, mask

    def forward(self, b) -> torch.Tensor:
        """B x z.  b: a FrameBatch or a policy_model.Batch."""
        h, mask = self.tokens(b)
        for block in self.token_blocks:
            h = block(h, None, mask)
        q = self.queries.expand(len(h), -1, -1)
        everyone = torch.ones_like(q[:, :, 0], dtype=torch.bool)
        for cross, latent in zip(self.cross, self.latent_blocks):
            q = latent(cross(q, h, mask), None, everyone)
        return self.to_z(self.lnf(q).flatten(1))


class FrameJepa(nn.Module):
    """The encoder and the projector that only training sees."""

    def __init__(self, cfg: EncoderConfig, vocabulary: int):
        super().__init__()
        self.encoder = FrameEncoder(cfg, vocabulary)
        w = cfg.projector
        self.projector = nn.Sequential(nn.Linear(cfg.z, w), nn.LayerNorm(w), nn.GELU(), nn.Linear(w, w), nn.LayerNorm(w), nn.GELU(), nn.Linear(w, cfg.projected))

    def embed_views(self, now: FrameBatch, later: FrameBatch, views: tuple[View, ...], generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Latents and projections, V x B x size, each view in its own forward pass."""
        z = torch.stack([self.encoder(apply_view(later if v.later else now, v, self.encoder.vocabulary - 1, generator)) for v in views])
        v, b, size = z.shape
        return z, self.projector(z.view(v * b, size)).view(v, b, -1)


# The loss -----------------------------------------------------------------------------------------------------------------------------------------------


def epps_pulley(x: torch.Tensor, points: int = 17, t_max: float = 3.0) -> torch.Tensor:
    """N x P -> P.  How far each column's empirical characteristic function is from a standard normal's, as N times a weighted squared distance."""
    t = torch.linspace(0.0, t_max, points, device=x.device, dtype=x.dtype)
    normal = torch.exp(-0.5 * t.square())
    dt = t_max / (points - 1)
    weights = torch.full((points,), 2.0 * dt, device=x.device, dtype=x.dtype)
    weights[0] = weights[-1] = dt
    weights = weights * normal
    at = x.unsqueeze(-1) * t
    error = (at.cos().mean(0) - normal).square() + at.sin().mean(0).square()
    return (error @ weights) * x.shape[0]


def sigreg(x: torch.Tensor, cfg: LossConfig, generator: torch.Generator | None = None) -> torch.Tensor:
    """N x K.  The Epps-Pulley statistic along random directions: a fresh set for each call, as a fixed set would leave the rest unchecked."""
    directions = torch.randn(x.shape[1], cfg.slices, device=x.device, dtype=x.dtype, generator=generator)
    directions = directions / directions.norm(dim=0, keepdim=True)
    return epps_pulley(x @ directions, cfg.points, cfg.t_max).mean()


def invariance(projected: torch.Tensor, global_views: int) -> torch.Tensor:
    """V x B x K, global views first.  Every view against the mean of the global views."""
    centre = projected[:global_views].mean(0)
    return (projected - centre).square().mean()


def lejepa_loss(projected: torch.Tensor, global_views: int, cfg: LossConfig, generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    pull = invariance(projected, global_views)
    spread = sigreg(projected.reshape(-1, projected.shape[-1]), cfg, generator)
    return (1.0 - cfg.weight) * pull + cfg.weight * spread, {"invariance": float(pull.detach()), "sigreg": float(spread.detach())}


# The data ------------------------------------------------------------------------------------------------------------------------------------------------


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
def encode_shards(encoder: FrameEncoder, paths: list[pathlib.Path], device: str, frames: int, batch: int = 128, each: int = 200) -> dict[str, np.ndarray]:
    """Latents of a sample, with what's known about each frame.  At most `each` frames, evenly spaced, from a shard, since one shard is only a few games."""
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


def save_encoder(path: pathlib.Path, model: FrameJepa, step: int) -> None:
    """What the main model would load: the encoder only."""
    torch.save({"cfg": asdict(model.encoder.cfg), "vocabulary": model.encoder.vocabulary, "encoder": model.encoder.state_dict(), "step": step, "feature_version": npnet.FEATURE_VERSION}, path)


def load_encoder(path: pathlib.Path, device: str = "cpu") -> FrameEncoder:
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["feature_version"] != npnet.FEATURE_VERSION:
        raise ValueError(f"{path}: feature version {saved['feature_version']}, this code uses {npnet.FEATURE_VERSION}")
    encoder = FrameEncoder(EncoderConfig(**saved["cfg"]), saved["vocabulary"])
    encoder.load_state_dict(saved["encoder"])
    return encoder.to(device).eval()


def _save_state(path: pathlib.Path, model: FrameJepa, optimiser: torch.optim.Optimizer, step: int) -> None:
    """All a run needs to go on from here, replacing the last.  Interrupted, it leaves the last as it was."""
    partial = path.with_suffix(".partial")
    torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(), "step": step}, partial)
    partial.replace(path)


def _learning_rate(cfg: TrainConfig, step: int) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    return 0.5 * cfg.lr * (1.0 + math.cos(math.pi * (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)))


def _collection(data: pathlib.Path) -> tuple[dict, list[pathlib.Path], list[int], dict[str, list[int]], tuple[np.ndarray, np.ndarray]]:
    """What tools/frame_data.py collect wrote: its manifest, shards, maps nothing fits on (tools/frame_world.py held_out_maps), groups of maps to report on, and how games ended."""
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf8"))
    if manifest.get("feature_version") != npnet.FEATURE_VERSION:
        raise ValueError(f"{data}: frames of feature version {manifest.get('feature_version')}, this code uses {npnet.FEATURE_VERSION}: collect them again")
    groups = {"real": [m for m, k in enumerate(manifest["kinds"]) if k == "real"], "made": [m for m, k in enumerate(manifest["kinds"]) if k == "made"], "wet": [m for m, w in enumerate(manifest["water"]) if w >= 0.03]}
    with np.load(data / "outcomes.npz") as ends:
        outcomes = (ends["game"], ends["result"])
    return manifest, sorted(data.glob("*-*.npz")), manifest["held_out"], groups, outcomes


def train(data: pathlib.Path, run: pathlib.Path, cfg: TrainConfig = TrainConfig(), encoder_cfg: EncoderConfig = EncoderConfig(), loss_cfg: LossConfig = LossConfig(),
          views: tuple[tuple[View, ...], tuple[View, ...]] = (GLOBAL_VIEWS, LOCAL_VIEWS), device: str | None = None) -> FrameJepa:
    """A `stop` file in the run's directory ends it at the next report.  A run's directory that holds a state goes on from there, with this run's random numbers."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    run.mkdir(parents=True, exist_ok=True)
    manifest, paths, held_out, groups, outcomes = _collection(data)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    shards = Shards(paths, rng, maps=tuple(m for m in range(len(manifest["maps"])) if m not in held_out))
    model = FrameJepa(encoder_cfg, len(tables.table().names) + 1).to(device)
    (run / "config.json").write_text(json.dumps({"train": asdict(cfg), "encoder": asdict(encoder_cfg), "loss": asdict(loss_cfg), "data": str(data)}, indent=1), encoding="utf8")
    decay = [p for p in model.parameters() if p.dim() > 1]
    plain = [p for p in model.parameters() if p.dim() <= 1]
    optimiser = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": plain, "weight_decay": 0.0}], lr=cfg.lr)
    chosen = (*views[0], *views[1])
    generator = torch.Generator(device).manual_seed(cfg.seed)
    started = time.time()
    first = 0
    if (run / "state.pt").exists():
        state = torch.load(run / "state.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        first = state["step"] + 1
    step = first - 1
    with open(run / "metrics.jsonl", "a", encoding="utf8") as log:
        for step in range(first, cfg.steps):
            for group in optimiser.param_groups:
                group["lr"] = _learning_rate(cfg, step)
            now, later = shards.sample(cfg.batch)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
                z, projected = model.embed_views(now.to(device), later.to(device), chosen, generator)
            # In full precision: the statistic is N times a small difference.
            loss, parts = lejepa_loss(projected.float(), len(views[0]), loss_cfg, generator)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            optimiser.step()
            if step % cfg.report_every == 0 or step == cfg.steps - 1:
                row = {"step": step, "loss": float(loss.detach()), "grad_norm": grad, "seconds": time.time() - started, "lr": optimiser.param_groups[0]["lr"], **parts, **collapse_report(z[0].detach())}
                if step % cfg.probe_every == 0:     # from the start, for what an untrained encoder gets
                    model.eval()
                    sample = encode_shards(model.encoder, [paths[int(i)] for i in rng.permutation(len(paths))], device, cfg.probe_frames)
                    row.update(probe_report(sample, tuple(manifest["probes"]), outcomes, held_out, groups))
                    model.train()
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(row, flush=True)
                if (run / "stop").exists() or (cfg.hours and time.time() - started > cfg.hours * 3600.0):
                    break
            if step and step % cfg.checkpoint_every == 0:
                save_encoder(run / f"encoder-{step:07d}.pt", model, step)
                _save_state(run / "state.pt", model, optimiser, step)
    save_encoder(run / "encoder-last.pt", model, step)
    _save_state(run / "state.pt", model, optimiser, step)
    return model


def ceiling(data: pathlib.Path, out: pathlib.Path | None = None, cfg: TrainConfig = TrainConfig(steps=4000, warmup=0), encoder_cfg: EncoderConfig = EncoderConfig(), device: str | None = None) -> dict[str, float]:
    """The most a run could score on the probes: this encoder trained on them directly, then scored the same way.  What it can't reach here isn't in the tokens, or not on unseen maps.  What it can, and a run doesn't, the run's objective left out."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest, paths, held_out, groups, outcomes = _collection(data)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    shards = Shards(paths, rng, maps=tuple(m for m in range(len(manifest["maps"])) if m not in held_out))
    encoder = FrameEncoder(encoder_cfg, len(tables.table().names) + 1).to(device)
    head = nn.Linear(encoder_cfg.z, len(manifest["probes"])).to(device)
    every = [*encoder.parameters(), *head.parameters()]
    optimiser = torch.optim.AdamW(every, lr=cfg.lr, weight_decay=cfg.weight_decay)
    known = np.concatenate([s["probes"] for s in shards.loaded])
    mean, std = torch.as_tensor(known.mean(0), device=device), torch.as_tensor(known.std(0) + 1e-6, device=device)
    for step in range(cfg.steps):
        frames, truth = shards.sample_probes(cfg.batch)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
            z = encoder(frames.to(device))
        loss = (head(z.float()) - (torch.as_tensor(truth, device=device) - mean) / std).square().mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(every, cfg.grad_clip)
        optimiser.step()
    encoder.eval()
    sample = encode_shards(encoder, [paths[int(i)] for i in rng.permutation(len(paths))], device, cfg.probe_frames)
    report = probe_report(sample, tuple(manifest["probes"]), outcomes, held_out, groups)
    if out is not None:
        out.write_text(json.dumps(report, indent=1), encoding="utf8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("ceiling")
    c.add_argument("--data", type=pathlib.Path, required=True)
    c.add_argument("--out", type=pathlib.Path)
    c.add_argument("--device", default=None)
    t = sub.add_parser("train")
    t.add_argument("--data", type=pathlib.Path, required=True)
    t.add_argument("--run", type=pathlib.Path, required=True)
    t.add_argument("--steps", type=int, default=TrainConfig.steps)
    t.add_argument("--batch", type=int, default=TrainConfig.batch)
    t.add_argument("--hours", type=float, default=0.0)
    t.add_argument("--z", type=int, default=EncoderConfig.z)
    t.add_argument("--device", default=None)
    t.add_argument("--sigreg-weight", type=float, default=LossConfig.weight)
    args = parser.parse_args()
    if args.command == "ceiling":
        print(json.dumps(ceiling(args.data, args.out, device=args.device), indent=1))
        return
    train(args.data, args.run, TrainConfig(batch=args.batch, steps=args.steps, hours=args.hours), EncoderConfig(z=args.z), LossConfig(weight=args.sigreg_weight), device=args.device)


if __name__ == "__main__":
    main()
