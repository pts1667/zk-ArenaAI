"""The actor on latents: a trained frame encoder's latent tokens in place of all the tokens the first actor read (policy_model.Actor).

    ~/zkml/bin/python tools/policy_train.py sim --run /mnt/d/zk-learned/runs/NAME --latent-encoder /mnt/d/zk-learned/runs/encoder-questions/final/vae-last.pt

* **The encoder is frozen.**  It's trained on its own (tools/frame_vae.py), and sees true names whatever names the actor's dropped this game.
* **A small transformer reads only the latents** and a token of the memory state, however many units and regions there are, so a decision's cost hardly grows with them.
* **Each own unit reads what that made,** with a query of its own raw features, so it knows exactly where it is and what it is.  Nothing else reads, or is read, unit by unit.
* **Pointers score against embeddings of the raw features** of what they point at, so a target's exact place is there for the choosing, whatever the latent blurred.
* **A key ends in sine codes of its place,** and a pointer's query ends in those of the unit's own, each band scaled by a gain the unit chooses.  Their dot product is then
  a sum over bands of gain x cos(band x (difference in place)): a learned function of how far away and in which direction, which a dot product of raw coordinates can't give.
* **Memory is as it was:** a global GRU, whose state makes the memory token, and one for each region, which reads the transformer's output with that region's query.

`units` from `encode` is B x U x (D + KEY): each token's reading, 0 for those that didn't read, then its key.  It keeps one tensor so the trainer can treat it as the first actor's.
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bots"))
import policy_model as pm  # noqa: E402
from frame_jepa import CrossBlock  # noqa: E402
from frame_vae import LatentEncoder, load_vae  # noqa: E402
from zkbot.policy import featurize as fz, net as npnet, rawframe as rf, tables  # noqa: E402

D = 128
HEADS = 4
LAYERS = 2
BANDS = 6                   # as the encoder's: the finest wave is 1/16 of the map's -1..1
CODE = 4 * BANDS            # sin and cos of each of x and z
KEY = D + CODE
UNIT_PLACE = (fz._I, fz._I + 2)
REGION_PLACE = (0, 2)
FIELDS = frozenset(("unit_x", "unit_name", "true_name", "unit_ally", "unit_player", "unit_mask", "reg_x", "reg_mask", "site_x", "site_mask", "feat_x", "feat_mask", "glob_x", "tok_kind", "ctrl_mask"))


class LatentActor(nn.Module):
    encoder_fields = FIELDS
    uses_pairs = False

    def __init__(self, vocabulary: int, encoder: LatentEncoder):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("bands", math.pi * 2.0 ** torch.arange(BANDS, dtype=torch.float32), persistent=False)
        self.latent_in = nn.Linear(encoder.latent, D)
        self.slots = nn.Parameter(torch.randn(encoder.tokens, D) / math.sqrt(D))    # a latent token's place in the list means something, and a shared layer loses it
        self.mem_in = nn.Linear(npnet.GLOBAL_STATE, D)
        self.blocks = nn.ModuleList(pm.Block(D, HEADS) for _ in range(LAYERS))
        self.lnf = nn.LayerNorm(D)
        self.name_embed = nn.Embedding(vocabulary, npnet.NAME_DIM)
        self.ally_embed = nn.Embedding(fz.MAX_ALLY_TEAMS + 1, npnet.ID_DIM)
        self.player_embed = nn.Embedding(fz.MAX_PLAYERS + 1, npnet.ID_DIM)
        self.kind_embed = nn.Embedding(fz.TOKEN_KINDS, D)
        self.unit_in0 = nn.Linear(fz.UNIT_DIM + CODE + npnet.NAME_DIM + 2 * npnet.ID_DIM, D)
        self.unit_in1 = nn.Linear(D, D)
        self.reg_in = nn.Linear(fz.REGION_DIM + CODE, D)
        self.reg_mem = nn.Linear(npnet.REGION_STATE, D)
        self.read_unit = CrossBlock(D, HEADS)
        self.read_region = CrossBlock(D, HEADS)
        self.gru_g = pm.Gru(D, npnet.GLOBAL_STATE)
        self.gru_r = pm.Gru(D, npnet.REGION_STATE)
        self.key_unit = nn.Linear(D, D)
        self.key_region = nn.Linear(D, D)
        self.head_kind = nn.Linear(D, rf.KIND_COUNT)
        self.q_enemy = nn.Linear(D, D)
        self.q_region = nn.Linear(D, D)
        self.gain_enemy = nn.Linear(D, 2 * BANDS)
        self.gain_region = nn.Linear(D, 2 * BANDS)
        self.q_type = nn.Linear(D, D)
        table = tables.table()
        self.type_key = nn.Linear(tables.TYPE_DIM, D)
        self.register_buffer("produce_stats", torch.from_numpy(table.stats[table.producible].copy()), persistent=False)
        self.fine0 = nn.Linear(D + KEY, D)
        self.fine1 = nn.Linear(D, fz.FINE_OPTIONS)
        self.route_scale = nn.Parameter(torch.tensor(npnet.ROUTE_SCALE_START))
        nn.init.normal_(self.head_kind.weight, std=npnet.HEAD_GAIN / math.sqrt(D))
        nn.init.zeros_(self.head_kind.bias)
        with torch.no_grad():
            self.head_kind.bias[rf.KEEP] = npnet.KEEP_START

    def code(self, place: torch.Tensor) -> torch.Tensor:
        """... x 2 to ... x CODE: every sine, then every cosine."""
        at = place.unsqueeze(-1) * self.bands
        return torch.cat((at.sin().flatten(-2), at.cos().flatten(-2)), -1)

    def encode(self, b, glob: torch.Tensor, regions: torch.Tensor, pair_grad: float = 1.0):
        """As `policy_model.Actor.encode`.  There's no pair bias for pair_grad to scale."""
        h = self.latent_in(self.encoder(b)) + self.slots
        h = torch.cat((h, self.mem_in(glob)[:, None]), 1)
        everyone = torch.ones_like(h[:, :, 0], dtype=torch.bool)
        for block in self.blocks:
            h = block(h, None, everyone)
        h = self.lnf(h)

        place = b.unit_x[..., UNIT_PLACE[0]:UNIT_PLACE[1]]
        raw = torch.cat((b.unit_x, self.code(place), self.name_embed(b.unit_name), self.ally_embed(b.unit_ally), self.player_embed(b.unit_player)), -1)
        hu = self.unit_in1(F.gelu(self.unit_in0(raw), approximate="tanh")) + self.kind_embed(b.tok_kind[:, :b.unit_x.shape[1]])
        count = min(b.ctrl_mask.shape[1], hu.shape[1])
        me = self.read_unit(hu[:, :count], h, everyone)
        units = torch.cat((F.pad(me, (0, 0, 0, hu.shape[1] - count)), self.key_unit(hu), self.code(place).to(hu.dtype)), -1)

        centre = b.reg_x[..., REGION_PLACE[0]:REGION_PLACE[1]]
        hr = self.reg_in(torch.cat((b.reg_x, self.code(centre)), -1)) + self.reg_mem(regions)
        new_regions = self.gru_r(self.read_region(hr, h, everyone), regions)
        new_glob = self.gru_g(h[:, -1], glob)
        return h, units, torch.cat((self.key_region(hr), self.code(centre).to(hr.dtype)), -1), new_glob, new_regions

    def _score(self, query: torch.Tensor, gain: torch.Tensor, own: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        scaled = gain.repeat(1, 1, 2) * own.to(gain.dtype)
        return (torch.cat((query, scaled), -1) @ keys.transpose(1, 2)).float() / math.sqrt(D)

    def heads(self, b, units: torch.Tensor, reg_h: torch.Tensor):
        """As `policy_model.Actor.heads`."""
        count = b.kind_mask.shape[1]
        me, keys = units[:, :count, :D], units[..., D:]
        own = self.code(b.unit_x[:, :count, UNIT_PLACE[0]:UNIT_PLACE[1]])
        kind = self.head_kind(me).float().masked_fill(~b.kind_mask, pm.ILLEGAL)
        region = self._score(self.q_region(me), self.gain_region(me), own, reg_h)
        region = region - F.softplus(self.route_scale).float() * b.route_bias
        enemy_h = torch.gather(keys, 1, b.enemy_tokens[:, :, None].expand(-1, -1, KEY))
        enemy = self._score(self.q_enemy(me), self.gain_enemy(me), own, enemy_h)
        enemy = enemy[:, None].expand(-1, b.ptr_enemy.shape[1], -1, -1).masked_fill(~b.ptr_enemy, pm.ILLEGAL)
        produce = (self.q_type(me) @ self.type_key(self.produce_stats).T).float() / math.sqrt(D)
        return kind, region, enemy, enemy_h, produce.masked_fill(~b.ptr_produce, pm.ILLEGAL)

    def fine(self, me: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.fine1(F.gelu(self.fine0(torch.cat((me[..., :D], target), -1)), approximate="tanh")).float()


def make(vocabulary: int, encoder_path: str | pathlib.Path) -> LatentActor:
    return LatentActor(vocabulary, LatentEncoder(load_vae(pathlib.Path(encoder_path))))


def trainable_parameters(actor: nn.Module) -> list[nn.Parameter]:
    return [p for p in actor.parameters() if p.requires_grad]


def parameter_report(actor: LatentActor) -> dict[str, int]:
    trained = sum(p.numel() for p in trainable_parameters(actor))
    frozen = sum(p.numel() for p in actor.encoder.parameters())
    return {"trained": trained, "encoder": frozen, "total": trained + frozen}

