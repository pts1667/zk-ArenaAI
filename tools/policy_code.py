"""An actor that chooses a code and reads no unit (the user, 2026-11-19; docs/learned-policy/05-order-autoencoder.md).

The frozen frame encoder's latents, a memory as policy_latent.LatentActor has it (an RDNN's or a GRU's state, told as one
vector), and a code head: a Gaussian over the order autoencoder's code (order_vae.py), whose frozen decoder says the
decision's commands (zkbot.policy.selections).  Nothing here reads a unit token, a region or a site.  `selections.decode`
does, to carry the commands out, and the frame encoder did, to make the latents.

    latents ──┬─────────────────────────────► 1 block among them ──► K code queries read them ──► mean (K x latent)
              └─► memory ◄── the last code                               ▲                           + a learnt spread
                    └────────────── its vector ───────────────────────────┘

It keeps the names of LatentActor's memory path that the trainer calls (`decide`, `acted`, `segments`, `memory_at`,
`initial_state`, `record`, `gru`, `memory`), so the collectors, the critics of the memory and GRPO go on as they are.
What differs is what a decision is: one code, with one probability.  `policy_latent.make` gives one for a size with an
order autoencoder (`Size.coder`; `--size code`).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import order_vae
import policy_latent as pl
import policy_model as pm

# A spread can't grow past the unit Gaussian's, which the autoencoder was pulled towards, or shrink past a least: a tenth of
# it unless the trainer says another (`CodeActor.log_std_least`, policy_train.Config.spread_floor).
LOG_STD_LEAST = math.log(0.1)
LOG_STD_MOST = 0.0


class CodeActor(nn.Module):
    def __init__(self, encoder: nn.Module, size: pl.Size):
        super().__init__()
        if not size.coder or not size.gru or size.gru % 2 or size.memory not in ("gru", "rdnn"):
            raise ValueError(f"a code's actor needs the order autoencoder's file and a memory of an even width: {size}")
        self.encoder, self.size = encoder, size
        self.d, self.gru, self.reach = size.d, size.gru, 1
        self.decoder = order_vae.load(size.coder)       # frozen, as the encoder is
        self.codes, self.latent = self.decoder.cfg.codes, self.decoder.cfg.latent
        self.code = self.codes * self.latent
        d = self.d
        self.latent_in = nn.Linear(encoder.latent, d)
        self.slots = nn.Parameter(torch.randn(encoder.tokens, d) / math.sqrt(d))        # which latent token this is
        self.z_in = nn.Linear(pl.Z_DIM, d)                                              # the objective a seat is told: nothing at first
        nn.init.zeros_(self.z_in.weight)
        nn.init.zeros_(self.z_in.bias)
        self.frame_blocks = nn.ModuleList(pm.Block(d, size.heads) for _ in range(size.layers))
        self.lnf = nn.LayerNorm(d)
        units = size.gru // 2 if size.memory == "rdnn" else size.gru
        wide = units // 2
        self.mem_units = units
        self.mem_frame = nn.Linear(encoder.tokens * encoder.latent, wide)
        self.mem_acts = nn.Linear(self.code, wide)      # what the decision before did: its code
        self.mem_ln = nn.LayerNorm(wide)
        self.memory = pl.RDNNCell(wide, units) if size.memory == "rdnn" else nn.GRUCell(wide, size.gru)
        self.mem_out = nn.Linear(units, d)
        self.code_queries = nn.Parameter(torch.randn(self.codes, d) / math.sqrt(d))
        self.code_blocks = nn.ModuleList(pl.PlanBlock(d, size.heads) for _ in range(size.plan_layers))
        self.code_ln = nn.LayerNorm(d)
        self.code_mean = nn.Linear(d, self.latent)
        # At first every code is the unit Gaussian's, whatever the frame: what the autoencoder was pulled towards.
        nn.init.zeros_(self.code_mean.weight)
        nn.init.zeros_(self.code_mean.bias)
        self.code_log_std = nn.Parameter(torch.zeros(self.codes, self.latent))
        # The spread's least, as a log: the trainer lowers it as a run goes on (the user, 2026-11-19: "Start with 0.1,
        # decreasing to 0.03 as the training run continues").  A buffer, so a collector's copy and a checkpoint have it.
        self.register_buffer("log_std_least", torch.tensor(LOG_STD_LEAST))

    @property
    def record(self) -> int:
        """What a seat keeps: the memory's state, then its last decision's code."""
        return self.gru + self.code

    def initial_state(self, seats: int, regions: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(seats, self.reach, self.record, device=device), torch.zeros(seats, self.reach, device=device)

    # The memory, as LatentActor's -----------------------------------------------------------------------------------------------------------------

    def _memory_in(self, latents: torch.Tensor, did: torch.Tensor) -> torch.Tensor:
        return self.mem_ln(self.mem_frame(latents.flatten(-2)) + self.mem_acts(did.to(latents.dtype)))

    def _step(self, taken: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        with torch.autocast(taken.device.type, enabled=False):
            return self.memory(taken.float(), state.float())

    def _steps(self, taken: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if isinstance(self.memory, pl.RDNNCell):
            with torch.autocast(taken.device.type, enabled=False):
                return self.memory.run(taken.float(), state.float())
        states = []
        for t in range(taken.shape[1]):
            state = self._step(taken[:, t], state)
            states.append(state)
        return torch.stack(states, 1)

    def memory_at(self, b, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A frame's latents and the memory's state after it, from what its seat kept: what a critic of the memory reads."""
        now = self.encoder(b).float()
        return now, self._step(self._memory_in(now, history[:, 0, self.gru:]), history[:, 0, :self.gru])

    # The code ----------------------------------------------------------------------------------------------------------------------------------------

    def mean(self, latents: torch.Tensor, state: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        """N x K x latent: each decision's code's mean, from its frame's latents (N x tokens x latent) and its memory's state."""
        added = self.z_in.bias[None] if z is None else self.z_in(z.to(self.z_in.weight.dtype))
        x = self.latent_in(latents) + self.slots + added[:, None]
        everyone = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        for block in self.frame_blocks:
            x = block(x, None, everyone)
        x = self.lnf(x)
        q = self.code_queries + self.mem_out(state[:, :self.mem_units]).to(x.dtype)[:, None]
        for block in self.code_blocks:
            q = block(q, x)
        # In float32 whatever the precision around it: a code's probability is of how far it is from this, to a hundredth.
        with torch.autocast(q.device.type, enabled=False):
            return self.code_mean(self.code_ln(q.float()))

    def log_std(self) -> torch.Tensor:
        """K x latent: the spread's log, within its bounds.  Past one, the parameter is read as the bound and still learns its
        way back."""
        raw = self.code_log_std
        return raw + (torch.maximum(raw, self.log_std_least).clamp(max=LOG_STD_MOST) - raw).detach()

    def log_prob(self, mean: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """N: each code's log density under its decision's Gaussian."""
        log_std = self.log_std()
        return (-0.5 * ((code - mean) / log_std.exp()).square() - log_std - 0.5 * math.log(2.0 * math.pi)).flatten(1).sum(1)

    def entropy(self) -> torch.Tensor:
        """The Gaussian's, the same at every decision: the spread is the actor's own, not the frame's."""
        return (self.log_std() + 0.5 * (1.0 + math.log(2.0 * math.pi))).sum()

    def sample(self, mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code = mean + torch.randn_like(mean) * self.log_std().exp()
        return code, self.log_prob(mean, code)

    def say(self, code: torch.Tensor) -> dict[str, torch.Tensor]:
        """The commands a code stands for, as order_vae.to_arrays has them: the frozen decoder's, in float32 as it was trained
        (a place in half precision is a 500th of the map off)."""
        with torch.autocast(code.device.type, enabled=False):
            return self.decoder.say(code.float())

    # Play and training -----------------------------------------------------------------------------------------------------------------------------

    def decide(self, b, history: torch.Tensor, seen: torch.Tensor, z: torch.Tensor | None = None):
        """A decision while playing: its code's mean, what to keep for next time (the code itself comes with `acted`), and what
        training will need of it."""
        now = self.encoder(b).float()
        state = self._step(self._memory_in(now, history[:, 0, self.gru:]), history[:, 0, :self.gru])
        kept = torch.cat((state, torch.zeros(len(state), self.code, device=state.device)), 1)[:, None]
        return self.mean(now, state, z), kept, torch.ones_like(seen), {"latents": now}

    def acted(self, code: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        out = history.clone()
        out[:, 0, self.gru:] = code.flatten(1).float()
        return out

    def segments(self, history: torch.Tensor, latents: torch.Tensor, lengths: list[int], codes: torch.Tensor, z: torch.Tensor | None = None,
                 keep: dict | None = None) -> torch.Tensor:
        """Training's: segments of decisions, each from the state kept at its start.  latents: B x T x tokens x latent, padded at
        the end; codes: B x T x K x latent, the codes drawn.  Returns each decision's mean, sum(lengths) x K x latent, segment by
        segment.  keep gets "memory": each decision's latents and the state after it."""
        count = latents.shape[1]
        did = torch.cat((history[:, :1, self.gru:].to(codes.dtype), codes.flatten(2)[:, :-1]), 1)      # what the decision before did
        states = self._steps(self._memory_in(latents, did), history[:, 0, :self.gru])
        real = torch.arange(count, device=latents.device)[None, :] < torch.as_tensor(lengths, device=latents.device)[:, None]
        told = None if z is None else z[:, None].expand(-1, count, -1)[real]
        own, after = latents[real], states[real]
        if keep is not None:
            keep["memory"] = (own, after)
        return self.mean(own, after, told)
