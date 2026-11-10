"""A decision round as CUDA graphs: the actor's whole forward pass while playing (`decide`, `sample_actions`, `acted`) and the
critic's, captured once for each padded shape and then replayed (policy_train.Player, with graphs on).

Eager, a round launched thousands of small kernels from Python.  Sampling alone is a loop of up to 16 actions of hundreds of
kernels each: about 0.25 s a round whatever its batch, with one core busy launching and the GPU mostly idle (2026-11-10).  A
replay launches the lot at once.

* **Shapes are padded** to a ladder of widths (`bucket`), each token group on its own, and the decisions to the same ladder.
  Sites, feature tokens and hidden units always pad to their caps.  Padding changes no decision's own outputs, as every reader
  masks it (policy_model.collate); only the random draws fall differently.
* **Each width only grows**: a batch pads to the widest of each group yet seen, so the shape changes only when a batch outgrows
  it.  Padded to its own buckets, a batch of 64 games' frames came to a new shape one round in five, as each group's widest
  game came and went: 95 captures in 494 rounds, at 0.7 s each, and 32 graphs' inputs kept (2026-11-10).  A wider shape costs
  the GPU a little more and launches the same.
* **All 16 action steps run**, done or not (policy_model.sample_actions, every_slot): a graph can't stop early.
* **The weights are read where they lie.**  A graph reads the parameters' memory, which the optimizer and `load_state_dict`
  update in place, so a graph stays right as the nets learn.  Autocast's cache is off while capturing, so the bfloat16 copies
  of the weights are made afresh at each replay.
* **Random numbers** are new at each replay: PyTorch's generator is graph-safe.
* **Memory**: every graph of a `Graphs` shares one pool, and a replay's outputs are copied out before the next.  The oldest
  graph goes past `keep` of them.  A collector of 16 games held 460 MB with 6 kept: 185 MB of pool, and the rest its nets, the
  kept graphs' inputs and outputs, and its seats' state.
"""
from __future__ import annotations

import pathlib
import sys
import time
from collections import OrderedDict

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bots"))
import policy_model as pm  # noqa: E402
from zkbot.policy import actions as act, featurize as fz  # noqa: E402

# Each step at most half as big again as the one before it.
LADDER = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
FIXED = {"s": fz.CAP_SITES, "f": fz.CAP_FEATURES, "h": fz.CAP_HIDDEN}    # featurize's caps: small, so always padded to


def bucket(n: int) -> int:
    """The smallest width on the ladder that holds n; past its top, n rounded up to a multiple of the top's half."""
    for w in LADDER:
        if w >= n:
            return w
    step = LADDER[-1] // 2
    return -(-n // step) * step


def padded(widths: dict[str, int], rows: int) -> tuple[int, dict[str, int]]:
    """The rows and widths a batch of these widths (policy_model.widths) is played at."""
    out = {k: max(v, FIXED[k]) if k in FIXED else bucket(v) for k, v in widths.items()}
    out["u"] = max(out["u"], out["c"])
    return bucket(rows), out


class _Graph:
    """One shape's graph: its inputs, laid out in one block that a single copy fills, and its outputs."""

    def __init__(self, nb: pm.Batch, history_shape: tuple, z_dim: int | None, device: str):
        self.fields = [f for f in pm.Batch.__dataclass_fields__ if getattr(nb, f) is not None]
        arrays = [getattr(nb, f) for f in self.fields]
        offsets, at = [], 0
        for a in arrays:
            offsets.append(at)
            at += -(-a.nbytes // 64) * 64
        # Not pinned: in WSL a collector's cuMemHostAlloc of 20 MB failed as out of memory, with 7 GB of the card free (2026-11-10).
        # A copy from pageable memory waits for the copy, a few ms a round, before the graph launches.
        self.host = torch.empty(max(at, 64), dtype=torch.uint8)
        self.there = torch.empty_like(self.host, device=device)
        flat = self.host.numpy()
        self.host_views = [flat[o:o + a.nbytes].view(a.dtype).reshape(a.shape) for a, o in zip(arrays, offsets)]
        views = {f: self.there[o:o + a.nbytes].view(torch.from_numpy(a[:0].reshape(-1)).dtype).view(a.shape)
                 for f, a, o in zip(self.fields, arrays, offsets)}
        self.batch = pm.Batch(**{f: views.get(f) for f in pm.Batch.__dataclass_fields__})
        self.history = torch.zeros(history_shape, device=device)
        self.seen = torch.zeros(history_shape[:2], device=device)
        self.z = None if z_dim is None else torch.zeros(history_shape[0], z_dim, device=device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.outputs: tuple[torch.Tensor, ...] = ()

    def load(self, nb: pm.Batch, history: torch.Tensor, seen: torch.Tensor, z: torch.Tensor | None) -> None:
        for view, f in zip(self.host_views, self.fields):
            view[...] = getattr(nb, f)
        self.there.copy_(self.host)
        n = len(history)
        self.history[:n].copy_(history)
        self.seen[:n].copy_(seen)
        if self.z is not None:
            self.z[:n].copy_(z)


class Graphs:
    """`policy_train.Player._forward` as CUDA graphs, for a CUDA actor and critic (or no critic)."""

    def __init__(self, actor, critic, slots: int = act.SLOTS, keep: int = 2):
        """keep: the graphs held.  As the widths only grow, a graph narrower than the widest is never played again: what's
        kept is the widest with z and without (a bootstrap or a test without it)."""
        self.actor, self.critic, self.slots, self.keep = actor, critic, slots, keep
        self.device = next(actor.parameters()).device
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: OrderedDict[tuple, _Graph] = OrderedDict()
        self.widest: dict[str, int] = {}        # each group's widest padded width yet, and "rows"
        self.captured = 0
        self.capture_s = 0.0
        self.replays = 0

    def _run(self, g: _Graph) -> tuple[torch.Tensor, ...]:
        b = g.batch
        ctx, new_history, new_seen, keep = self.actor.decide(b, g.history, g.seen, g.z)
        a = pm.sample_actions(self.actor, b, ctx, self.slots, every_slot=True)
        new_history = self.actor.acted(b, a, new_history)
        value = self.critic(b) if self.critic is not None else torch.zeros(len(g.history), device=self.device)
        ints = torch.stack([getattr(a, f) for f in act.SLOT_FIELDS], 1)
        floats = torch.cat((a.logp.float(), value.float()[:, None], keep["latents"].float().flatten(1)), 1)
        return ints, floats, new_history.float(), new_seen.float()

    def _capture(self, key: tuple, nb: pm.Batch, history: torch.Tensor, seen: torch.Tensor, z: torch.Tensor | None) -> _Graph:
        began = time.perf_counter()
        while len(self.graphs) >= self.keep:
            self.graphs.popitem(last=False)
        rows = len(nb.glob_x)
        g = _Graph(nb, (rows, *history.shape[1:]), None if z is None else z.shape[1], self.device)
        g.load(nb, history, seen, z)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(side), autocast:
            for _ in range(2 if not self.captured else 1):      # the first also makes cuBLAS's workspaces, and kind_in's tables
                self._run(g)
        torch.cuda.current_stream().wait_stream(side)
        g.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g.graph, pool=self.pool), autocast:
            g.outputs = self._run(g)
        self.graphs[key] = g
        self.captured += 1
        self.capture_s += time.perf_counter() - began
        return g

    @torch.no_grad()
    def forward(self, items: list[fz.Tensors], history: torch.Tensor, seen: torch.Tensor, z: torch.Tensor | None = None):
        """As Player._forward: numpy actions and latents, numpy values, and the new state (on the device)."""
        n = len(items)
        rows, sizes = padded(pm.widths(items), n)
        sizes["rows"] = rows
        self.widest = {k: max(v, self.widest.get(k, 0)) for k, v in sizes.items()}
        rows = self.widest["rows"]
        nb = pm.collate(items, sizes=self.widest, rows=rows)
        key = (z is None, *(None if v is None else v.shape for v in vars(nb).values()))
        g = self.graphs.get(key)
        if g is None:
            g = self._capture(key, nb, history, seen, z)
        else:
            self.graphs.move_to_end(key)
            g.load(nb, history, seen, z)
        g.graph.replay()
        self.replays += 1
        ints, floats, new_history, new_seen = g.outputs
        ints, floats = ints[:n].cpu().numpy(), floats[:n].cpu().numpy()
        out = {f: ints[:, i] for i, f in enumerate(act.SLOT_FIELDS)}
        out["logp"] = floats[:, :self.slots]
        value = floats[:, self.slots]
        out["latents"] = floats[:, self.slots + 1:].reshape(n, self.actor.encoder.tokens, self.actor.encoder.latent)
        # Copies: the next replay of this graph writes over its outputs.
        return out, value, new_history[:n].clone(), new_seen[:n].clone()

    def stats(self) -> dict[str, float]:
        return {"graphs_captured": self.captured, "graphs_capture_s": self.capture_s, "graph_replays": self.replays}
