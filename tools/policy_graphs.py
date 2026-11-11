"""A decision round as CUDA graphs: the actor's whole forward pass while playing (`decide`, `sample_actions`, `acted`) and the
critic's, captured once for each padded shape and then replayed (policy_train.Player, with graphs on).

Eager, a round launched thousands of small kernels from Python.  Sampling alone is a loop of up to 16 actions of hundreds of
kernels each: about 0.25 s a round whatever its batch, with one core busy launching and the GPU mostly idle (2026-11-10).  A
replay launches the lot at once.

* **Shapes are padded** to a ladder of widths (`bucket`), each token group on its own, and the decisions to the same ladder.
  Sites and hidden units always pad to their caps; feature tokens, whose cap is 512, take the ladder as units do.  Padding
  changes no decision's own outputs, as every reader masks it (policy_model.collate); only the random draws fall differently.
* **Each width only grows**: a batch pads to the widest of each group yet seen, so the shape changes only when a batch outgrows
  it.  Padded to its own buckets, a batch of 64 games' frames came to a new shape one round in five, as each group's widest
  game came and went: 95 captures in 494 rounds, at 0.7 s each, and 32 graphs' inputs kept (2026-11-10).  A wider shape costs
  the GPU a little more and launches the same.
* **The action steps stop at the first of `CUTS` where every decision is done.**  A graph can't stop of itself, so a shape has
  several: the decision with the steps up to the first cut (`head`), the steps from each cut to the next (`chunks`), and what
  ends the round (`tail`: `acted`, and the outputs gathered).  At each cut the host asks whether any decision is still going.
  The steps write in place (policy_model.Sampling), so each graph reads what the one before left.
  Measured with full-1's PPO actor at iteration 88 (2026-11-10): a decision took 3.2 steps on average, but in a round of 64
  the longest took 14 at the median, and 1.3% of decisions took all 16.  A step cost about 1 ms however few its rows (it's
  hundreds of small kernels), 1.6 ms at 64, beside 41 ms for the decision itself.  So these cuts came to 21 ms of steps a
  round instead of 25 at 64 rows, and 12 instead of 19 at 16 (a collector's share of 64 games), where steps are most of it.
  Stepping only the rows still going after a cut (gathered, stepped, put back) came to 17 and 11 ms, but took a graph for
  each count of rows at each cut: 4 collectors held 1.8 GB more of the card, outside PyTorch's count and each one's cap.
* **The weights are read where they lie.**  A graph reads the parameters' memory, which the optimizer and `load_state_dict`
  update in place, so a graph stays right as the nets learn.  Autocast's cache is off while capturing, so the bfloat16 copies
  of the weights are made afresh at each replay.
* **Random numbers** are new at each replay: PyTorch's generator is graph-safe.
* **Memory**: every graph of a `Graphs` shares one pool, and a replay's outputs are copied out before the next.  What one graph
  leaves for the next (the decision's context, the steps' state, the tail's outputs) is held, so that no later capture puts
  anything there; the rest of a graph's memory is its own only while it plays, and a round plays one shape's graphs.  The
  oldest shape goes past `keep` of them.  A collector of 16 games held 460 MB with 6 graphs kept: 185 MB of pool, and the
  rest its nets, the kept graphs' inputs and outputs, and its seats' state.
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
FIXED = {"s": fz.CAP_SITES, "h": fz.CAP_HIDDEN}    # featurize's caps: small, so always padded to
CUTS = (4, 6, 8, 10, 12, 14)    # where the action steps may stop: the head runs to the first, each chunk to the next


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


class _Shape:
    """One shape's graphs and their inputs, laid out in one block that a single copy fills; and what each graph leaves for the
    next: the decision's context, the steps' state, and the tail's outputs."""

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
        self.head: torch.cuda.CUDAGraph | None = None
        self.chunks: dict[int, torch.cuda.CUDAGraph] = {}     # by the cut it starts at
        self.tail: torch.cuda.CUDAGraph | None = None
        self.ctx = self.sampling = None
        self.decided: tuple[torch.Tensor, ...] = ()     # the head's: new history, new seen, latents, value
        self.outputs: tuple[torch.Tensor, ...] = ()     # the tail's

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

    def __init__(self, actor, critic, slots: int = act.SLOTS, keep: int = 2, cuts: tuple[int, ...] = CUTS):
        """keep: the shapes held.  As the widths only grow, a shape narrower than the widest is never played again: what's
        kept is the widest with z and without (a bootstrap or a test without it)."""
        self.actor, self.critic, self.slots, self.keep = actor, critic, slots, keep
        self.cuts = tuple(c for c in cuts if 0 < c < slots) + (slots,)
        self.device = next(actor.parameters()).device
        self.pool = torch.cuda.graph_pool_handle()
        self.shapes: OrderedDict[tuple, _Shape] = OrderedDict()
        self.widest: dict[str, int] = {}        # each group's widest padded width yet, and "rows"
        self.captured = 0
        self.capture_s = 0.0
        self.replays = 0
        self.steps = 0                          # the action steps played, over every replay

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)

    def _capture(self, run, warmups: int = 1) -> tuple[torch.cuda.CUDAGraph, object]:
        """run, warmed up on a side stream and then captured; with what the capture returned (memory the graph writes)."""
        began = time.perf_counter()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(side), self._autocast():
            for _ in range(warmups):
                run()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, pool=self.pool), self._autocast():
            out = run()
        self.capture_s += time.perf_counter() - began
        return graph, out

    def _head(self, shape: _Shape):
        b = shape.batch
        ctx, new_history, new_seen, keep = self.actor.decide(b, shape.history, shape.seen, shape.z)
        s = pm.start_sampling(b, ctx, self.slots)
        pm.sample_steps(self.actor, b, ctx, s, range(self.cuts[0]))
        value = self.critic(b) if self.critic is not None else torch.zeros(len(shape.history), device=self.device)
        return ctx, s, (new_history, new_seen, keep["latents"], value)

    def _tail(self, shape: _Shape) -> tuple[torch.Tensor, ...]:
        new_history, new_seen, latents, value = shape.decided
        a = shape.sampling.actions()
        new_history = self.actor.acted(shape.batch, a, new_history)
        ints = torch.stack([getattr(a, f) for f in act.SLOT_FIELDS], 1)
        floats = torch.cat((a.logp.float(), value.float()[:, None], latents.float().flatten(1)), 1)
        return ints, floats, new_history.float(), new_seen.float()

    def _new_shape(self, nb: pm.Batch, history: torch.Tensor, seen: torch.Tensor, z: torch.Tensor | None) -> _Shape:
        while len(self.shapes) >= self.keep:
            self.shapes.popitem(last=False)
        shape = _Shape(nb, (len(nb.glob_x), *history.shape[1:]), None if z is None else z.shape[1], self.device)
        shape.load(nb, history, seen, z)
        # The first warm-up also makes cuBLAS's workspaces, and kind_in's tables.
        shape.head, (shape.ctx, shape.sampling, shape.decided) = self._capture(lambda: self._head(shape), 2 if not self.captured else 1)
        self.captured += 1
        return shape

    def _chunk(self, shape: _Shape, i: int) -> torch.cuda.CUDAGraph:
        """The steps from cut i to the next, captured when first needed, once the graphs before it have played this round.  Its
        warm-up plays those steps on the round's own state, which is put back before the replay."""
        s = shape.sampling
        state = [*s.out.values(), s.logp, s.taken, s.going, s.vectors]
        saved = [t.clone() for t in state]
        steps = range(self.cuts[i], self.cuts[i + 1])
        graph, _ = self._capture(lambda: pm.sample_steps(self.actor, shape.batch, shape.ctx, s, steps))
        for t, before in zip(state, saved):
            t.copy_(before)
        shape.chunks[i] = graph
        return graph

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
        shape = self.shapes.get(key)
        if shape is None:
            shape = self.shapes[key] = self._new_shape(nb, history, seen, z)
        else:
            self.shapes.move_to_end(key)
            shape.load(nb, history, seen, z)
        shape.head.replay()
        steps = self.cuts[0]
        for i in range(len(self.cuts) - 1):
            if not bool(shape.sampling.going[:n].any()):
                break
            graph = shape.chunks.get(i)
            if graph is None:
                graph = self._chunk(shape, i)
            graph.replay()
            steps = self.cuts[i + 1]
        if shape.tail is None:
            shape.tail, shape.outputs = self._capture(lambda: self._tail(shape))
        shape.tail.replay()
        self.replays += 1
        self.steps += steps
        ints, floats, new_history, new_seen = shape.outputs
        ints, floats = ints[:n].cpu().numpy(), floats[:n].cpu().numpy()
        out = {f: ints[:, i] for i, f in enumerate(act.SLOT_FIELDS)}
        out["logp"] = floats[:, :self.slots]
        value = floats[:, self.slots]
        out["latents"] = floats[:, self.slots + 1:].reshape(n, self.actor.encoder.tokens, self.actor.encoder.latent)
        # Copies: the next replay of this graph writes over its outputs.
        return out, value, new_history[:n].clone(), new_seen[:n].clone()

    def stats(self) -> dict[str, float]:
        """So far, as totals, so that several collectors' add up."""
        return {"graphs_captured": self.captured, "graphs_capture_s": self.capture_s, "graph_replays": self.replays, "graph_steps": self.steps}
