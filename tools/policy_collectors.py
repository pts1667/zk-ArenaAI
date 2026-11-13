"""Collector processes for PPO (policy_train.py --collectors N): each plays its share of the games, with its own copy of the
nets, and its share of the tests.  A decision round in one process is mostly Python on one core (stepping the sim, the scripts'
advice, the actions' decoding, a forward pass's launches), so N processes do N rounds at once.

* **Forked before the GPU is touched**, after the map pool is loaded, as the data workers are: they share the pool's maps, and
  each starts CUDA for itself.  Each holds at most `collector_gb` of the card, which the learner's own limit leaves room for.
* **The weights go through shared memory** (`Weights`): before each collection the learner writes its nets' whole state into a
  block, which every collector copies into its own nets.  The tests' weights, the moving average, have a block of their own.
* **Segments come back by pipe**, each naming its map by its place in the pool, as the learner's data workers hold it.
* **Each plays with CUDA graphs** (policy_graphs.py), and featurizes in its own process.
* **With --concurrent** the next iteration is collected while this one trains, by the weights from before its update: as the
  in-process `Ahead`, but on other cores.
* **A collector that fails** sends its traceback, which the learner raises.  They are daemons: they end with the learner.
"""
from __future__ import annotations

import functools
import multiprocessing
import os
import pathlib
import signal
import sys
import time
import traceback
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.multiprocessing  # noqa: F401  (registers the reductions that send a shared tensor down a pipe)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import policy_latent  # noqa: E402
import policy_model as pm  # noqa: E402
import policy_train as pt  # noqa: E402


def split(total: int, parts: int) -> list[int]:
    """total as `parts` whole numbers, the first ones a larger by one where it doesn't divide."""
    return [total // parts + (k < total % parts) for k in range(parts)]


def seed_of(seed: int, k: int) -> int:
    """Collector k's own seed from a run's."""
    return int(np.random.SeedSequence((seed, k)).generate_state(1)[0])


class Weights:
    """Some modules' whole state, in one block of shared memory, each tensor at its place in `layout`."""

    def __init__(self, modules: dict[str, torch.nn.Module]):
        self.layout: list[tuple[str, str, torch.dtype, tuple[int, ...], int, int]] = []
        at = 0
        for section, m in modules.items():
            for name, t in m.state_dict().items():
                size = t.numel() * t.element_size()
                self.layout.append((section, name, t.dtype, tuple(t.shape), at, size))
                at += -(-size // 64) * 64
        self.block = torch.zeros(max(at, 64), dtype=torch.uint8).share_memory_()

    def _views(self) -> dict[tuple[str, str], torch.Tensor]:
        return {(s, name): self.block[at:at + size].view(dtype).view(shape) for s, name, dtype, shape, at, size in self.layout}

    @torch.no_grad()
    def write(self, modules: dict[str, torch.nn.Module]) -> None:
        views = self._views()
        for section, m in modules.items():
            for name, t in m.state_dict().items():
                views[(section, name)].copy_(t)

    @torch.no_grad()
    def read(self, modules: dict[str, torch.nn.Module]) -> None:
        """Into the modules' own tensors, in place: a CUDA graph reads them where they are."""
        views = self._views()
        for section, m in modules.items():
            for name, t in m.state_dict().items():
                t.copy_(views[(section, name)])


@torch.no_grad()
def fingerprint(modules: dict[str, torch.nn.Module]) -> float:
    """A sum over every number of the modules' state, in float64: the same for the same weights, near enough."""
    return float(sum(t.double().abs().sum() + t.double().sum() for m in modules.values() for t in m.state_dict().values() if t.is_floating_point()))


class Shell:
    """What a collector holds in place of the learner, for policy_train.Collector and policy_train.evaluate: its nets, as last
    sent, and where the run had got to then."""

    def __init__(self, cfg: pt.Config, actor, critic, device: str, vocabulary: int, bundles: list, graphs: bool):
        self.cfg, self.actor, self.critic, self.device, self.vocabulary = cfg, actor, critic, device, vocabulary
        self.term_names = tuple(name for name, _ in cfg.terms)
        self.player = pt.Player(actor, critic, pt.DataWorkers(0, bundles, cfg.enabled_kinds), device, graphs)
        self.evaluator = self.player        # with the tests' weights loaded
        self.progress = 0.0
        self.economy_share = 0.0
        self.advising = False


class _Serving:
    """A collector process's side: commands from the learner's pipe, one at a time."""

    def __init__(self, conn, k: int, cfg: pt.Config, make_env):
        self.conn, self.k, self.cfg, self.make_env = conn, k, cfg, make_env

    def run(self) -> None:
        while True:
            command, args = self.conn.recv()
            if command == "stop":
                return
            getattr(self, command)(**args)

    def start(self, size: dict, vocabulary: int, games: int, seed: int, iteration: int, play: Weights, test: Weights, threads: int,
              device: str, gb: float, economy: float, self_play: float, lead: float) -> None:
        cfg = self.cfg
        if device == "cuda":
            torch.cuda.set_per_process_memory_fraction(min(1.0, gb * 2**30 / torch.cuda.mem_get_info()[1]))
        torch.manual_seed(seed)
        actor = policy_latent.make(vocabulary, cfg.latent_encoder, cfg.decide_every, policy_latent.Size(**size)).to(device)
        critic = pm.Critic(vocabulary, bins=cfg.critic_bins, span=cfg.critic_span, terms=len(cfg.terms)).to(device)
        for p in list(actor.parameters()) + list(critic.parameters()):
            p.requires_grad_(False)
        self.nets = {"actor": actor, "critic": critic}
        self.play, self.test, self.device = play, test, device
        self.play.read(self.nets)
        self.make_env = functools.partial(self.make_env, threads=threads)
        env = self.make_env(games, seed)
        env.policy_economy_share, env.self_play_share, env.lead_share = economy, self_play, lead
        self.index = {id(b): i for i, b in enumerate(env.bundles())}
        self.shell = Shell(cfg, actor, critic, device, vocabulary, env.bundles(), cfg.graphs)
        self.collector = pt.Collector(self.shell, env, np.random.default_rng((cfg.seed, iteration, self.k)))
        self.conn.send(("started", None))

    def _memory(self) -> dict[str, float]:
        return {"collector_reserved_gb": torch.cuda.max_memory_reserved() / 2**30} if self.device == "cuda" else {}

    def collect(self, want: int, progress: float, economy: float, self_play: float, advising: bool, lead: float) -> None:
        began = time.perf_counter()
        self.play.read(self.nets)
        weights = fingerprint(self.nets)
        self.shell.progress, self.shell.economy_share, self.shell.advising = progress, economy, advising
        env = self.collector.env
        env.policy_economy_share, env.self_play_share, env.lead_share = economy, self_play, lead
        segments, tally = self.collector.collect_tally(want)
        segments = [replace(s, bundle=self.index[id(s.bundle)]) for s in segments]
        self.conn.send(("collected", {"segments": segments, "tally": tally, "seconds": time.perf_counter() - began, "weights": weights,
                                      **self._memory()}))

    def evaluate(self, opponent: str, games: int, seed: int, economy: float, replays: dict | None, kind: str = "1v1") -> None:
        self.test.read(self.nets)
        self.shell.economy_share = economy
        result = pt.evaluate(self.shell, self.make_env, opponent, games, seed, replays, kind)
        self.conn.send(("evaluated", result))


def _serve(conn, k: int, cfg: pt.Config, make_env) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl-C is the learner's: it stops these
    # Before any op: a child forked after the parent ran a parallel CPU op hangs in OpenMP's first parallel region, whose threads
    # it doesn't have (2026-11-10).  It has little CPU work for torch anyway.
    torch.set_num_threads(1)
    try:
        _Serving(conn, k, cfg, make_env).run()
    except EOFError:        # the learner is gone
        pass
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except OSError:
            pass


class Collectors:
    """The learner's side: `cfg.collectors` processes, forked now.  `start` once the learner exists, then `collect` and `result`
    each iteration, and `evaluate` for a test."""

    def __init__(self, cfg: pt.Config, make_env, bundles: list):
        self.cfg, self.bundles, self.count = cfg, bundles, cfg.collectors
        context = multiprocessing.get_context("fork")
        self.conns, self.procs = [], []
        for k in range(self.count):
            mine, theirs = context.Pipe()
            proc = context.Process(target=_serve, args=(theirs, k, cfg, make_env), name=f"collector-{k}", daemon=True)
            proc.start()
            theirs.close()
            self.conns.append(mine)
            self.procs.append(proc)
        self.stash: list[list[tuple[str, object]]] = [[] for _ in range(self.count)]
        self.collecting = False
        self.sent = 0.0

    def _send(self, k: int, message: tuple) -> None:
        """To collector k; one that has died says why, if it could."""
        try:
            self.conns[k].send(message)
        except OSError:
            self._expect(k, "nothing")      # raises with its traceback, or how it ended
            raise

    def _expect(self, k: int, tag: str):
        """Collector k's next message of this kind; any other that comes first is kept for its turn."""
        for i, (t, payload) in enumerate(self.stash[k]):
            if t == tag:
                return self.stash[k].pop(i)[1]
        while True:
            try:
                t, payload = self.conns[k].recv()
            except (EOFError, OSError):
                self.procs[k].join(5)
                raise RuntimeError(f"collector {k} died (exit code {self.procs[k].exitcode})") from None
            if t == "error":
                raise RuntimeError(f"collector {k} failed:\n{payload}")
            if t == tag:
                return payload
            self.stash[k].append((t, payload))

    def _nets(self, player) -> dict[str, torch.nn.Module]:
        return {"actor": player.actor, "critic": player.critic}

    def start(self, learner: pt.Learner, seed: int, device: str) -> None:
        """Each collector's nets and games: its share of `cfg.games`, from its own seed."""
        nets = self._nets(learner.player)
        self.play, self.test = Weights(nets), Weights(nets)
        self.play.write(nets)
        threads = max(2, (os.cpu_count() or 16) // self.count)
        for k, games in enumerate(split(self.cfg.games, self.count)):
            self._send(k, ("start", {"size": asdict(learner.actor.size), "vocabulary": learner.vocabulary, "games": games, "seed": seed_of(seed, k),
                                 "iteration": learner.iteration, "play": self.play, "test": self.test, "threads": threads, "device": device,
                                 "gb": self.cfg.collector_gb, "economy": learner.economy_share, "self_play": learner.self_play.share,
                                 "lead": learner.lead_share}))
        for k in range(self.count):
            self._expect(k, "started")

    def collect(self, learner: pt.Learner, want: int) -> None:
        """Starts a collection of about `want` decisions, by the learner's weights as they are now."""
        assert not self.collecting
        nets = self._nets(learner.player)
        self.play.write(nets)
        self.sent_weights = fingerprint(nets)
        for k, part in enumerate(split(want, self.count)):
            self._send(k, ("collect", {"want": part, "progress": learner.progress, "economy": learner.economy_share, "self_play": learner.self_play.share,
                                       "advising": learner.advising, "lead": learner.lead_share}))
        self.collecting, self.sent = True, time.perf_counter()

    def result(self) -> tuple[list[pt.Segment], dict[str, float], float]:
        """The collection's segments, its metrics, and how long it took the slowest collector."""
        parts = [self._expect(k, "collected") for k in range(self.count)]
        self.collecting = False
        for k, part in enumerate(parts):
            if abs(part["weights"] - self.sent_weights) > 1e-6 * abs(self.sent_weights):
                raise RuntimeError(f"collector {k} played by other weights than were sent: {part['weights']!r} against {self.sent_weights!r}")
        segments = [replace(s, bundle=self.bundles[s.bundle]) for part in parts for s in part["segments"]]
        metrics = pt.Tally.merge([part["tally"] for part in parts]).metrics()
        seconds = [part["seconds"] for part in parts]
        metrics["collector_seconds_min"], metrics["collector_seconds_max"] = min(seconds), max(seconds)
        if any("collector_reserved_gb" in part for part in parts):
            metrics["collector_reserved_gb"] = max(part.get("collector_reserved_gb", 0.0) for part in parts)
        return segments, metrics, max(seconds)

    def evaluate(self, learner: pt.Learner, opponent: str, games: int, seed: int, replays: dict | None = None, kind: str = "1v1") -> dict:
        """policy_train.evaluate, its games shared out; with replays, all in the first collector, whose games the replays number."""
        self.test.write(self._nets(learner.evaluator))
        shares = [games] if replays is not None else split(games, self.count)
        asked = [k for k, n in enumerate(shares) if n]
        for k in asked:
            self._send(k, ("evaluate", {"opponent": opponent, "games": shares[k], "seed": seed_of(seed, k), "economy": learner.economy_share,
                                             "replays": replays, "kind": kind}))
        parts = [self._expect(k, "evaluated") for k in asked]
        total = sum(p["games"] for p in parts)
        out = {name: sum(p[name] * p["games"] for p in parts) / max(total, 1) for name in ("win_rate", "draw_rate", "loss_rate")}
        return {**out, "games": total, "records": [r for p in parts for r in p["records"]]}

    def close(self) -> None:
        """Each stops; one still busy with a collection no one will read is ended."""
        for conn in self.conns:
            try:
                conn.send(("stop", {}))
            except OSError:
                pass
        deadline = time.perf_counter() + (0.0 if self.collecting else 10.0)
        for proc in self.procs:
            proc.join(max(0.0, deadline - time.perf_counter()))
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
