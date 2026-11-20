"""The actor that chooses a code (policy_code.py), cloned from the scripts before it learns by RL (the user, 2026-11-20:
"start the actor cloned"; docs/learned-policy/05-order-autoencoder.md, "A cloned start").

    policy_code_clone.py record --out DIR --decisions N [--actor FILE] [--spread S] [--modes mirror] [--workers 6] [--games 16]
        [--seed 1] [--maps /mnt/d/zk-learned/maps] [--kinds 1v1:0.5 teams:0.2 ffa:0.2 chicken:0.1]
    policy_code_clone.py train --data DIR [DIR ...] --out DIR [--steps 3000] [--spread 0.3] [--init FILE]

`record` plays the trainer's games (policy_train.make_sim: the sim at every slot, a decision every 3 s), each of the policy's
seats with its own economy and the scripts advising it, and keeps of each seat's game every decision's latents, as the
frozen frame encoder gave them to the trainer's own pass, and the scripts' advice there as commands (selections.encode), up
to 16.  Without --actor the advice is what's carried out, said as commands and read back: the scripts' own games, in the
commands' terms.  With one, that actor plays, at its own spread or --spread, and the advice is only kept: what the scripts
would have done where the clone's play leads (DAgger), beside the code it drew.  A worker writes part-wNN.npz; summary.json
says what was played.  Its workers share the map pool and each takes --gb of the card: nothing beside it.

`train` teaches an actor's mean to say the advice.  A step is --batch seats' games side by side, --length decisions of
each, from where the step before left each one's memory, as a run's segments follow one another.  Each decision's mean,
off by a draw of up to --spread, is read by the frozen decoder and set against the advice's commands
(order_vae.reconstruction, the autoencoder's own loss), and --code-weight of its squared distance from the advice's own
code is added.  What the memory is told the decision before did is the code of what was done: the advice's own, or the
actor's draw.  It writes actor.pt for `policy_train.py --init`, its spread --spread, and log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "bots"), str(ROOT / "sim" / "py")]
import order_vae  # noqa: E402
import policy_latent as pl  # noqa: E402
import policy_sim_env as se  # noqa: E402
import policy_train as pt  # noqa: E402
from zkbot.policy import actions as act, selections as sel, tables  # noqa: E402

DECIDE_EVERY = 6        # slots a decision: 3 s, when the scripts look, as the runs have it (results/launch-code.sh)
ENCODER = "/mnt/d/zk-learned/runs/full-4/encoder/vae-last.pt"
EMPTY = np.zeros(0, sel.COMMAND_DTYPE)
# The most of the card `train` may hold (AGENTS.md, "Traps": past the card the driver spills into system memory), and the
# decisions that go through the autoencoder at once outside a step.
CARD_GB = 6.0
CHUNK = 4096


def _env(args: argparse.Namespace, games: int, seed: int, threads: int):
    return pt.make_sim(games, seed, modes=list(args.modes), decide_every=DECIDE_EVERY, maps=args.maps, fine_sim=True, threads=threads,
                       kinds=se.parse_kinds(args.kinds) if args.kinds else None, slots=0)


def load_actor(path: str, encoder: str, coder: str = "") -> torch.nn.Module:
    """An actor that chooses a code, from a file with its size and weights: `train`'s actor.pt, or a run's checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    size = pl.Size(**{**state["size"], **({"coder": coder} if coder else {})})
    actor = pl.make(len(tables.table().names) + 1, encoder, DECIDE_EVERY, size)
    actor.load_state_dict(state["actor"])
    return actor


def _record_part(args: argparse.Namespace, worker: int, decisions: int) -> dict:
    torch.set_num_threads(1)        # before any op: a forked child hangs in the parent's parallel regions (policy_collectors._serve)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gb * 2**30 / torch.cuda.mem_get_info()[1]))
    seed = args.seed * 1000 + worker
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = _env(args, args.games, seed, max(1, args.threads // args.workers))
    env.advise, env.policy_economy_share = True, 1.0
    vocabulary = len(tables.table().names) + 1
    own = bool(args.actor)
    if own:
        actor = load_actor(args.actor, args.encoder, args.coder)
        if args.spread > 0.0:
            with torch.no_grad():
                actor.code_log_std.fill_(math.log(args.spread))
    else:       # read for its frame encoder's latents alone, as the trainer's pass gives them
        actor = pl.make(vocabulary, args.encoder, DECIDE_EVERY, replace(pl.CODE_SIZE, **({"coder": args.coder} if args.coder else {})))
    actor = actor.to(device).requires_grad_(False)
    player = pt.Player(actor, None, pt.DataWorkers(0, env.bundles(), se.KINDS, args.featurize_threads), device, graphs=True)
    seats = pt.Seats(env.seat_count, device, actor)
    stash: dict[int, tuple] = {}            # this decision's: seat -> its tensors, its map, its units' places and last orders
    said: dict[int, np.ndarray] = {}        # and the advice that was carried out, as commands

    def through(seat: int, advice: list) -> list:
        if seat not in stash:
            return []
        t, bundle, x, z, last = stash[seat]
        said[seat] = sel.encode(advice, t, bundle, x, z, slots=None, last=last)[0][:sel.SLOTS]
        return sel.decode(said[seat], t, bundle, x, z)

    if not own:
        env.through = through
    env.begin()
    games: dict[int, dict] = {}             # seat -> its game so far
    out = {"latents": [], "counts": [], "commands": [], "played": [], "lengths": [], "map": []}
    total = commands_said = fights = 0
    won = 0.0
    began = time.perf_counter()

    def close(seat: int) -> None:
        game = games.pop(seat, None)
        seats.clear(seat)
        if game and game["latents"]:
            out["latents"].append(np.stack(game["latents"]))
            out["counts"].append(np.array([len(c) for c in game["labels"]], np.uint8))
            out["commands"].extend(game["labels"])
            out["played"].extend(game["played"])
            out["lengths"].append(len(game["latents"]))
            out["map"].append(game["map"])

    while total < decisions:
        acting = env.policy_seats_to_act()
        stash.clear()
        said.clear()
        if acting:
            for seat in acting:
                at = int(env.decisions[env.game_of(seat)])
                if seat in games and at <= games[seat]["at"]:       # a new game that no end told of
                    close(seat)
                if seat not in games:
                    bundle = env.bundle_of(seat)
                    games[seat] = {"latents": [], "labels": [], "played": [], "names": pt.name_map(rng, vocabulary),
                                   "map": (float(bundle.map_w), float(bundle.map_h))}
                games[seat]["at"] = at
            frames = [(env.frame(seat), env.bundle_of(seat), games[seat]["names"]) for seat in acting]
            index = torch.as_tensor(acting, device=device)
            items, got, _, new_history, new_seen = player.act(frames, seats.history[index], seats.seen[index])
            commands = pt.said_commands(got)
            for k, seat in enumerate(acting):
                t, bundle = items[k], frames[k][1]
                x, z = act.world_positions(t, bundle)
                stash[seat] = (t, bundle, x, z, sel.last_orders(t, frames[k][0].units))
                if own:
                    env.give_orders(seat, sel.decode(commands[k], t, bundle, x, z))
                    commands_said += len(commands[k])
            if own:
                seats.history[index], seats.seen[index] = new_history, new_seen
        env.step(1.0)
        if acting:
            (army, ao), (eco, eo) = env.army_orders, env.eco_orders
            for k, seat in enumerate(acting):
                label = said.get(seat, EMPTY)
                if own:
                    advice = np.concatenate((army[ao[seat]:ao[seat + 1]], eco[eo[seat]:eo[seat + 1]]))
                    if len(advice):
                        t, bundle, x, z, last = stash[seat]
                        label = sel.encode(se.policy_orders(advice), t, bundle, x, z, slots=None, last=last)[0][:sel.SLOTS]
                    games[seat]["played"].append(got["code"][k].astype(np.float32))
                games[seat]["latents"].append(got["latents"][k].astype(np.float16))
                games[seat]["labels"].append(label)
            total += len(acting)
        for seat, mode, result in env.finished_seats():
            if mode == "fight" and env.team_of(seat) == 0:
                fights, won = fights + 1, won + result
            close(seat)
    for seat in list(games):        # what was still being played: a game's start is a game's start
        close(seat)
    arrays = {"latents": np.concatenate(out["latents"]), "counts": np.concatenate(out["counts"]),
              "commands": np.concatenate(out["commands"]) if out["commands"] else EMPTY, "lengths": np.array(out["lengths"], np.int32),
              "map": np.array(out["map"], np.float32).reshape(-1, 2)}
    if own:
        arrays["played"] = np.stack(out["played"])
    np.savez(pathlib.Path(args.out) / f"part-w{worker:02d}.npz", **arrays)
    counts = arrays["counts"]
    return {"decisions": int(len(counts)), "games": int(len(arrays["lengths"])), "commands": int(counts.sum()), "empty": int((counts == 0).sum()),
            "actor_commands": commands_said, "fights": fights, "fights_won": won, "seconds": time.perf_counter() - began,
            "card_gb": torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else 0.0}


def record(args: argparse.Namespace) -> None:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _env(args, 1, args.seed, 1)         # the map pool, loaded once before the workers fork, with no thread of the sim's running
    share = -(-args.decisions // args.workers)
    if args.workers == 1:
        parts = [_record_part(args, 0, share)]
    else:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
            parts = list(pool.map(_record_part, [args] * args.workers, range(args.workers), [share] * args.workers))
    summary = {k: sum(p[k] for p in parts) for k in ("decisions", "games", "commands", "empty", "actor_commands", "fights", "fights_won")}
    summary.update(seconds=max(p["seconds"] for p in parts), card_gb=max(p["card_gb"] for p in parts), actor=args.actor, spread=args.spread,
                   modes=list(args.modes), maps=args.maps, kinds=list(args.kinds or ()), seed=args.seed, workers=args.workers, games_at_once=args.games)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


# Training ------------------------------------------------------------------------------------------------------------------------------------------


def load_data(dirs: list[str], coder: order_vae.OrderVAE, device: str) -> dict:
    """Recordings as arrays in memory, a row a decision, seats' games one after another: `latents` (float16), `labels` (the
    advice, as order_vae.to_arrays has decisions), `codes` (the advice's own code: the autoencoder's mean of it), `played`
    (the code of what was done: the actor's draw, where one played, and else the advice's own), `map` (the decision's map,
    in elmos), and each game's `starts` and `lengths`."""
    parts = [dict(np.load(p)) for d in dirs for p in sorted(pathlib.Path(d).glob("part-*.npz"))]
    assert parts, f"no parts in {dirs}"
    slots = coder.cfg.slots
    decisions = []
    for p in parts:
        decisions += np.split(p["commands"], np.cumsum(p["counts"].astype(np.int64))[:-1]) if len(p["counts"]) else []
    labels = order_vae.to_arrays(decisions, slots)
    n = len(decisions)
    codes = np.zeros((n, coder.cfg.codes, coder.cfg.latent), np.float32)
    with torch.no_grad():
        for at in range(0, n, CHUNK):
            b = {k: torch.as_tensor(v[at:at + CHUNK]).to(device) for k, v in labels.items()}
            codes[at:at + CHUNK] = coder.encode(b)[0].cpu().numpy()
    played, at = codes.copy(), 0
    for p in parts:
        if "played" in p:
            played[at:at + len(p["counts"])] = p["played"]
        at += len(p["counts"])
    lengths = np.concatenate([p["lengths"] for p in parts]).astype(np.int64)
    return {"latents": np.concatenate([p["latents"] for p in parts]), "labels": labels, "codes": codes, "played": played,
            "map": np.repeat(np.concatenate([p["map"] for p in parts]), lengths, 0), "starts": np.cumsum(lengths) - lengths, "lengths": lengths}


class Streams:
    """Seats' games side by side, each gone through `length` decisions a step from where the step before left its memory,
    as a run's segments follow one another (policy_train.Segment.history).  With rng, games drawn without end, the first of
    each stream from somewhere in it with nothing kept, so that a step isn't all openings; without, each of `games` once, in
    order, and a stream with none left drops out."""

    def __init__(self, data: dict, games: np.ndarray, batch: int, length: int, record: int, device: str, rng: np.random.Generator | None = None):
        self.starts, self.lengths, self.length, self.rng, self.device = data["starts"][games], data["lengths"][games], length, rng, device
        self.queue = list(range(len(games)))[::-1]
        self.game = np.full(batch, -1, np.int64)
        self.at = np.zeros(batch, np.int64)
        self.history = torch.zeros(batch, 1, record, device=device)
        for s in range(batch):
            self._next(s, first=True)

    def _next(self, s: int, first: bool = False) -> None:
        self.history[s] = 0.0
        self.at[s] = 0
        if self.rng is None:
            self.game[s] = self.queue.pop() if self.queue else -1
        else:
            self.game[s] = self.rng.integers(len(self.starts))
            if first:
                self.at[s] = self.rng.integers(self.lengths[self.game[s]])

    def step(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The streams still going, each one's decisions this step (rows of the data, -1 past its game's end), and how many."""
        rows = np.flatnonzero(self.game >= 0)
        n = np.minimum(self.lengths[self.game[rows]] - self.at[rows], self.length)
        index = (self.starts[self.game[rows]] + self.at[rows])[:, None] + np.arange(self.length)[None]
        index[np.arange(self.length)[None] >= n[:, None]] = -1
        return rows, index, n

    def advance(self, rows: np.ndarray, n: np.ndarray, kept: torch.Tensor) -> None:
        """kept: what each of the step's streams keeps after its last decision, as the actor's record has it."""
        self.history[torch.as_tensor(rows, device=self.device), 0] = kept
        self.at[rows] += n
        for s in rows[self.at[rows] >= self.lengths[self.game[rows]]]:
            self._next(int(s))


def means(actor, data: dict, streams: Streams, autocast) -> tuple[torch.Tensor, np.ndarray] | None:
    """A step of the streams: each decision's mean (decisions x K x latent) and its row of the data, stream by stream; None
    when no stream is left."""
    rows, index, n = streams.step()
    if not len(rows):
        return None
    device = streams.device
    at = np.maximum(index, 0)
    latents = torch.as_tensor(data["latents"][at]).to(device).float()
    played = torch.as_tensor(data["played"][at]).to(device)
    kept: dict = {}
    with autocast:
        mean = actor.segments(streams.history[torch.as_tensor(rows, device=device)], latents, n.tolist(), played, None, keep=kept)
    last = torch.as_tensor(np.cumsum(n) - 1, device=device)
    did = played[torch.arange(len(rows), device=device), torch.as_tensor(n - 1, device=device)].flatten(1)
    streams.advance(rows, n, torch.cat((kept["memory"][1][last].detach().float(), did), 1))
    return mean, index[index >= 0]


def batch_of(data: dict, at: np.ndarray, device: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """These decisions' advice, as the autoencoder takes it, and its own code."""
    return {k: torch.as_tensor(v[at]).to(device) for k, v in data["labels"].items()}, torch.as_tensor(data["codes"][at]).to(device)


@torch.no_grad()
def held_out(actor, data: dict, games: np.ndarray, batch: int, length: int, device: str, autocast) -> dict[str, float]:
    """On games kept from training, each gone through whole: what the mean itself says against the advice
    (order_vae.Agreement), its loss by field, and how far it is from the advice's own code, a number of the code."""
    streams = Streams(data, games, min(batch, len(games)), length, actor.record, device)
    tally = order_vae.Agreement(actor.decoder.cfg)
    parts: dict[str, float] = {}
    count = 0
    while (got := means(actor, data, streams, autocast)) is not None:
        mean, at = got
        b, code = batch_of(data, at, device)
        tally.add(actor.say(mean), b, torch.as_tensor(data["map"][at], device=device))
        with torch.autocast(device, enabled=False):
            lost = order_vae.reconstruction(actor.decoder, actor.decoder.decode(mean.float()), b)
        lost["code_off"] = (mean.float() - code).square().mean().sqrt()
        for k, v in lost.items():
            parts[k] = parts.get(k, 0.0) + float(v) * len(at)
        count += len(at)
    out = {k: v / count for k, v in parts.items()}
    out["loss"] = sum(v for k, v in out.items() if k != "code_off")
    sums = tally.sums
    return {**out, **tally.rates(), "commands_a_decision": sums["commands"] / sums["decisions"], "said_a_decision": sums["said in all"] / sums["decisions"]}


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, CARD_GB * 2**30 / torch.cuda.mem_get_info()[1]))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.init:
        actor = load_actor(args.init, args.encoder, args.coder)
    else:
        actor = pl.make(len(tables.table().names) + 1, args.encoder, DECIDE_EVERY, replace(pl.CODE_SIZE, **({"coder": args.coder} if args.coder else {})))
    actor = actor.to(device)
    began = time.time()
    data = load_data(args.data, actor.decoder, device)
    games = rng.permutation(len(data["lengths"]))
    keep = max(1, round(len(games) * args.held_out))
    test, learn = np.sort(games[:keep]), games[keep:]
    counts = data["labels"]["present"].sum(1)
    print(f"{len(counts)} decisions of {len(games)} seats' games, {int(data['lengths'][test].sum())} of them in the {keep} held out; "
          f"{counts.mean():.2f} commands a decision, {(counts == 0).mean():.3f} with none; read in {time.time() - began:.0f} s", flush=True)
    trained = [p for p in actor.parameters() if p.requires_grad]
    print(f"{sum(p.numel() for p in trained)} parameters trained", flush=True)
    opt = torch.optim.AdamW(trained, lr=args.lr, weight_decay=0.0)
    warm = min(300, max(1, args.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(s / args.steps, 1.0)))))
    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda")
    streams = Streams(data, learn, args.batch, args.length, actor.record, device, rng)
    log = (out / "log.jsonl").open("w")
    sums: dict[str, float] = {}
    seen = steps = 0
    began = time.time()

    def save() -> None:
        with torch.no_grad():       # what it starts a run with (policy_train's --spread-start gives another)
            actor.code_log_std.fill_(math.log(args.spread))
        torch.save({"size": asdict(actor.size), "actor": actor.state_dict(), "spread": args.spread, "data": list(args.data)}, out / "actor.pt")

    for step in range(args.steps + 1):
        mean, at = means(actor, data, streams, autocast)
        b, code = batch_of(data, at, device)
        # Off by a draw of the spread it will play at, or less: each decision's own share of it, from none to all.
        drawn = mean + args.spread * torch.rand(len(mean), 1, 1, device=device) * torch.randn_like(mean)
        parts = order_vae.reconstruction(actor.decoder, actor.decoder.decode(drawn), b)
        parts["code"] = (mean - code).square().sum((1, 2)).mean()
        total = sum(v for k, v in parts.items() if k != "code") + args.code_weight * parts["code"]
        opt.zero_grad(set_to_none=True)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(trained, args.grad_clip)
        opt.step()
        sched.step()
        seen, steps = seen + len(at), steps + 1
        for k, v in {"loss": total, "grad_norm": norm, **parts}.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach())
        if step and step % args.report == 0 or step == args.steps:
            row = {"step": step, "decisions": seen, "seconds": time.time() - began, **{k: v / steps for k, v in sums.items()}}
            sums, steps = {}, 0
            if step % (args.report * args.test_every) == 0 or step == args.steps:
                row["held_out"] = held_out(actor, data, test, args.batch, args.length, device, autocast)
                save()
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "held_out"}), flush=True)
            if "held_out" in row:
                print("  held out:", json.dumps({k: round(v, 3) for k, v in row["held_out"].items()}), flush=True)
    save()
    if device == "cuda":
        print(f"card: {torch.cuda.max_memory_reserved() / 2**30:.2f} GB of {CARD_GB:g}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("--decisions", type=int, required=True)
    r.add_argument("--actor", default="", help="an actor that plays (train's actor.pt, a run's checkpoint), the scripts only advising; without, the advice is played")
    r.add_argument("--spread", type=float, default=0.0, help="with --actor: the spread it plays at; 0: its own")
    r.add_argument("--modes", nargs="+", default=["mirror"], help="policy_sim_env's: mirror (every player's team is recorded), fight (team 0, against the scripts)")
    r.add_argument("--games", type=int, default=16, help="a worker's, at a time")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--threads", type=int, default=16)
    r.add_argument("--featurize-threads", type=int, default=4)
    r.add_argument("--gb", type=float, default=0.8, help="of the card, a worker")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--maps", default="")
    r.add_argument("--kinds", nargs="*", default=None)
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--init", default="", help="an actor to go on from (an earlier actor.pt); without, a new one")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=64, help="seats' games side by side")
    t.add_argument("--length", type=int, default=32, help="decisions of each a step: 64 x 32 hold 2.5 GB of the card, 128 x 64 didn't fit 6")
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--spread", type=float, default=0.3, help="the spread the actor will start a run at: its mean is trained off by up to a draw of it")
    t.add_argument("--code-weight", type=float, default=0.05, help="of the mean's squared distance from the advice's own code")
    t.add_argument("--held-out", type=float, default=0.05, help="the share of the seats' games kept from training")
    t.add_argument("--report", type=int, default=50)
    t.add_argument("--test-every", type=int, default=10, help="reports between looks at the held-out games")
    t.add_argument("--seed", type=int, default=1)
    for p in (r, t):
        p.add_argument("--encoder", default=ENCODER, help="the frozen frame encoder")
        p.add_argument("--coder", default="", help="the order autoencoder; without, the actor's own (policy_latent.ORDER_CODER)")
    args = parser.parse_args(argv)
    record(args) if args.command == "record" else train(args)


if __name__ == "__main__":
    main()
