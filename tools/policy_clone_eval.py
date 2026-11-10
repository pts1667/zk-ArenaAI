"""A cloned actor playing the army script in the sim: how often it wins, and how its economy and army grow beside the script's
(04-literature.md C8).  With --record, its games are kept for DAgger (P3): every frame it saw, with what the scripts would have
told it there (they only advise its side), as policy_clone_data records the scripts' own games.

    ~/zkml/bin/python tools/policy_clone_eval.py --actor /mnt/d/zk-learned/runs/clone-7/actor.pt --maps /mnt/d/zk-learned/maps --out RESULT.json

Side 0 is the actor, side 1 the script, each game.  Both sides' army scripts draw their knobs (policy_variants.py): the script's
side plays its own, and the actor's side's advises with its own (and, with --z variant, the actor is told them as its z).  The
maps are the pool's training maps, as the data was recorded on.  --policy-economy is the share of games where the actor runs its
own economy (the economy script only advises); in the rest the economy script builds for it too, after its own orders.

The sim steps a slot at a time (2 a second), as the cloning data was recorded, and the actor decides every `decide_every` of its
checkpoint, when the scripts look: so its games are recorded as the scripts' were, and the trainer reads them the same way.

Each game slot counts its first game only, as policy_train.evaluate: the games that end soonest would otherwise count most.
At minutes 3, 6 and 10 of those games it notes each side's metal income, finished extractors and value (zksim stats).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import policy_clone_data as cd  # noqa: E402
import policy_clone_train as ct  # noqa: E402
import policy_latent  # noqa: E402
import policy_model as pm  # noqa: E402
import policy_sim_env as se  # noqa: E402
import policy_train as pt  # noqa: E402
import zksim  # noqa: E402
from zkbot.policy import actions as act, rawframe as rf, tables  # noqa: E402

MINUTES = (3, 6, 10)


def make_env(games: int, seed: int, maps: str | None, decide_every: int, policy_economy: float, advise: bool) -> se.SimEnv:
    """Actor against script, with variants, on the pool's training maps (all of `se.pool` without --maps)."""
    pool = weights = None
    if maps:
        from frame_world import held_out_maps, pool_weights
        pool = se.load_pool(maps)
        weights = pool_weights(pool)
        weights[held_out_maps(pool)] = 0.0
        weights = weights / weights.sum()
        se.share_maps(pool)
    env = se.SimEnv(games, seed, ["fight"], decide_every=decide_every, maps=pool, weights=weights, variants=True)
    env.policy_economy_share = policy_economy
    env.advise = advise
    return env


class Notes:
    """Each game slot's first game: its minutes' measures and how it ended."""

    def __init__(self, env: se.SimEnv):
        self.env = env
        seconds = env.ticks * zksim.TICK_FRAMES / zksim.FRAMES_PER_SECOND
        self.at = {int(round(m * 60.0 / seconds)): m for m in MINUTES}
        self.extractor = env.eco_data.extractor
        self.minutes: dict[int, dict] = {g: {} for g in range(env.count)}
        self.results: dict[int, dict] = {}

    def after_step(self) -> None:
        env = self.env
        for g in range(env.count):
            m = self.at.get(int(env.decisions[g]))
            if m is None or g in self.results:
                continue
            row = {}
            for side, who in ((0, "actor"), (1, "script")):
                u = env.batch.records[g, side, :int(env.batch.counts[g, side])]
                own = u[(u["player"] == side) & ((u["flags"] & rf.BEING_BUILT) == 0)]
                row[who] = {"income": float(env.batch.globs[g, side, rf.G["metal_income"]]),
                            "extractors": int(self.extractor[env.table.rows(own["def_id"])].sum()), "value": float(env.batch.stats[g, side, 0])}
            self.minutes[g][m] = row

    def finished(self, seats: list[tuple[int, str, float]]) -> None:
        for seat, _, won in seats:
            g = seat // 2
            if seat % 2 == 0 and g not in self.results:
                self.results[g] = {"won": won, **self.env.ended.get(seat, {}), "minutes": self.minutes[g]}

    def summary(self) -> dict:
        r = np.array([v["won"] for v in self.results.values()], np.float32)
        out = {"games": len(r), "win_rate": float((r == 1).mean()), "draw_rate": float((r == 0.5).mean()), "loss_rate": float((r == 0).mean())}
        for m in MINUTES:
            for who in ("actor", "script"):
                rows = [v["minutes"][m][who] for v in self.results.values() if m in v["minutes"]]
                for key in ("income", "extractors", "value"):
                    out[f"{who}_{key}_{m}"] = round(float(np.mean([x[key] for x in rows])), 2) if rows else None
            out[f"reached_{m}"] = sum(m in v["minutes"] for v in self.results.values())
        out["how"] = {h: sum(v.get("how") == h for v in self.results.values()) for h in ("eliminated", "lead", "limit")}
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor", required=True, help="a cloning checkpoint (policy_clone_train's actor.pt)")
    ap.add_argument("--maps", help="the map pool; without it, the five training maps")
    ap.add_argument("--out", required=True, help="where the result goes, as JSON")
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--policy-economy", type=float, default=1.0, help="share of games where the actor runs its own economy")
    ap.add_argument("--z", choices=("none", "variant"), default="none",
                    help="what the actor is told of its strategy: nothing, or its side's knobs (the counts unknown)")
    ap.add_argument("--record", help="keep the games here for DAgger: the actor's frames with the scripts' advice as orders")
    ap.add_argument("--record-games", type=int, default=0, help="with --record, play on until this many games are kept (default --games)")
    ap.add_argument("--record-prefix", default="game-dagger",
                    help="the kept games' file names start with this (it must start with game-), so that rounds can share a directory")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--decide-every", type=int, default=se.SCRIPT_EVERY, help="for a PPO checkpoint, which doesn't say")
    ap.add_argument("--encoder", default=policy_latent.ENCODER)
    ap.add_argument("--gpu-memory", type=float, default=0.85)
    args = ap.parse_args()
    assert args.record_prefix.startswith("game-"), "policy_clone_data.sides lists only game-*"
    began = time.perf_counter()
    state = torch.load(args.actor, map_location="cpu", weights_only=False)
    decide_every = int(state["config"]["decide_every"]) if "config" in state else args.decide_every     # PPO's has none
    assert se.SCRIPT_EVERY % decide_every == 0
    env = make_env(args.games, args.seed, args.maps, 1, args.policy_economy, advise=bool(args.record))
    workers = pt.DataWorkers(args.workers, env.bundles(), se.KINDS)       # before the GPU is touched: they fork
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory)
    table = tables.table()
    vocabulary = len(table.names) + 1
    actor = ct.load_actor(args.actor, vocabulary, args.encoder, decide_every, device).eval()
    pm.TEMPERATURE = args.temperature
    player = pt.Player(actor, None, workers, device)
    env.begin()
    rec = cd.Recorder(env, pathlib.Path(args.record), args.record_prefix) if args.record else None
    seats = pt.Seats(env.seat_count, device, actor)
    names = np.arange(vocabulary, dtype=np.int32)
    notes = Notes(env)
    want_kept = args.record_games or args.games
    decisions = 0
    while len(notes.results) < args.games or (rec is not None and not rec.closed):
        if rec is not None and not rec.closing and len(notes.results) >= args.games and rec.games >= want_kept:
            rec.close()
        acting = [s for s in env.policy_seats_to_act() if env.decisions[s // 2] % decide_every == 0]
        frames = [(env.frame(seat), env.bundle_of(seat), names) for seat in acting]
        index = torch.as_tensor(acting, device=device)
        z = None
        if args.z == "variant" and acting:
            z = torch.as_tensor(np.stack([cd.variant_z(env.styles[s // 2][s % 2]) for s in acting]), device=device)
        items, out, _, new_history, new_seen = player.act(frames, seats.history[index], seats.seen[index], z) if acting else ([], {}, None, None, None)
        for k, seat in enumerate(acting):
            t = items[k]
            d = act.to_decision(act.Slots(*(out[h][k] for h in act.SLOT_FIELDS)), t.controllable)
            b = env.bundle_of(seat)
            x, zz = act.world_positions(t, b)
            env.give_orders(seat, act.decode(d, t, b, x, zz))
        if acting:
            seats.history[index] = new_history
            seats.seen[index] = new_seen
        decisions += len(acting)
        if rec is not None:
            finished = rec.step()
        else:
            env.step(1.0)
            notes.after_step()
            finished = env.finished_seats()
        if rec is not None:
            notes.after_step()
        notes.finished(finished)
        for seat, _, _ in finished:
            seats.clear(seat)
    result = {**notes.summary(), "actor": args.actor, "temperature": args.temperature, "policy_economy": args.policy_economy, "z": args.z,
              "seconds": round(time.perf_counter() - began, 1), "decisions": decisions, "kept_games": rec.games if rec else 0,
              "records": list(notes.results.values())}
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    workers.close()
    os._exit(0)


if __name__ == "__main__":
    main()
