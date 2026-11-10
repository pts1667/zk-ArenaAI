"""A full training run, stage after stage: the frame encoder (optional), cloning, DAgger rounds, then PPO from the last clone.

    ~/zkml/bin/python tools/policy_pipeline.py --root /mnt/d/zk-learned/runs/full-1 --train-encoder
    ~/zkml/bin/python tools/policy_pipeline.py --root /mnt/d/zk-learned/runs/full-1 --dry-run     # the commands, nothing run

Each stage is one of the tools' own commands (HANDOFF.md section 4), run one at a time, so no two load the map pool at once and
each hands the GPU back when it ends.  Its output goes to ROOT/logs/STAGE.out.  ROOT/pipeline.json lists the stages done, so the
same command again carries on from the first that isn't; --redo STAGE runs that one and every one after it again.  The encoder
(frame_vae) and PPO resume from their own checkpoints; a cloning stage starts over, its old log.jsonl kept beside the new one.

A `stop` file in ROOT halts the pipeline once the running stage ends.  One in a training's own directory (ROOT/encoder,
ROOT/ppo) stops that training at its next report, and the pipeline with it, that stage not marked done.

The stages:
  frames       with --train-encoder, if --frames has no manifest.json: frame_data.py collect, pool5's recipe (1M frames each of
               the frame games, scenarios and the C++ scripts' games; 03-training.md 11.17)
  encoder      with --train-encoder: frame_vae.py train with encoder-big's settings, 25,000 steps, to ROOT/encoder
  zindex       the cloning data's z.npz, if it has none
  clone        policy_clone_train.py with clone-8's settings (z, the economy at 2.5, the last 10% on winners), to ROOT/clone
  record-N     DAgger round N: the last clone plays the army script, told its variant as z, with the scripts advising it, and
               --dagger-games of its games are kept (policy_clone_eval.py --record).  Every round's go into ROOT/dagger, so each
               round learns from them all.  The first 64 games are the last clone's closed-loop eval too: ROOT/evals/record-N.json
  zindex-N     z.npz over every round's games
  dagger-N     policy_clone_train.py --init the last clone: half its samples from ROOT/dagger, half from --data, to ROOT/dagger-N
  eval         the last clone against the army script, z from its variant: ROOT/evals/final.json
  ppo          policy_train.py --init the last clone, --z variant, --fine-sim, with the PPO smoke run's flags (03-training.md
               11.22) at a run's size, to ROOT/ppo

--clone-args, --dagger-args and --ppo-args are added at the end of those commands, so they override what's set here.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"

# encoder-big's (03-training.md 11.17).
ENCODER_FLAGS = ["--autoencoder", "--region-pool", "32", "--two-part", "--questions", "--muon-lr", "0.008", "--queries", "40",
                 "--d", "144", "--rounds", "3"]
# pool5's (03-training.md 11.17).
FRAMES_FLAGS = ["--script-frames", "1000000", "--scenario-frames", "1000000", "--army-frames", "1000000", "--games", "256",
                "--workers", "8"]
# clone-8's (03-training.md 11.22), at the actor's decision every 3 s.
CLONE_FLAGS = ["--decide-every", "6", "--samples", "8", "--accumulate", "8", "--workers", "15", "--z", "--z-dropout", "0.2",
               "--economy-weight", "2.5"]
# The PPO smoke run's (03-training.md 11.22): a learning rate of 1e-4 drifted from the clone too far; a token budget of 165,000
# didn't fit in 85% of the card.
PPO_FLAGS = ["--variants", "--z", "variant", "--fine-sim", "--decide-every", "6", "--segment", "16", "--token-budget", "40000",
             "--no-concurrent", "--critic-bins", "51", "--critic-span", "1.5", "--kl-start", "0.5", "--kl-end", "0.1",
             "--clone-weight", "0.5", "--dual-clip", "3", "--economy-start", "0.25", "--economy-end", "1", "--income-weight", "0.5",
             "--lr-start", "1e-5", "--lr-end", "3e-6", "--entropy-floor", "0"]


@dataclass
class Stage:
    name: str
    argv: list[str]
    own: pathlib.Path | None = None             # a training's directory, where its own `stop` file goes
    before: list = field(default_factory=list)  # what to do first, each time it runs: (callable, description)


def tool(name: str, *args) -> list[str]:
    return [sys.executable, str(TOOLS / name), *(str(a) for a in args)]


def default_encoder() -> str:
    sys.path[:0] = [str(ROOT / "bots"), str(TOOLS), str(ROOT / "sim" / "py")]
    import policy_latent       # torch: only when the encoder isn't trained here
    return policy_latent.ENCODER


def fresh(directory: pathlib.Path):
    """A collection goes into an empty directory: frame_data numbers its shards from 0 again, and doesn't carry on."""
    def run() -> None:
        if any(directory.glob("*.npz")):
            sys.exit(f"{directory} holds shards but no manifest.json, from a collection that didn't finish: clear it, or give "
                     f"--frames another directory, and run again")
    return run, f"check that {directory} holds no shards"


def set_aside(path: pathlib.Path):
    """A stage that starts over keeps what its last attempt logged, renamed."""
    def run() -> None:
        if path.exists():
            path.rename(path.with_name(f"{path.stem}-before-{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}"))
    return run, f"set aside {path} if it's there"


def drop_round(directory: pathlib.Path, prefix: str):
    """A round recorded again replaces its own games: those an interrupted attempt left would otherwise stay among the new."""
    def run() -> None:
        for p in directory.glob(f"{prefix}-*"):
            p.unlink()
    return run, f"delete {directory}/{prefix}-* if any"


def stages(args: argparse.Namespace) -> list[Stage]:
    root = args.root
    out: list[Stage] = []
    if args.train_encoder:
        frames = args.frames or root / "frames"
        if not (frames / "manifest.json").exists():
            out.append(Stage("frames", tool("frame_data.py", "collect", "--maps", args.maps, "--out", frames, *FRAMES_FLAGS),
                             before=[fresh(frames)]))
        out.append(Stage("encoder", tool("frame_vae.py", "train", "--data", frames, "--run", root / "encoder", "--steps",
                                         args.encoder_steps, *ENCODER_FLAGS), own=root / "encoder"))
        encoder = str(root / "encoder" / "vae-last.pt")
    else:
        encoder = args.encoder or default_encoder()
    data = pathlib.Path(args.data)
    if not (data / "z.npz").exists():
        out.append(Stage("zindex", tool("policy_clone_data.py", "zindex", "--data", data, "--workers", 15)))
    clone = root / "clone"
    out.append(Stage("clone", tool("policy_clone_train.py", "--data", data, "--maps", args.maps, "--run", clone, "--encoder", encoder,
                                   "--steps", args.clone_steps, "--check-every", args.clone_check_every, "--lr", "6e-4",
                                   "--muon-lr", "0.008", "--finish-winners", "0.1", *CLONE_FLAGS, *shlex.split(args.clone_args)),
                     before=[set_aside(clone / "log.jsonl")]))
    last = clone / "actor.pt"
    dagger = root / "dagger"
    evals = root / "evals"
    for r in range(1, args.dagger_rounds + 1):
        prefix = f"game-dagger{r}"
        out.append(Stage(f"record-{r}", tool("policy_clone_eval.py", "--actor", last, "--maps", args.maps, "--encoder", encoder,
                                             "--games", 64, "--record-games", args.dagger_games, "--workers", 12, "--z", "variant",
                                             "--record", dagger, "--record-prefix", prefix, "--out", evals / f"record-{r}.json"),
                         before=[drop_round(dagger, prefix)]))
        out.append(Stage(f"zindex-{r}", tool("policy_clone_data.py", "zindex", "--data", dagger, "--workers", 15)))
        run = root / f"dagger-{r}"
        out.append(Stage(f"dagger-{r}", tool("policy_clone_train.py", "--data", data, "--mix", dagger, "--mix-share", args.dagger_share,
                                             "--maps", args.maps, "--run", run, "--init", last, "--encoder", encoder,
                                             "--steps", args.dagger_steps, "--check-every", max(1, args.dagger_steps // 2),
                                             "--lr", "2e-4", "--muon-lr", "0.003", *CLONE_FLAGS, *shlex.split(args.dagger_args)),
                         before=[set_aside(run / "log.jsonl")]))
        last = run / "actor.pt"
    out.append(Stage("eval", tool("policy_clone_eval.py", "--actor", last, "--maps", args.maps, "--encoder", encoder, "--games", 64,
                                  "--workers", 12, "--z", "variant", "--out", evals / "final.json")))
    if args.ppo:
        ppo = root / "ppo"
        out.append(Stage("ppo", tool("policy_train.py", "--run", ppo, "--init", last, "--latent-encoder", encoder, "--maps", args.maps,
                                     "--games", args.ppo_games, "--iteration", args.ppo_iteration, "--decisions", args.ppo_decisions,
                                     "--hours", args.ppo_hours, "--critic-warmup", args.ppo_critic_warmup, *PPO_FLAGS,
                                     *shlex.split(args.ppo_args)), own=ppo))
    return out


def available_gb() -> float | None:
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


class State:
    """ROOT/pipeline.json: which stages are done, and when each ran."""

    def __init__(self, root: pathlib.Path):
        self.path = root / "pipeline.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"done": [], "runs": []}

    def save(self) -> None:
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, indent=1))
        os.replace(temp, self.path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, required=True, help="the run's directory: every stage's output goes under it")
    ap.add_argument("--maps", default="/mnt/d/zk-learned/maps")
    ap.add_argument("--data", default="/mnt/d/zk-learned/clone/army-6", help="the scripts' recorded games to clone")
    ap.add_argument("--train-encoder", action="store_true", help="train the frame encoder first, and read it after")
    ap.add_argument("--frames", type=pathlib.Path, help="with --train-encoder, the frames to train it on, collected there first if "
                    "they aren't yet (default ROOT/frames).  Frames are stored featurized: after a change to featurize, such as "
                    "the ids' sizes (MAX_PLAYERS, MAX_ALLY_TEAMS), collect new ones rather than give frames/pool5")
    ap.add_argument("--encoder-steps", type=int, default=25_000)
    ap.add_argument("--encoder", help="without --train-encoder, the encoder the actor reads (default policy_latent.ENCODER)")
    ap.add_argument("--clone-steps", type=int, default=10_000)
    ap.add_argument("--clone-check-every", type=int, default=500, help="held-out measures and a checkpoint this many steps apart")
    ap.add_argument("--dagger-rounds", type=int, default=2)
    ap.add_argument("--dagger-games", type=int, default=256, help="games kept each round")
    ap.add_argument("--dagger-steps", type=int, default=500, help="fine-tuning steps each round")
    ap.add_argument("--dagger-share", type=float, default=0.5, help="the share of a fine-tuning's samples from the rounds' games")
    ap.add_argument("--ppo", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ppo-games", type=int, default=64)
    ap.add_argument("--ppo-iteration", type=int, default=4096, help="decisions collected an iteration")
    ap.add_argument("--ppo-decisions", type=int, default=2_000_000)
    ap.add_argument("--ppo-hours", type=float, default=0.0, help="stop PPO after this long as well; 0 for no limit")
    ap.add_argument("--ppo-critic-warmup", type=int, default=5, help="iterations where only the new critic learns")
    ap.add_argument("--clone-args", default="", help="more flags for the clone stage, last so they win")
    ap.add_argument("--dagger-args", default="", help="more flags for each DAgger fine-tuning")
    ap.add_argument("--ppo-args", default="", help="more flags for PPO")
    ap.add_argument("--redo", help="run this stage and every one after it again")
    ap.add_argument("--dry-run", action="store_true", help="print the stages' commands and what's done; run nothing")
    args = ap.parse_args()
    args.root = args.root.resolve()

    plan = stages(args)
    names = [s.name for s in plan]
    state = State(args.root) if (args.root / "pipeline.json").exists() or not args.dry_run else None
    done = set(state.data["done"]) if state else set()
    if args.redo:
        if args.redo not in names:
            ap.error(f"--redo {args.redo}: the stages are {', '.join(names)}")
        done -= set(names[names.index(args.redo):])
    if args.dry_run:
        for s in plan:
            print(f"# {s.name}{' (done)' if s.name in done else ''}")
            for _, what in s.before:
                print(f"#   first: {what}")
            print(shlex.join(s.argv) + f"  > {args.root / 'logs' / (s.name + '.out')} 2>&1\n")
        return

    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "logs").mkdir(exist_ok=True)
    (args.root / "evals").mkdir(exist_ok=True)
    state.data["done"] = [n for n in names if n in done]
    state.data["plan"] = names
    state.data["args"] = {k: str(v) for k, v in vars(args).items()}
    state.save()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "bots"), str(TOOLS), str(ROOT / "sim" / "py")] +
                                        ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for s in plan:
        if s.name in done:
            continue
        if (args.root / "stop").exists():
            print(f"stopped before {s.name}: {args.root / 'stop'} is there", flush=True)
            return
        for run, _ in s.before:
            run()
        log = args.root / "logs" / f"{s.name}.out"
        free = available_gb()
        began = time.time()
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s.name}: {shlex.join(s.argv)}"
              + (f"  [{free:.0f} GB free]" if free is not None else ""), flush=True)
        with open(log, "a") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(s.argv)}\n")
            f.flush()
            code = subprocess.run(s.argv, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env).returncode
        seconds = round(time.time() - began)
        state.data["runs"].append({"stage": s.name, "began": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(began)),
                                   "seconds": seconds, "exit": code})
        state.save()
        if code != 0:
            sys.exit(f"{s.name} failed (exit {code}) after {seconds} s: see {log}")
        if s.own is not None and (s.own / "stop").exists():
            print(f"{s.name} was stopped by {s.own / 'stop'}: not marked done; remove the file and run again to go on", flush=True)
            return
        state.data["done"].append(s.name)
        state.save()
        print(f"{s.name} done in {seconds / 60:.1f} minutes", flush=True)
    print(f"all done: {args.root}", flush=True)


if __name__ == "__main__":
    main()
