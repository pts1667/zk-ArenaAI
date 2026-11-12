"""A full training run, stage after stage: the frame encoder (optional), cloning, DAgger rounds, then PPO from the last clone.

    ~/zkml/bin/python tools/policy_pipeline.py --root /mnt/d/zk-learned/runs/full-1 --train-encoder
    ~/zkml/bin/python tools/policy_pipeline.py --root /mnt/d/zk-learned/runs/full-1 --dry-run     # the commands, nothing run

Each stage is one of the tools' own commands (AGENTS.md, "Running"), run one at a time, so no two load the map pool at once and
each hands the GPU back when it ends.  Its output goes to ROOT/logs/STAGE.out.  ROOT/pipeline.json lists the stages done, so the
same command again carries on from the first that isn't; --redo STAGE runs that one and every one after it again.  The encoder
(frame_vae) and PPO resume from their own checkpoints; a cloning stage starts over, its old log.jsonl kept beside the new one.

A `stop` file in ROOT halts the pipeline once the running stage ends.  One in a training's own directory (ROOT/encoder,
ROOT/ppo) stops that training at its next report, and the pipeline with it, that stage not marked done.

While it runs it serves a dashboard at http://localhost:8765 (--port; 0 for none): the stages, the log as it comes (the
pipeline's own in ROOT/logs/pipeline.out), each stage's results, and replays of test games, which the tests keep in
ROOT/replays/STAGE: 4 games of each closed-loop test (record-N, eval) and 4 every hour of PPO (--replay-games, --replay-hours).
It ends with the pipeline; tools/policy_dashboard.py --root ROOT serves it again, during the run or after.

The stages:
  record       if --data holds no finished recording: the scripts' games recorded there (policy_clone_data.py record), as army-6
               was, --record-decisions of them, in the kinds of game --kinds says
  frames       with --train-encoder, if --frames has no manifest.json: frame_data.py collect, pool5's recipe (1M frames each of
               the frame games, scenarios and the C++ scripts' games; 03-training.md, "Frame data"), the army games of --kinds
  encoder      with --train-encoder: frame_vae.py train with encoder-big's settings, 25,000 steps, to ROOT/encoder
  zindex       the cloning data's z.npz, if it has none
  clone        policy_clone_train.py with clone-8's settings (z, the economy at 2.5, the last 10% on winners), to ROOT/clone
  record-N     DAgger round N: the last clone plays the army script, told its variant as z, with the scripts advising it, and
               --dagger-games of its games are kept (policy_clone_eval.py --record), of the kinds --kinds says.  Every round's go
               into ROOT/dagger, so each round learns from them all.  The first 64 games are the last clone's closed-loop eval too:
               ROOT/evals/record-N.json, by kind where there's more than 1v1s
  zindex-N     z.npz over every round's games
  dagger-N     policy_clone_train.py --init the last clone: half its samples from ROOT/dagger, half from --data, to ROOT/dagger-N
  eval         the last clone against the army script, z from its variant, in 1v1s, the headline: ROOT/evals/final.json
  eval-KIND    the same in games of each of --test-kinds alone: ROOT/evals/final-KIND.json
  ppo          policy_train.py --init the last clone, --z variant, --fine-sim, with the PPO smoke run's flags (03-training.md,
               "From a clone") at a run's size, playing --kinds and testing --test-kinds besides the 1v1 test, to ROOT/ppo

Every stage that plays games plays the kinds of game --kinds says (1v1s, team games, free-for-alls, chicken games:
policy_sim_env.GAME_KINDS), in batches with --slots teams a game, and chicken games at --chicken-difficulties.  The tests' headline
stays a 1v1 against the script, so that runs stay comparable; --test-kinds adds a test of each kind it names.

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

# encoder-big's (03-training.md, "The frame encoder").
ENCODER_FLAGS = ["--autoencoder", "--region-pool", "32", "--two-part", "--questions", "--muon-lr", "0.008", "--queries", "40",
                 "--d", "144", "--rounds", "3"]
# pool5's (03-training.md, "Frame data").
FRAMES_FLAGS = ["--script-frames", "1000000", "--scenario-frames", "1000000", "--army-frames", "1000000", "--games", "256",
                "--workers", "8"]
# army-6's (03-training.md, "Script data"): 12 batches of 64 games, each side's army script its own knobs.
RECORD_FLAGS = ["--games", "64", "--workers", "12", "--threads", "12", "--seed", "6", "--variants"]
# clone-8's (03-training.md, "Cloning"), at the actor's decision every 3 s; but 4 samples a pass and 16 passes a step, where clone-8
# had 8 and 8: the same 64 samples a step.  A sample from late in an 80-minute game takes 1.3 to 2 GB of the card, so 8 ran out of
# 85% of it (measured 2026-11-10).  4 didn't always fit either: on army-7 a pass of 4 padded to the caps took up to 8.6 GB, and one
# ran out (2026-11-11).  So the trainer puts a step's 64 into passes by size (--pass-gb), and splits one that runs out anyway
# (03-training.md, "Cloning").  Compiled with symbolic shapes: 13% faster on army-7, after about 5 minutes of compiling (2026-11-11).
CLONE_FLAGS = ["--decide-every", "6", "--samples", "4", "--accumulate", "16", "--workers", "15", "--z", "--z-dropout", "0.2",
               "--economy-weight", "2.5", "--compile", "dynamic"]
# The PPO smoke run's (03-training.md, "From a clone"): a learning rate of 1e-4 drifted from the clone too far; a token budget of 165,000
# didn't fit in 85% of the card.  The new critic learns at 3e-4, as every earlier critic did: at the actor's 1e-5 it learned nothing.
# A win sooner is worth up to 0.1 more (the user, 2026-11-10); the lead rule fades by policy_train's default --lead-fade.
PPO_FLAGS = ["--variants", "--z", "variant", "--fine-sim", "--decide-every", "6", "--segment", "16", "--token-budget", "40000",
             "--no-concurrent", "--critic-bins", "51", "--critic-span", "1.5", "--kl-start", "0.5",
             "--clone-weight", "0.5", "--economy-start", "0.25", "--economy-end", "1", "--income-weight", "0.5",
             "--speed-weight", "0.1", "--lr-start", "1e-5", "--lr-end", "3e-6", "--critic-lr", "3e-4", "--entropy-floor", "0"]


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


def no_games(directory: pathlib.Path):
    """A recording goes into a directory with no games: the recorder numbers its files from 0 again, and doesn't carry on."""
    def run() -> None:
        if next(directory.glob("game-*"), None) is not None:
            sys.exit(f"{directory} holds games but no summary.json or sides.npz, from a recording that didn't finish: clear it, or "
                     f"give --data another directory, and run again")
    return run, f"check that {directory} holds no games"


def recorded(data: pathlib.Path) -> bool:
    """Whether --data holds a finished recording: the recorder's summary.json, or the sides.npz that reading it leaves (army-6
    was stopped before it wrote its summary)."""
    return (data / "summary.json").exists() or (data / "sides.npz").exists()


def kinds_flags(args: argparse.Namespace, kinds: list[str] | None = None, difficulties: bool = True, slots: bool = True) -> list:
    """A stage's flags for the kinds of game it plays: the run's (--kinds), or these; and, where the tool takes them, the slots
    a game and the chicken difficulties."""
    out = ["--kinds", *(args.kinds if kinds is None else kinds)]
    if slots and args.slots:
        out += ["--slots", args.slots]
    if difficulties and args.chicken_difficulties:
        out += ["--chicken-difficulties", *args.chicken_difficulties]
    return out


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


def replays(args: argparse.Namespace, stage: str, actor: pathlib.Path) -> list:
    """A closed-loop test's flags to keep its first games for the dashboard."""
    if not args.replay_games:
        return []
    return ["--replays", args.root / "replays" / stage, "--replay-games", args.replay_games,
            "--replay-label", f"{actor.parent.name} against the army script"]


def drop_replays(directory: pathlib.Path):
    """A test run again replaces its replays, rather than listing them twice."""
    def run() -> None:
        for p in [*directory.glob("*.json.gz"), directory / "index.jsonl"]:
            p.unlink(missing_ok=True)
    return run, f"delete {directory}'s replays if any"


def stages(args: argparse.Namespace) -> list[Stage]:
    root = args.root
    out: list[Stage] = []
    data = pathlib.Path(args.data)
    if not recorded(data):
        out.append(Stage("record", tool("policy_clone_data.py", "record", "--out", data, "--maps", args.maps, "--decisions",
                                        args.record_decisions, *RECORD_FLAGS, *kinds_flags(args)), before=[no_games(data)]))
    if args.train_encoder:
        frames = args.frames or root / "frames"
        if not (frames / "manifest.json").exists():
            # frame_data.py's army games take the kinds (its own --kinds), not the slots or the chicken difficulties.
            out.append(Stage("frames", tool("frame_data.py", "collect", "--maps", args.maps, "--out", frames, *FRAMES_FLAGS,
                                            *kinds_flags(args, difficulties=False, slots=False)), before=[fresh(frames)]))
        out.append(Stage("encoder", tool("frame_vae.py", "train", "--data", frames, "--run", root / "encoder", "--steps",
                                         args.encoder_steps, *ENCODER_FLAGS), own=root / "encoder"))
        encoder = str(root / "encoder" / "vae-last.pt")
    else:
        encoder = args.encoder or default_encoder()
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
                                             "--record", dagger, "--record-prefix", prefix, "--out", evals / f"record-{r}.json",
                                             *kinds_flags(args), *replays(args, f"record-{r}", last)),
                         before=[drop_round(dagger, prefix), drop_replays(root / "replays" / f"record-{r}")]))
        out.append(Stage(f"zindex-{r}", tool("policy_clone_data.py", "zindex", "--data", dagger, "--workers", 15)))
        run = root / f"dagger-{r}"
        out.append(Stage(f"dagger-{r}", tool("policy_clone_train.py", "--data", data, "--mix", dagger, "--mix-share", args.dagger_share,
                                             "--maps", args.maps, "--run", run, "--init", last, "--encoder", encoder,
                                             "--steps", args.dagger_steps, "--check-every", max(1, args.dagger_steps // 2),
                                             "--lr", "2e-4", "--muon-lr", "0.003", *CLONE_FLAGS, *shlex.split(args.dagger_args)),
                         before=[set_aside(run / "log.jsonl")]))
        last = run / "actor.pt"
    out.append(Stage("eval", tool("policy_clone_eval.py", "--actor", last, "--maps", args.maps, "--encoder", encoder, "--games", 64,
                                  "--workers", 12, "--z", "variant", "--out", evals / "final.json", *replays(args, "eval", last)),
                     before=[drop_replays(root / "replays" / "eval")]))
    for kind in args.test_kinds:
        out.append(Stage(f"eval-{kind}", tool("policy_clone_eval.py", "--actor", last, "--maps", args.maps, "--encoder", encoder,
                                              "--games", 64, "--workers", 12, "--z", "variant", "--out", evals / f"final-{kind}.json",
                                              *kinds_flags(args, [kind]), *replays(args, f"eval-{kind}", last)),
                         before=[drop_replays(root / "replays" / f"eval-{kind}")]))
    if args.ppo:
        ppo = root / "ppo"
        out.append(Stage("ppo", tool("policy_train.py", "--run", ppo, "--init", last, "--latent-encoder", encoder, "--maps", args.maps,
                                     "--games", args.ppo_games, "--iteration", args.ppo_iteration, "--decisions", args.ppo_decisions,
                                     "--hours", args.ppo_hours, "--critic-warmup", args.ppo_critic_warmup,
                                     "--replays", root / "replays" / "ppo", "--replay-hours", args.replay_hours if args.replay_games else 0,
                                     "--replay-games", args.replay_games, *kinds_flags(args),
                                     *(["--test-kinds", *args.test_kinds] if args.test_kinds else []), *PPO_FLAGS,
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
    ap.add_argument("--data", default="/mnt/d/zk-learned/clone/army-6", help="the scripts' recorded games to clone, recorded there "
                    "first if it holds no finished recording (the record stage)")
    ap.add_argument("--record-decisions", type=int, default=100_000_000, help="the record stage's size, in decisions of every side "
                    "together: army-6 is 300M, 105 GB at 375 bytes a decision; games of several teams take about 1,000 bytes")
    ap.add_argument("--kinds", nargs="+", default=["1v1:0.5", "teams:0.2", "ffa:0.2", "chicken:0.1"], metavar="KIND:WEIGHT",
                    help="the kinds of game every stage that plays games plays, and how likely each (policy_sim_env.GAME_KINDS) "
                         "(default: %(default)s)")
    ap.add_argument("--slots", type=int, default=0, help="teams a game has room for in those stages; 0: what the biggest of --kinds needs")
    ap.add_argument("--test-kinds", nargs="*", default=[], help="kinds of game the last clone (eval-KIND) and PPO are tested at too, "
                    "besides the 1v1 test that's the headline")
    ap.add_argument("--chicken-difficulties", nargs="+", metavar="NAME:WEIGHT",
                    help="how likely each chicken difficulty is, as very-easy:1 easy:1 normal:1 (default the tools' own, "
                         "policy_sim_env.CHICKEN_DIFFICULTY_WEIGHTS)")
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
    ap.add_argument("--ppo-critic-warmup", type=int, default=10, help="iterations where only the new critic learns")
    ap.add_argument("--clone-args", default="", help="more flags for the clone stage, last so they win")
    ap.add_argument("--dagger-args", default="", help="more flags for each DAgger fine-tuning")
    ap.add_argument("--ppo-args", default="", help="more flags for PPO")
    ap.add_argument("--replay-games", type=int, default=4, help="games kept for the dashboard from each test, and each --replay-hours "
                    "of PPO; 0 for none")
    ap.add_argument("--replay-hours", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8765, help="the dashboard's (tools/policy_dashboard.py); 0 for none")
    ap.add_argument("--host", default="127.0.0.1", help="where the dashboard listens: 0.0.0.0 for other machines too")
    ap.add_argument("--map-archives", help="the maps' archives (.sd7, .sdz), for the dashboard's map pictures (default its own)")
    ap.add_argument("--zero-k", help="Zero-K's repository, for the dashboard's unit icons (default ../Zero-K)")
    ap.add_argument("--redo", help="run this stage and every one after it again")
    ap.add_argument("--dry-run", action="store_true", help="print the stages' commands and what's done; run nothing")
    args = ap.parse_args()
    args.root = args.root.resolve()
    sys.path[:0] = [str(ROOT / "bots"), str(TOOLS), str(ROOT / "sim" / "py")]
    import policy_sim_env       # what the stages will be given, checked now rather than hours in
    try:
        policy_sim_env.parse_kinds(args.kinds)
        policy_sim_env.parse_kinds(args.test_kinds)
        if args.chicken_difficulties:
            policy_sim_env.difficulty_weights(args.chicken_difficulties)
    except ValueError as e:
        ap.error(str(e))

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
    state.data.pop("current", None)
    state.save()
    said = open(args.root / "logs" / "pipeline.out", "a")

    def say(text: str) -> None:
        print(text, flush=True)
        said.write(text + "\n")
        said.flush()
    if args.port:
        try:
            import policy_dashboard
            extra = {k: v for k, v in (("archives", args.map_archives), ("zero_k", args.zero_k)) if v}
            board = policy_dashboard.Dashboard(args.root, args.host, args.port, maps=args.maps, **extra).start()
            say(f"dashboard: {board.url}")
        except OSError as e:        # the port taken, say: the run goes on without it
            say(f"no dashboard on port {args.port}: {e}")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "bots"), str(TOOLS), str(ROOT / "sim" / "py")] +
                                        ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    for s in plan:
        if s.name in done:
            continue
        if (args.root / "stop").exists():
            say(f"stopped before {s.name}: {args.root / 'stop'} is there")
            return
        for run, _ in s.before:
            run()
        log = args.root / "logs" / f"{s.name}.out"
        free = available_gb()
        began = time.time()
        say(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s.name}: {shlex.join(s.argv)}" + (f"  [{free:.0f} GB free]" if free is not None else ""))
        state.data["current"] = {"stage": s.name, "began": began, "pid": os.getpid()}
        state.save()
        with open(log, "a") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(s.argv)}\n")
            f.flush()
            code = subprocess.run(s.argv, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env).returncode
        seconds = round(time.time() - began)
        state.data.pop("current", None)
        state.data["runs"].append({"stage": s.name, "began": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(began)),
                                   "seconds": seconds, "exit": code})
        state.save()
        if code != 0:
            say(f"{s.name} failed (exit {code}) after {seconds} s: see {log}")
            sys.exit(1)
        if s.own is not None and (s.own / "stop").exists():
            say(f"{s.name} was stopped by {s.own / 'stop'}: not marked done; remove the file and run again to go on")
            return
        state.data["done"].append(s.name)
        state.save()
        say(f"{s.name} done in {seconds / 60:.1f} minutes")
    say(f"all done: {args.root}")


if __name__ == "__main__":
    main()
