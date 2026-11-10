# The learned policy: overview

A Zero-K AI meant to beat CircuitAI, built as one network in the manner of Pluto (the CoG 2026 StarCraft winner,
https://davechurchill.ca/starcraft/cog/results/2026/PLUTO_README.txt).  It is trained in a simplified simulator of Zero-K, first by
cloning scripted players and then by PPO, and is meant to play the real game through the bot client.  `HANDOFF.md` has the
current state and next steps.  This folder is the reference: what's there and why, with the numbers that decided it.

| Doc | What it holds |
|---|---|
| [01-simulator.md](01-simulator.md) | The C++ sim: what it models, how closely it matches the engine, and the scripted players that run in it |
| [02-policy.md](02-policy.md) | The contract between the game and the network: the raw frame, the featurizer's tensors, the action kinds and their masks |
| [../model.md](../model.md) | The network: the frozen frame encoder, the time transformer, and actions chosen one after another |
| [03-training.md](03-training.md) | Data, the encoder's training, cloning, DAgger, PPO, the pipeline that runs them, what each measured, and the dead ends |
| [04-literature.md](04-literature.md) | A literature review: what was taken from it, and what's left |

## The pipeline

    map pool --> frame games and scenarios --> frames --> frame encoder (trained once, then frozen)
                                                               |
    army + economy scripts playing each other --> recorded games --> cloning --> DAgger rounds --> PPO --> engine (not built)

`tools/policy_pipeline.py` runs all of it, one stage at a time (03-training.md, "The pipeline").

## Where it stands

* **Built and run:** the sim and its scripts, frame collection and the frame encoder, recording script games, cloning, and the
  closed-loop eval of a clone against the script.
* **Built, not yet run at scale:** DAgger and PPO from a clone (smoke runs only).
* **Not built:** play in the engine.  `bots/zkbot` hosts the policy and sends no commands.  It still needs a CPU forward of the
  encoder and actor, the opening and the economy script on the engine side, and decoded orders sent as commands.
* **Unvalidated:** the sim's fights.  The replay gate failed and the user waived it (2026-11-05; 01-simulator.md, "Validation").
  So the first test of whether what the sim teaches carries over is in the engine.
* **The rules bot is gone** (2026-11-08).  It was the hand-written bot that beat CircuitAI Hard at high game speed, plus ML stages
  1-4 around it.  It's kept at the local git tag `rules-bot-final`, and any engine A/B against it needs that tag checked out.

## Decisions that stand

| Topic | Decision |
|---|---|
| Timing | Engine 30 frames a second.  The sim ticks every 5 frames, and a slot is 3 ticks (15 frames, 0.5 s).  The actor decides every 6 slots (3 s), when the army script looks; the sim still steps every slot (`--fine-sim`), as the data was recorded |
| What the actor sees | What a player sees: its own units, enemies in sight, radar blips and remembered structures.  The critic sees everything, in training only |
| Units | One token per unit: type stats from the catalog, a name embedding with 30% dropout to "unknown", and instance state.  No logic keyed on a unit's name |
| Players | One pool of units.  Each unit carries its player and allyteam, numbered from the viewer: up to 64 players and 8 allyteams, for free-for-all later.  The sim plays only 1v1 |
| Raw frame | The sim and the live bot emit the same frame (`bots/zkbot/policy/rawframe.py`), and one featurizer reads it in both |
| Weapons | No projectiles.  Beams hit at once; other weapons' damage lands after the flight time, if the target is still within the blast |
| Re-ordering | Costs what it costs in the engine: a plain move stops a unit until its next slow update, and a fight order costs nothing |
| Reward | Zero-sum: the change in the value balance, faded into the result as training goes on (03-training.md, "Reward") |
| Economy | The policy runs its own economy too (the user, 2026-11-09).  Constructors and the commander are anchors like any unit.  An idle constructor stays idle until told: there's no default job.  There's no grid feature: the encoder reads every generator and pylon, and income from overdrive is in the reward |
| Opening | A fixed script on both sides (`tools/policy_opening.py`), not the policy's |
| Strategy z | The actor can be told a side's strategy (the army script's knobs).  The user goes on with `clone-8` told its variant (2026-11-10) |
| Opponents | The scripts first, and self-play once tests reach 80% wins against them.  A league only if that stalls |
| Terraform | Off |
| Storage | Runs and data go on `D:\zk-learned` (`/mnt/d/zk-learned`), not C:.  C: backs WSL's disk, even though `df` in WSL shows far more free than C: has |

## Hardware

RTX 4070 (12 GB), Ryzen 7 5800X3D (8 cores, 16 threads), WSL capped at 40 GB of RAM.  PyTorch in WSL (`~/zkml`); the bot runs
on Windows Python.  Every GPU script caps its memory at 0.85 of the card: past the card, the Windows driver spills into system
RAM (30 GB once) instead of failing.

## Sources

Read-only, beside this repo in `C:\Users\Thoma\Documents\sources`: `RecoilEngine` (CircuitAI under `AI/Skirmish/CircuitAI`),
`Zero-K` and `ControllerAI`.
