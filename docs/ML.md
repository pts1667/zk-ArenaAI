# Learned components (ML stages 1-4)

The bot stays rules-based for execution (micro, build placement, orders).
Learned models replace hand-tuned *judgements* where the rules were shown to
fail and the real engine can supply data cheaply.  Everything is gated by
`ZKBOT_FLAGS` and A/B-tested with `tools/prun.py` like any other change.

| stage | what | data | code |
|---|---|---|---|
| 1 | fight evaluator: metal each side loses in the next 10/20 s | `fights-team<N>.jsonl` (bot) | `bots/zkbot/fightnet.py`, `tools/train_fightnet.py` |
| 2 | game value: win probability, metal ratio 5 min ahead | `state-team<N>.jsonl` (bot) + demo team stats | `bots/zkbot/statelog.py`, `tools/train_value.py` |
| 3 | CircuitAI macro (build order, spending, expansion) | `observe-team<N>.jsonl` (observer allied with CircuitAI) | `bots/observer/main.py`, `prun.py --observe` |
| 4 | parameter tuning in the real engine | paired prun games | `tools/tune.py` (SPSA over `P()` params) |

Why not one large end-to-end model trained by RL in a simplified sim: what
beats us (outranging, turrets, line of sight, terrain, reclaim, explosions)
is what a simplified sim leaves out, policy RL from scratch needs far more
games than the engine can play (about 4,500/day at max speed), and the data
supports models of 0.1-10M parameters, not 100M.  See HANDOFF.md
(h124 onwards) for the reasoning and results.

## Environment (WSL, PyTorch with torch.compile)

Training runs in WSL (Ubuntu) on the RTX 4070; the bot itself never needs
PyTorch (stage 1 runs a numpy forward pass of exported weights).

```sh
python3 -m venv ~/zkml
~/zkml/bin/pip install torch numpy --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
# torch.compile (Triton) needs Python.h; without sudo, extract python3.12-dev:
mkdir -p ~/pydev && cd ~/pydev && apt-get download libpython3.12-dev python3.12-dev
for d in *.deb; do dpkg-deb -x "$d" root; done
# ~/zkml/bin/zkpy: sets CPATH to those headers and TRITON_DEFAULT_FP_FUSION=1, then runs the venv python
```

Run trainers from the repo under `/mnt/c/...`:

```sh
cd /mnt/c/Users/Thoma/Documents/sources/zk-ArenaAI
~/zkml/bin/zkpy tools/train_fightnet.py results/d1 --export     # stage 1 -> bots/zkbot/data/fightnet.npz
~/zkml/bin/zkpy tools/train_value.py results/d1 --export        # stage 2 -> bots/zkbot/data/valuenet.pt
```

## Logs written by the bot (all passive)

* `deaths-team<N>.jsonl`: one line per lost unit: killer (attacker name;
  `~name` inferred from the weapon def; `own:name` friendly fire or our own
  explosions; `boom:<explosion>` death explosions; `env:<source>` debris,
  collisions...), branch, position, local balance.  Weapon def ids are mapped
  at runtime by querying each unit def once (`learn_weapon_defs`).
* `fights-team<N>.jsonl` (flag `nofightlog` disables): entity snapshots of
  fight sites every 2 s and the measured losses after 10 and 20 s, from
  UNIT_DAMAGED (topic 9) and ENEMY_DAMAGED (topic 17).
* `state-team<N>.jsonl` (flag `nostatelog`): whole-game state every 10 s,
  including `spent`: cumulative metal committed per spend category (each
  unit's full cost when it first appears, dead or not), bucketed like the
  observer logs (`statelog.spend_category`) so the two mixes compare.

## Stage notes

1. The fight evaluator's input is every unit near the fight, including
   turrets, air, remembered structures and radar blips, with per-type stat
   features (cost, HP, DPS, range, speed, role, ability tags), so new or
   rebalanced units are not unknown to it; a hashed name embedding adds
   per-type detail and is dropped 30% of the time in training.
2. The value model follows arXiv 2501.03832 (a transformer over a time
   series of a game's feature planes) with an 8x8 value grid per 10 s step.
   Its first use is as a low-variance objective for stage 4.
3. Observer games: `python tools/prun.py o2 --observe --games 24 --jobs 2`.
   The observer's own commander idles at its start; CircuitAI's units are
   the friendly units that are not the observer's.
   `python tools/observe_dataset.py results/o2 --ours results/<batch>`
   prints both spend mixes by phase.  The first comparison (HANDOFF 6d.1)
   showed CircuitAI putting 16-27% of its metal into turrets from minute 3
   (we: 2-7%) and 4-6% into riots (we: 15-22%).  Two arms copy that:
   * `circdef`: CircuitAI's per-cluster porcupine (MilitaryManager::
     MakeDefence): each owned metal sector gets turrets from CircuitAI's
     land chain (Lotus, Lotus, Picket, Stinger, ...) while their value is
     below `def.cluster_sec` (23) seconds of income; front sectors get the
     whole chain, quiet ones one turret.  Replaces the extractor and
     forward turret rules; base turrets and AA are unchanged.
   * `lessriot`: two of the three riot slots in the late production cycle
     become an assault and an artillery unit.
   * `morebp`: CircuitAI's measured constructor count (~0.6 mobile
     constructors per metal/s; its config's 1.25 BP ratio is not what it
     fields). h127 showed turret jobs going stale for lack of builders.
   * `spendmix`: production role quotas on metal started in the last
     4 minutes instead of standing head counts (which refilled dying
     Glaives without end), with CircuitAI's late army metal mix as target.
4. `python tools/tune.py NAME --params army.attack_ratio,retreat.hp,...`
   runs SPSA with paired +/- games per iteration; state in
   `results/tune-NAME/state.json`.
