# The policy runs the economy: plan

2026-11-09.  The user's decision: the learned policy should run the economy too, not only the army.  Until now the economy script
(`sim/scripts/src/eco.cpp`, described in `tools/policy_eco.py`) runs every side's constructors, the policy's included, and the
policy was given none (`featurize._select`: controllable meant a mobile fighter or something that acts in place).  This is stage 3 of
`02-policy.md` (M5, "economy heads"); this file is how to get there from the actor of `docs/model.md`.

`clone-6` was stopped at step 2,000 of 20,000 for this (`/mnt/d/zk-learned/runs/clone-6/actor.pt`; held-out anchor right 0.675,
kind right 0.989, NLL of actions 3.89, already `clone-5`'s best at 11,500).

## 1. What there is already

| Piece | State |
|---|---|
| Sim | Obeys `ZK_ORDER_BUILD` (type at x, z), `ZK_ORDER_ASSIST` (a unit), `ZK_ORDER_RECLAIM` (a feature) from anyone: the opening and the economy script give them now.  No REPAIR (kind 12 isn't in `capi.h`); assisting a finished, damaged unit would be it. |
| Raw frame | Kinds BUILD 10, ASSIST 11, REPAIR 12, RECLAIM 13 are numbered, and a unit's current order shows them, so the frozen encoder has seen them on constructors in every recorded frame. |
| Featurizer | Sites (metal spots, with status free / ours / enemy / building), wrecks, `idle_builders` and `unfinished_structures` in the globals.  No build masks; constructors aren't controllable. |
| Actor | Kinds, a type head (`produce`: `q_type(me)` against type keys of `table().producible`, masked per anchor), region and fine heads.  No pointer to own units or to wrecks. |
| C++ scripts | `zks_step` keeps the economy script's orders apart (`seat.eco_out`) but returns only the army script's as `army`, which is what's recorded as labels.  The economy script looks every `ECO_EVERY` 4 slots (2 s); the actor decides every 6 (3 s). |
| Placement | `Ground.fits` / `Ground.site` (Python) and `Seat::site` (C++) find where a footing can stand near a point, clear of what's taken.  The opening and the economy script place everything that way. |

## 2. Design

One actor, more kinds, as Pluto and AlphaStar do: constructors, the commander after the opening, factories and fighters are all
anchors of the same decision, so whom to tell and what to build compete in one list of actions.

### Actions

| Kind | Who | What it chooses | Decoded as |
|---|---|---|---|
| BUILD | a constructor or the commander | type (the type head, over the anchor's build options); a place, a pointer over our own unit tokens and then the sites (`place_tokens`, `site_pos`); an offset from it, 25 options: on it, or one of 8 compass points at 64, 128 or 256 elmos, north towards the enemy | the type at that point if it fits, else the nearest place it fits (`SimEnv.place_build`: rings of samples, then `Ground.site`), clear of our structures and of metal spots; an extractor at the free spot nearest the point |
| ASSIST | the same | one of our units being built, or a factory: the target pointer (`ptr_enemy`, a fourth kind; `enemy_tokens` now holds these own tokens too) | help build it, or help a factory with what it makes |
| RECLAIM | a constructor (not the commander) | a wreck: a pointer over the feature tokens (`feat_ids`, `feat_pos`; the `place` field holds its index) | reclaim that feature |
| PRODUCE | a factory, a silo | type, as now | as now |

* **Type head shared.**  `producible` becomes what anything of the side could make, mobile or not, and each anchor's mask is its own
  options (`EcoData.options`; no geothermal, only what the sim models).  One `q_type` scores both a factory's choice and a
  constructor's.
* **Place beside a token, then snap.**  First planned as region and fine, as moves are; step 1 measured that it loses what the
  script meant (half of all builds over 119 elmos off), since the script builds beside what it has.  So a BUILD points at one of our
  unit tokens or a site and takes an offset from it; the exact spot is still the legaliser's, precise to the footing and never
  illegal.  In a batch, each decision's places are its own units then its sites, contiguous (`policy_model.Batch.places`, sites as
  -1 - index), so a place index means the same however the batch is padded.
* **Masks.**  BUILD legal for a builder with any option it can afford to start, while unfinished structures are under the cap
  (`02-policy.md` 4.3: 2 + builders / 3); region masked by where its move class can reach and the footing fits.  ASSIST legal with
  something unfinished of ours, or a factory, in reach.
* **Group orders.**  Not for these kinds at first: a constructor's order is its own.  Assist as a group (all constructors in a
  radius help this) is a later option: it's what the script's "finish" mostly is.
* **Action tokens.**  `act_type` embeds a built type as it does a produced one; `act_place` the chosen place's key and
  `act_offset` the offset; ASSIST's target is the target pointer's, as ATTACK_UNIT's is (`act_enemy`).
* **Actions a decision.**  16 now; the army script gave more than 16 in 0.3% of its looks.  Economy orders add to that; step 0
  measures by how much before changing `SLOTS`.

### What a unit nobody tells does

Units no action picks keep what they're doing, constructors too: an idle constructor stands idle until told (the user's choice,
2026-11-09: no default).  The script never leaves one idle, so cloning teaches keeping them busy; in PPO, idle build power costs
what it doesn't build.

### What the policy can't see yet

* **Energy grids.**  Overdrive, and Cerberus and Lucifer firing, depend on what each grid makes, and the raw frame has no grid
  number (`01-simulator.md` 8.9).  The user's choice (2026-11-09): no grid feature.  Every own generator, pylon and extractor is a
  unit token the encoder reads, so the grid is there to infer from where they stand; the economy script's pylons are labels to
  learn it from; and in PPO the metal overdrive makes is wired into the reward.  Step 1 checks that own energy buildings and pylons
  are kept as tokens and not cut into the summaries by the 592-unit cap.
* Everything else the script decides from (income, storage, metal floating, spots and who holds them, enemy defences near a spot,
  the front) is in the frame already.

## 3. Training

Cloning first, from the economy script's orders, then PPO, as for the army.

* **Labels.**  `zks_step` returns `eco_out` as a second label stream; the recorder writes it (data version 6), tagged so that the
  army and economy orders can be counted apart.  `actions.encode` learns BUILD (type, then the nearest region and fine to the place
  among the legal ones), ASSIST and RECLAIM (target token).
* **Timing.**  The economy script looks every 4 slots and the actor every 6, so 2 of 3 of its looks fall between the actor's
  decisions.  Record with `ECO_EVERY` 6, so both scripts look at the same frames as the actor.  It changes the script's play a
  little (a constructor waits up to 3 s, not 2, for its next job); the alternative, folding the looks in between into the next
  decision's labels, gives labels from frames the actor didn't see.
* **Data.**  A new dataset, `army-6` (both scripts' orders), recorded as `army-5` was, which it replaces: `army-5` is deleted
  first (the user's choice, 2026-11-09), as `army-6` holds all it does.
* **Then `clone-7`** with the same trainer, its checks split by army and economy kinds (anchor right among constructors, type
  right, place error in elmos after the snap).
* **PPO.**  The policy's seat runs with the economy script off (`SimEnv(economy=False)` for that seat only, which needs the flag per
  seat), the opponent's with it on.  Reward as now; the diversity bonus already counts what the policy makes, and would count
  structures too, and the metal overdrive makes is added to it (the user's choice).  `policy_train.py` must load the cloned actor
  and run at `--decide-every 6` (both already on the list).

## 4. Steps

| # | Step | Check (one decisive test each) | Size |
|---|---|---|---|
| 0 | Measure: economy orders per actor decision, by kind; how often more than 16 orders a look with both scripts | a sim run of a few hundred games, `ECO_EVERY` 6 | small |
| 1 | Encode/decode: BUILD, ASSIST, RECLAIM in `actions`, the snap, the masks in `featurize`; constructors and the commander (after the opening) controllable | round trip: the economy script's orders, encoded and decoded, give the same type and a place within a footprint of the script's for ≥ 95% of them; none missed for want of a mask | medium |
| 2 | Actor: shared type head over all buildable types, own-unit and wreck pointers, their action-token terms; `KINDS` in the sim | the existing actor tests, plus a sampled decision of an untrained actor plays a sim game with BUILD orders the sim obeys | medium |
| 3 | Record: `eco_out` as labels, data version 6, `ECO_EVERY` 6, with reclaim; `army-6` | `check` on 20 sides: no order missed, kinds counted, RECLAIM among them; a first 600-game set before the whole | medium, plus ~2 h recording |
| 4 | Clone (`clone-7`) | held-out: kinds and types right as the army's are; constructors picked as often as fighters; place error | ~5 h a run |
| 5 | Play the cloned actor in the sim against the scripts, with the economy script off on its side | income and army value curves near the script's; win rate against the script | small |
| 6 | PPO from the clone | win rate against the script rises past cloning's | large |
| 7 | The opening, as the policy's too (or kept by rule: it's 1-2 minutes and the same every game) | | later |

Steps 1 and 2 are made and tested on fresh sim frames before anything is recorded.

**Done so far.**  `zks_eco_orders` (libzkscripts ABI 5) gives the economy script's orders alone, by seat; `SimEnv.eco_orders` holds
them, and the recorder writes them after the army script's (data version 6).  `ECO_EVERY` is 6, and `SimEnv` now scales it by
`decide_every` as it does the army script's: at `--decide-every 6` the economy script had looked only every 12 s.

**Step 0** (64 games of 20 minutes, variants on, 51,302 looks of a side where the scripts look):

| Per look | Mean | p50 | p99 | Max | Looks with any | Looks over 16 | Orders cut by 16 |
|---|---|---|---|---|---|---|---|
| Economy script | 1.13 | 0 | 14 | 22 | 0.368 | 0.0042 | 0.0089 |
| Army script (actions) | 1.11 | 1 | 9 | 30 | 0.507 | 0.0007 | 0.0032 |
| Both | 2.24 | 1 | 17 | 41 | 0.660 | 0.0112 | 0.0229 |

* Of the economy script's 57,843 orders, 39,049 were ASSIST and 18,794 BUILD; none RECLAIM, none MOVE.  So there was nothing to
  clone RECLAIM from; the script has reclaimed since (below, "Reclaim").
* Both scripts together exceed 16 actions in 1.1% of looks and lose 2.3% of orders to the cut; the economy's are mostly a dozen
  ASSISTs at once to one project.  `SLOTS` stays 16 for now: recordings keep raw orders and labels are made when they're read, so
  raising it (24?) or making ASSIST a group order needs no new recording, and can wait for `clone-7`'s numbers.
* Every own generator, pylon and extractor was a unit token in the frames looked at (12,798 of 12,798): none cut into summaries.

**Found on the way: cloning and PPO featurize with `economy=False`** (`policy_clone_train._sample`, `policy_train` workers): metal,
energy, income, storage, unfinished structures and idle builders read as zero, and every site as free.  That was right when the sim
had no economy (`02-policy.md`); it has had one since 2026-11-06, the frozen encoder was trained with `economy=True`
(`frame_data.py`), and `clone-5` and `clone-6` read it without.  For an actor that runs the economy it has to be on: step 1 turns it
on everywhere the actor's frames are made.

**Step 1** (16 games x 1,200 decisions, the scripts' orders encoded as the policy's choices and decoded again):

* Economy orders missed 0.58%, all for the commit lock (a unit given an order this look can't be told again yet); the army's, none.
* ASSIST: the same target 99.26%.  BUILD: the same type 99.65%; extractors on the script's spot 99.2%; other builds p50 19 elmos
  from the script's place, p90 57, 99.1% within 150.  (Region and fine had given p50 119, p90 364.)
* Constructors are controllable (`featurize._select`, `table.constructor`): 187 producible types, 127 from factories and silos and
  60 built by constructors, none both.

**Step 2.**  The actor has the place pointer (`q_place` / `gain_place` against `key_site`'s site keys and the own units' keys) and
an offset head (`offset0` / `offset1`, from the anchor's reading and the chosen place's key); `policy_model._choices` samples
place, then offset, for a BUILD.  Checks: the test suite (319); an untrained actor, sampled in the sim, tells four constructors
for 8 decisions, and every structure that goes up is one it told them to build; a place index picks the same key in a batch of
frames of different sizes.  A 30-step cloning run on 11 recorded games ran end to end (NLL of a BUILD 14.8, untrained).  The
clone trainer's held-out checks add `nll_KIND`, the whole action's NLL for each kind (for a BUILD its type, place and offset).

**Reclaim** (the user, 2026-11-09: before recording `army-6`, so that its labels have it).

* **The script** (`eco.cpp`, libzkscripts ABI 6: `zks_step_args` gains each seat's features in sight).  Up to 2 constructors, one
  for every 3 it has, reclaim full time; any other constructor with nothing better to do reclaims within 1,200 elmos rather than
  assist a factory.  It picks the best metal / (distance + 400) among wrecks of at least 30 metal on its own half (nearer its base
  than 1.15 x the enemy's), reachable, clear of enemy defences and armed units, and not already taken by another reclaimer.  A
  reclaimer below half health, or with an armed enemy within 500 elmos, goes home.  The commander doesn't reclaim.
* **The policy.**  `rf.RECLAIM` is a kind for constructors with any feature token in sight; a wreck head (`q_wreck` / `gain_wreck`
  against `key_feat`'s feature keys) points at one, and `act_wreck` embeds it in the action token.  `Actions.place` carries the
  feature's index for a RECLAIM, as it does the place's for a BUILD.
* **Measured** (64 games x 2,400 decisions, scripts): of the economy's orders, 36,631 ASSIST, 17,741 BUILD, 5,605 RECLAIM and 1,727
  going home.  Reclaim gives a side 1.42 metal a second on average (p90 5.0), 4.0% of its metal income; 20% of samples have some.
  Wreck metal in sight, in pieces of 30 or more: mean 414, p90 1,086 a side.
* **Round trip** (16 games x 1,200, as step 1): every RECLAIM decoded to the same wreck (521 of 521), none missed for want of a
  feature token; economy orders missed 0.49%, all the commit lock.

**No encoder retrain** is needed for this plan: unit tokens keep their layout, and the encoder already reads frames with the
economy on.  Feature version 6 changed only the actor's tensors and which unit tokens come first; the encoder doesn't depend on
the tokens' order (shuffled, its latents agree to 1.3e-5), so `ENCODER_FEATURE_VERSIONS` (5, 6) lets the version-5 encoder load.  At whatever retrain comes next (the user, 2026-11-09): id embeddings for up to 64 teams (players) and 8 allyteams, so
that FFA and games of many teams can be played later.  The encoder, actor and critic embed ids relative to the viewer now, with
`featurize.MAX_PLAYERS` 32 and `MAX_ALLY_TEAMS` 16; the sim plays only 1v1, so a game of more teams has to be in the data too.

## 5. Decided (the user, 2026-11-09)

1. **Idle constructors:** no default; idle until told.
2. **Grids:** no grid feature; the encoder reads every energy building and pylon, the script's grid-building is learned from, and
   overdrive's metal goes into PPO's reward.
3. **Opening:** stays a fixed script.
4. **Data:** `army-6` replaces `army-5`, which is deleted.
5. **`ECO_EVERY` 6** for recording: yes.

## 6. From the literature

From `05-literature.md` section C, which agrees with this plan on the main lines: one network (AlphaStar, SCC and Pluto learn the
whole game; OpenAI Five scripted its items and called that a limit), a type head like produce, and placement by region and fine
snapped by the script's placer rather than a 2-D map head.  What it adds, to take up at the step named:

* **Step 4 (cloning):** track constructor metrics apart, and up-weight their loss 2-3x if they lag.  Also P1 there: the anchor
  pointer's unit keys come from each unit's own features only, so whom to tell can't use what's around a unit; that holds for
  constructors as much as fighters.
* **Step 5:** DAgger, the scripts labelling the states the cloned actor reaches in the sim, which also gives economy labels where
  the script never goes.
* **Step 6 (PPO):** hand the economy over gradually (the policy runs it in a growing share of games, or the script plays the first
  T minutes, T annealed); keep the KL to the clone longer on the economy heads.  The user's overdrive-metal reward: the literature
  warns that agents overfit shaped rewards (Gym-µRTS), so keep it small, zero-sum and annealed, and never reward structures.
* **Later:** a queued head (append or replace), a build-order z to condition on, a site pointer for extractors if the snap loses
  what the script meant.
