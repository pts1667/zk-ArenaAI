# The policy model

How the learned policy works as of 2026-11-09: what it reads, how it reads it, and how it chooses orders.  The code is
`tools/policy_latent.py` (the actor), `tools/frame_vae.py` and `tools/frame_model.py` (the frozen frame encoder),
`tools/policy_model.py` (batches, sampling, the critic) and `bots/zkbot/policy/featurize.py` (raw frame to tensors).  Results and
the history behind each choice are in `docs/learned-policy/03-training.md` section 11.

## 1. The whole of it

A decision is made every 90 game frames (every 3 s): `decide_every` 6 of the sim's 15-frame slots, the rate the army script looks
at, which the actor is cloned from (2026-11-09; it was every slot, 2 a second).  The actor reads only the frames it decides at.  The
seat's raw frame becomes tokens; a frozen encoder squeezes them into 40 latent tokens; the actor reads those latents together with
the latents and actions of the frames before, over a sliding window in time.  It then gives its orders as a player does, one action after another, up to 16, until it chooses done: each action picks a
unit, its anchor, which reads the frame and seven heads choose its order, which may go as a group order to the units about it.
Units no action picks keep what they're doing.  What was done is summarised into 4 action tokens, which later frames read.

```mermaid
flowchart LR
    RF["raw frame<br/>(rawframe.Frame:<br/>units, sites, features, globals)"]
    FZ["featurizer<br/>featurize.py"]
    TOK["tokens<br/>units · regions · sites ·<br/>features · global"]
    ENC["frame encoder<br/>(frozen, 1.65M)"]
    LAT["40 latent tokens<br/>x 32 numbers"]
    TIME["time transformer<br/>2 layers, 4-frame window"]
    HIST[("kept history:<br/>6 frames of latents<br/>+ action tokens")]
    PLAN["plan: the next action<br/>reads the frame and<br/>the actions before it"]
    ANC["anchor: a unit, or done"]
    READ["the anchor reads<br/>its frame's latents"]
    HEADS["heads: kind · region ·<br/>enemy · fine · produce ·<br/>group · roles"]
    ORD["orders<br/>actions.decode"]
    ACT["4 action tokens"]

    RF --> FZ --> TOK --> ENC --> LAT --> TIME
    HIST --> TIME
    TIME --> PLAN --> ANC --> READ --> HEADS --> ORD
    HEADS -- "next action" --> PLAN
    HEADS --> ACT --> HIST
    LAT --> HIST
    TOK -. "raw unit features<br/>(queries and pointer keys)" .-> READ
```

| | |
|---|---|
| Size | `policy_latent.Size`, kept in each checkpoint: width `d` 256, 8 heads, 4 time layers, 4 plan layers (the user, 2026-11-09; it was 128, 4 heads, 2 and 2, now `SMALL`).  Diagrams and numbers below that say 128 or 2 layers are from then |
| Latent tokens a frame | 40, of 32 numbers each, LayerNormed without weights |
| Action tokens a frame | 4 (`ACTION_TOKENS`) |
| Decisions | every 3 s (`decide_every` 6; `window_frames` and the sim's `SimEnv(decide_every=)` take it) |
| Window | 4 frames: `WINDOW_SECONDS` 8 at a decision every 3 s, rounded, the frame itself included (9 s back) |
| Reach | 12 frames: each of the 4 layers reaches 3 further back, so a decision depends on the 12 before it (36 s) |
| Actions a decision | up to 16 (`actions.SLOTS`), then done |
| Parameters | 10,537,808 trained, 1,650,624 frozen (the encoder).  At `SMALL`, 1,924,568 trained: 1,550,374 before the economy, of which 435,737 choose the actions in turn; 101,029 for a build's place and offset, the site keys and their action-token terms; 57,996 for reclaim's wreck pointer, the feature keys and its action-token term; 215,169 for the units' reading before the anchor is picked, their anchor keys and whether each is told anything |
| Critic (PPO) | width 256, 4 layers of 8 heads: 3,221,769 parameters (was 128, 2 of 4: 430,473); with `--critic-bins`, its value is a distribution trained by HL-Gauss |
| Strategy z | 32 numbers a side, optional (`--z` in cloning; `Z_DIM`): see section 4 |

## 2. What it reads: the featurizer

`featurize.featurize(frame, bundle, table)` turns a raw frame into fixed-normalisation numpy tensors.  Its per-unit loops
(selection, the unit rows, the regions' live columns, the masks) are numba kernels in `featurize_kernels.py`, the same to the bit
as numpy, 0.4 ms a frame; without numba they run in plain Python, ~50x slower.  Tokens come in a fixed order:

    controllable units | every other unit, most important first | summaries | regions | sites | features | global

* **Units** (up to 592, of which up to 256 controllable, plus 12 summaries of what the caps left out): 127 numbers each. These are
  type stats from the type table, instance state (place, velocity, health, build, flags, ...), the unit's current order and its
  age, 24 terrain rays, and summary counts.  Each unit also has a type name, an alliance and a player, numbered from the viewer.
  Its token kind says what is known of it: seen, blip, memory or summary.
* **Regions** (the map's, about 120 at most): 34 numbers each.  The first 27 are static (place, terrain, spots); the rest say what
  is there now (own value, seen and radar enemy value, DPS, wreck metal, ...).
* **Sites** (metal spots, up to 24): 9 numbers.  **Features** (wrecks in sight, up to 32): 5 numbers.  **Global**: 16 numbers
  (resources, storage, commander, time since contact, ...).
* **Masks for the heads**: which tokens are controllable, which kinds each may choose (`kind_mask`), which enemies each pointer
  kind may aim at, which regions each unit may move, jump or wait to, what each factory may produce, and a route cost to every
  region.
* **Privileged extras for the critic only**: enemy units the seat can't see (`hidden_*`, up to 64).

Names are dropped per game for the actor (`NAME_DROPOUT` 0.3 of the vocabulary becomes "unknown"), so it learns to go by stats
too.  The frozen encoder always sees the true names.

## 3. The frame encoder (frozen)

A Perceiver-style reader, trained on its own as an autoencoder (`frame_vae.py`, run `encoder-big`: width 144, 3 rounds, 40 queries), then frozen: its decoder
is thrown away and the actor keeps `LatentEncoder` (the encoder, `to_mean` and a weightless LayerNorm).

```mermaid
flowchart TB
    subgraph tokens ["Tokens of one frame (embedded to 144)"]
        U["units: features + sine codes of place<br/>+ name, alliance, player embeddings<br/>→ 2-layer MLP + token-kind embedding"]
        R["regions: features + sine codes<br/>+ region pool: their units' sum (32)"]
        S["sites"]
        F["features (wrecks)"]
        G["global"]
    end
    Q["40 learned queries"]
    C1["cross-attention: queries read all tokens"]
    S1["self-attention among the 40"]
    C2["cross-attention"]
    S2["self-attention"]
    LN["LayerNorm"]
    M["to_mean: 144 → 32"]
    N["LayerNorm, no weights"]
    OUT["40 x 32 latent tokens"]

    Q --> C1
    tokens --> C1 --> S1 --> C2 --> S2 --> LN --> M --> N --> OUT
    tokens --> C2
```

* Cost grows with tokens × 40, not tokens squared: there's no self-attention over the tokens themselves (`token_layers` 0).
* Region tokens carry what their units sum to (`region_pool`, squashed by asinh), since attention averages and can't count.
* What taught the latent: a decoder asked each region what's in it now (value by side and class), the globals, and questions about
  a few units and wrecks per frame ("where is a unit like this?", "what's at this exact place?").  Terrain isn't asked: the actor
  knows the map.  See the docstring of `frame_vae.py` and `03-training.md` 11.5.

## 4. Frames in time: the actor's sequence

The actor lays frames out in order, as GPT-2 lays out text, each frame's 40 latent tokens followed by its 4 action tokens:

```mermaid
flowchart LR
    subgraph f1 ["frame t-2"]
        L1["40 latent"] --> A1["4 action"]
    end
    subgraph f2 ["frame t-1"]
        L2["40 latent"] --> A2["4 action"]
    end
    subgraph f3 ["frame t (deciding now)"]
        L3["40 latent"] --> A3["4 action<br/>(empty until chosen)"]
    end
    f1 --> f2 --> f3
```

* A latent token's input is `latent_in(latents) + slots[i]` (which of the 40 it is).  An action token's is the frame's action
  summary + `act_slots[j]`.
* Frames from before the game began are learned embeddings (`before`, `act_before`), so every decision sees a full reach.
* **A strategy, z** (optional; `05-literature.md` P8): 32 numbers for the side, added through `z_in` to every token of every frame.
  Two flags say what's known; then the army script's role weights and knobs, its preset, and in hindsight the side's extractors,
  energy, factories, defences and army at minutes 3, 6 and 10 (`policy_clone_data.Z_*`).  `z_in` starts at zero, so a z changes
  nothing until it's learned, and an unknown z (zeros) is trained for by dropping it in a share of samples (`--z-dropout`).

### Who attends to whom

Each `TimeBlock` is a pre-LayerNorm transformer block (attention, then a 4x MLP) with a learned bias on each head by how many
frames back a key is (`age`, one value per frame of the window, 4 per head), and a learned bias per head for keys that are action
tokens (`act_key`).  The mask is by frame, not by token:

| Query ↓ / key → | latents, earlier frame in window | actions, earlier frame in window | latents, same frame | actions, same frame | any later frame, or out of window |
|---|---|---|---|---|---|
| latent token | yes | yes | yes (all 40, both ways) | **no** | no |
| action token | yes | yes | yes | yes (all 4, both ways) | no |

So within a frame the latents attend to each other in both directions.  They don't see that frame's actions, which are what's
being chosen from them.  The next frame's tokens do see them.  The 4 action tokens of a frame attend to each other in both
directions too.

```mermaid
flowchart LR
    subgraph past ["frames t-3 … t-1 (in the window)"]
        PL["latents"]
        PA["actions"]
    end
    subgraph now ["frame t"]
        NL["latents"]
        NA["actions"]
    end
    PL --> NL
    PA --> NL
    PL --> NA
    PA --> NA
    NL <--> NL
    NL --> NA
    NA <--> NA
```

(Arrows run from key to query: what's read to what reads it.)

With a window of 4 frames and 2 layers, a decision depends on exactly the 6 frames before it (`reach`).  In layer 1, frame t
reads back to t-3 and frame t-3 to t-6.  In layer 2, frame t reads what t-3 took in.  Play keeps those 6 frames (section 7).

At the 10M size (4 layers, a decision every 3 s: a window of 4 frames), a decision depends on the 12 frames before it.

The attention is `F.scaled_dot_product_attention` with a dense float mask (age bias + action bias, `-inf` where unseen) of
heads × queries × keys.  The age bias is looked up a pair of frames at a time and spread over their tokens (`_age_bias`): looked up
a pair of tokens at a time, its backward pass took 77% of a cloning step (`03-training.md` 11.20).

In cloning on CUDA it's flex attention instead (`--attention flex`, `03-training.md` 11.21), the same function: a block mask of
the window (`frame_mask`, a `Frames` layout) skips the ~93% of key blocks no query sees, and the biases ride in 18 dimensions
appended to each head's queries and keys (`TimeBlock.bias_dims`): a query holds its frame's age bias for keys of each frame mod
the window, then `act_key`; a key, its frame mod the window one-hot, then whether it's an action token.  Their dot product is the
bias for every key in the window.  Play and PPO keep the dense mask.

## 5. From the sequence to orders: actions, one after another

As Pluto plays StarCraft (2026-11-08): not every unit choosing its own order at once, but a few orders, each to a unit or a group
about it, chosen in turn so that each knows what the ones before it did.  Zero-K's unit AI is made to need little micro, so what
a player does is mostly a few orders to groups.

```mermaid
flowchart LR
    S0["slot 1: start"] --> P1["plan"] --> A1["anchor: unit or done"] --> R1["anchor reads<br/>the frame"] --> H1["heads: its order"]
    H1 -- "action vector" --> S1["slot 2"] --> P2["plan"] --> A2["anchor: unit or done"] --> R2["..."]
```

### Plan: what the next action reads

A small decoder (`plan`, 2 `PlanBlock`s, width 128) runs over the slots: slot 1 is a learned start, and slot k + 1 is the vector of
action k (`action_vectors`, the same vectors the action tokens are made of, section 6), each with a learned embedding of its place
in the list.  In each block a slot attends to the frame's 40 latent tokens (`now`, what the time transformer left for this frame)
and to the slots up to its own, then a 4x MLP.

### The anchor, or done

The slot's output scores every controllable unit, `q_anchor(slot) · anchor key`, and done, `head_done(slot)`.  A unit may be
picked if it may be told something other than keep, and no action before picked it.  Done starts 3 nats up (`DONE_START`), so an
untrained actor mostly does nothing.  After done, or 16 actions (`actions.SLOTS`), the decision is over.

**Anchor keys know the frame** (`05-literature.md` P1, 2026-11-09).  Once a decision, before any unit is picked, every controllable
unit reads the 40 latents with a cross-attention block of its own (`unit_read`, its query the unit's raw embedding `hu`), and its
anchor key is `key_anchor` of that reading with its sine code (`Context.anchor_keys`).  Before, a unit's key was its own features
alone, so "my region is threatened" couldn't reach whom to tell.  From the same reading, `head_ordered` gives each unit the logit
that it's told anything at all this decision (`Context.ordered`): cloning learns it as a side loss (P2), a label for every unit at
every decision; nothing samples from it.

### The anchor reads its frame

The anchor then reads the latents with a cross-attention block (`read_unit`, in `reading`), its query an embedding of its **own raw
features** (`hu`: the same kind of embedding as the encoder's, but the actor's own weights) plus a projection of its slot.  So it
knows exactly where and what it is, whatever the latents blurred, and asks the frame what matters to it for this action.

```mermaid
flowchart LR
    X["anchor's raw features<br/>+ sine codes of place<br/>+ name / ally / player"] --> HU["hu (128)<br/>MLP + token kind"]
    SL["its slot (plan)"] --> PJ["slot_unit"]
    HU --> ADD(("+"))
    PJ --> ADD
    ADD --> RU["read_unit<br/>cross-attention<br/>over the 40 latents"]
    NOW["now: 40 x 128"] --> RU
    RU --> ME["me (128): the action's reading"]
    HU --> KU["unit key: key_unit(hu) ‖ sine code (152)"]
    RX["region features + centre code"] --> KR["region key: key_region ‖ sine code (152)"]
```

Every unit token (own or enemy) has a **key**: `key_unit(hu)` followed by the 24-number sine code of its place.  Regions get keys
the same way.  Pointers score against these keys, built from raw features, so a target's exact place is there to choose.

### The heads

What the anchor reads (`me`) goes to the heads, with the anchor's own masks (`policy_model.per_action` takes them out of the batch as
if the actions were units):

```mermaid
flowchart TB
    ME["me (each action's anchor)"]
    K["kind head<br/>linear → 21 kinds<br/>(illegal and keep masked)"]
    RG["region pointer<br/>(q_region(me) ‖ gain · own code) · region keys<br/>− softplus(route_scale) × route cost"]
    EN["target pointers, one per kind:<br/>attack unit · attack ground · manual fire · assist<br/>(q_enemy(me) ‖ gain · own code) · target keys"]
    FI["fine head<br/>MLP(me ‖ target key) → 9 options"]
    PR["produce head<br/>q_type(me) · type keys of producible types<br/>(what factories make and constructors build)"]
    PL["place pointer (build)<br/>(q_place(me) ‖ gain · own code) · keys of<br/>our unit tokens, then the sites"]
    OF["offset head (build)<br/>MLP(me ‖ place key) → 25 options"]
    WR["wreck pointer (reclaim)<br/>(q_wreck(me) ‖ gain · own code) · feature keys"]
    GR["group head<br/>MLP(me + kind embedding) → alone or 10 radii"]
    RO["roles head<br/>MLP(me + kind + radius embeddings) → a logit per role"]

    ME --> K
    ME --> RG
    ME --> EN
    ME --> FI
    ME --> PR
    ME --> PL
    PL --> OF
    ME --> WR
    ME --> GR
    GR --> RO
```

* **Kind**: move, fight, attack unit, attack ground, stop, manual fire, jump, cloak on/off, produce, throw, push, pull, place
  beacon, wait at beacon, build, assist, reclaim, ... (21 in all; the sim's games enable 17 with keep, `policy_sim_env.KINDS`).
  Never keep: an action tells its unit something, and a unit no action picks keeps.  Constructors and the commander are
  controllable for build and assist, constructors for reclaim too (`docs/learned-policy/04-economy.md`).
* **Pointers with place-aware keys**: a pointer's query ends in the anchor's own sine code, each band scaled by a gain it chooses
  (`gain_region`, `gain_enemy`).  Its dot product with a key's sine code is then a sum over bands of gain × cos(band × difference in
  place): a learned function of distance and direction, which a dot product of raw coordinates can't give.
* **Region** also subtracts a learned multiple of the walking cost to each region (`route_bias`), so near regions come first.
* **Fine** (9 options): option 0 is the target itself; options 1-8 are the compass points around it.  For a region, these are at
  half its equivalent radius, with north towards the enemy; for attack ground, they are 200 elmos from the enemy target.  It reads
  the chosen target's key, so it's chosen after the target.
* **Produce**: a factory's or silo's choice among producible types, scored against embeddings of their stats; a build's type is
  the same head, over what the constructor can build (187 producible types: 127 made by factories and silos, 60 built).
* **Assist** is the target pointer's fourth kind: one of our units being built, or a factory (`enemy_tokens` holds those own
  tokens too).
* **Place and offset** (build only): the place pointer chooses one of our unit tokens or a site, with the same place-aware keys;
  the offset head then chooses where beside it, from its key: on it, or one of 8 compass points at 64, 128 or 256 elmos, north
  towards the enemy.  The sim puts the building at the nearest place its footing fits (`SimEnv.place_build`), an extractor at
  the nearest free spot.
* **Wreck** (reclaim only): a pointer over the feature tokens (wrecks in sight), with keys from each feature's numbers and sine
  code (`feat_in`, `key_feat`); the action's `place` field holds the chosen feature's index.
* **Group** (move, fight and attack only): alone, or as a group order of one of 10 radii, 60 to 2,250 elmos
  (`bots/zkbot/policy/groups.py`): the sim then gives the order as well to every unit of the side of the chosen roles within that
  radius of the anchor.  It starts at alone (`ALONE_START`).
* **Roles** (a group only): each of the 15 roles with a mobile type in or out, a Bernoulli each, given the radius.

### Sampling, in order

```mermaid
sequenceDiagram
    participant P as plan (slot k)
    participant H as heads (its anchor)
    participant S as sample
    P->>S: anchor logits: done, or each unit that may be picked
    S-->>P: anchor (done ends the decision)
    H->>S: kind logits (the anchor's kind_mask, keep out)
    S-->>H: kind
    H->>S: region logits, masked by what the kind may reach<br/>(ptr_region, ptr_jump for jump/throw, ptr_wait)
    S-->>H: region (used only by kinds with a region)
    H->>S: the kind's enemy logits (attack unit / ground / manual fire)
    S-->>H: enemy (used only by those kinds)
    H->>S: fine logits from me ‖ chosen target's key
    S-->>H: fine (used only by kinds with a fine option)
    H->>S: produce logits (masked by what it can make or build)
    S-->>H: type (used only by produce and build)
    H->>S: place logits, then offset logits from the chosen place's key
    S-->>H: place and offset (used only by build)
    H->>S: wreck logits over the feature tokens
    S-->>H: wreck (used only by reclaim)
    H->>S: group logits from me and the kind
    S-->>H: alone or a radius (used only by move, fight, attack)
    H->>S: role logits from me, the kind and the radius
    S-->>H: roles (used only by a group)
    H-->>P: the action's vector, slot k + 1's input
```

An action's log-probability is its anchor's (or done's) plus the heads its kind uses; a decision's is the sum over its actions.
Sampling is Gumbel-max.  `actions.to_decision` makes the actions what each unit was told, and `actions.decode` orders for the sim or
the engine.  In training all the slots are scored in one pass (`evaluate_actions`): the decoder is causal over them, so each scores
as when it was chosen.

`policy_model.each_unit_heads` gives the heads for every unit as if it were the first action's anchor, to look at what the policy
makes of each unit.

## 6. Action tokens: what was done

What was done at a frame becomes that frame's 4 action tokens (`action_tokens`).  Only the policy's own choices go in (in cloning,
both scripts' orders, which are the labels): the opening's orders, and the economy script's where it still runs on the policy's
side, show only in later frames, as each unit's current order.

```mermaid
flowchart LR
    subgraph each ["each action (not done)"]
        E["kind embedding<br/>+ act_unit(anchor's hu)<br/>+ act_enemy(target key)<br/>+ act_region(region key)<br/>+ act_fine(option)<br/>+ act_type(type stats)<br/>+ act_place(place key) + act_offset(option)<br/>+ act_wreck(feature key)<br/>+ act_group(radius) + act_roles(roles)"] --> G["GELU → act_out<br/>(action_vectors)"]
    end
    NOTH["learned 'nothing' vector<br/>(always there)"]
    Q4["4 learned queries<br/>+ act_count(log(1 + actions))"]
    CR["act_read<br/>cross-attention"]
    OUT["4 action tokens (128 each)"]
    G --> CR
    NOTH --> CR
    Q4 --> CR --> OUT
```

However many actions there were (0 to 16; in cloning, every order the script gave), the frame's actions are summarised into 4
tokens; a frame where the policy was done at once still has 4 tokens, which read only "nothing" and the count.  They are a
fixed-size, unordered summary.  The same action vectors are what the decoder reads of the actions before (section 5).

## 7. Play and training

### In play: one decision at a time

A seat keeps the 6 frames before the present, newest first: each frame's raw latents (40 × 32) and its 4 action tokens, flat
(`record` numbers a frame).  `decide` runs the encoder on the new frame, lays out the 7 frames, and runs the sequence.  The first
layer runs over every token.  The last runs only the newest frame's latent tokens as queries (`newest_only`).  The actions are then
chosen in turn (`policy_model.sample_actions`), and `acted` writes their tokens into the newest frame's slot.

```mermaid
sequenceDiagram
    participant Env as sim / engine
    participant Seat as seat state (6 frames)
    participant Pol as actor
    Env->>Pol: featurized frame t
    Pol->>Pol: encoder → 40 latents
    Seat->>Pol: frames t-6 … t-1 (latents + action tokens)
    Pol->>Pol: sequence (last layer: frame t's latents only)
    Pol->>Pol: actions in turn: plan, anchor, its heads
    Pol->>Env: orders
    Pol->>Seat: push frame t's latents, then its action tokens (acted)
```

### In training: a stretch in one pass

Training takes a stretch of decisions with the 6 frames before it and runs the whole sequence at once.  Because the mask is
causal by frame, every decision in the stretch sees exactly what it would have seen in play.

* **Cloning** (`policy_clone_train.py`): stretches of 48 of the actor's decisions from recorded script games
  (`policy_clone_data.py`).  Games are recorded every slot; the actor's decision j is recorded frame 6j (`--decide-every`), each one
  where the script looked.  The script's orders become the labels (`actions.encode`), then actions in a random order, at most 16
  (`actions.to_slots`), and the action tokens are built from all of them.  The loss is the negative log-probability of each action,
  done included.  The script's orders have no order of their own, so an action's anchor scores as any unit the script told
  something and no action before it picked (`evaluate_actions`' `labelled`).  A step's samples go through the encoder and into
  action tokens together, one call each; a worker decodes only the side and the part of a game its sample needs.
* **PPO** (`policy_train.py`): segments of 32 decisions from the sim, each started from the state kept at collection.  It must run
  at the actor's rate, `--decide-every 6`, with `--fine-sim` (the sim steps every slot, as the data was recorded); `--init` starts
  it from a cloning checkpoint, with a frozen copy for an annealed KL to it and a cloning loss on the scripts' advice (`03-training.md`
  11.22).  Each action
  gets its own clipped ratio, averaged within a decision (done included), with one advantage per decision.  The **critic** is a separate network,
  never in the bot.  It is a 2-layer self-attention encoder over the privileged frame, including the enemy units the seat can't
  see, and reads the value off its global token.

```mermaid
flowchart LR
    subgraph enc ["trained once, then frozen"]
        FD["frames<br/>frame_data.py"] --> VAE["frame VAE<br/>frame_vae.py"] --> E["LatentEncoder"]
    end
    subgraph clone ["cloning"]
        REC["recorded script games<br/>.zkr (policy_clone_data.py)"] --> CT["policy_clone_train.py"]
    end
    subgraph rl ["RL"]
        SIM["sim games<br/>policy_sim_env.py"] --> PPO["policy_train.py<br/>actor + critic"]
    end
    E --> CT
    E --> PPO
    CT -- "--init" --> PPO
```

## 8. Where it stands

* Cloning with the economy, on `army-6` (`03-training.md` 11.21, 11.22): `clone-7`, 1,000 steps, held out anchor right 0.856,
  kind right 0.938, NLL of actions 2.84.  In the sim against the army script (`policy_clone_eval.py`, 64 games) it wins 3% and
  draws 17%: it keeps up for 3 minutes, then stops expanding (a third of the script's extractors at minute 10).  `clone-8` (z,
  economy weight 2.5, finishing on winners) is even with it held out; the small actor is worse on every measure held out.  In
  play all of them, at temperature 1.0 or 0.8 and with z or without, win 2-5% and stall the same way.
* Cloning with actions in turn on `army-5` (`03-training.md` 11.20): at a decision every slot (`clone-5`, stopped at 11,500 of
  20,000 steps), the first unit picked was one the script told 68% of the time and the order's kind was right 99% of the time.
  `clone-6`, at a decision every 3 s, cloned about 1,050 of the script's decisions a second where `clone-5` cloned 52, and reached
  the same at 2,000 steps (anchor right 0.675, kind right 0.989); it was stopped there, as the policy is to run the economy too
  (`docs/learned-policy/04-economy.md`).
* Nothing plays in the engine yet.  The bot (`bots/zkbot`) hosts the policy but sends no commands, until a numpy forward of the
  actor and encoder, and the opening and economy script, are there.
* Open questions, from the user:
  * full attention over all past frames through summary tokens, in place of the 9 s window;
  * FlexAttention, with one token per acting unit and causal attention along them in place of the 4 summary tokens.  Within a
    decision the actions now are such tokens (the decoder); between frames they're still the 4 summary tokens.
