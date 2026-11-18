# The policy model

How the learned policy works: what it reads, how it reads it, and how it chooses orders.  The code:

* `tools/policy_latent.py`: the actor;
* `tools/frame_vae.py` and `tools/frame_model.py`: the frozen frame encoder;
* `tools/policy_model.py`: batches, sampling and the critic;
* `bots/zkbot/policy/featurize.py`: raw frame to tensors.

The observations and actions are specified in `docs/learned-policy/02-policy.md`, and training and its results in
`docs/learned-policy/03-training.md`.

## The whole of it

A decision is made every 90 game frames (3 s): `decide_every` 6 of the sim's 15-frame slots.  That's when the army script, which
the actor is cloned from, looks.  The actor reads only the frames it decides at.

1. The seat's raw frame becomes tokens.
2. A frozen encoder squeezes them into 40 latent tokens.
3. The actor reads those latents with the latents and actions of the frames before, over a sliding window in time.
4. It gives its orders as a player does: one action after another, up to 16, until it chooses done.  Each action picks a unit,
   its anchor.  The anchor reads the frame, and the heads choose its order.  That order may go as a group order to the units
   around it.
5. Units no action picks keep what they're doing.  What was done is summarised into 4 action tokens, which later frames read.

```mermaid
flowchart LR
    RF["raw frame<br/>(rawframe.Frame:<br/>units, sites, features, globals)"]
    FZ["featurizer<br/>featurize.py"]
    TOK["tokens<br/>units · regions · sites ·<br/>features · global"]
    ENC["frame encoder<br/>(frozen, 1.65M)"]
    LAT["40 latent tokens<br/>x 32 numbers"]
    TIME["time transformer<br/>4 layers, 4-frame window"]
    HIST[("kept history:<br/>12 frames of latents<br/>+ action tokens")]
    PLAN["plan: the next action<br/>reads the frame and<br/>the actions before it"]
    ANC["anchor: a unit, or done"]
    READ["the anchor reads<br/>its frame's latents"]
    HEADS["heads: kind · region · target ·<br/>fine · type · place · offset ·<br/>wreck · group · roles"]
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
| Size | `policy_latent.SIZE`, kept in each checkpoint: width `d` 256, 8 heads, 4 time layers, 4 plan layers, no unit layers.  `SMALL` is 128, 4 heads, 2 and 2.  `WIDE` (the user, 2026-11-14; `full-4`, `policy_clone_train.py --size wide`) is 384, 12 heads, 4 and 4, and 2 unit layers.  `GRU` (the user, 2026-11-17; `policy_train.py --size gru`) is 256, 8 heads, a GRU of 1,024 in place of the time layers, 1 block within the frame and 2 plan layers: "A GRU for memory" below.  `RDNN` (the user, 2026-11-17; `--size rdnn`) is `GRU` with a Recurrent Divisive Normalization Network of 1,024 units where the GRU is, its state told to what reads the frame as a vector, not as tokens: "An RDNN in the GRU's place" |
| Parameters | 10,546,864 trained (`SMALL`: 1,929,088; `WIDE`: 27,053,667, 3,548,928 of them in the unit layers), and 1,650,816 frozen in the encoder at `encoder-big`'s recipe (2,893,776 at `full-4`'s, 192 wide with 64 queries) |
| Latent tokens a frame | 40, of 32 numbers each, LayerNormed without weights |
| Action tokens a frame | 4 (`ACTION_TOKENS`) |
| Decisions | every 3 s (`decide_every` 6; `window_frames` and `SimEnv(decide_every=)` take it) |
| Window | 4 frames, the frame itself included: `WINDOW_SECONDS` 8 at a decision every 3 s, rounded (9 s back) |
| Reach | 12 frames: each of the 4 layers reaches 3 further back, so a decision depends on the 12 before it (36 s) |
| Actions a decision | up to 16 (`actions.SLOTS`), then done |
| Who owns a unit | an allyteam and a player numbered from the viewer, in tables of 8 and 64 (`featurize.MAX_ALLY_TEAMS`, `MAX_PLAYERS`), 8 numbers each |
| Critic (PPO) | width 256, 8 heads, 32 learnt queries and the global token reading every token in 4 rounds: 6,655,081 parameters (6,667,931 with 51 bins); with `--critic-bins`, its value is a distribution trained by HL-Gauss |
| Critic of the memory (bootstrapped GRPO, `--critic-size memory`) | for the actor with a GRU: a net for each of the reward's terms, two layers of 256 on the frame's latents, flat, and the GRU's state: 0.87M parameters a term with 51 bins |
| Strategy z | 32 numbers a side, optional (`--z` in cloning; `Z_DIM`): see "Frames in time" |

## What it reads: the featurizer

`featurize.featurize(frame, bundle, table)` turns a raw frame into fixed-normalisation numpy tensors (02-policy.md, "The
featurizer").  Tokens come in a fixed order:

    controllable units | every other unit, most important first | summaries | regions | sites | features | global

* **Units**: up to 1,280, of which up to 512 are controllable, plus 12 summaries of what the caps left out.  Each has 129 numbers:
  * type stats from the type table;
  * instance state: place, velocity, health, build, flags;
  * its current order and the order's age;
  * 24 terrain rays;
  * summary counts.
  
  Each unit also has a type name, an allyteam and a player, numbered from the viewer.  Its token kind says what is known of it:
  seen, blip or summary.  A building seen once and now out of sight was a token too, of kind "memory", till 2026-11-18
  (the user): the actor's own memory keeps what it saw (docs/learned-policy/02-policy.md, "Nothing remembered").
* **Regions** (about 120 at most): 34 numbers each.  The first 27 are static (place, terrain, spots).  The rest say what is there
  now: own value, seen and radar enemy value, DPS, wreck metal, and so on.
* **Sites** (metal spots, up to 24): 9 numbers.
* **Features** (wrecks in sight, up to 512): 5 numbers.
* **Global**: 16 numbers (resources, storage, commander, time since contact, ...).
* **Masks for the heads**:
  * which tokens are controllable;
  * which kinds each may choose (`kind_mask`);
  * which targets each pointer kind may aim at;
  * which regions each unit may move, jump or wait to;
  * what each factory may produce and each constructor build;
  * a route cost to every region.
* **For the critic only**: enemy units the seat can't see (`hidden_*`, up to 256, nearest the seat's army first; 64 before
  2026-11-11).

Names are dropped per game for the actor (`NAME_DROPOUT`: 30% of the vocabulary becomes "unknown"), so it learns to go by stats
too.  The frozen encoder always sees the true names.

## The frame encoder (frozen)

A Perceiver-style reader, trained on its own as an autoencoder (`frame_vae.py`), then frozen.  The recipe is `encoder-big`'s
(03-training.md, "The frame encoder"): width 144, 3 rounds, 40 queries.  Its decoder is thrown away, and the actor keeps
`LatentEncoder`: the encoder, `to_mean` and a weightless LayerNorm.

```mermaid
flowchart TB
    subgraph tokens ["Tokens of one frame (embedded to 144)"]
        U["units: features + sine codes of place<br/>+ name, allyteam, player embeddings<br/>→ 2-layer MLP + token-kind embedding"]
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

(Two of the three rounds are drawn.)

* **Cost** grows with tokens × 40, not tokens squared: there's no self-attention over the tokens themselves (`token_layers` 0).
* **Region tokens** carry what their units sum to (`region_pool`, squashed by asinh), since attention averages and can't count.
* **What taught the latent**: a decoder asked each region what's in it now (value by side and class), and the globals.  It also
  asked questions about a few units and wrecks a frame: "where is a unit like this?" and "what's at this exact place?".
  Terrain isn't asked: the actor knows the map.

## Frames in time: the actor's sequence

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

* **Inputs**: a latent token's is `latent_in(latents) + slots[i]` (which of the 40 it is).  An action token's is the frame's
  action summary + `act_slots[j]`.
* **Before the game**: frames from before it began are learned embeddings (`before`, `act_before`), so every decision sees a full
  reach.
* **A strategy, z** (optional): 32 numbers for the side, added through `z_in` to every token of every frame.
  * Two flags say what's known.  Then come the army script's role weights and knobs, its preset, and in hindsight the side's
    extractors, energy, factories, defences and army at minutes 3, 6 and 10 (`policy_clone_data.Z_*`).
  * `z_in` starts at zero, so a z changes nothing until it's learned.
  * An unknown z (zeros) is trained for by dropping it in a share of samples (`--z-dropout`).  No z (None) is the same as zeros.
  * In play and PPO, `--z variant` tells the actor its side's knobs and preset, with the counts unknown.

### Who attends to whom

Each `TimeBlock` is a pre-LayerNorm transformer block: attention, then a 4x MLP.  It has two learned biases on each head:

* by how many frames back a key is (`age`, one value per frame of the window);
* for keys that are action tokens (`act_key`).

The mask is by frame, not by token:

| Query ↓ / key → | latents, earlier frame in window | actions, earlier frame in window | latents, same frame | actions, same frame | any later frame, or out of window |
|---|---|---|---|---|---|
| latent token | yes | yes | yes (all 40, both ways) | **no** | no |
| action token | yes | yes | yes | yes (all 4, both ways) | no |

So within a frame, the latents attend to each other in both directions.  They don't see that frame's actions, which are what's
being chosen from them; the next frame's tokens do see them.  The 4 action tokens of a frame attend to each other in both
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

With a window of 4 frames, each layer reaches 3 frames further back.  So after 4 layers, a decision depends on exactly the 12
frames before it (`reach`), and play keeps those 12.

**Attention, two ways, the same function:**

* **Dense** (play and PPO): `F.scaled_dot_product_attention` with a float mask of heads × queries × keys (age bias + action bias,
  `-inf` where unseen).  The age bias is looked up a pair of frames at a time and spread over their tokens (`_age_bias`).
* **Flex** (cloning on CUDA, `--attention flex`):
  * a block mask of the window (`frame_mask`, a `Frames` layout) skips the ~93% of key blocks no query sees;
  * the biases ride in 18 dimensions appended to each head's queries and keys (`TimeBlock.bias_dims`).  A query holds its
    frame's age bias for keys of each frame mod the window, then `act_key`.  A key holds its frame mod the window, one-hot, then
    whether it's an action token.  Their dot product is the bias.

## From the sequence to orders: actions, one after another

As Pluto plays StarCraft: not every unit choosing its own order at once, but a few orders, each to a unit or a group around it,
chosen in turn so that each knows what the ones before it did.  Zero-K's unit AI is made to need little micro, so what a player
does is mostly a few orders to groups.

```mermaid
flowchart LR
    S0["slot 1: start"] --> P1["plan"] --> A1["anchor: unit or done"] --> R1["anchor reads<br/>the frame"] --> H1["heads: its order"]
    H1 -- "action vector" --> S1["slot 2"] --> P2["plan"] --> A2["anchor: unit or done"] --> R2["..."]
```

### Plan: what the next action reads

A small decoder (`plan`: 4 `PlanBlock`s at width 256) runs over the slots.  Slot 1 is a learned start, and slot k + 1 is the
vector of action k (`action_vectors`, the same vectors the action tokens are made of; see "Action tokens").  Each slot also has a
learned embedding of its place in the list.  In each block, a slot attends to the frame's 40 latent tokens (`now`, what the time
transformer left for this frame) and to the slots up to its own, then a 4x MLP.

### The anchor, or done

The slot's output scores every controllable unit, `q_anchor(slot) · anchor key`, and done, `head_done(slot)`.

* **Who may be picked**: a unit that may be told something other than keep, and that no action before picked.
* **Done** starts 3 nats up (`DONE_START`), so an untrained actor mostly does nothing.  After done, or 16 actions, the decision
  is over.

**Anchor keys know the frame.**  Once a decision, before any unit is picked, every controllable unit reads the 40 latents with a
cross-attention block of its own (`unit_read`, its query the unit's raw embedding `hu`).  With `Size.unit_layers` (2 in `WIDE`),
the controllable units first read each other in that many self-attention blocks (`LatentActor.among`, padding masked): who's
beside whom, which the 40 latents blur.  The picked unit's later reading of the frame (`reading`) starts from that row too
(`Context.mine`); the action tokens of earlier frames still read raw embeddings, having no controllable mask.  Its anchor key is `key_anchor` of that
reading, with its sine code (`Context.anchor_keys`).  So "my region is threatened" reaches whom to tell.  From the same reading,
`head_ordered` gives each unit the logit that it's told anything at all this decision (`Context.ordered`).  Cloning learns it as
a side loss, a label for every unit at every decision.  Nothing samples from it.

### The anchor reads its frame

The anchor then reads the latents with a cross-attention block (`read_unit`, in `reading`).  Its query is an embedding of its
**own raw features** (`hu`: the same kind of embedding as the encoder's, with the actor's own weights), plus a projection of its
slot.  So it knows exactly where and what it is, whatever the latents blurred, and asks the frame what matters to it for this
action.

```mermaid
flowchart LR
    X["anchor's raw features<br/>+ sine codes of place<br/>+ name / allyteam / player"] --> HU["hu (256)<br/>MLP + token kind"]
    SL["its slot (plan)"] --> PJ["slot_unit"]
    HU --> ADD(("+"))
    PJ --> ADD
    ADD --> RU["read_unit<br/>cross-attention<br/>over the 40 latents"]
    NOW["now: 40 x 256"] --> RU
    RU --> ME["me (256): the action's reading"]
    HU --> KU["unit key: key_unit(hu) ‖ sine code (280)"]
    RX["region features + centre code"] --> KR["region key: key_region ‖ sine code (280)"]
```

Every unit token, own or enemy, has a **key**: `key_unit(hu)` followed by the 24-number sine code of its place.  Regions, sites
and features get keys the same way.  Pointers score against these keys, built from raw features, so a target's exact place is
there to choose.

### The heads

What the anchor reads (`me`) goes to the heads, with the anchor's own masks.  `policy_model.per_action` takes the masks out of
the batch as if the actions were units.

```mermaid
flowchart TB
    ME["me (each action's anchor)"]
    K["kind head<br/>linear → 24 kinds<br/>(illegal and keep masked)"]
    RG["region pointer<br/>(q_region(me) ‖ gain · own code) · region keys<br/>− softplus(route_scale) × route cost"]
    EN["target pointers, one per kind:<br/>attack unit · attack ground · manual fire · assist · load<br/>(q_enemy(me) ‖ gain · own code) · target keys"]
    FI["fine head<br/>MLP(me ‖ target key) → 9 options"]
    PR["type head<br/>q_type(me) · type keys of producible types<br/>(what factories make and constructors build)"]
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

* **Kind**: one of 24 (02-policy.md, "Kinds"); the sim's games enable 20 with keep (`policy_sim_env.KINDS`).  Never keep: an
  action tells its unit something, and a unit no action picks keeps.
* **Pointers with place-aware keys**: a pointer's query ends in the anchor's own sine code, each band scaled by a gain it chooses
  (`gain_region`, `gain_enemy`).  Its dot product with a key's sine code is then a sum over bands of gain × cos(band × difference
  in place): a learned function of distance and direction, which a dot product of raw coordinates can't give.
* **Region** also subtracts a learned multiple of the walking cost to each region (`route_bias`), so near regions come first.
* **Fine** (9 options): option 0 is the target itself, and options 1-8 the compass points around it.  For a region, these are at
  half its equivalent radius, with north towards the enemy.  For attack ground, they are 200 elmos from the target.  It reads
  the chosen target's key, so it's chosen after the target.
* **Type**: what a factory or silo produces, or what a constructor builds.  It's scored against embeddings of the types' stats,
  over 187 producible types: 127 made by factories and silos, 60 built.
* **Assist** is the target pointer's fourth kind: one of our units being built, or a factory (`enemy_tokens` holds those own
  tokens too).
* **Load** is its fifth, an air transport's: one of our units it can lift, or another alliance's in sight that stands still.
  **Unload** chooses a region and a fine option as a jump does, over where the load can stand; **drop** chooses nothing.
* **Place and offset** (build only): the place pointer chooses one of our unit tokens or a site, with the same place-aware keys.
  The offset head then chooses where beside it, from its key: on it, or one of 8 compass points at 64, 128 or 256 elmos, north
  towards the enemy.
* **Wreck** (reclaim only): a pointer over the feature tokens, with keys from each feature's numbers and sine code (`feat_in`,
  `key_feat`).  The action's `place` field holds the chosen feature's index.
* **Group** (move, fight and attack only): alone, or a group order of one of 10 radii, 60 to 2,250 elmos
  (`bots/zkbot/policy/groups.py`).  It starts at alone (`ALONE_START`).
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
    H->>S: region logits, masked by what the kind may reach<br/>(ptr_region, ptr_jump for jump/throw/unload, ptr_wait)
    S-->>H: region (used only by kinds with a region)
    H->>S: the kind's target logits (attack unit / ground / manual fire / assist / load)
    S-->>H: target (used only by those kinds)
    H->>S: fine logits from me ‖ chosen target's key
    S-->>H: fine (used only by kinds with a fine option)
    H->>S: type logits (masked by what it can make or build)
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

* **Log-probability**: an action's is its anchor's (or done's) plus the heads its kind uses.  A decision's is the sum over its
  actions.
* **Sampling** is Gumbel-max.
* **To orders**: `actions.to_decision` makes the actions what each unit was told, and `actions.decode` makes orders for the sim
  or the engine.
* **In training**, all the slots are scored in one pass (`evaluate_actions`).  The decoder is causal over them, so each scores as
  when it was chosen.
* `policy_model.each_unit_heads` gives the heads for every unit as if it were the first action's anchor, to look at what the
  policy makes of each unit.

## Action tokens: what was done

What was done at a frame becomes that frame's 4 action tokens (`action_tokens`).  Only the policy's own choices go in.  In
cloning, both scripts' orders go in, since they're the labels.  The opening's orders, and the economy script's where it still
runs on the policy's side, show only in later frames, as each unit's current order.

```mermaid
flowchart LR
    subgraph each ["each action (not done)"]
        E["kind embedding<br/>+ act_unit(anchor's hu)<br/>+ act_enemy(target key)<br/>+ act_region(region key)<br/>+ act_fine(option)<br/>+ act_type(type stats)<br/>+ act_place(place key) + act_offset(option)<br/>+ act_wreck(feature key)<br/>+ act_group(radius) + act_roles(roles)"] --> G["GELU → act_out<br/>(action_vectors)"]
    end
    NOTH["learned 'nothing' vector<br/>(always there)"]
    Q4["4 learned queries<br/>+ act_count(log(1 + actions))"]
    CR["act_read<br/>cross-attention"]
    OUT["4 action tokens (256 each)"]
    G --> CR
    NOTH --> CR
    Q4 --> CR --> OUT
```

However many actions there were (0 to 16; in cloning, every order the scripts gave), the frame's actions are summarised into 4
tokens: a fixed-size, unordered summary.  A frame where the policy was done at once still has 4 tokens, which read only "nothing"
and the count.  The same action vectors are what the decoder reads of the actions before.

## Play and training

### In play: one decision at a time

A seat keeps the 12 frames before the present, newest first: each frame's raw latents (40 × 32) and its 4 action tokens, flat
(`record` numbers a frame).

1. `decide` runs the encoder on the new frame, lays out the 13 frames, and runs the sequence.  The first layers run over every
   token.  The last runs only the newest frame's latent tokens as queries (`newest_only`).
2. The actions are chosen in turn (`policy_model.sample_actions`).
3. `acted` writes their tokens into the newest frame's slot.

```mermaid
sequenceDiagram
    participant Env as sim / engine
    participant Seat as seat state (12 frames)
    participant Pol as actor
    Env->>Pol: featurized frame t
    Pol->>Pol: encoder → 40 latents
    Seat->>Pol: frames t-12 … t-1 (latents + action tokens)
    Pol->>Pol: sequence (last layer: frame t's latents only)
    Pol->>Pol: actions in turn: plan, anchor, its heads
    Pol->>Env: orders
    Pol->>Seat: push frame t's latents, then its action tokens (acted)
```

### In training: a stretch in one pass

Training takes a stretch of decisions with the 12 frames before it and runs the whole sequence at once.  The mask is causal by
frame, so every decision in the stretch sees exactly what it would have seen in play.

* **Cloning** (`policy_clone_train.py`):
  * Stretches of 48 of the actor's decisions come from recorded script games (`policy_clone_data.py`).  Games are recorded every
    slot, and the actor's decision j is recorded frame 6j, each one where the script looked.
  * The scripts' orders become the labels (`actions.encode`): actions in a random order, at most 16 (`actions.to_slots`).  The
    action tokens are built from all of them.
  * The loss is the negative log-probability of each action, done included.  The orders have no order of their own, so an
    action's anchor scores as any unit still to be told (`evaluate_actions`' `labelled`).
* **PPO** (`policy_train.py`):
  * Segments of 32 decisions from the sim, each started from the state kept at collection.  It runs at the actor's rate,
    `--decide-every 6`, with `--fine-sim` (the sim steps every slot, as the data was recorded).
  * `--init` starts it from a cloning checkpoint, with a frozen copy for a KL to it and a cloning loss on the scripts'
    advice, both fading by iteration.  `--z variant` tells the actor its side's knobs, kept on each step so the update reads
    what play did.
  * The policy loss is Simple Policy Optimization's (SPO), not PPO's clip.  Each action gets its own ratio, averaged within a
    decision (done included), with one advantage per decision.
  * The **critic** is a separate network, never in the bot: learnt queries reading the privileged frame, including the enemy
    units the seat can't see, as the frame encoder reads (so its cost grows with the tokens, not their square).  The global
    token is one of the queries, and it reads the value off that.  Muon trains its hidden matrices.

```mermaid
flowchart LR
    subgraph enc ["trained once, then frozen"]
        FD["frames<br/>frame_data.py"] --> VAE["frame autoencoder<br/>frame_vae.py"] --> E["LatentEncoder"]
    end
    subgraph clone ["cloning"]
        REC["recorded script games<br/>.zkr (policy_clone_data.py)"] --> CT["policy_clone_train.py"]
    end
    subgraph dagger ["DAgger rounds"]
        DR["the clone plays, the scripts advise<br/>policy_clone_eval.py --record"] --> DT["fine-tuning<br/>policy_clone_train.py --mix --init"]
    end
    subgraph rl ["RL"]
        SIM["sim games<br/>policy_sim_env.py"] --> PPO["policy_train.py<br/>actor + critic"]
    end
    E --> CT
    E --> PPO
    CT -- "--init" --> DR
    DT -- "next round" --> DR
    DT -- "--init" --> PPO
```

`tools/policy_pipeline.py` runs all of it in order, one stage at a time, and carries on where it left off (03-training.md, "The
pipeline").

## A GRU for memory (`Size.gru`)

The user's, 2026-11-17, as Pluto has it (a GRU and what was done before, not a window of frames): for RL from nothing
(`policy_train.py --size gru`, `policy_latent.GRU`), not for cloning, whose stretches the actor's `sequence` reads.  Of the
wide actor's 27M trained parameters the time transformer was 7.1M, and in play its first three layers ran again over 13
frames of 68 tokens at every decision.

```mermaid
flowchart LR
    LAT["latent tokens<br/>of the frame"]
    DID["what the last decision did<br/>(its actions' vectors averaged,<br/>and how many)"]
    GRUC["GRU, 1,024<br/>(the state: all a seat keeps)"]
    MEM["4 memory tokens<br/>read off the state"]
    FRAME["the frame: latents + memory tokens<br/>1 block among themselves"]
    PLAN["plan, anchor, heads<br/>(as before, 2 plan layers)"]
    CRIT["a critic a term<br/>(latents, flat, + state)"]

    LAT --> GRUC
    DID --> GRUC
    GRUC --> MEM --> FRAME
    LAT --> FRAME --> PLAN
    PLAN -- "acted" --> DID
    LAT -.-> CRIT
    GRUC -. "detached" .-> CRIT
```

* **The state is all a seat keeps** (`reach` 1): the GRU's 1,024 numbers, then a summary of what its last decision did
  (`action_summary`: the action vectors of its actions averaged, and their count; `acted` fills it in).  A decision's
  latents, flat, and that summary go into the GRU (`_memory_in`, `_step`, in float32).
* **A decision's frame** is its own latent tokens and 4 memory tokens read off the new state (`MEMORY_TOKENS`, `_frame`),
  after `Size.layers` blocks of them attending to each other.  The plan, the anchor's reading and the heads are as they
  were, the plan at half the depth.
* **Not made**: the time transformer, the frames kept and their slots, the action tokens and their reader.  The layers of
  controllable units reading each other, whose cost grows with the square of the units, are `WIDE`'s alone (the user asked
  for them out, 2026-11-17: `GRU` has none; the code stays for `full-4`'s checkpoints).
* **Training** runs the GRU through a segment of 16 decisions from the state kept at its start, a step a decision, and
  everything else for all its decisions at once (`segments`): each decision reads what it read in play
  (`test_with_a_gru_training_reads_a_segment_as_play_did_from_the_state_kept_at_its_start`).  Nothing reaches back past the
  segment's start but through that kept state, which the weights of its time made.
* **Its critic** (`--critic-size memory`, `policy_model.MemoryCritic`; the user, 2026-11-17, as Pluto's): a small net for
  each of the reward's terms, each reading the frame's latents, flat, and the GRU's state after it, so what the actor has
  seen and nothing privileged.  03-training.md, "GRPO from branched games" ("Bootstrapped").
* **Speed**: about 870 decisions a second where the wide actor made 119, and 1,250 with bigger passes and minibatches
  (03-training.md, "Playing").

## An RDNN in the GRU's place (`Size.memory` "rdnn")

The user's, 2026-11-17, after arXiv 2608.01947 ("Divisive Normalization Shapes Low-Rank Slow Manifolds for Continuous
Working Memory"): a Recurrent Divisive Normalization Network where the GRU was, everything around it the same
(`policy_train.py --size rdnn`, `policy_latent.RDNN`, `RDNNCell`).  The paper shows it on two toy tasks of continuous
working memory (an angle integrated from its turns, or held through a delay; 1 to 3 numbers in, 64 to 256 units,
supervised, the gradient through all 256 to 512 steps), where it keeps a ring that a GRU breaks into points.  It has no
RL and nothing of this size, and that's what the run `rdnn-1` is for.

* **The step** (its appendix B), for units' rates R and their inhibitory pools G, with I the memory's input through a
  linear layer:

      R' = R + aR ((J relu(R) + I) / (eta + G) - R)
      G' = G + aG (W relu(R) - G)

  J, W (units x units) and eta are positive, the softplus of what's learnt; aR and aG are each unit's own, a sigmoid.
  A pool that starts at 0 never goes below it, so nothing is divided by less than eta.
* **The state** a seat keeps is the rates and then the pools: 2,048 numbers for `RDNN`'s 1,024 units (`Size.gru` is the
  state's width, so twice the units).  What the rest of the actor is told of it is read off the rates, as the paper
  reads its output; the critics read the whole state.  2.6M parameters where the GRU had 4.7M.
* **Told as a vector** (`Size.memory_as` "vector", `RDNN`'s; the user, 2026-11-17: "pass in memory input as a vector"):
  one vector of the actor's width off the rates (`mem_out`), added to what reads the frame: each controllable unit as it
  reads it for its key and again as an action's anchor (`context`, `reading`), and each of the plan's slots (`plan`).
  The frame is then its 64 latent tokens alone, which stay tokens: each reader picks among them.  As 4 tokens among the
  frame's 68 (`GRU`, "tokens") the memory is what a reader's attention gives it, a sixteenth at first; how much the GRU
  run's readers came to give it wasn't measured.  In the code the vector rides as a last row of what `_frame` and
  `segments` return, which `context` takes off, so nothing that calls them changed.  It isn't LayerNormed: at first it's
  about a tenth in each place, the size of the plan's own slots.
* **At first**, as the paper has it: each weight of J and W about 1 / units (the softplus of -ln(units) and a spread of
  1 / sqrt(units)), so a unit's drive is the units' mean rate and what it takes in, over eta and that mean; the rates a
  tenth of the way a decision (a sigmoid of -2.2).  In numpy at 1,024 units with random frames the state stays where it
  starts over 1,700 decisions (rates' spread 0.14, pools 0.08 to 0.13), and of one decision's difference it keeps 9%
  after 16 decisions and 1.2% after 64, where a GRU at PyTorch's first weights keeps 0.03% and nothing.
* **Mine, where the paper is silent or doesn't fit**:
  * relu and not tanh, with which a pool could go negative and the division blow up;
  * no noise on the state (the paper trains with 0.1): play and training must step a state alike here;
  * eta 1 at first, and a spread of 1 in the rates before their sigmoid (a twentieth of units keep a fiftieth of the way
    a decision, another twentieth a third): the paper gives neither;
  * an Adam of their own for what's learnt of J and W (`RDNNCell.recurrent`, `Learner.recurrent`), with eps 1e-8 and
    the schedule's rate, or `--recurrent-lr` (the user, 2026-11-17: 5e-3, the paper's best, in `rdnn-1` from iteration
    146), whatever the KL guard leaves the rest of the actor.  The gradient on them is the weight's
    times about 1 / units, far under the policy's Adam eps of 1e-5: in `rdnn-1`'s first smoke run, in the actor's own
    Adam at 2e-3 and scaled up by sqrt(units), a step moved J by 0.013% of its mean and after three iterations by
    nothing.  With a small eps a step moves a weight by about the rate, as a share of the weight.  `memory_step` in
    the trainer's metrics is how far a step moved J, as a share of its mean.
* **Training** cuts the gradient at a segment's start.  That was 16 decisions, as with the GRU; from iteration 146 of
  `rdnn-1` it's a whole stretch (`--segment 128`; the user, 2026-11-17, after the paper's task showed the RDNN learning
  nothing with the gradient cut at 16).  A segment ends with its stretch, 60 to 180 s of a game, so that's 20 to 60
  decisions and the most `--segment` can give: for more, the memory's inputs would have to be kept from the game's
  start and its steps run again from there (not built).
* **The loop over a segment's steps** (`RDNNCell.run`, `LatentActor._steps`) makes what doesn't change from step to
  step once (the weights, J over W as one matrix, the rates, the input layer for every step), and with
  `--compile-memory` goes through torch.compile (`policy_latent.compile_memory`; the user: "I'd keep the loop compiled",
  "Consider using dynamic shapes").  A loop in Python is unrolled when it's compiled, so its length can't be a shape
  left open: what's compiled is 16 steps (`RDNN_CHUNK`), called as often as a segment needs, the last chunk filled out
  with steps that take in nothing and are dropped; the number of segments is open.  Forward and back at 1,024 units
  (`results/bench_memory_loop.py`): 5 segments of 60 took 109 ms a step at a time, 72 as `run` and 24 compiled; 18 of
  16 took 34, 25 and 7.  The states are the same and the gradients within 5e-7 of their largest.  It compiles three
  times as the learner starts, 30 to 60 s each: a first chunk, a later one (whose state carries a gradient), and one
  segment alone.  The collectors don't use it: they step the memory once a decision inside their graphs.
* **Tests**: `test_an_rdnn_steps_as_the_paper_has_it_and_never_divides_by_less_than_its_semi_saturation`,
  `test_a_memory_told_as_a_vector_is_no_token_of_the_frame_and_reaches_the_units_and_the_plan`, the GRU's
  play-against-training test run for all three (GRU, RDNN, RDNN told as a vector), and the trainer's for `--size rdnn`.
  The first checks `run` against its steps, whole and a chunk at a time.  `results/rdnn_check.py` trains it and a GRU
  on the paper's first task with the gradient cut at 16.

## Open questions (the user's)

* **Memory**: full attention over all past frames, through summary tokens, in place of the 9 s window.
* **Action tokens**: one token per action between frames too, in place of the 4 summary tokens.  Within a decision, the actions
  are such tokens already (the decoder).
