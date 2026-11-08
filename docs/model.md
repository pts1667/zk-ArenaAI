# The policy model

How the learned policy works as of 2026-11-08: what it reads, how it reads it, and how it chooses orders.  The code is
`tools/policy_latent.py` (the actor), `tools/frame_vae.py` and `tools/frame_model.py` (the frozen frame encoder),
`tools/policy_model.py` (batches, sampling, the critic) and `bots/zkbot/policy/featurize.py` (raw frame to tensors).  Results and
the history behind each choice are in `docs/learned-policy/03-training.md` section 11.

## 1. The whole of it

A decision is made every 15 game frames (2 a second).  The seat's raw frame becomes tokens; a frozen encoder squeezes them into 24
latent tokens; the actor reads those latents together with the latents and actions of the frames before, over a sliding window in
time; each of the seat's own units then reads the result and five heads choose its order.  What was chosen is summarised into 4
action tokens, which later frames read.

```mermaid
flowchart LR
    RF["raw frame<br/>(rawframe.Frame:<br/>units, sites, features, globals)"]
    FZ["featurizer<br/>featurize.py"]
    TOK["tokens<br/>units · regions · sites ·<br/>features · global"]
    ENC["frame encoder<br/>(frozen, 0.92M)"]
    LAT["24 latent tokens<br/>x 32 numbers"]
    TIME["time transformer<br/>2 layers, 8 s window"]
    HIST[("kept history:<br/>32 frames of latents<br/>+ action tokens")]
    READ["each own unit reads<br/>its frame's latents"]
    HEADS["heads: kind · region ·<br/>enemy · fine · produce"]
    ORD["orders<br/>actions.decode"]
    ACT["4 action tokens"]

    RF --> FZ --> TOK --> ENC --> LAT --> TIME
    HIST --> TIME
    TIME --> READ --> HEADS --> ORD
    HEADS --> ACT --> HIST
    LAT --> HIST
    TOK -. "raw unit features<br/>(queries and pointer keys)" .-> READ
```

| | |
|---|---|
| Width `D` | 128, 4 heads |
| Latent tokens a frame | 24, of 32 numbers each, LayerNormed without weights |
| Action tokens a frame | 4 (`ACTION_TOKENS`) |
| Time layers | 2 (`LAYERS`) |
| Window | 17 frames: 8 s at 2 decisions a second, the frame itself included |
| Reach | 32 frames: each layer reaches 16 further back, so a decision depends on the 32 before it |
| Parameters | 1,082,199 trained (the actor), 915,920 frozen (the encoder) |

## 2. What it reads: the featurizer

`featurize.featurize(frame, bundle, table)` turns a raw frame into fixed-normalisation numpy tensors.  Tokens come in a fixed order:

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

A Perceiver-style reader, trained on its own as an autoencoder (`frame_vae.py`, run `encoder-questions`), then frozen: its decoder
is thrown away and the actor keeps `LatentEncoder` (the encoder, `to_mean` and a weightless LayerNorm).

```mermaid
flowchart TB
    subgraph tokens ["Tokens of one frame (embedded to 128)"]
        U["units: features + sine codes of place<br/>+ name, alliance, player embeddings<br/>→ 2-layer MLP + token-kind embedding"]
        R["regions: features + sine codes<br/>+ region pool: their units' sum (32)"]
        S["sites"]
        F["features (wrecks)"]
        G["global"]
    end
    Q["24 learned queries"]
    C1["cross-attention: queries read all tokens"]
    S1["self-attention among the 24"]
    C2["cross-attention"]
    S2["self-attention"]
    LN["LayerNorm"]
    M["to_mean: 128 → 32"]
    N["LayerNorm, no weights"]
    OUT["24 x 32 latent tokens"]

    Q --> C1
    tokens --> C1 --> S1 --> C2 --> S2 --> LN --> M --> N --> OUT
    tokens --> C2
```

* Cost grows with tokens × 24, not tokens squared: there's no self-attention over the tokens themselves (`token_layers` 0).
* Region tokens carry what their units sum to (`region_pool`, squashed by asinh), since attention averages and can't count.
* What taught the latent: a decoder asked each region what's in it now (value by side and class), the globals, and questions about
  a few units and wrecks per frame ("where is a unit like this?", "what's at this exact place?").  Terrain isn't asked: the actor
  knows the map.  See the docstring of `frame_vae.py` and `03-training.md` 11.5.

## 4. Frames in time: the actor's sequence

The actor lays frames out in order, as GPT-2 lays out text, each frame's 24 latent tokens followed by its 4 action tokens:

```mermaid
flowchart LR
    subgraph f1 ["frame t-2"]
        L1["24 latent"] --> A1["4 action"]
    end
    subgraph f2 ["frame t-1"]
        L2["24 latent"] --> A2["4 action"]
    end
    subgraph f3 ["frame t (deciding now)"]
        L3["24 latent"] --> A3["4 action<br/>(empty until chosen)"]
    end
    f1 --> f2 --> f3
```

* A latent token's input is `latent_in(latents) + slots[i]` (which of the 24 it is).  An action token's is the frame's action
  summary + `act_slots[j]`.
* Frames from before the game began are learned embeddings (`before`, `act_before`), so every decision sees a full reach.

### Who attends to whom

Each `TimeBlock` is a pre-LayerNorm transformer block (attention, then a 4x MLP) with a learned bias on each head by how many
frames back a key is (`age`, 17 values per head), and a learned bias per head for keys that are action tokens (`act_key`).  The
mask is by frame, not by token:

| Query ↓ / key → | latents, earlier frame in window | actions, earlier frame in window | latents, same frame | actions, same frame | any later frame, or out of window |
|---|---|---|---|---|---|
| latent token | yes | yes | yes (all 24, both ways) | **no** | no |
| action token | yes | yes | yes | yes (all 4, both ways) | no |

So within a frame the latents attend to each other in both directions.  They don't see that frame's actions, which are what's
being chosen from them.  The next frame's tokens do see them.  The 4 action tokens of a frame attend to each other in both
directions too.

```mermaid
flowchart LR
    subgraph past ["frames t-16 … t-1 (in the window)"]
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

With a window of 17 frames and 2 layers, a decision depends on exactly the 32 frames before it (`reach`).  In layer 1, frame t
reads back to t-16 and frame t-16 to t-32.  In layer 2, frame t reads what t-16 took in.  Play keeps those 32 frames (section 7).

The attention is `F.scaled_dot_product_attention` with a dense float mask (age bias + action bias, `-inf` where unseen) of
heads × queries × keys.

## 5. From the sequence to orders

### Each unit reads its frame

After the last layer and a final LayerNorm, the newest frame's 24 latent tokens (`now`, B × 24 × 128) are what the frame holds.
Every controllable unit then reads them with a cross-attention block (`read_unit`), using as its query an embedding of its **own raw
features** (`hu`: the same kind of embedding as the encoder's, but the actor's own weights).  So each unit knows exactly where and
what it is, whatever the latents blurred, and asks the frame what matters to it.

```mermaid
flowchart LR
    X["unit's raw features<br/>+ sine codes of place<br/>+ name / ally / player"] --> HU["hu (128)<br/>MLP + token kind"]
    HU --> RU["read_unit<br/>cross-attention<br/>over the 24 latents"]
    NOW["now: 24 x 128"] --> RU
    RU --> ME["me (128): the unit's reading"]
    HU --> KU["unit key: key_unit(hu) ‖ sine code (152)"]
    RX["region features + centre code"] --> KR["region key: key_region ‖ sine code (152)"]
```

Every unit token (own or enemy) also gets a **key**: `key_unit(hu)` followed by the 24-number sine code of its place.  Regions get
keys the same way.  Pointers score against these keys, built from raw features, so a target's exact place is there to choose.

### The heads

```mermaid
flowchart TB
    ME["me (each controllable unit)"]
    K["kind head<br/>linear → 21 kinds<br/>(illegal masked)"]
    RG["region pointer<br/>(q_region(me) ‖ gain · own code) · region keys<br/>− softplus(route_scale) × route cost"]
    EN["enemy pointers, one per kind:<br/>attack unit · attack ground · manual fire<br/>(q_enemy(me) ‖ gain · own code) · enemy keys"]
    FI["fine head<br/>MLP(me ‖ target key) → 9 options"]
    PR["produce head<br/>q_type(me) · type keys of producible types"]

    ME --> K
    ME --> RG
    ME --> EN
    ME --> FI
    ME --> PR
```

* **Kind**: keep, move, fight, attack unit, attack ground, stop, manual fire, jump, cloak on/off, produce, throw, push, pull, place
  beacon, wait at beacon, ... (21 in all; the sim's games enable 14, `policy_sim_env.KINDS`).  The kind head's bias starts at keep
  (`KEEP_START`) and its weights near zero, so an untrained policy keeps.
* **Pointers with place-aware keys**: a pointer's query ends in the unit's own sine code, each band scaled by a gain the unit
  chooses (`gain_region`, `gain_enemy`).  Its dot product with a key's sine code is then a sum over bands of gain × cos(band ×
  difference in place): a learned function of distance and direction, which a dot product of raw coordinates can't give.
* **Region** also subtracts a learned multiple of the walking cost to each region (`route_bias`), so near regions come first.
* **Fine** (9 options): option 0 is the target itself; options 1-8 are the compass points around it.  For a region, these are at
  half its equivalent radius, with north towards the enemy; for attack ground, they are 200 elmos from the enemy target.  It reads
  the chosen target's key, so it's chosen after the target.
* **Produce**: a factory's or silo's choice among producible types, scored against embeddings of their stats.

### Sampling, in order

Every unit samples independently and at once, given the frame; the heads are conditioned on each other only through what's chosen
for that same unit:

```mermaid
sequenceDiagram
    participant H as heads (per unit)
    participant S as sample
    H->>S: kind logits (masked by kind_mask)
    S-->>H: kind
    H->>S: region logits, masked by what the kind may reach<br/>(ptr_region, ptr_jump for jump/throw, ptr_wait)
    S-->>H: region (used only by kinds with a region)
    H->>S: the kind's enemy logits (attack unit / ground / manual fire)
    S-->>H: enemy (used only by those kinds)
    H->>S: fine logits from me ‖ chosen target's key
    S-->>H: fine (used only by kinds with a fine option)
    H->>S: produce logits (masked by what it can make)
    S-->>H: type (used only by produce)
```

A unit's log-probability is the sum over the heads its kind uses.  Sampling is Gumbel-max.  `actions.decode` turns the choices into
orders for the sim or the engine.

## 6. Action tokens: what was done

What the units were told at a frame becomes that frame's 4 action tokens (`action_tokens`).  Only the policy's own choices go in:
the economy script's and the opening's orders show only in later frames, as each unit's current order.

```mermaid
flowchart LR
    subgraph each ["each unit told something other than keep"]
        E["kind embedding<br/>+ act_unit(hu)<br/>+ act_enemy(target key)<br/>+ act_region(region key)<br/>+ act_fine(option)<br/>+ act_type(type stats)"] --> G["GELU → act_out"]
    end
    NOTH["learned 'nothing' vector<br/>(always there)"]
    Q4["4 learned queries<br/>+ act_count(log(1 + units that acted))"]
    CR["act_read<br/>cross-attention"]
    OUT["4 action tokens (128 each)"]
    G --> CR
    NOTH --> CR
    Q4 --> CR --> OUT
```

However many units acted (0 to 256), the frame's actions are summarised into 4 tokens; a frame where everything kept still has 4
tokens, which read only "nothing" and the count.  They are a fixed-size, unordered summary, not one token per order.

## 7. Play and training

### In play: one decision at a time

A seat keeps the 32 frames before the present, newest first: each frame's raw latents (24 × 32) and its 4 action tokens, flat
(`record` numbers a frame).  `decide` runs the encoder on the new frame, lays out the 33 frames, and runs the sequence.  The first
layer runs over every token.  The last runs only the newest frame's latent tokens as queries (`newest_only`).  The units then read,
and the heads sample.  `acted` then writes the chosen actions' tokens into the newest frame's slot.

```mermaid
sequenceDiagram
    participant Env as sim / engine
    participant Seat as seat state (32 frames)
    participant Pol as actor
    Env->>Pol: featurized frame t
    Pol->>Pol: encoder → 24 latents
    Seat->>Pol: frames t-32 … t-1 (latents + action tokens)
    Pol->>Pol: sequence (last layer: frame t's latents only)
    Pol->>Pol: units read, heads sample
    Pol->>Env: orders
    Pol->>Seat: push frame t's latents, then its action tokens (acted)
```

### In training: a stretch in one pass

Training takes a stretch of decisions with the 32 frames before it and runs the whole sequence at once.  Because the mask is
causal by frame, every decision in the stretch sees exactly what it would have seen in play.

* **Cloning** (`policy_clone_train.py`): stretches of 48 decisions from recorded script games (`policy_clone_data.py`).  The
  script's orders become the labels (`actions.encode`), and the action tokens are built from those labels.  The loss is the
  negative log-probability of the script's choices per unit.  Keep is weighted down (`--keep-weight`, 0.1), and keep's logit gets
  −log w added back when playing (`keep_shift`).  By default only decisions where the script looked count (every 6th).
* **PPO** (`policy_train.py`): segments of 32 decisions from the sim, each started from the state kept at collection.  Each unit
  gets its own clipped ratio, averaged within a decision, with one advantage per decision.  The **critic** is a separate network,
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
    CT -- "--init-actor" --> PPO
```

## 8. Where it stands

* Cloning on `army-3` (11.9) was still learning after 3,000 steps.  An act's kind was right about a quarter of the time, with no
  measurable difference with or without action tokens.
* Nothing plays in the engine yet.  The bot (`bots/zkbot`) hosts the policy but sends no commands, until a numpy forward of the
  actor and encoder, and the opening and economy script, are there.
* Open questions, from the user:
  * full attention over all past frames through summary tokens, in place of the 8 s window;
  * FlexAttention, with one token per acting unit and causal attention along them in place of the 4 summary tokens.
