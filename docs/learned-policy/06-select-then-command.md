# Select, then command: orders as a player gives them

The user's, 2026-11-19: "Starcraft's Pluto includes player-like unit selection before issuing group commands.  I think
this is feasible and likely to be more efficient.  As an example (in token form): `<latent tokens> <SELECT_CIRCLE
CENTRE=(x,z) RADIUS=r TYPE=combat> <ATTACK_MOVE AT=(x,z)> <END_FRAME_ORDERS>`."  Then:

* "I think all the single-unit orders can be made to work with multiple units."
* "Proceed with the BUILD change.  Additionally, make RECLAIM an area reclaim order (default to a small radius to ensure
  compatibility).  The circle select could have a 'portion' parameter, to select say QUARTER, HALF or FULL."
* "Rather than selecting or positional orders based on region, consider a spatial encoding or finer subregions."

**Built**: the commands themselves (`bots/zkbot/policy/selections.py`), the two changes to the sim, and the check that
the scripts play as well when their orders are said this way ("The check", below).  What chooses commands is the actor
of 05-order-autoencoder.md, by way of a code: the user's word was to go straight to the autoencoder of them, so no actor
has selection heads of its own.

The form, the selection before the command, every order working on several units, the portion, the area reclaim and
places as coordinates are the user's.  The roles as the type, each rule of a selection and how the scripts' orders are
put into commands are mine.

## A command

A side's orders at a decision are a few commands, in turn, 16 at most (`SLOTS`).  Each is one selection and one thing
told to it (`COMMAND_DTYPE`):

| | Field | What it is |
|---|---|---|
| **Select** | `roles` | which roles, any set of the type table's 19 (`groups.roles`): raid, riot, skirm, assault, arty, aa, scout, heavy, bomb, snipe, antiheavy, support, other, builder, factory and the rest |
| | `cx`, `cz` | the circle's middle, each a fraction of the map |
| | `radius` | one of 11 (`RADII`): the one unit nearest the middle, or a circle of 60 to 2,250 elmos |
| | `portion` | of what the circle holds: all, the nearest half or the nearest quarter (`PORTIONS`), rounded up |
| **Command** | `kind` | one of 21 (`KINDS`).  The scripts' eleven and the cancel: MOVE, FIGHT, ATTACK_UNIT, MANUAL_FIRE, BUILD, ASSIST, CANCEL_BUILD, RECLAIM, PRODUCE, LOAD, UNLOAD, DROP.  And the nine no script gives: JUMP, THROW, CLOAK_ON, CLOAK_OFF, PUSH, PULL, PLACE_BEACON, WAIT_AT_BEACON, AREA_MEX |
| | `x`, `z` | its place, each a fraction of the map |
| | `type` | what to BUILD or PRODUCE: its place among the 187 producible types |
| | `extent` | a RECLAIM's or an AREA_MEX's area: one of the same 11 radii, the first the sim's least (32 elmos) |

**Nothing in a command points into a frame.**  There's no unit token, no region, no site.  Who a place means is found
when the command is carried out, so a decision's commands can be written, read and autoencoded with no frame beside
them.

* **The kinds are every kind the sim obeys from the policy** (`policy_sim_env.KINDS`, but keep).  Nine of them no
  script gives; the autoencoder learns them from commands made up (the user, 2026-11-19: "The training data for the
  autoencoder should also include randomly/synthetically generated actions to ensure coverage of cases/orders not seen
  in the script"; 05, "Its data").
* **What's plausible** (`selections.able`; the user, 2026-11-19: "commands should be plausible/valid").  By the type
  table alone, with no frame: a role can be told a kind if some type of the role ever can, by what the frame's kind
  mask asks of a type (a FIGHT: 13 of the 19 roles; a BUILD, an ASSIST, a RECLAIM, an area mex: the builders; a
  PRODUCE: the factories and the static builders; a LOAD: the two roles with a transport; a THROW: the one with the
  thrower), and a role can make a type if one of its types builds or produces it.  The autoencoder's made-up commands
  are drawn from these, and its decoder says nothing else (05).  `decode` itself carries out whatever it's given: a
  command to roles that can't obey just finds nobody.
* **A selection is never empty** unless the side has no unit of those roles: with none in the circle it's the one
  nearest the middle.  That, with a radius of "the nearest one", is how one constructor or one factory is told something.
* **A unit that obeyed is in no later selection** of the decision.  So the order of the commands means something: these
  raiders first, then every fighter that's left.

## Carrying one out (`selections.decode`)

Each selected unit obeys if the command is legal for it by the frame's own masks (`kind_mask`, the pointers'), and
otherwise keeps what it's doing, as a player's units ignore a command they haven't got.  So the commit lock and "a build
sticks" hold with no rule of their own here: a unit on a build obeys only the cancel.

| Kind | What a selection of several does |
|---|---|
| MOVE, FIGHT | each goes to the place.  Where there's no way, as near as a straight line gets (the sim's, and what the scripts' own orders rely on) |
| ATTACK_UNIT, MANUAL_FIRE | the target is the one nearest the place that any of them may aim at; each that may, does |
| ASSIST | the same, of what can be helped: a frame, or a factory |
| BUILD | each that builds the type is sent to the one point, and they build one thing ("The sim", below) |
| PRODUCE | each factory that makes the type is set to it |
| RECLAIM | each takes what lies in the area, the nearest to itself first; no order where no feature in sight lies in it |
| CANCEL_BUILD | each that's on a build stops |
| LOAD | the thing nearest the place that can be lifted, by the one transport nearest it |
| UNLOAD, DROP | each loaded transport, where its load can stand |
| JUMP, THROW | each whose own reach lands in the place's region; a throw past a unit's reach goes as far as it can the same way |
| PLACE_BEACON | each that can walk to the place's region |
| WAIT_AT_BEACON | the beacon of ours nearest the place that any of them can walk to; each that can, goes |
| CLOAK_ON, CLOAK_OFF, PUSH, PULL | each that has the switch; the place means nothing |
| AREA_MEX | each that builds extractors takes the free spots in the area, the nearest to itself first (`SimEnv._area_builds`) |

## The sim (2026-11-19; libzksim ABI 32)

* **Several told to build one thing build one** (`World::run_builder`, `frame_to_join`): a BUILD that arrives where an
  unfinished frame of its type and allyteam stands, within the type's footprint of the order's place, works on that
  frame.  Before, each constructor placed its own.  And BUILDs of one type told to one point in a decision are given one
  place (`SimEnv._placed`), an extractor's spot among them.
* **A RECLAIM is an area's** (`World::reclaim_step`, `wreck_in_area`; capi.h `ZK_ORDER_RECLAIM`): the unit takes what
  lies in it, the nearest to itself first, and the order ends when nothing's left there.  Named by a feature's id, as
  the scripts and the old actor name theirs, the area is 32 elmos round that feature (`ZK_RECLAIM_RADIUS`, mine): the
  one feature and what lies on it.  Named by a place, `target` is minus the radius, and with nothing in it the order
  isn't taken.

## The scripts' orders as commands (`selections.encode`)

The scripts order unit by unit, and group what they can.  `encode` finds commands that give each unit what the orders
gave it:

1. Each unit's order is put in a command's terms.  Who it's aimed at becomes the place it's at.  An attack on something
   the side has no token of (a building it only remembers) becomes a FIGHT at its place.  An order to a unit on a build
   becomes the cancel of that build, and the order itself is left out.
2. Units told the same thing are one command where a selection takes them and nobody else: their roles, in the smallest
   circle round their middle that holds them, whole, or its nearest half or quarter.
3. Where none does, they're halved across the way they're spread and each half tried, down to a unit on its own.
4. A circle may also take in a unit whose last order was this very MOVE or FIGHT and that's told nothing else now:
   told again, it goes on as it was.  The army script's own group orders do the same (army.cpp, `cover`).
5. The most units first, so that what's cut at 16 commands is the least.

## The check: the scripts through commands, against the scripts

`results/checks/language_check.py`: team 0 of 128 1v1s is a seat the scripts only advise, and what's carried out is
their advice said as commands and read back (`SimEnv.through`); team 1 is the scripts as they are.  Set beside the same
road with the advice left as it is.

| | The advice as it is | As commands and back |
|---|---|---|
| team 0 won, lost, drew | 67, 57, 4 | 62, 64, 2 |
| its games' result (a draw half) | 0.539 ± 0.044 | 0.492 ± 0.044 |

Level within what 128 games can show (-0.047 ± 0.062); not shown the same.  In the games played in commands:

* **Of the units the scripts told something at a look, 71% were told the same by the commands** (1.29M of 1.82M).  What
  the rest lost:
  * a quarter of the commands were past the 16 a decision (200,061 of 810,856), all in the 3% of looks with 20 or more:
    late in long games, a big army spread out among others of its roles that it leaves alone;
  * 10% of the scripts' orders couldn't be said: 4.6% were to units that aren't among those the side's frame gives it
    to command, and 5.4% where a unit's own command picked another or wasn't obeyed (mostly FIGHT and ASSIST).  Neither
    is looked into.
* **A look is few commands**: one in 28% of looks, two in 21%, three in 14%, eight or fewer in 90%.
* **A third of the commands are circles**: 67% select one unit, 24% a whole circle, 5% its nearest half and 4% its
  nearest quarter.  The units told one thing with others took 3.0 units a command for a FIGHT and 2.8 for an ASSIST.
* **209,351 units were told again what they were on already**, swept in by a circle (rule 4 above).
* By kind, of the commands: FIGHT 60%, ASSIST 14%, BUILD 9%, PRODUCE 7%, MOVE 5%, RECLAIM 4%, CANCEL_BUILD 0.6%, the
  transports' three 0.8%, ATTACK_UNIT 0.3%.

On the way there the check caught two faults of mine, both fixed before these numbers: every MOVE and FIGHT read back at
the map's corner (team 0 then won 9 of 128), and a wide circle claiming units that had an order of their own that look.

## What may go wrong

* **A circle is blunt.**  It can't take these six raiders and leave those four of the same role beside them, but for a
  half or a quarter by nearness.
* **A portion is seldom the scripts' meaning.**  They never choose one; 9% of their commands have one because that's
  what took exactly the units they told.
* **An untrained chooser moves its army about at random**, which the old actor's one unit a decision didn't.
* **Places are frame-blind.**  A code says a place whatever is there, and who's nearest takes it.  That's by design,
  and it's also the thing the policy has to learn through a frozen decoder.

## For the user to decide

1. **The roles as the type.**  The user wrote `TYPE=combat`; here it's any set of the table's roles, which says "combat"
   as every fighting role at once and says the scripts' own groups exactly.  A single class would be simpler to choose.
2. **Whether "idle" should narrow a selection**, as a player's select-idle does.  Not built.
3. **The least reclaim area**, 32 elmos, which is also an AREA_MEX's least: too small for one to find a spot.
4. **Attack on ground** isn't a command: the sim obeys none from the policy.
