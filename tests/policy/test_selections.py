"""Orders as selections and commands (zkbot.policy.selections): whom a command selects, what its orders are, and that orders
given unit by unit, as the scripts give theirs, come back the same from the commands that say them."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from zkbot.policy import actions, featurize as fz, groups as gr, rawframe as rf, selections as sel

from conftest import ENEMY, MEMORY, make_frame, make_units

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim" / "py"), str(ROOT / "tools")]

EVERY = tuple(range(1, rf.KIND_COUNT))
BUSY = rf.IN_LOS | rf.TYPE_KNOWN
IDLE = BUSY | rf.IDLE


def read(units, small_map, table):
    t = fz.featurize(make_frame(units), small_map, table, enabled_kinds=EVERY, economy=True)
    return t, *actions.world_positions(t, small_map)


def role_bit(table, name):
    return 1 << int(gr.roles()[1][table.names.index(name)])


def told(orders):
    return {o.uid: (o.kind, o.target) for o in orders}


def test_a_command_selects_its_roles_in_its_circle_or_the_nearest_and_whoever_obeyed_is_in_no_later_one(small_map, table):
    units = make_units(*({"uid": u, "x": x, "z": 500.0, "flags": IDLE} for u, x in ((1, 410.0), (2, 460.0), (3, 520.0), (4, 900.0))),
                       {"uid": 5, "name": "cloakriot", "x": 430.0, "z": 500.0, "flags": IDLE})
    t, x, z = read(units, small_map, table)
    w, h = small_map.map_w, small_map.map_h
    raid = role_bit(table, "cloakraid")

    def picked(radius, portion=0, cx=460.0, cz=500.0, roles=raid):
        c = sel.command(roles, cx / w, cz / h, radius, rf.FIGHT, 0.5, 0.5, portion=portion)
        return [int(t.uids[i]) for i in sel.selected(c, sel.unit_roles(t), np.ones(t.controllable, bool), x, z, w, h)]

    within_135 = sel.RADII.index(135.0)
    assert picked(within_135) == [2, 1, 3] and picked(within_135, portion=1) == [2, 1] and picked(within_135, portion=2) == [2]
    assert picked(0) == [2] and picked(sel.RADII.index(60.0), cx=2000.0, cz=2000.0) == [4]      # none in the circle: the nearest
    assert set(picked(within_135, roles=raid | role_bit(table, "cloakriot"))) == {1, 2, 3, 5}
    # The three by the middle are told the first thing, and the wide circle after it takes the one that's left.
    everyone = sel.RADII.index(2250.0)
    orders = sel.decode(np.array([sel.command(raid, 460.0 / w, 500.0 / h, within_135, rf.FIGHT, 0.25, 0.75),
                                  sel.command(raid, 460.0 / w, 500.0 / h, everyone, rf.MOVE, 0.75, 0.25)]), t, small_map, x, z)
    assert {o.uid: (o.kind, round(o.x), round(o.z)) for o in orders} == {
        **{u: (rf.FIGHT, round(0.25 * w), round(0.75 * h)) for u in (1, 2, 3)}, 4: (rf.MOVE, round(0.75 * w), round(0.25 * h))}


def test_the_library_decodes_the_rarer_kinds_as_the_python_does(small_map, table):
    """libzkpolicy's zkp_decode_commands against selections.decode where the golden frames have no such unit
    (test_zkpolicy.py has the rest): a reclaim by its features, a load by one transport, a throw as far as it reaches, a
    beacon placed and waited at, a jump."""
    import zkpolicy as zp
    from policy_parity import orders_diff, random_commands

    ready = BUSY | rf.JUMP_READY
    (ax, az), (bx, bz) = (tuple(float(v) for v in small_map.centre[i]) for i in (0, 1))
    units = make_units(*({"uid": u, "x": ax + 40.0 * u, "z": az + 60.0, "flags": IDLE} for u in range(1, 4)),
                       *({"uid": 10 + u, "name": "cloakcon", "x": ax - 50.0 * u, "z": az, "flags": IDLE} for u in range(2)),
                       {"uid": 20, "name": "gunshiptrans", "x": ax, "z": az - 80.0}, {"uid": 21, "name": "gunshiptrans", "x": ax + 300.0, "z": az},
                       {"uid": 30, "name": "amphlaunch", "x": ax + 60.0, "z": az, "flags": ready},
                       {"uid": 31, "name": "amphtele", "x": ax - 20.0, "z": az + 30.0}, {"uid": 32, "name": "tele_beacon", "x": bx, "z": bz},
                       {"uid": 33, "name": "jumpraid", "x": ax + 20.0, "z": az - 30.0, "flags": ready},
                       {"uid": 40, "name": "cloakriot", "side": ENEMY, "x": bx + 100.0, "z": bz}, {"uid": 41, "side": ENEMY, "x": ax + 500.0, "z": az})
    features = np.zeros(3, rf.FEATURE_DTYPE)
    features["feat_id"], features["x"], features["z"], features["metal"] = (7, 8, 9), (ax - 150.0, ax + 400.0, bx), (az + 100.0, az, bz + 90.0), 50.0
    frame = make_frame(units)
    frame.features = features
    t = fz.featurize(frame, small_map, table, enabled_kinds=EVERY, economy=True)
    x, z = actions.world_positions(t, small_map)
    rng, kinds = np.random.default_rng(11), set()
    for _ in range(150):
        commands = random_commands(t, small_map, x, z, rng)
        want = sel.decode(commands, t, small_map, x, z)
        assert orders_diff(want, zp.decode_commands(commands, t, small_map, x, z, table)) is None
        kinds |= {o.kind for o in want}
    assert {rf.RECLAIM, rf.LOAD, rf.THROW, rf.PLACE_BEACON, rf.WAIT_AT_BEACON, rf.JUMP, rf.BUILD, rf.ASSIST} <= kinds, sorted(kinds)


def test_orders_given_unit_by_unit_come_back_from_the_commands_that_say_them(small_map, table):
    solar, raider = table.names.index("energysolar"), table.names.index("cloakraid")
    units = make_units(*({"uid": u, "x": x, "z": 500.0, "flags": IDLE} for u, x in ((1, 500.0), (2, 480.0), (3, 520.0))),
                       {"uid": 4, "name": "cloakcon", "x": 700.0, "z": 700.0, "flags": IDLE},
                       {"uid": 5, "name": "factorycloak", "x": 800.0, "z": 800.0},
                       # on a build, past its commit lock: nothing is legal for it but the cancel
                       {"uid": 6, "name": "cloakcon", "x": 900.0, "z": 900.0, "ord_kind": rf.BUILD, "ord_age": 900.0, "flags": BUSY},
                       {"uid": 7, "name": "cloakriot", "side": ENEMY, "x": 1500.0, "z": 500.0},
                       {"uid": 8, "name": "staticmex", "side": MEMORY, "x": 2200.0, "z": 2200.0},
                       {"uid": 9, "x": 2000.0, "z": 2000.0, "flags": IDLE}, {"uid": 10, "x": 1400.0, "z": 400.0, "flags": IDLE})
    t, x, z = read(units, small_map, table)
    raid = role_bit(table, "cloakraid")
    orders = [actions.Order(1, rf.FIGHT, 1400.0, 600.0, radius=135.0, roles=raid),          # a group order: units 1 to 3
              actions.Order(4, rf.BUILD, 720.0, 760.0, solar), actions.Order(5, rf.PRODUCE, target=raider),
              actions.Order(6, rf.MOVE, 100.0, 100.0),                                       # to one on a build: the cancel
              actions.Order(9, rf.ATTACK_UNIT, 2200.0, 2200.0, 8),                           # at what it only remembers: a fight there
              actions.Order(10, rf.ATTACK_UNIT, 1500.0, 500.0, 7)]
    missed: dict = {}
    commands, lost = sel.encode(orders, t, small_map, x, z, missed=missed)
    assert lost == 0 and len(commands) == 6
    # The three told one thing are one command, the first: their role, in the smallest circle that holds them.
    assert commands[0]["kind"] == rf.FIGHT and commands[0]["roles"] == raid and sel.RADII[commands[0]["radius"]] == 60.0
    assert (commands["radius"][1:] == 0).all()
    back = sel.decode(commands, t, small_map, x, z)
    assert told(back) == {1: (rf.FIGHT, -1), 2: (rf.FIGHT, -1), 3: (rf.FIGHT, -1), 4: (rf.BUILD, solar), 5: (rf.PRODUCE, raider),
                          6: (rf.CANCEL_BUILD, -1), 9: (rf.FIGHT, -1), 10: (rf.ATTACK_UNIT, 7)}
    places = {o.uid: (o.x, o.z) for o in back}
    assert places[1] == pytest.approx((1400.0, 600.0), abs=0.01) and places[4] == pytest.approx((720.0, 760.0), abs=0.01)
    assert places[9] == pytest.approx((2200.0, 2200.0), abs=0.01)


def test_a_decision_comes_back_from_its_arrays_and_any_code_says_commands_that_can_be_carried_out(small_map, table):
    torch = pytest.importorskip("torch")
    import order_vae as ov

    units = make_units(*({"uid": u, "x": 400.0 + 40.0 * u, "z": 500.0, "flags": IDLE} for u in range(1, 5)),
                       {"uid": 5, "name": "cloakcon", "x": 700.0, "z": 700.0, "flags": IDLE}, {"uid": 6, "name": "factorycloak", "x": 800.0, "z": 800.0})
    t, x, z = read(units, small_map, table)
    raid, builder = role_bit(table, "cloakraid"), role_bit(table, "cloakcon")
    commands = np.array([sel.command(raid, 0.2, 0.2, 4, rf.FIGHT, 0.6, 0.3, portion=1),
                         sel.command(builder, 0.27, 0.27, 0, rf.BUILD, 0.3, 0.3, type=7),
                         sel.command(builder, 0.3, 0.3, 2, rf.RECLAIM, 0.4, 0.4, extent=3)])
    back = ov.to_commands(ov.to_arrays([commands, commands[:0]]))
    assert (back[0] == commands).all() and len(back[1]) == 0
    torch.manual_seed(0)
    model = ov.OrderVAE(ov.sizes(ov.Config(d=32, layers=1)))
    with torch.no_grad():
        arrays = {k: v.numpy() for k, v in model.say(torch.randn(64, model.cfg.codes, model.cfg.latent) * 3.0).items()}
    said = ov.to_commands(arrays)
    # Whatever an untrained decoder says, some type of each role it selects can be told the kind, and can make the type.
    assert any(len(c) for c in said) and ov.plausible(arrays).all()
    # And commands made up, of every kind the sim obeys and plausible too: each is carried out or ignored, and the same back
    # from its arrays.
    made = ov.synthetic(np.random.default_rng(3), 300, model.cfg.slots, model.cfg.roles, model.cfg.types)
    made_up = ov.to_commands(made)
    assert set(np.concatenate(made_up)["kind"].tolist()) == set(sel.KINDS) and ov.plausible(made)[made["present"]].all()
    fight = made["present"] & (made["kind"] == ov.KIND_INDEX[rf.FIGHT])
    assert not (made["roles"][fight] & role_bit(table, "staticmex")).any()      # nobody tells an extractor to fight
    again = ov.to_arrays(made_up)
    assert all((again[k] == made[k]).all() for k in made)
    kinds_told = set()
    for c in said + made_up:
        assert np.isin(c["kind"], sel.KINDS).all() and ((c["cx"] >= 0.0) & (c["cx"] <= 1.0)).all()
        kinds_told |= {o.kind for o in sel.decode(c, t, small_map, x, z)}
    assert {rf.MOVE, rf.FIGHT, rf.BUILD, rf.PRODUCE} <= kinds_told
