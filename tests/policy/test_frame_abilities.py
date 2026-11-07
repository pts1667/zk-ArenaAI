"""What frames should show of the units that do something out of the ordinary: tools/frame_world.py, tools/frame_data.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim" / "py"), str(ROOT / "tools"), str(ROOT / "bots")]

try:
    import frame_data as fd
    import frame_world as fw
    import map_gen as mg
    import policy_sim_env as se
    import zksim
    from policy_script import SENT
    from zkbot.policy import rawframe as rf
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)

DRY = mg.Rules(waters=(("none", 1.0),), sides=(8,), side_weights=(1.0,), square=1.0)


@pytest.fixture(scope="module")
def land(tmp_path_factory):
    out = tmp_path_factory.mktemp("maps")
    made = mg.make(1, DRY)
    mg.write_entry(made.map, "made", made.layouts, made.features, out, made.style)
    _, bundle, facts = se.load_pool(out)[0]
    return fw.Ground(bundle, facts), facts


@pytest.fixture(scope="module")
def roster():
    return fw.Roster(se.tables.table(), se.catalog())


@pytest.fixture(scope="module")
def data(roster):
    return fw.fight_data(roster.table, se.catalog())


def units(*rows, flags=0):
    """(uid, name, x, z)"""
    cat = se.catalog()
    out = np.zeros(len(rows), rf.UNIT_DTYPE)
    for i, (uid, name, x, z) in enumerate(rows):
        out[i]["uid"], out[i]["def_id"], out[i]["hp_frac"], out[i]["x"], out[i]["z"] = uid, cat.units[cat.index[name]]["def_id"], 1.0, x, z
    out["flags"] = flags
    return out


def script(data, land):
    ground, facts = land
    a, b = facts.layouts[0][0].point, facts.layouts[0][1].point
    return fw.FrameFight(data, a, b, ground=ground), a, b


NOBODY = np.zeros(0, rf.UNIT_DTYPE)


def test_what_the_sim_only_partly_models_can_be_on_the_board(roster):
    names = {roster.table.names[r] for r in roster.mobile}
    assert {"jumpsumo", "amphlaunch", "amphtele", "jumpblackhole", "veharty"} <= names
    built = {roster.table.names[r] for r in roster.structures}
    assert {"energysingu", "energyfusion", "staticarty", "staticmissilesilo", "turretimpulse"} <= built


def test_a_scenario_can_hold_the_big_things_and_no_superweapon(roster):
    table = roster.table
    most = fd.ScenarioRules().most_structure_cost
    for name in ("energysingu", "staticarty", "staticheavyarty", "turretheavy"):
        assert table.cost[table.names.index(name)] <= most
    for name in ("mahlazer", "raveparty", "staticnuke"):
        assert table.cost[table.names.index(name)] > most


def test_what_each_can_do_comes_from_the_catalog(data, roster):
    row = roster.table.names.index
    assert data.throw_radius[row("amphlaunch")] > 0.0 and data.throw_reach[row("amphlaunch")] > 0.0
    assert data.throw_radius[row("cloakraid")] == 0.0
    assert data.teleports[row("amphtele")] and not data.teleports[row("amphlaunch")]
    assert data.beacon[row("tele_beacon")] and not data.beacon[row("amphtele")]
    assert data.switches[row("turretimpulse")] and data.switches[row("jumpsumo")]
    assert not data.switches[row("cloakraid")]


def test_a_lobster_throws_who_is_around_it_along_the_way(data, land):
    s, a, b = script(data, land)
    own = units((1, "amphlaunch", *a), (2, "cloakassault", a[0] + 50.0, a[1]), flags=rf.JUMP_READY)
    s.state[:] = SENT
    out = s.orders(own, NOBODY, 0.0)
    assert out["kind"][out["unit"] == 1][0] == rf.MANUAL_FIRE
    assert out["target"][out["unit"] == 1][0] == -1
    assert np.hypot(out["x"][0] - a[0], out["z"][0] - a[1]) <= data.throw_reach[s.data.table.names.index("amphlaunch")]


def test_alone_or_reloading_it_walks(data, land):
    s, a, _ = script(data, land)
    s.state[:] = SENT
    alone = s.orders(units((1, "amphlaunch", *a), flags=rf.JUMP_READY), NOBODY, 0.0)
    assert alone["kind"][0] != rf.MANUAL_FIRE
    s, a, _ = script(data, land)
    s.state[:] = SENT
    reloading = s.orders(units((1, "amphlaunch", *a), (2, "cloakassault", a[0] + 50.0, a[1])), NOBODY, 0.0)
    assert reloading["kind"][0] != rf.MANUAL_FIRE


def test_what_pushes_and_pulls_changes_over_now_and_then_even_if_it_is_a_tower(data, land):
    s, a, _ = script(data, land)
    own = units((4, "turretimpulse", *a), (5, "turretimpulse", a[0] + 100.0, a[1]))
    assert not len(s.orders(own, NOBODY, 10.0))
    first = s.orders(own, NOBODY, 61.0)
    assert list(first["kind"]) == [zksim.ORDER_ON_OFF] * 2
    assert sorted(first["target"]) == [0, 1]
    assert not len(s.orders(own, NOBODY, 62.0))
    second = s.orders(own, NOBODY, 125.0)
    assert list(second["target"]) == list(1 - first["target"])


def test_a_djinn_leaves_a_beacon_then_goes_ahead_and_others_wait_to_come_over(data, land):
    s, a, b = script(data, land)
    s.state[:] = SENT
    djinn = (1, "amphtele", *a)
    pal = (2, "cloakassault", a[0] + 40.0, a[1])
    out = s.orders(units(djinn, pal), NOBODY, 0.0)
    placing = out[out["unit"] == 1][0]
    assert placing["kind"] == zksim.ORDER_PLACE_BEACON
    # It's let be while it does, and tries again only long after.
    assert not (s.orders(units(djinn, pal), NOBODY, 5.0)["unit"] == 1).any()
    beacon = (3, "tele_beacon", float(placing["x"]), float(placing["z"]))
    out = s.orders(units(djinn, pal, beacon), NOBODY, 20.0)
    going = out[out["unit"] == 1][0]
    assert going["kind"] == rf.MOVE
    mid = (a[0] + 0.5 * (b[0] - a[0]), a[1] + 0.5 * (b[1] - a[1]))
    assert np.hypot(going["x"] - mid[0], going["z"] - mid[1]) < 500.0
    # There, and standing: those about the beacon go to it.
    there = (1, "amphtele", float(going["x"]), float(going["z"]))
    own = units(there, pal, beacon)
    own["flags"][0] = rf.IDLE
    out = s.orders(own, NOBODY, 40.0)
    assert not (out["unit"] == 1).any()
    waiting = out[out["unit"] == 2][0]
    assert waiting["kind"] == zksim.ORDER_WAIT_AT_BEACON and waiting["target"] == 3


def test_a_silo_makes_missiles_and_goes_between_them(roster, land):
    ground, facts = land
    base = fw.BaseScript(roster, ground, facts.layouts[0][0].point, facts.layouts[0][1].point, np.random.default_rng(1))
    silo = units((1, "staticmissilesilo", *facts.layouts[0][0].point))
    out = base.orders(silo, NOBODY, np.zeros(0, rf.FEATURE_DTYPE), 0.0)
    assert len(out) == 1 and out[0]["kind"] == rf.PRODUCE
    assert se.catalog().units[out[0]["target"]]["flags"] & zksim.MISSILE
    assert not len(base.orders(silo, NOBODY, np.zeros(0, rf.FEATURE_DTYPE), 1.0))
