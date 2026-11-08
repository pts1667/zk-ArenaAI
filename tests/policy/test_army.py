"""The army script and the production mix behind it, on records made by hand and in short sim games."""
from __future__ import annotations

import collections
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "tools"))
try:
    import policy_army as pa
    import policy_mix as pm
    import policy_script as ps
    import policy_sim_env as se
    from zkbot import catalog as bc
    from zkbot.policy import rawframe as rf
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)

RALLY, ENEMY_START = (500.0, 500.0), (4000.0, 4000.0)
CLOAK, VEH = "factorycloak", "factoryveh"


@pytest.fixture(scope="module")
def data():
    return ps.ScriptData(se.tables.table(), se.catalog())


def _def_id(name: str) -> int:
    return int(se.catalog().units[se.catalog().index[name]]["def_id"])


def _units(*specs) -> np.ndarray:
    """Each (uid, name, x, z[, hp_frac])."""
    out = np.zeros(len(specs), rf.UNIT_DTYPE)
    for i, spec in enumerate(specs):
        uid, name, x, z, *hp = spec
        out[i]["uid"], out[i]["def_id"], out[i]["x"], out[i]["z"] = uid, _def_id(name), x, z
        out[i]["hp_frac"] = hp[0] if hp else 1.0
    return out


def _view(*specs) -> pa.View:
    """Enemies as (uid, x, z, cost, reach, kind), kind being building, extractor, defence, threat or heavy (a riot heavy)."""
    uid, x, z, cost, reach, kind = (np.array(col) for col in zip(*specs))
    return pa.View(uid=uid.astype(np.int64), x=x.astype(float), z=z.astype(float), cost=cost.astype(float), reach=reach.astype(float),
                   building=np.isin(kind, ["building", "extractor"]), defence=kind == "defence", threat=np.isin(kind, ["threat", "heavy"]),
                   heavy_riot=kind == "heavy", slow_heavy=np.zeros(len(uid), bool), extractor=kind == "extractor")


def _named(data, pred) -> str:
    """The first modelled unit, by name, that has this property of its table row."""
    return sorted(n for n, row in data.row_of_name.items() if data.modelled[row] and pred(row))[0]


def _script(data) -> pa.ArmyScript:
    return pa.ArmyScript(data, RALLY, ENEMY_START)


def _goals(out: np.ndarray) -> dict[int, tuple[int, float, float]]:
    return {int(o["unit"]): (int(o["kind"]), float(o["x"]), float(o["z"])) for o in out}


def _no_enemies() -> np.ndarray:
    return np.zeros(0, rf.UNIT_DTYPE)


def _build(data, mix: pm.Mix, factory: str, times: list[float]) -> list[str | None]:
    """The role of each unit the factory starts, at each of these times, as the factory would start it."""
    row = data.row_of_name[factory]
    roles = []
    for seconds in times:
        pick = mix.choose(row, seconds)
        if pick is None:
            roles.append(None)
            continue
        roles.append(data.role_of[pick])
        mix.record(data.role_of[pick])
    return roles


# The production mix.

def test_a_factory_builds_only_raiders_for_the_first_four_minutes(data):
    roles = _build(data, pm.Mix(data), CLOAK, [30.0 * i for i in range(8)])
    assert roles == ["raid"] * 8


def test_the_mix_at_ten_minutes_follows_the_shares(data):
    mix = pm.Mix(data)
    roles = _build(data, mix, CLOAK, [600.0] * 40)
    count = collections.Counter(roles)
    share = pm.shares(600.0, pm.Seen())
    total = sum(share.values())
    for role, w in share.items():
        if w > 0.0:
            assert abs(count[role] - w / total * 40) <= 2, (role, count)
    assert set(count) <= {"raid", "riot", "assault"}


def test_skirmishers_wait_until_a_kitable_unit_has_been_seen(data):
    assert "skirm" not in pm.shares(600.0, pm.Seen())
    mix = pm.Mix(data)
    mix.observe(_units((9, "cloakriot", 0.0, 0.0)))
    assert mix.seen.kitable and pm.shares(600.0, mix.seen)["skirm"] == pm.SKIRM_SHARE
    assert "skirm" in _build(data, mix, CLOAK, [600.0] * 40)


def test_aa_waits_until_the_enemy_has_been_seen_with_air(data):
    air = _named(data, lambda row: bool(data.table.air[row] and data.table.mobile[row]))
    assert "aa" not in pm.shares(600.0, pm.Seen())
    assert "aa" not in set(_build(data, pm.Mix(data), CLOAK, [600.0] * 40))
    mix = pm.Mix(data)
    mix.observe(_units((9, air, 0.0, 0.0)))
    assert mix.seen.air
    assert "aa" in set(_build(data, mix, CLOAK, [600.0] * 40))


def test_artillery_comes_late_and_snipers_only_after_a_slow_heavy_and_480_seconds(data):
    assert pm.shares(300.0, pm.Seen())["arty"] == 0.0
    assert "arty" not in set(_build(data, pm.Mix(data), CLOAK, [60.0 * i for i in range(8)]))
    assert "arty" in set(_build(data, pm.Mix(data), CLOAK, [900.0] * 40))
    heavy = _named(data, lambda row: bool(data.slow_heavy[row]))
    seen = pm.Seen(heavy=True)
    assert "snipe" not in pm.shares(400.0, seen)
    assert "snipe" not in pm.shares(600.0, pm.Seen())
    mix = pm.Mix(data)
    mix.observe(_units((9, heavy, 0.0, 0.0)))
    assert pm.shares(600.0, mix.seen)["snipe"] == pm.SNIPE_SHARE
    assert "snipe" in set(_build(data, mix, CLOAK, [600.0] * 40))


def test_scouts_come_one_at_a_time_from_two_minutes(data):
    mix = pm.Mix(data)
    assert _build(data, mix, VEH, [100.0]) == ["raid"]
    assert _build(data, mix, VEH, [130.0]) == ["scout"]
    assert _build(data, mix, VEH, [130.0]) != ["scout"]
    assert _build(data, mix, VEH, [600.0]) == ["scout"]


def test_each_role_on_a_roster_is_its_cheapest_modelled_unit(data):
    mix = pm.Mix(data)
    for factory in (CLOAK, VEH):
        menu = mix.menu(data.row_of_name[factory])
        for name in bc.roster(factory):
            row = data.row_of_name.get(name)
            if row is None or not data.modelled[row]:
                continue
            assert data.role_of[row] in menu
            assert data.table.cost[menu[data.role_of[row]]] <= data.table.cost[row]


# The sim: what a recorder reads.

def test_a_factory_is_told_what_to_build_only_when_that_changes_and_it_is_its_own():
    env = se.SimEnv(8, 3, ["scripts"], threads=4)
    env.begin()
    last: dict[tuple, str] = {}
    told = 0
    for _ in range(400):
        env.step(0.0)
        for seat in range(16):
            g, side = divmod(seat, 2)
            script = env.scripts[g][side]
            for uid, name in env.script_production(seat):
                assert uid == env.seats[g][side].factory_uid
                key = (seat, id(script), uid)
                assert last.get(key) != name
                last[key] = name
                told += 1
                role = env.script_data.role_of[env.script_data.row_of_name[name]]
                assert role in pm.ROLE_ORDER + (pm.SCOUT,)
    assert told > 0


def test_the_army_gives_the_same_orders_on_any_number_of_threads():
    runs = []
    for threads in (1, 4):
        env = se.SimEnv(8, 5, ["scripts"], threads=threads)
        env.begin()
        for _ in range(60):
            env.step(0.0)
            env.finished_seats()
        runs.append(env.batch.records.copy())
    assert runs[0].tobytes() == runs[1].tobytes()


# Production and orders for one factory, on records made by hand.

def test_a_factory_is_given_a_produce_only_when_its_next_type_changes(data):
    script = _script(data)
    factory = _units((7, CLOAK, 100.0, 100.0))
    first = script.orders(factory, _no_enemies(), 60.0)
    assert [(int(o["unit"]), int(o["kind"])) for o in first] == [(7, rf.PRODUCE)]
    assert int(first[0]["target"]) == int(data.catalog_row[data.row_of_name["cloakraid"]])
    assert len(script.orders(factory, _no_enemies(), 90.0)) == 0


def test_scouts_move_between_waypoints(data):
    script = _script(data)
    out = _goals(script.orders(_units((5, "vehscout", 600.0, 600.0)), _no_enemies(), 60.0))
    assert out[5][0] == rf.MOVE


def test_a_snipe_attacks_the_nearest_slow_heavy(data):
    heavy = _named(data, lambda row: bool(data.slow_heavy[row]))
    script = _script(data)
    script.state[5] = ps.SENT
    out = {int(o["unit"]): o for o in script.orders(_units((5, "cloaksnipe", 2000.0, 2000.0)), _units((50, heavy, 3000.0, 2000.0)), 600.0)}
    assert int(out[5]["kind"]) == rf.ATTACK_UNIT and int(out[5]["target"]) == 50


def test_a_skirmisher_backs_off_from_an_armed_unit_close_to_its_range(data):
    reach = float(data.table.range[data.row_of_name["cloakskirm"]])
    script = _script(data)
    script.state[5] = ps.SENT
    out = _goals(script.orders(_units((5, "cloakskirm", 2000.0, 2000.0)), _units((50, "cloakriot", 2000.0 + 0.3 * reach, 2000.0)), 600.0))
    assert out[5][0] == rf.MOVE and out[5][1] < 2000.0


def test_artillery_stands_at_three_quarters_of_its_range_from_the_nearest_building(data):
    reach = float(data.table.range[data.row_of_name["cloakarty"]])
    script = _script(data)
    script.state[5] = ps.SENT
    kind, x, z = _goals(script.orders(_units((5, "cloakarty", 2000.0, 2000.0)), _units((50, "staticmex", 3000.0, 3000.0)), 600.0))[5]
    assert kind == rf.FIGHT
    assert math.hypot(x - 3000.0, z - 3000.0) == pytest.approx(pa.ARTY_STAND * reach)


def test_riots_hold_behind_the_goal_when_no_assault_leads(data):
    script = _script(data)
    script.state[1] = ps.SENT
    kind, x, z = _goals(script.orders(_units((1, "cloakriot", 3000.0, 3000.0)), _no_enemies(), 600.0))[1]
    assert kind == rf.FIGHT
    assert (x, z) == pytest.approx(pa._along(ENEMY_START, RALLY, pa.BEHIND))


def test_riots_hold_behind_the_assaults_when_there_are_some(data):
    script = _script(data)
    script.state[1], script.state[2] = ps.SENT, ps.SENT
    out = _goals(script.orders(_units((1, "cloakassault", 3000.0, 3000.0), (2, "cloakriot", 2000.0, 2000.0)), _no_enemies(), 600.0))
    assert out[1][1:] == pytest.approx(ENEMY_START)
    assert out[2][1:] == pytest.approx(pa._along((3000.0, 3000.0), RALLY, pa.BEHIND))


def test_before_groups_begin_the_army_goes_straight_out(data):
    script = _script(data)
    out = _goals(script.orders(_units((1, "cloakriot", 600.0, 600.0)), _no_enemies(), ps.GROUPS_FROM - 10.0))
    assert out[1][1:] == pytest.approx(pa._along(ENEMY_START, RALLY, pa.BEHIND))


# Raiders.

def test_raiders_go_for_an_extractor_before_other_buildings(data):
    script = _script(data)
    raiders = _units((1, "cloakraid", 600.0, 600.0), (2, "cloakraid", 600.0, 600.0), (3, "cloakraid", 600.0, 600.0))
    out = _goals(script.orders(raiders, _units((50, "staticmex", 3000.0, 3000.0), (51, "energysolar", 1000.0, 1000.0)), 600.0))
    assert all(out[u][1:] == (3000.0, 3000.0) for u in (1, 2, 3))


def test_a_squad_passes_by_a_target_it_cannot_take(data):
    view = _view((50, 3000.0, 3000.0, 60.0, 0.0, "extractor"), (51, 1000.0, 1000.0, 60.0, 0.0, "building"), (52, 3000.0, 3100.0, 1000.0, 300.0, "defence"))
    script = _script(data)
    assert script._choose(np.array([600.0, 600.0]), 195.0, 50.0, view, set()) == 51


def test_a_squad_does_not_go_for_a_target_a_riot_heavy_is_near(data):
    view = _view((50, 3000.0, 3000.0, 60.0, 0.0, "extractor"), (52, 3000.0, 3700.0, 1000.0, 300.0, "heavy"))
    assert _script(data)._choose(np.array([600.0, 600.0]), 195.0, 50.0, view, set()) is None


def test_a_defence_that_outranges_a_squad_counts_four_times_over():
    view = _view((1, 0.0, 0.0, 100.0, 500.0, "defence"))
    cover, heavy = pa._cover(view, np.array([300.0]), np.array([0.0]), 100.0)
    assert cover[0] == pytest.approx(400.0) and not heavy[0]
    assert pa._cover(view, np.array([700.0]), np.array([0.0]), 100.0)[0][0] == 0.0


def test_mobile_armies_count_for_cover_only_when_asked():
    view = _view((1, 300.0, 0.0, 100.0, 50.0, "threat"))
    assert pa._cover(view, np.array([0.0]), np.array([0.0]), 50.0)[0][0] == pytest.approx(100.0)
    assert pa._cover(view, np.array([0.0]), np.array([0.0]), 50.0, armies=False)[0][0] == 0.0


def test_a_probe_stops_short_of_a_defence_it_cannot_take(data):
    view = _view((1, 600.0, 0.0, 1000.0, 300.0, "defence"))
    script = pa.ArmyScript(data, (0.0, 0.0), (4000.0, 0.0))
    assert script._probe(np.array([0.0, 0.0]), 200.0, 50.0, view) == (100.0, 0.0)


def test_a_probe_holds_when_all_of_the_ground_towards_the_enemy_is_covered(data):
    view = _view((1, 50.0, 0.0, 1000.0, 300.0, "defence"))
    script = pa.ArmyScript(data, (0.0, 0.0), (4000.0, 0.0))
    assert script._probe(np.array([0.0, 0.0]), 200.0, 50.0, view) == (0.0, 0.0)


def test_patrol_counts_follow_the_share_and_the_minimum(data):
    script = _script(data)
    assert script._patrol_count(20, 600.0, 3) == 5
    assert script._patrol_count(10, 60.0, 3) == 3
    assert script._patrol_count(4, 100.0, 2) == 1
    assert script._patrol_count(8, 100.0, 0) == 0


def test_a_patroller_waits_by_its_extractor_and_answers_only_threats_near_it(data):
    script = _script(data)
    near = _view((7, 1200.0, 1000.0, 100.0, 200.0, "threat"))
    far = _view((7, 3000.0, 1000.0, 100.0, 200.0, "threat"))
    assert script._patrol_orders(near, [5], [(1000.0, 1000.0)], {5: False})[5][1:3] == (1200.0, 1000.0)
    assert script._patrol_orders(far, [5], [(1000.0, 1000.0)], {5: False})[5][1:3] == pytest.approx(
        pa._along((1000.0, 1000.0), ENEMY_START, pa.PATROL_OFFSET))


def test_a_target_given_a_long_way_off_takes_longer_to_give_up_on(data):
    script = _script(data)
    view = _view((5, 3000.0, 3000.0, 60.0, 0.0, "extractor"))
    script.since = {1: (0.0, 2000.0)}
    centre = np.array([1000.0, 1000.0])
    assert script._patient(1, 5, centre, 150.0, view)
    assert not script._patient(1, 5, centre, 170.0, view)


def test_squads_that_probe_together_do_not_become_a_blob(data):
    script = _script(data)
    raiders = _units(*[(i, "cloakraid", 600.0, 600.0) for i in range(1, 7)])
    script.orders(raiders, _no_enemies(), 600.0)
    counts = collections.Counter(script.squad_of.values())
    assert max(counts.values()) <= 2 * pa.SQUAD_SIZE


def test_an_order_is_not_given_again_unless_it_matters(data):
    script = _script(data)
    out: list[tuple] = []
    script._emit(5, (rf.FIGHT, 100.0, 100.0, -1), np.array([0.0, 0.0]), False, out)
    assert len(out) == 1
    script._emit(5, (rf.FIGHT, 100.0, 100.0, -1), np.array([0.0, 0.0]), False, out)
    assert len(out) == 1
    script._emit(5, (rf.FIGHT, 100.0 + pa.REORDER + 10.0, 100.0, -1), np.array([0.0, 0.0]), False, out)
    assert len(out) == 2
    script._emit(5, (rf.FIGHT, 100.0 + pa.REORDER + 10.0, 100.0, -1), np.array([5000.0, 0.0]), True, out)
    assert len(out) == 3
