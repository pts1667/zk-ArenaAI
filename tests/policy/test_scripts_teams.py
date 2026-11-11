"""The army and economy scripts (libzkscripts) in games of more than two teams: alliances, free-for-all, a team the scripts don't
play (the chickens, and the army's posture against them).  These frames are made by hand, as zksim.Batch holds them ([games, teams,
cap] records, a team's own units first, then its allies'); the scripts read nothing else but a chicken game's state.  And a runner
with room for more teams plays a 1v1 as before."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
try:
    import policy_scripts_cpp as psc
    psc.lib()
except OSError:
    pytest.skip("needs the built libzkscripts", allow_module_level=True)
import policy_sim_env as se  # noqa: E402
import zksim  # noqa: E402
from zkbot.policy import rawframe as rf  # noqa: E402

CAP = 64


@pytest.fixture(scope="module")
def world():
    """The scripts' data and a training map's ground (agalta, 6144 x 5120)."""
    table = se.tables.table()
    data = psc.ScriptsData(se.ScriptData(table, se.catalog()), se.EcoData(table, se.catalog()))
    _, bundle, facts = se.pool()[0]
    return data, se.Ground(bundle, facts), bundle


def _at(bundle, fx: float, fz: float) -> tuple[float, float]:
    """A point at these fractions of the map, in whole elmos, so that the records' float32 hold it exactly."""
    return float(round(fx * bundle.map_w)), float(round(fz * bundle.map_h))


def _def(name: str) -> int:
    return int(se.catalog().units[se.catalog().index[name]]["def_id"])


def _batch(ally_of: list[int], units: list[tuple], chickens: dict | None = None) -> SimpleNamespace:
    """One game's frames.  units: (uid, name, x, z, team).  Each team's frame lists its own units, then its allies' in sight; what
    it knows of an enemy is read from that enemy's own frame.  chickens: the game's chicken state (`_chickens`), for a batch that
    has one."""
    n = len(ally_of)
    slots = n
    records = np.zeros((1, slots, CAP), rf.UNIT_DTYPE)
    counts = np.zeros((1, slots), np.int32)
    for t in range(n):
        listed = [u for u in units if u[4] == t] + [u for u in units if u[4] != t and ally_of[u[4]] == ally_of[t]]
        for i, (uid, name, x, z, owner) in enumerate(listed):
            r = records[0, t, i]
            r["uid"], r["def_id"], r["x"], r["z"], r["hp_frac"] = uid, _def(name), x, z, 1.0
            r["flags"], r["knowledge"], r["ord_kind"] = rf.IDLE, rf.KNOW_LOS, -1
            r["player"], r["ally_team"] = owner, ally_of[owner]
        counts[0, t] = len(listed)
    globs = np.zeros((1, slots, len(rf.GLOBAL_FIELDS)), np.float32)
    globs[..., rf.G["metal_storage"]] = globs[..., rf.G["energy_storage"]] = 500.0
    batch = SimpleNamespace(records=records, cap=CAP, counts=counts, globs=globs, stats=np.zeros((1, slots, zksim.STATS), np.float32),
                            features=np.zeros((1, slots, 8), rf.FEATURE_DTYPE), feature_counts=np.zeros((1, slots), np.int32))
    if chickens is not None:
        batch.chicken_state = lambda game: chickens
    return batch


def _chickens(team: int, anger: float = 0.0, until_queen: float = 3000.0, queen: int = 0) -> dict:
    """A chicken game's state as zksim.Batch.chicken_state gives it."""
    state = dict.fromkeys(zksim.CHICKEN_STATE, 0.0)
    state.update(queen=float(queen), until_queen=until_queen if queen == 0 else 0.0, aggro=anger, difficulty=3.0, team=float(team))
    return state


def _runner(world, starts: list[tuple[float, float]], ally_of: list[int], script_teams=None, economy=False) -> psc.Scripts:
    data, ground, _ = world
    runner = psc.Scripts(data, 1, 1, teams=len(starts))
    flat = [v for s in starts for v in s]
    runner.reset(0, ground, flat, flat, economy=economy, aimer=False, ally_of=ally_of, script_teams=script_teams)
    return runner


def _orders(runner: psc.Scripts, batch, seconds: float) -> tuple[list[dict], list[np.ndarray]]:
    """Every seat's army script orders this step as {unit: (kind, x, z)}, and its economy script's."""
    seats = runner.teams
    decisions = np.array([round(seconds * zksim.FRAMES_PER_SECOND / (se.TICKS_PER_DECISION * zksim.TICK_FRAMES))], np.int32)
    _, _, army, offsets, _, _ = runner.step(batch, decisions, np.full(seats, psc.SEAT_ARMY, np.int32), np.full(seats, -1, np.int32),
                                            se.TICKS_PER_DECISION, 1, 1, group_orders=False)
    eco, eco_offsets = runner.eco_orders(1)
    told = [{int(o["unit"]): (int(o["kind"]), float(o["x"]), float(o["z"])) for o in army[offsets[s]:offsets[s + 1]]} for s in range(seats)]
    return told, [eco[eco_offsets[s]:eco_offsets[s + 1]] for s in range(seats)]


def test_with_three_alliances_a_seat_sees_both_others_units_as_enemies(world):
    """Two raider squads each go for the extractor nearest them, one of each enemy."""
    _, _, b = world
    p, q = _at(b, 0.3, 0.3), _at(b, 0.7, 0.3)
    units = [(uid, "cloakraid", p[0], p[1], 0) for uid in (1, 2, 3)] + [(uid, "cloakraid", q[0], q[1], 0) for uid in (4, 5, 6)]
    units += [(20, "staticmex", p[0] + 100.0, p[1] + 100.0, 1), (30, "staticmex", q[0] + 100.0, q[1] + 100.0, 2)]
    starts = [_at(b, 0.5, 0.1), _at(b, 0.1, 0.9), _at(b, 0.9, 0.9)]
    told, _ = _orders(_runner(world, starts, [0, 1, 2]), _batch([0, 1, 2], units), 60.0)
    assert {told[0][u] for u in (1, 2, 3)} == {(rf.FIGHT, p[0] + 100.0, p[1] + 100.0)}
    assert {told[0][u] for u in (4, 5, 6)} == {(rf.FIGHT, q[0] + 100.0, q[1] + 100.0)}


def test_an_allys_units_are_neither_attacked_nor_counted_as_own(world):
    _, _, b = world
    p, q = _at(b, 0.3, 0.3), _at(b, 0.7, 0.6)
    units = [(uid, "cloakraid", p[0], p[1], 0) for uid in (1, 2, 3)]
    units += [(20, "staticmex", p[0] + 100.0, p[1] + 100.0, 1), (21, "cloakraid", p[0] + 50.0, p[1], 1), (30, "staticmex", *q, 2)]
    starts = [_at(b, 0.2, 0.2), _at(b, 0.4, 0.2), _at(b, 0.8, 0.8)]
    told, _ = _orders(_runner(world, starts, [0, 0, 1]), _batch([0, 0, 1], units), 60.0)
    assert set(told[0]) == {1, 2, 3} and set(told[1]) == {21}
    assert {told[0][u] for u in (1, 2, 3)} == {(rf.FIGHT, *q)}       # past the ally's extractor, to the enemy's


def test_the_army_heads_for_the_nearer_enemys_start_and_for_the_others_once_the_nearer_has_no_units(world):
    _, _, b = world
    starts = [_at(b, 0.2, 0.2), _at(b, 0.6, 0.2), _at(b, 0.2, 0.9)]
    army = [(1, "cloakassault", *starts[0], 0), (2, "cloakassault", starts[0][0] + 40.0, starts[0][1], 0)]
    near, far = (20, "staticmex", *starts[1], 1), (30, "staticmex", *starts[2], 2)
    runner = _runner(world, starts, [0, 1, 2])
    told, _ = _orders(runner, _batch([0, 1, 2], army + [near, far]), 60.0)
    assert told[0][1] == told[0][2] == (rf.FIGHT, *starts[1])
    told, _ = _orders(runner, _batch([0, 1, 2], army + [far]), 63.0)
    assert told[0][1] == told[0][2] == (rf.FIGHT, *starts[2])


def test_allied_economies_split_the_spots_between_them(world):
    """Two allies' constructors stand together between their starts: each goes to the nearest spot that's nearer its own start."""
    _, _, b = world
    starts = [_at(b, 0.25, 0.5), _at(b, 0.75, 0.5), _at(b, 0.5, 0.95)]
    middle = _at(b, 0.5, 0.3)          # the very middle of agalta is a hill a constructor can't reach the spots from
    units = [(1, "cloakcon", *middle, 0), (2, "cloakcon", *middle, 1), (30, "energysolar", *starts[2], 2)]
    _, eco = _orders(_runner(world, starts, [0, 0, 1], economy=True), _batch([0, 0, 1], units), 60.0)
    built = [(float(e[0]["x"]), float(e[0]["z"])) for e in eco[:2]]
    assert all(len(e) == 1 and e[0]["kind"] == zksim.ORDER_BUILD for e in eco[:2])
    assert built[0] != built[1]
    for team, other in ((0, 1), (1, 0)):
        x, z = built[team]
        assert np.hypot(x - starts[team][0], z - starts[team][1]) < np.hypot(x - starts[other][0], z - starts[other][1])


def test_a_team_outside_script_teams_gets_no_orders_and_is_fought(world):
    """The chickens: no base where their start is, so the army doesn't walk there; it goes for the one near its buildings, and holds
    at the rally while there are none."""
    _, _, b = world
    starts = [_at(b, 0.2, 0.2), _at(b, 0.9, 0.9)]
    army = [(1, "cloakassault", *starts[0], 0), (2, "cloakassault", starts[0][0] + 40.0, starts[0][1], 0)]
    mex = (20, "staticmex", starts[0][0] + 600.0, starts[0][1] + 300.0, 0)
    chicken = (50, "chickena", starts[0][0] + 800.0, starts[0][1] + 300.0, 1)
    runner = _runner(world, starts, [0, 1], script_teams=[0], economy=True)
    told, eco = _orders(runner, _batch([0, 1], army + [mex, chicken]), 60.0)
    assert told[0][1] == told[0][2] == (rf.FIGHT, chicken[2], chicken[3])
    assert not told[1] and not len(eco[1]) and runner.made(0, 1) == {}
    told, _ = _orders(runner, _batch([0, 1], army + [mex]), 63.0)
    assert told[0][1] == told[0][2] == (rf.FIGHT, *starts[0])


def _guarding(world):
    """A chicken game on agalta: our army a little off its rally (our start), an extractor of ours, a burrow near our base and
    another far off.  The army's assaults go straight for the army's goal."""
    _, _, b = world
    starts = [_at(b, 0.2, 0.2), _at(b, 0.9, 0.9)]
    x, z = starts[0]
    army = [(1, "cloakassault", x + 300.0, z + 300.0, 0), (2, "cloakassault", x + 340.0, z + 300.0, 0)]
    mex = (20, "staticmex", x + 600.0, z, 0)
    near, far = (60, "roost", x + 1200.0, z + 900.0, 1), (61, "roost", *_at(b, 0.8, 0.8), 1)
    return starts, army + [mex, near, far], near, far


def test_in_a_chicken_game_the_army_hunts_the_burrow_nearest_its_base_only_while_a_kill_costs_the_queen_no_time(world):
    """A burrow killed brings the queen sooner by 15 s times the chickens' anger: none at 0 or below, so the army hunts then, and
    holds at its rally once they're angry."""
    starts, units, near, _ = _guarding(world)
    told, _ = _orders(_runner(world, starts, [0, 1], script_teams=[0]), _batch([0, 1], units, _chickens(1, anger=0.0)), 60.0)
    assert told[0][1] == told[0][2] == (rf.FIGHT, near[2], near[3])
    told, _ = _orders(_runner(world, starts, [0, 1], script_teams=[0]), _batch([0, 1], units, _chickens(1, anger=0.5)), 60.0)
    assert told[0][1] == told[0][2] == (rf.FIGHT, *starts[0])


def test_in_a_chicken_game_the_army_defends_its_buildings_before_hunting_unless_the_anger_is_below_zero(world):
    """Below 0 the chickens' kinds come sooner every wave, so a free kill comes before the chickens at our extractor."""
    starts, units, near, _ = _guarding(world)
    chicken = (50, "chickena", units[2][2] + 300.0, units[2][3], 1)
    told, _ = _orders(_runner(world, starts, [0, 1], script_teams=[0]), _batch([0, 1], units + [chicken], _chickens(1, anger=0.0)), 60.0)
    assert told[0][1] == told[0][2] == (rf.FIGHT, chicken[2], chicken[3])
    told, _ = _orders(_runner(world, starts, [0, 1], script_teams=[0]), _batch([0, 1], units + [chicken], _chickens(1, anger=-0.25)), 60.0)
    assert told[0][1] == told[0][2] == (rf.FIGHT, near[2], near[3])


def test_in_a_chicken_game_the_army_hunts_whatever_the_anger_from_minute_30_to_bring_the_queen_on(world):
    starts, units, near, _ = _guarding(world)
    runner = _runner(world, starts, [0, 1], script_teams=[0])
    for seconds, goal in ((60.0, starts[0]), (1797.0, starts[0]), (1800.0, near[2:4])):
        told, _ = _orders(runner, _batch([0, 1], units, _chickens(1, anger=2.0)), seconds)
        assert told[0][1] == told[0][2] == (rf.FIGHT, *goal), seconds


def test_in_a_chicken_game_the_whole_army_goes_for_the_queen_wherever_she_is(world):
    """Her death is the game: before the chickens at our extractor and the burrow, and the AA to her rather than with the army."""
    _, _, b = world
    starts, units, _, _ = _guarding(world)
    chicken = (50, "chickena", units[2][2] + 300.0, units[2][3], 1)
    queen = (70, "chickenflyerqueen", *_at(b, 0.6, 0.7), 1)
    aa = (3, "cloakaa", starts[0][0] + 320.0, starts[0][1] + 340.0, 0)
    batch = _batch([0, 1], units + [chicken, queen, aa], _chickens(1, queen=1))
    told, _ = _orders(_runner(world, starts, [0, 1], script_teams=[0]), batch, 60.0)
    assert told[0][1] == told[0][2] == told[0][3] == (rf.FIGHT, queen[2], queen[3])


def test_in_a_chicken_game_plants_make_aa_from_five_minutes_before_the_queen_is_due(world):
    """With the plant's quotas of constructors and scouts met and the chickens seen with air, at minute 20 a plant makes raiders
    (the largest share) while the queen is ten minutes off, and AA once she's four minutes off."""
    data, _, b = world
    table = se.tables.table()
    plant = table.names.index("factorycloak")
    menu = {role: int(data._arrays["menu"][plant, k]) for k, role in enumerate(data.roles)}
    starts = [_at(b, 0.2, 0.2), _at(b, 0.9, 0.9)]
    x, z = starts[0]
    units = [(1, "factorycloak", x, z, 0)] + [(10 + k, table.names[menu["builder"]], x + 200.0, z + 20.0 * k, 0) for k in range(6)]
    units += [(20 + k, table.names[menu["scout"]], x + 250.0, z + 20.0 * k, 0) for k in range(2)]
    units.append((50, "chickenf", *_at(b, 0.5, 0.5), 1))
    decisions = np.array([1200 * zksim.FRAMES_PER_SECOND // (se.TICKS_PER_DECISION * zksim.TICK_FRAMES)], np.int32)
    made = {}
    for until in (600.0, 240.0):
        runner = _runner(world, starts, [0, 1], script_teams=[0])
        _, _, orders, offsets, _, _ = runner.step(_batch([0, 1], units, _chickens(1, until_queen=until)), decisions,
                                                  np.full(2, psc.SEAT_ARMY, np.int32), np.full(2, -1, np.int32), se.TICKS_PER_DECISION, 1, 1,
                                                  group_orders=False)
        produce = [o for o in orders[offsets[0]:offsets[1]] if o["unit"] == 1 and o["kind"] == zksim.ORDER_PRODUCE]
        assert len(produce) == 1
        made[until] = int(produce[0]["target"])
    assert made[600.0] == se.catalog().index[table.names[menu["raid"]]]
    assert made[240.0] == se.catalog().index[table.names[menu["aa"]]]


def test_the_scripts_data_knows_the_chicken_queen_in_both_her_forms(world):
    """And the White Dragon, the queen at Very Easy and Beginner (a miniqueen at the others, as big a threat)."""
    data, _, _ = world
    table = se.tables.table()
    bits = data._arrays["bits"]
    assert {table.names[r] for r in np.flatnonzero(bits & psc.QUEEN)} == {"chickenflyerqueen", "chickenlandqueen", "chicken_dragon"}


def test_a_four_slot_runner_with_a_two_team_game_in_it_gives_the_same_orders_as_a_two_slot_runner():
    """Scripted sim games, each step's frames given as well to a runner with room for four teams, slots 2 and 3 empty."""
    env = se.SimEnv(3, 17, ["scripts"], threads=2, script_threads=2, variants=True)
    wide = psc.Scripts(env.cpp_data, env.count, 2, teams=4)
    narrow_reset, narrow_step = env.cpp.reset, env.cpp.step
    compared = 0

    def reset(*a, **kw):
        narrow_reset(*a, **kw)
        wide.reset(*a, **kw)

    def widen(a: np.ndarray, fill=0) -> np.ndarray:
        out = np.full((a.shape[0], 4, *a.shape[2:]), fill, a.dtype)
        out[:, :2] = a
        return out

    def by_seat(a: np.ndarray, fill) -> np.ndarray:
        return widen(np.asarray(a).reshape(-1, 2), fill).reshape(-1)

    def offsets(o: np.ndarray) -> np.ndarray:      # 2 n + 1 offsets as 4 n + 1, the empty slots' ranges empty
        return np.concatenate(([0], by_seat(np.diff(o), 0).cumsum())).astype(np.int32)

    def step(batch, decisions, flags, busy, ticks, script_every, eco_every, **kw):
        nonlocal compared
        out = [np.copy(a) for a in narrow_step(batch, decisions, flags, busy, ticks, script_every, eco_every, **kw)]
        eco, counts = env.cpp.eco_orders(env.count), env.cpp.seat_counts(env.count)
        big = SimpleNamespace(records=widen(batch.records), cap=batch.cap, counts=widen(batch.counts), globs=widen(batch.globs),
                              stats=widen(batch.stats), features=widen(batch.features), feature_counts=widen(batch.feature_counts))
        for part in ("pre", "post"):
            if kw.get(part) is not None:
                kw[part] = (kw[part][0], offsets(kw[part][1]))
        got = wide.step(big, decisions, by_seat(flags, 0), by_seat(busy, -1), ticks, script_every, eco_every, **kw)
        orders, order_offsets, army, army_offsets, groups, group_offsets = out
        assert got[0].tobytes() == orders.tobytes() and (got[1] == order_offsets).all()
        assert got[2].tobytes() == army.tobytes() and (got[3] == offsets(army_offsets)).all()
        assert got[4].tobytes() == groups.tobytes() and (got[5] == group_offsets).all()
        wide_eco = wide.eco_orders(env.count)
        assert wide_eco[0].tobytes() == eco[0].tobytes() and (wide_eco[1] == offsets(eco[1])).all()
        assert (wide.seat_counts(env.count) == widen(counts.reshape(-1, 2, 4)).reshape(-1, 4)).all()
        compared += len(orders)
        return tuple(out)

    env.cpp.reset, env.cpp.step = reset, step
    env.begin()
    for _ in range(900):
        env.step(0.0)
        env.finished_seats()
    assert compared > 1000
    assert all(wide.made(g, t) == env.cpp.made(g, t) for g in range(env.count) for t in (0, 1))
