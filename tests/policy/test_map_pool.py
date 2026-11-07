"""Made-up maps, and the frames that use them: do they hold together, and can things stand where they're put?"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
try:
    import frame_data as fd
    import frame_world as fw
    import map_gen as mg
    import policy_sim_env as se
    import zksim
    from zkbot.policy import bundle as bd, rawframe as rf, tables
    from zkbot import terrain as tr
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)

SMALL = dict(sides=(8,), side_weights=(1.0,), square=1.0)
DRY = mg.Rules(waters=(("none", 1.0),), **SMALL)
ISLANDS = mg.Rules(waters=(("islands", 1.0),), lava=0.0, **SMALL)
SHIP = bd.MOVE_CLASSES.index("ship")


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    """A small pool: two dry maps and two nearly all water."""
    out = tmp_path_factory.mktemp("maps")
    for seed, rules in ((1, DRY), (2, DRY), (3, ISLANDS), (4, ISLANDS)):
        made = mg.make(seed, rules)
        mg.write_entry(made.map, "made", made.layouts, made.features, out, made.style)
    return se.load_pool(out)


def _wet(pool):
    return [m for m in pool if m[2].water > 0.3]


def _dry(pool):
    return [m for m in pool if m[2].water == 0.0]


# Maps -----------------------------------------------------------------------------------------------------------------------------------------------------


def test_a_seed_makes_one_map():
    a, b = mg.make(5, DRY), mg.make(5, DRY)
    assert np.array_equal(a.map.corners, b.map.corners) and a.map.spots == b.map.spots and a.layouts == b.layouts


@pytest.mark.parametrize("symmetry", ("rot180", "mirror", "diagonal"))
def test_a_symmetric_map_is_its_own_image(symmetry):
    rules = mg.Rules(symmetries=((symmetry, 1.0),), waters=(("river", 1.0),), lava=0.0, **SMALL)
    made = mg.make(6, rules)
    c = mg.Canvas(np.random.default_rng(1), (made.map.corners.shape[1] - 1) * 8.0, (made.map.corners.shape[0] - 1) * 8.0, symmetry)
    # Which way it's flipped was drawn inside, so any of the ways this symmetry can go will do.
    images = []
    for axis, flip in (("x", False), ("z", False), ("", False), ("", True)):
        c.axis, c.flip = axis, flip
        images.append(np.abs(c.reflect(made.map.corners) - made.map.corners).max() if made.map.corners.shape == c.reflect(made.map.corners).shape else np.inf)
    assert min(images) < 1.0


@pytest.mark.parametrize("seed", range(10, 16))
def test_a_walker_can_get_from_start_to_start_on_a_map_meant_for_walkers(seed):
    made = mg.make(seed, DRY)
    ground = mg.Ground.of(made.map.corners, 0.0, False)
    can = ground.can("kbot")
    labels, _ = tr.label_runs(can)
    boxes = made.layouts[0]
    assert len(boxes) >= 2
    parts = {int(labels[mg._cell(*b["startpoints"][0], can.shape)]) for b in boxes}
    assert len(parts) == 1 and -1 not in parts
    for box in boxes:
        x, z = box["startpoints"][0]
        assert mg.tm_buildable(ground, x, z)
        (x0, z0), _, (x1, z1), _ = box["polygons"][0]
        assert min(x0, x1) <= x <= max(x0, x1) and min(z0, z1) <= z <= max(z0, z1)


def test_sea_maps_are_mostly_sea_with_their_starts_on_land():
    for seed in (20, 21):
        made = mg.make(seed, mg.Rules(waters=(("sea", 1.0),), lava=0.0, **SMALL))
        ground = mg.Ground.of(made.map.corners, 0.0, False)
        assert (ground.height < 0.0).mean() > 0.6
        for box in made.layouts[0]:
            assert mg.tm_buildable(ground, *box["startpoints"][0])


def test_spots_are_on_the_map_and_some_are_under_water_on_a_map_of_water():
    made = mg.make(3, ISLANDS)
    ground = mg.Ground.of(made.map.corners, 0.0, False)
    spots = np.array(made.map.spots)
    w, d = (made.map.corners.shape[1] - 1) * 8.0, (made.map.corners.shape[0] - 1) * 8.0
    assert ((0.0 <= spots[:, 0]) & (spots[:, 0] <= w) & (0.0 <= spots[:, 1]) & (spots[:, 1] <= d)).all()
    assert (spots[:, 2] > 0.0).all()
    i, j = np.minimum((spots[:, 1] / 16.0).astype(int), ground.height.shape[0] - 1), np.minimum((spots[:, 0] / 16.0).astype(int), ground.height.shape[1] - 1)
    assert (ground.height[i, j] < 0.0).any()


def test_a_written_map_loads_and_plays(pool):
    assert [m[2].kind for m in pool] == ["made"] * 4
    env = fw.FrameGames(8, 3, 2, pool, fw.pool_weights(pool))
    env.begin()
    for _ in range(60):
        env.step(0.0)
        env.finished_seats()
    assert (env.batch.counts[:, 0] > 0).all()


def test_weights_share_between_kinds_and_favour_water(pool):
    facts = [m[2] for m in pool]
    weights = fw.pool_weights(pool)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[[f.water > 0.03 for f in facts]].mean() > weights[[f.water <= 0.03 for f in facts]].mean()
    other = [(m[0], m[1], se.MapFacts(m[2].layouts, m[2].features, "x", "real" if i < 1 else "made", m[2].water)) for i, m in enumerate(pool)]
    assert fw.pool_weights(other)[0] == pytest.approx(0.5)


def test_titanduel_is_held_out_and_a_large_pool_loses_about_a_tenth():
    facts = [(None, None, se.MapFacts([], np.zeros((0, 6)), name)) for name in ("titanduel_2_2", "agalta_v1_0")]
    assert fw.held_out_maps(facts) == [0]
    many = [(None, None, se.MapFacts([], np.zeros((0, 6)), f"gen_{i:05d}")) for i in range(1000)]
    assert 60 < len(fw.held_out_maps(many)) < 140


# What stands where ------------------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def roster():
    return fw.Roster(tables.table(), se.catalog())


def _row(roster, name):
    return roster.table.names.index(name)


def test_the_data_says_where_things_stand(pool, roster):
    """A shipyard needs deep water, a solar can't be under any, and a floating turret doesn't mind."""
    _, bundle, facts = _wet(pool)[0]
    ground = fw.Ground(bundle, facts)
    ship = ground.mask(roster.footing[_row(roster, "factoryship")].low, roster.footing[_row(roster, "factoryship")].high)
    solar = ground.mask(roster.footing[_row(roster, "energysolar")].low, roster.footing[_row(roster, "energysolar")].high)
    either = ground.mask(roster.footing[_row(roster, "turretlaser")].low, roster.footing[_row(roster, "turretlaser")].high)
    assert ship.any() and solar.any()
    assert not (ship & solar).any()
    assert either[ship].all() and either[solar].all()
    scale = int(bundle.cell / bd.PASS_CELL)
    deep = np.repeat(np.repeat(bundle.height, scale, 0), scale, 1)[:ship.shape[0], :ship.shape[1]]
    assert (deep[ship] <= -roster.footing[_row(roster, "factoryship")].low).all() and (deep[solar] >= 0.0).all()


def test_a_shipyard_goes_to_the_water_however_far_the_base_is_from_it(pool, roster):
    _, bundle, facts = _wet(pool)[0]
    ground = fw.Ground(bundle, facts)
    footing = roster.footing[_row(roster, "factoryship")]
    box = facts.layouts[0][0]
    place = ground.site(np.random.default_rng(1), footing, *box.point, np.zeros((0, 3), np.float32))
    assert place is not None and ground.fits(*place, footing)
    assert not fw.Ground(*_dry(pool)[0][1:]).has_room(footing.low, footing.high)


def test_factories_suit_the_map(pool, roster):
    names = lambda ground: {roster.table.names[r] for r in roster.factories_for(ground)[0]}  # noqa: E731
    dry = names(fw.Ground(*_dry(pool)[0][1:]))
    wet_ground = fw.Ground(*_wet(pool)[0][1:])
    wet = names(wet_ground)
    assert "factoryship" not in dry and "factorycloak" in dry
    assert "factoryship" in wet and "factoryhover" in wet and "factoryplane" in wet
    rows, chances = roster.factories_for(wet_ground)
    by_name = {roster.table.names[r]: c for r, c in zip(rows, chances)}
    assert chances.sum() == pytest.approx(1.0)
    assert by_name["factoryship"] > by_name.get("factorycloak", 0.0)


@pytest.mark.parametrize("which", (0, 1))
def test_scenarios_on_water_put_everything_where_it_can_be(pool, roster, which):
    _, bundle, facts = _wet(pool)[which]
    ground = fw.Ground(bundle, facts)
    rng = np.random.default_rng(which)
    box = facts.layouts[0]
    table = roster.table
    ships = wrecks = air = floating = 0
    for _ in range(30):
        side = fd.draw_side(rng, fd.ScenarioRules(), roster, ground, box[0].point, box[1].point)
        mobile = table.mobile[side.rows]
        moving = mobile & ~table.air[side.rows]
        assert fd._on_passable(bundle, side.x[moving], side.z[moving], table.move_class[side.rows[moving]].astype(np.int32)).all()
        for r, x, z in zip(side.rows[~mobile], side.x[~mobile], side.z[~mobile]):
            if not roster.is_extractor[r]:
                assert ground.fits(float(x), float(z), roster.footing[int(r)])
                floating += bundle.ground_at(float(x), float(z)) < 0.0
        ships += int((table.move_class[side.rows] == SHIP).sum())
        air += int(table.air[side.rows].sum())
        wrecks += len(side.wrecks)
        if len(side.wrecks):
            assert (side.wrecks[:, 2] > 0.0).all() and (side.wrecks[:, 3] > 0.0).all() and side.wrecks.shape[1] == 6
    assert ships > 0 and air > 0 and wrecks > 0 and floating > 0


def test_a_scenario_has_factories_at_work_and_builders_with_jobs(pool):
    env = fd.ScenarioEnv(8, 5, fd.ScenarioRules(side_value=(5000.0, 5000.0), has_factory=1.0, structure_share=(0.4, 0.4), decisions=200), threads=2, maps=pool, weights=fw.pool_weights(pool))
    env.begin()
    orders = [o for g in range(8) for s in (0, 1) for o in env.script_orders(g, s)]
    kinds = {int(o["kind"]) for o in orders}
    assert rf.PRODUCE in kinds and rf.BUILD in kinds
    for _ in range(40):
        env.step(0.0)
        env.finished_seats()
    built = [env.batch.frame(g, s).units for g in range(8) for s in (0, 1)]
    assert any(((u["flags"] & rf.BEING_BUILT) != 0).any() for u in built)


def test_a_scenario_side_remembers_some_of_what_it_cannot_see(pool):
    env = fd.ScenarioEnv(8, 6, fd.ScenarioRules(side_value=(8000.0, 8000.0), scouted=(1.0, 1.0)), threads=2, maps=pool, weights=fw.pool_weights(pool))
    env.begin()
    env.step(0.0)
    memory = 0
    for seat in range(16):
        frame = env.frame(seat)
        u = frame.units
        rows = env.table.rows(u["def_id"])
        assert not ((u["knowledge"] == rf.KNOW_MEMORY) & env.table.mobile[rows]).any()
        at_the_start = u["uid"] < len(env.scouted[seat // 2][seat % 2])    # and not what's been begun since
        assert not ((u["knowledge"] == rf.KNOW_HIDDEN) & ~env.table.mobile[rows] & at_the_start).any()      # the whole base was seen
        memory += int((u["knowledge"] == rf.KNOW_MEMORY).sum())
    assert memory > 0
    never = fd.ScenarioEnv(8, 6, fd.ScenarioRules(side_value=(8000.0, 8000.0), scouted=(0.0, 0.0)), threads=2, maps=pool, weights=fw.pool_weights(pool))
    never.begin()
    never.step(0.0)
    assert sum(int((never.frame(s).units["knowledge"] == rf.KNOW_MEMORY).sum()) for s in range(16)) < memory / 4     # what the sim itself remembers, of a first moment's glimpse


# Games ----------------------------------------------------------------------------------------------------------------------------------------------------------


def test_a_ships_orders_stay_on_the_water(pool, roster):
    _, bundle, facts = _wet(pool)[0]
    ground = fw.Ground(bundle, facts)
    data = fw.fight_data(roster.table, se.catalog())
    land = facts.layouts[0][1].point
    assert not ground.can(SHIP, np.array([land[0]]), np.array([land[1]]))[0]
    script = fw.FrameFight(data, land, land, ground=ground)
    unit = np.zeros(1, rf.UNIT_DTYPE)
    unit["uid"], unit["def_id"], unit["hp_frac"] = 3, se.catalog().units[se.catalog().index["shipriot"]]["def_id"], 1.0
    place = ground.near(SHIP, *land)
    unit["x"], unit["z"] = place
    out = script.orders(unit, np.zeros(0, rf.UNIT_DTYPE), 0.0)
    assert len(out) == 1 and ground.can(SHIP, out["x"], out["z"])[0]


def test_builders_are_left_to_their_work(roster):
    data = fw.fight_data(roster.table, se.catalog())
    table = roster.table
    assert not data.fights[table.builder].any()
    assert data.fights[_row(roster, "cloakraid")]


def test_a_base_changes_what_it_makes_and_its_builders_find_work(pool, roster):
    world_map, bundle, facts = _dry(pool)[0]
    ground = fw.Ground(bundle, facts)
    cat = se.catalog()
    x, z = facts.layouts[0][0].point
    base = fw.BaseScript(roster, ground, (x, z), facts.layouts[0][1].point, np.random.default_rng(1))
    units = np.zeros(4, rf.UNIT_DTYPE)
    units["uid"] = np.arange(4)
    units["x"], units["z"] = x, z
    units["hp_frac"] = 1.0
    for i, name in enumerate(("factorycloak", "cloakcon", "cloakcon", "cloakcon")):
        units["def_id"][i] = cat.units[cat.index[name]]["def_id"]
    units["flags"][1:] = rf.IDLE
    none = np.zeros(0, rf.FEATURE_DTYPE)
    orders = base.orders(units, np.zeros(0, rf.UNIT_DTYPE), none, 0.0)
    assert (orders["kind"] == rf.PRODUCE).sum() == 1
    made = int(orders["target"][orders["kind"] == rf.PRODUCE][0])
    assert cat.units[made]["name"] in fw.bot_catalog.roster("factorycloak")
    assert set(orders["unit"][orders["kind"] != rf.PRODUCE].tolist()) <= {1, 2, 3}
    assert (orders["kind"] != rf.PRODUCE).sum() >= 2        # all have something to do, on a map with spots
    assert len(base.orders(units[:1], np.zeros(0, rf.UNIT_DTYPE), none, 1.0)) == 0      # not yet time to make something else
    assert (base.orders(units[:1], np.zeros(0, rf.UNIT_DTYPE), none, 500.0)["kind"] == rf.PRODUCE).all()


def test_a_commander_in_its_opening_is_left_alone(pool, roster):
    _, bundle, facts = _dry(pool)[0]
    cat = se.catalog()
    base = fw.BaseScript(roster, fw.Ground(bundle, facts), facts.layouts[0][0].point, facts.layouts[0][1].point, np.random.default_rng(1))
    commander = next(n for n in (u["name"] for u in cat.units if u["flags"] & zksim.COMMANDER) if n in roster.table.names)
    unit = np.zeros(1, rf.UNIT_DTYPE)
    unit["uid"], unit["def_id"], unit["flags"], unit["hp_frac"] = 0, cat.units[cat.index[commander]]["def_id"], rf.IDLE, 1.0
    unit["x"], unit["z"] = facts.layouts[0][0].point
    assert len(base.orders(unit, np.zeros(0, rf.UNIT_DTYPE), np.zeros(0, rf.FEATURE_DTYPE), 0.0, busy=0)) == 0


def test_games_on_water_use_it(pool):
    wet = _wet(pool)
    env = fw.FrameGames(16, 8, 2, wet, None)
    env.begin()
    names = {se.catalog().units[seat.factory_type]["name"] for seats in env.seats for seat in seats}
    assert "factoryship" in names
    table, ship = env.table, 0
    for _ in range(400):
        env.step(0.0)
        env.finished_seats()
    for g in range(16):
        for side in (0, 1):
            u = env.batch.frame(g, side).units
            own = u[u["player"] == side]
            ship += int((table.move_class[table.rows(own["def_id"])] == SHIP).sum())
    assert ship > 0


def test_a_dry_map_never_opens_with_ships(pool):
    env = fw.FrameGames(16, 9, 2, _dry(pool), None)
    env.begin()
    names = {se.catalog().units[seat.factory_type]["name"] for seats in env.seats for seat in seats}
    assert "factoryship" not in names and len(names) >= 4


def test_something_afloat_stands_on_the_water_and_the_rest_on_the_sea_bed():
    depth = 60.0
    cells = 128
    world_map = zksim.Map(np.full((cells, cells), -depth, np.float32), np.ones((6, cells, cells), np.float32))
    w = zksim.World(se.catalog(), world_map, seed=1)
    afloat, sunk = w.add("turretlaser", 0, 500.0, 500.0), w.add("factoryamph", 0, 900.0, 900.0)
    w.step(1)
    u = w.view(0).units
    assert u["y"][u["uid"] == afloat][0] == pytest.approx(0.0, abs=1.0)
    assert u["y"][u["uid"] == sunk][0] == pytest.approx(-depth, abs=1.0)


# Collecting ------------------------------------------------------------------------------------------------------------------------------------------------


def test_collecting_on_a_pool_says_what_it_took(pool, tmp_path):
    facts = fd.collect(tmp_path, 120, 120, games=8, seed=3, threads=2, script_every=6, frames_per_shard=60, maps=pool, workers=2)
    assert len(facts["maps"]) == len(pool) and sorted(facts["kinds"]) == ["made"] * len(pool)
    assert sorted(facts["frames_per_second"]) == ["scenario", "script"]
    assert facts["seen_in"]["scenario"]["sea_share"] > 0.0
    assert len(list(tmp_path.glob("script-w1-*.npz"))) >= 1 and len(list(tmp_path.glob("script-0*.npz"))) >= 1
    games = {}
    for path in tmp_path.glob("*-*.npz"):
        with np.load(path) as d:
            games.setdefault("w1" in path.name, set()).update(d["game"].tolist())
            assert d["probes"].shape[1] == len(fd.PROBES)
    assert not games[True] & games[False]


# Which factories suit a map ---------------------------------------------------------------------------------------------------------------------------------


KBOT, TANK, HOVER, AMPH = (bd.MOVE_CLASSES.index(n) for n in ("kbot", "tank", "hover", "amph"))
LAND = sum(1 << c for c in range(len(bd.MOVE_CLASSES)) if c != SHIP)
SEA = (1 << AMPH) | (1 << HOVER) | (1 << SHIP)
ANYWHERE = fw.Footing(32.0, -10000.0, 10000.0)
FLAT_CELLS = 64


def _ground(land: np.ndarray, spots: list[tuple[float, float]], reach=None, liquid: str = "water") -> fw.Ground:
    """A map made of cells: true for land, sea for the rest.  A cell is a passability cell."""
    size = land.shape[0] * bd.PASS_CELL
    bundle = SimpleNamespace(passable=np.where(land, LAND, SEA).astype(np.int32), height=np.where(land, 50.0, -100.0).astype(np.float32), cell=bd.PASS_CELL,
                             spot_list=np.array([(x, z, 2.0) for x, z in spots], np.float32).reshape(-1, 3), map_w=float(size), map_h=float(size))
    return fw.Ground(bundle, SimpleNamespace(water=float((~land).mean()), liquid=liquid, reach=reach))


def _at(col: float, row: float) -> tuple[float, float]:
    return (col + 0.5) * bd.PASS_CELL, (row + 0.5) * bd.PASS_CELL


def _two_islands() -> np.ndarray:
    land = np.zeros((FLAT_CELLS, FLAT_CELLS), bool)
    land[:, :12] = land[:, -12:] = True
    return land


def test_land_units_dont_suit_islands_they_cant_leave():
    ground = _ground(_two_islands(), [_at(c, 32) for c in (5, 30, 58)])
    a, b = _at(5, 32), _at(58, 32)
    assert not ground.suits(ANYWHERE, KBOT, a, b)
    assert ground.suits(ANYWHERE, HOVER, a, b) and ground.suits(ANYWHERE, AMPH, a, b)
    assert ground.suits(ANYWHERE, tables.NO_CLASS, a, b)


def test_water_that_is_only_decoration_changes_nothing():
    land = np.zeros((FLAT_CELLS, FLAT_CELLS), bool)
    land[:, :40] = True
    ground = _ground(land, [_at(c, 32) for c in (5, 15, 25, 35)])
    a, b = _at(5, 10), _at(35, 50)
    assert ground.facts.water > 0.3
    assert ground.suits(ANYWHERE, KBOT, a, b) and ground.suits(ANYWHERE, TANK, a, b) and ground.suits(ANYWHERE, HOVER, a, b)


def test_a_class_that_gets_to_little_of_the_metal_doesnt_suit():
    """Joined up, but most of the metal is under the sea."""
    land = np.zeros((FLAT_CELLS, FLAT_CELLS), bool)
    land[:12, :] = True
    ground = _ground(land, [_at(5, 5)] + [_at(c, 40) for c in range(10, 50, 5)])
    a, b = _at(5, 5), _at(58, 5)
    assert ground.joined(KBOT, a, b)
    assert ground.metal_reach(KBOT, a) < 0.2 and ground.metal_reach(HOVER, a) == 1.0
    assert not ground.suits(ANYWHERE, KBOT, a, b) and ground.suits(ANYWHERE, HOVER, a, b)


def test_tanks_that_start_on_an_island_of_their_own_dont_suit():
    """What the map's own labels are for: kbots can go a way that tanks can't."""
    land = np.ones((FLAT_CELLS, FLAT_CELLS), bool)
    tanks = np.full(land.shape, 1, np.int16)
    tanks[:, :20] = 0
    tanks[:, 20:24] = -1
    kbots = np.zeros(land.shape, np.int16)
    reach = {"tank": tanks, "kbot": kbots}
    ground = _ground(land, [_at(c, 32) for c in (5, 30, 50, 60)], reach)
    ground.bundle.passable[:, 20:24] &= ~(1 << TANK)
    a, b = _at(5, 32), _at(60, 32)
    assert not ground.suits(ANYWHERE, TANK, a, b) and ground.suits(ANYWHERE, KBOT, a, b)
    assert ground.metal_reach(TANK, a) == 0.25
    assert ground.suits(ANYWHERE, TANK, b, _at(50, 5))      # those on the larger part are fine


def test_ships_want_a_real_sea_with_both_bases_on_it():
    deep = fw.Footing(32.0, 15.0, 10000.0)
    ground = _ground(_two_islands(), [_at(5, 32), _at(58, 32)])
    assert ground.suits(deep, SHIP) and ground.suits(deep, SHIP, _at(10, 32), _at(53, 32))
    pond = np.ones((FLAT_CELLS, FLAT_CELLS), bool)
    pond[28:36, 28:36] = False
    assert not _ground(pond, [_at(5, 32)]).suits(deep, SHIP)
    # A base a long way inland has no use for one.
    shore = np.ones((256, 256), bool)
    shore[:, 200:] = False
    ground = _ground(shore, [_at(5, 32)])
    assert not ground.suits(deep, SHIP, _at(5, 128), _at(190, 128))
    assert ground.suits(deep, SHIP, _at(195, 20), _at(195, 230))


def test_what_comes_without_labels_is_labelled_from_where_it_can_go():
    """The training maps have none on file, and all cells of a class aren't one piece."""
    ground = _ground(_two_islands(), [_at(5, 32)])
    labels = ground.labels(KBOT)
    assert labels[32, 5] >= 0 and labels[32, 58] >= 0 and labels[32, 5] != labels[32, 58] and labels[32, 30] == -1


def test_no_metal_on_a_map_rules_out_nothing():
    ground = _ground(np.ones((FLAT_CELLS, FLAT_CELLS), bool), [])
    assert ground.suits(ANYWHERE, KBOT, _at(5, 5), _at(50, 50))


def test_seats_only_take_factories_that_suit_their_start(pool):
    env = se.SimEnv(16, 11, ["scripts"], threads=2, maps=_wet(pool))
    asked, choices = [], env.opening.choices

    def noted(*args):
        asked.append(choices(*args)[0])
        return choices(*args)

    env.opening.choices = noted
    env.begin()
    seats = [seat for pair in env.seats for seat in pair]
    assert len(asked) == len(seats) == 32
    for chosen, seat in zip(asked, seats):
        assert not chosen or next(k for k, (f, _) in enumerate(env.opening.factories) if f == seat.factory_type) in chosen


def test_every_start_of_the_training_maps_has_a_ground_factory():
    env = se.SimEnv(1, 3, ["scripts"], threads=1)
    for _, bundle, facts in se.pool():
        ground = env.grounds[id(bundle)]
        for boxes in facts.layouts:
            assert len(env.opening.choices(ground, boxes[0].point, boxes[1].point)[0]) >= 4


def test_training_doesnt_favour_water_but_frames_do(pool, roster):
    _, bundle, facts = _wet(pool)[0]
    ground = fw.Ground(bundle, facts)
    boxes = facts.layouts[0]
    plain = se.Opening(se.catalog()).choices(ground, boxes[0].point, boxes[1].point)[1]
    assert len(set(np.round(plain, 6).tolist())) == 1
    assert fw.WideOpening.favour_water


# Jumping in frames ------------------------------------------------------------------------------------------------------------------------------------------


def _pyro(flags: int, x: float, z: float) -> np.ndarray:
    unit = np.zeros(1, rf.UNIT_DTYPE)
    unit["uid"], unit["def_id"], unit["hp_frac"], unit["flags"] = 3, se.catalog().units[se.catalog().index["jumpraid"]]["def_id"], 1.0, flags
    unit["x"], unit["z"] = x, z
    return unit


def test_in_frames_what_is_charged_jumps_towards_where_it_was_going(pool, roster):
    _, bundle, facts = _dry(pool)[0]
    ground = fw.Ground(bundle, facts)
    data = fw.fight_data(roster.table, se.catalog())
    a, b = facts.layouts[0][0].point, facts.layouts[0][1].point
    script = fw.FrameFight(data, b, b, ground=ground)
    out = script.orders(_pyro(rf.IN_LOS | rf.JUMP_READY, *a), np.zeros(0, rf.UNIT_DTYPE), 700.0)
    assert out["kind"][0] == rf.JUMP
    assert np.hypot(out["x"][0] - a[0], out["z"][0] - a[1]) <= roster.table.jump_range[_row(roster, "jumpraid")]
    out = fw.FrameFight(data, b, b, ground=ground).orders(_pyro(rf.IN_LOS, *a), np.zeros(0, rf.UNIT_DTYPE), 700.0)
    assert out["kind"][0] != rf.JUMP
