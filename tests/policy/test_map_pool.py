"""Made-up maps, and the frames that use them: do they hold together, and can things stand where they're put?"""
from __future__ import annotations

import json
import pathlib
import shutil
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
    import map_modes
    import map_rampart as mr
    import policy_opening as po
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
def made_maps():
    """Two dry maps and two nearly all water, by seed."""
    return {seed: mg.make(seed, rules) for seed, rules in ((1, DRY), (2, DRY), (3, ISLANDS), (4, ISLANDS))}


@pytest.fixture(scope="module")
def pool_dir(made_maps, tmp_path_factory):
    """A small pool: those four."""
    out = tmp_path_factory.mktemp("maps")
    for made in made_maps.values():
        mg.write_entry(made.map, "made", made.layouts, made.features, out, made.style)
    return out


@pytest.fixture(scope="module")
def pool(pool_dir):
    return se.load_pool(pool_dir)


def _wet(pool):
    return [m for m in pool if m[2].water > 0.3]


def _dry(pool):
    return [m for m in pool if m[2].water == 0.0]


# Maps -----------------------------------------------------------------------------------------------------------------------------------------------------


def test_a_seed_makes_one_map(made_maps):
    a, b = made_maps[1], mg.make(1, DRY)
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


@pytest.mark.parametrize("seed", (1, 2, 10, 11))
def test_a_walker_can_get_from_start_to_start_on_a_map_meant_for_walkers(made_maps, seed):
    made = made_maps[seed] if seed in made_maps else mg.make(seed, DRY)
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


def test_spots_are_on_the_map_and_some_are_under_water_on_a_map_of_water(made_maps):
    made = made_maps[3]
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


# Kinds of game ----------------------------------------------------------------------------------------------------------------------------------------------


def _modes(one: bool, teams: int, ffa: int, chickens: bool = False) -> dict:
    return {"1v1": one, "teams": teams, "ffa": ffa, "chickens": chickens, "chickens_or_teams": chickens or teams > 0}


def _of(boxes: int, area: float = 0.0, listing=None, names=("some map",), points: int = 1, variant: bool = False) -> dict:
    return map_modes.modes(map_modes.Known(boxes, points, area, None if listing is None else tuple(listing), tuple(names), variant))


def test_a_map_zero_k_lists_holds_the_team_games_and_free_for_alls_it_is_listed_for_as_far_as_its_boxes_go():
    assert _of(4, 196.0, ["1v1", "teams", "ffa (up to 4)"]) == _modes(True, 4, 4)
    assert _of(5, 400.0, ["ffa (up to 11)"]) == _modes(True, 0, 5)
    assert _of(9, 400.0, ["ffa (up to 11)"])["ffa"] == 8          # the sim's most teams
    # Listed for a free-for-all, but the game its facts came from made two boxes: a free-for-all of two is a 1v1.
    assert _of(2, 400.0, ["ffa (up to 9)"]) == _modes(True, 0, 0)


def test_any_playable_map_with_two_boxes_or_more_is_a_1v1_map_whatever_zero_k_lists_it_for():
    assert _of(2, 256.0, ["teams"]) == _modes(True, 4, 0)
    assert _of(6, 256.0, ["ffa (up to 6)"])["1v1"] and not _of(1, 256.0, ["1v1"])["1v1"]
    # But not the many variants a generator makes of one map, which would crowd out the rest.
    assert not _of(5, 576.0, ["ffa (up to 11)"], variant=True)["1v1"]


def test_a_map_zero_k_doesnt_list_is_a_team_map_with_two_boxes_and_a_free_for_all_with_more():
    assert _of(2, 100.0) == _of(2, 100.0, []) == _modes(True, 2, 0)
    assert _of(4, 256.0) == _modes(True, 0, 4)
    assert _of(1) == _modes(False, 0, 0)


def test_a_team_map_holds_more_a_side_the_larger_it_is_and_never_more_than_four():
    assert [map_modes.team_size(side * side) for side in (6, 8, 10, 12, 14, 24)] == [2, 2, 2, 3, 4, 4]
    assert map_modes.team_size(100.0, points=3) == 3     # its boxes list a start point for each of three


def test_a_map_named_for_this_space_or_zero_wars_plays_nothing():
    for name in ("THISspace", "thisspace_v2", "ZeroWars Crossfire 1.36.9", "zerowars_crossfire_1.36.9"):
        assert _of(4, 256.0, ["1v1", "teams"], names=("x", name)) == _modes(False, 0, 0)
    assert map_modes.unplayable("ZEROWARS 2") and not map_modes.unplayable("Zero Wars", "Space This")


def test_a_map_named_for_chickens_or_made_for_them_plays_chicken_games_only_whatever_its_boxes():
    chickens = _modes(False, 0, 0, True)
    assert _of(2, 96.0, ["chickens"]) == _of(4, 96.0, None, names=("chicken_nuggets_v5",)) == chickens
    assert _of(1, 96.0, ["1v1", "teams"], names=("x", "Chicken Coop v2")) == chickens
    # Listed for chickens among other things, it plays those too.
    assert _of(2, 196.0, ["1v1", "teams", "chickens"]) == {**_modes(True, 4, 0), "chickens": True}


def test_chicken_games_may_go_on_team_maps_by_the_separate_fact():
    team = _of(2, 196.0, ["1v1", "teams"])
    assert not team["chickens"] and team["chickens_or_teams"]
    assert not _of(2, 196.0, ["1v1"])["chickens_or_teams"]


def test_a_facts_file_goes_by_zero_k_s_listing_found_by_its_archive_s_name_or_its_generator_s():
    table = map_modes.catalog()
    for m in json.loads((ROOT / "maps.json").read_text("utf8")):
        assert map_modes.lookup(pathlib.Path(m["File"]).stem, table) == (m["Name"], m["Modes"]) == map_modes.lookup(m["Name"], table)
    box = {"polygons": [], "startpoints": [[0.0, 0.0]]}
    assert map_modes.known({"name": "Chicken_Nuggets_v5", "kind": "real", "layouts": [[box, box]]}, 96.0, table).listing == ("chickens",)
    rampart = map_modes.known({"name": "violet_rampart_5_bases_seed_1", "kind": "real", "layouts": [[box] * 5], "style": {"generator": mr.NAME}}, 576.0, table)
    assert rampart.listing == ("ffa (up to 11)",) and rampart.variant
    assert map_modes.known({"name": "gen_00001", "kind": "made", "layouts": [[box, box]]}, 64.0, table).listing is None


def test_a_start_box_keeps_every_start_point_it_lists_with_the_first_as_its_point():
    square = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    a, b = se._layouts({"layouts": [[{"polygons": [square], "startpoints": [[10, 10], [50, 50], [90, 90]]},
                                     {"polygons": [square], "startpoints": [[60, 60]]}]]})[0]
    assert a.point == (10.0, 10.0) and a.points == ((10.0, 10.0), (50.0, 50.0), (90.0, 90.0)) and b.points == (b.point,)
    assert se.MapFacts([[a, b]], np.zeros((0, 6))).modes == _modes(True, 3, 0)


def test_written_maps_say_their_kinds_of_game_and_facts_without_them_read_the_same_from_their_boxes(pool_dir, tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    for path in pool_dir.iterdir():
        shutil.copy(path, old / path.name)
        if path.name.endswith(".facts.json"):
            facts = json.loads(path.read_text("utf8"))
            assert facts["modes"] == _of(len(facts["layouts"][0]), 64.0, names=(facts["name"],))      # 8 x 8, so 2 a side
            del facts["modes"]
            (old / path.name).write_text(json.dumps(facts), "utf8")
    derived = {f.name: f.modes for _, _, f in se.load_pool(old)}
    mg.update_modes(old, tmp_path / "new")
    written = {p.name[:-len(".facts.json")]: json.loads(p.read_text("utf8"))["modes"] for p in (tmp_path / "new").glob("*.facts.json")}
    assert written == derived and len(written) == 4
    assert all(f.modes == derived[f.name] for _, _, f in se.load_pool(pool_dir))


def test_the_training_maps_without_modes_on_file_are_1v1_and_two_a_side_maps():
    assert all(f.modes == _modes(True, 2, 0) for _, _, f in se.pool())


# Violet Rampart ---------------------------------------------------------------------------------------------------------------------------------------------


def test_violet_rampart_makes_one_map_for_a_seed_and_a_number_of_bases():
    a, b = mr.generate(5, 314159), mr.generate(5, 314159)
    assert np.array_equal(a.corners, b.corners) and a.spots == b.spots and a.boxes == b.boxes
    assert not np.array_equal(a.corners, mr.generate(5, 271828).corners)


@pytest.mark.parametrize("bases", (3, 4, 8))
def test_violet_rampart_gives_each_base_its_spots_and_a_level_start_in_its_box_that_walks_to_every_other(bases):
    r = mr.generate(bases, 271828)
    assert len(r.spots) == 11 * bases and len(r.boxes) == bases
    assert sorted(m for _, _, m in r.spots[:11]) == [1.5] * 5 + [2.0] * 5 + [3.0]
    ground = mg.Ground.of(r.corners.astype(np.float32), mr.WATER_DAMAGE, False)
    can = ground.can("kbot")
    labels, _ = tr.label_runs(can)
    parts = {int(labels[mg._cell(*start, can.shape)]) for _, start in r.boxes}
    assert len(parts) == 1 and -1 not in parts
    assert all(int(labels[mg._cell(x, z, can.shape)]) in parts for x, z, _ in r.spots)
    for box, start in r.boxes:
        assert mg.tm_buildable(ground, *start) and se._inside([list(p) for p in box], *start)
        assert ground.height[mg._cell(*start, can.shape)] == pytest.approx(mr.RAMPART)


def test_violet_rampart_with_four_bases_is_its_own_quarter_turn():
    r = mr.generate(4, 161803)
    assert (np.rot90(r.corners) != r.corners).mean() < 0.001        # a corner right on an edge may fall either way
    spots = np.array(r.spots)
    c = mr.MAP_SIZE / 2
    turned = np.stack((c - (spots[:, 1] - c), c + (spots[:, 0] - c)), 1)       # a quarter turn about the centre
    gaps = np.hypot(turned[:, None, 0] - spots[None, :, 0], turned[:, None, 1] - spots[None, :, 1]).min(1)
    assert gaps.max() <= 23.0       # spots are put on building centres, 16 apart: a tie in each may round either way


def test_violet_rampart_is_a_free_for_all_of_its_bases_and_replaces_the_pool_s_old_map():
    box = {"polygons": [], "startpoints": [[0.0, 0.0]]}
    for bases in (3, 8):
        facts = {"name": f"violet_rampart_{bases}_bases_seed_1", "kind": "real", "layouts": [[box] * bases], "style": {"generator": mr.NAME}}
        assert map_modes.of_facts(facts, 576.0, map_modes.catalog()) == _modes(False, 0, bases)
    assert mg.REPLACED == ("violet_rampart_generator_v1_4",)


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
    place = ground.site(np.random.default_rng(1), footing, *box.point, np.zeros((0, 4), np.float32))
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
    seats = [g * env.slots + t for g in range(env.count) for t in range(int(env.n_teams[g])) if t != env.chickens[g]]
    for seat in seats:
        frame = env.frame(seat)
        u = frame.units
        rows = env.table.rows(u["def_id"])
        assert not ((u["knowledge"] == rf.KNOW_MEMORY) & env.table.mobile[rows]).any()
        at_the_start = u["uid"] < len(env.scouted[seat // env.slots][seat % env.slots])    # and not what's been begun since
        assert not ((u["knowledge"] == rf.KNOW_HIDDEN) & ~env.table.mobile[rows] & at_the_start).any()      # the whole base was seen
        memory += int((u["knowledge"] == rf.KNOW_MEMORY).sum())
    assert memory > 0
    never = fd.ScenarioEnv(8, 6, fd.ScenarioRules(side_value=(8000.0, 8000.0), scouted=(0.0, 0.0)), threads=2, maps=pool, weights=fw.pool_weights(pool))
    never.begin()
    never.step(0.0)
    seats = [g * never.slots + t for g in range(never.count) for t in range(int(never.n_teams[g])) if t != never.chickens[g]]
    assert sum(int((never.frame(s).units["knowledge"] == rf.KNOW_MEMORY).sum()) for s in seats) < memory / 4     # what the sim itself remembers, of a first moment's glimpse


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


# Where the game lets a structure be built ----------------------------------------------------------------------------------------------------------------


FACTORY = fw.Footing(56.0, -10000.0, 0.0, 56.0, 80.0, 10.72)    # a Cloakbot Factory's: 112 by 160 elmos, maxSlope 15
SOLAR = fw.Footing(40.0, -10000.0, 0.0, 40.0, 40.0, 13.0)


def _slope(rise: float, base: float = 0.0) -> fw.Ground:
    """Land all over, this much higher each cell towards +x from `base`."""
    ground = _ground(np.ones((FLAT_CELLS, FLAT_CELLS), bool), [])
    ground._height = np.ascontiguousarray(np.tile(base + rise * np.arange(FLAT_CELLS, dtype=np.float32), (FLAT_CELLS, 1)))
    return ground


def test_the_catalog_says_each_footprint_and_how_level_it_wants_the_ground():
    cat = se.catalog()
    factory, solar = (fw.Footing.of(cat.units[cat.index[name]]) for name in ("factorycloak", "energysolar"))
    assert (factory.half_x, factory.half_z, factory.radius) == (56.0, 80.0, 56.0)     # footprint 7 x 10, of 16 elmos
    assert factory.height_dif == pytest.approx(40.0 * np.tan(np.radians(15.0)), abs=1e-3)      # the engine's maxHeightDif for maxSlope 15
    assert (solar.half_x, solar.half_z) == (40.0, 40.0) and solar.height_dif > factory.height_dif
    assert fw.Footing(32.0, 0.0, 0.0).half_x == fw.Footing(32.0, 0.0, 0.0).half_z == 32.0          # the radius both ways, if not said


def test_a_build_is_tested_against_its_footprints_rectangle():
    ground = _slope(0.0)
    size = FLAT_CELLS * bd.PASS_CELL
    # At the map's edge: 56 from it along x is room, and along z, where it's 80 deep, isn't.
    assert ground.fits(60.0, 500.0, FACTORY) and not ground.fits(500.0, 60.0, FACTORY) and ground.fits(500.0, 80.0, FACTORY)
    assert not ground.fits(size - 50.0, 500.0, FACTORY) and ground.fits(size - 56.0, 500.0, FACTORY)
    # Beside another: footprints may touch, and the factory's is deeper than its circle.
    taken = fw.footprints(500.0, 500.0, FACTORY.half_x, FACTORY.half_z)
    apart = lambda x, z: po.apart(taken, x, z, 24.0, 24.0)  # noqa: E731
    assert apart(500.0 + 56.0 + 24.0, 500.0) and not apart(500.0 + 79.0, 500.0)
    assert not apart(500.0, 500.0 + 56.0 + 24.0), "80 along z is inside the footprint, though clear of a 56 circle"
    assert apart(500.0, 500.0 + 80.0 + 24.0) and apart(500.0 + 80.0, 500.0 + 60.0)
    assert po.apart(np.zeros((0, 4), np.float32), 1.0, 1.0, 24.0, 24.0)


def test_a_build_needs_ground_within_its_height_limit_of_the_ground_at_its_middle():
    flat, slope = _slope(0.0), _slope(4.0)
    assert flat.fits(504.0, 504.0, FACTORY) and flat.fits(504.0, 504.0, SOLAR)
    # 4 a cell: the factory's footprint reaches 4 cells from its middle's, 16 up, past its 10.72; the solar's 3, 12 of its 13.
    assert not slope.fits(504.0, 504.0, FACTORY) and slope.fits(504.0, 504.0, SOLAR)
    assert not _slope(-4.0, 1000.0).fits(504.0, 504.0, FACTORY), "down as much as up"
    assert slope.fits(504.0, 504.0, fw.Footing(56.0, -10000.0, 0.0, 56.0, 80.0)), "no limit in its data, none here"
    # Nowhere on a slope too steep for it, so `site` finds none; on one it can take, somewhere.
    none = np.zeros((0, 4), np.float32)
    assert slope.site(np.random.default_rng(1), FACTORY, 504.0, 504.0, none) is None
    place = _slope(2.0).site(np.random.default_rng(1), FACTORY, 504.0, 504.0, none)
    assert place is not None and _slope(2.0).fits(*place, FACTORY)
    # One whose middle is under water floats or sits on the sea bed: the ground there isn't asked.
    sea = _ground(np.zeros((FLAT_CELLS, FLAT_CELLS), bool), [])
    sea._height = np.ascontiguousarray(np.tile(-100.0 - 8.0 * np.arange(FLAT_CELLS, dtype=np.float32), (FLAT_CELLS, 1)))
    assert sea.fits(504.0, 504.0, fw.Footing(64.0, 15.0, 10000.0, 64.0, 96.0, 10.72))


def test_a_ground_judges_level_by_the_sim_maps_heights_when_given_them(pool):
    world_map, bundle, facts = _dry(pool)[0]
    assert world_map.height.shape == bundle.passable.shape, "the sim map's cells are the passability cells"
    fine, coarse = fw.Ground(bundle, facts, world_map.height), fw.Ground(bundle, facts)
    assert fine.heights is not coarse.heights and np.shares_memory(fine.heights, world_map.height)
    assert coarse.heights.shape == fine.heights.shape and np.abs(coarse.heights - fine.heights).mean() < 20.0


def test_the_economy_script_builds_only_where_the_game_would_let_it():
    """libzkscripts' own test of a site (Ground::fits) against this one, on what it builds in whole games."""
    env = se.SimEnv(4, 7, ["scripts"], threads=2)
    env.begin()
    ed = env.eco_data
    row_of = {int(c): r for r, c in enumerate(ed.catalog_row) if c >= 0}
    checked, big = 0, 0
    for _ in range(900):
        env.step(0.0)
        orders, offsets = env.eco_orders
        for seat in range(env.seat_count):
            for o in orders[offsets[seat]:offsets[seat + 1]]:
                row = row_of.get(int(o["target"]), -1) if o["kind"] == zksim.ORDER_BUILD else -1
                if row < 0 or ed.extractor[row]:
                    continue
                ground, footing = env.grounds[id(env.bundle_of(seat))], ed.footing[row]
                assert ground.fits(float(o["x"]), float(o["z"]), footing), env.table.names[row]
                spots = np.asarray(ground.bundle.spot_list, np.float32).reshape(-1, 3)
                assert po.apart(fw.footprints(spots[:, 0], spots[:, 1], ed.spot_half, ed.spot_half), float(o["x"]), float(o["z"]), footing.half_x, footing.half_z)
                checked += 1
                big += footing.half_z > footing.half_x
        env.finished_seats()
    assert checked > 40 and big > 0, (checked, big)


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
