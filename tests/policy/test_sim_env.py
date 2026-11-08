"""The sim as an environment: which maps it plays, and when a match ends."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "tools"))
try:
    import policy_sim_env as se
    import policy_opening as po
    import zksim
    from zkbot.policy import featurize as fz, rawframe as rf
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)


def _env_with_values(a: float, b: float, decide_every: int = 1) -> se.SimEnv:
    """Both sides stay alive at these metal values, whatever the sim does."""
    env = se.SimEnv(1, 3, ["mirror"], units=2, threads=1, decide_every=decide_every)
    env.begin()
    real = env.batch.step

    def step(*args, **kwargs):
        real(*args, **kwargs)
        env.batch.stats[0] = ((a, 5.0), (b, 5.0))
    env.batch.step = step
    return env


def _steps_to_end(env: se.SimEnv, most: int) -> tuple[int, list]:
    for n in range(1, most + 1):
        env.step(0.0)
        if env.done:
            return n, env.done
    return most + 1, []


def test_training_plays_only_its_maps():
    assert len(se.pool()) == len(se.TRAINING_MAPS)
    assert max(b.regions for _, b, _ in se.pool()) <= 35


@pytest.mark.parametrize("a, b, seat_a, seat_b", ((16000.0, 6000.0, 1.0, 0.0), (6000.0, 16000.0, 0.0, 1.0)))
def test_a_side_far_ahead_wins_at_two_minutes(a, b, seat_a, seat_b):
    env = _env_with_values(a, b)
    n, done = _steps_to_end(env, se.EARLY_END_DECISIONS)
    assert n == se.EARLY_END_DECISIONS
    assert sorted(done) == [(0, "mirror", seat_a), (1, "mirror", seat_b)]
    assert env.result[0] == (1.0 if a > b else -1.0)


@pytest.mark.parametrize("a, b", ((2600.0, 1000.0), (15000.0, 6000.0), (24000.0, 14000.5)))
def test_a_lead_short_of_the_ratio_or_the_metal_plays_on(a, b):
    env = _env_with_values(a, b)
    assert _steps_to_end(env, se.EARLY_END_DECISIONS + 5)[1] == []


def test_exactly_the_metal_lead_is_enough():
    env = _env_with_values(10000.0, 0.0)
    assert _steps_to_end(env, se.EARLY_END_DECISIONS)[0] == se.EARLY_END_DECISIONS


def test_two_minutes_stay_two_minutes_at_a_slower_decision_rate():
    env = _env_with_values(16000.0, 6000.0, decide_every=2)
    assert _steps_to_end(env, se.EARLY_END_DECISIONS)[0] == se.EARLY_END_DECISIONS // 2


def _sides(env: se.SimEnv):
    for world_map, bundle, facts in se.pool():
        for boxes in facts.layouts:
            yield bundle, boxes[0].point, boxes[1].point
            yield bundle, boxes[1].point, boxes[0].point


def _in_box(box: se.StartBox, x: float, z: float) -> bool:
    """A start point can be a corner of its box, as FrostyCove's are."""
    return (x, z) == box.point or any(se._inside(polygon, x, z) for polygon in box.polygons)


def test_every_training_map_has_boxes_and_all_the_metal_the_game_reports():
    facts = json.loads((zksim.DATA / "map_facts.json").read_text(encoding="utf8"))
    for slug, (_, bundle, known) in zip(se.TRAINING_MAPS, se.pool()):
        assert known.layouts and all(len(boxes) >= 2 for boxes in known.layouts)
        assert np.allclose(bundle.spot_list, np.array(facts[slug]["spots"], np.float32)), slug
        for box in (box for boxes in known.layouts for box in boxes):
            assert min(np.hypot(*(np.array(v) - box.point)) for polygon in box.polygons for v in polygon) < 1.0 or _in_box(box, *box.point)
            for polygon in box.polygons:
                assert all(0.0 <= x <= bundle.map_w and 0.0 <= z <= bundle.map_h for x, z in polygon)


def test_a_start_is_in_its_box_on_ground_a_factory_could_use():
    rng = np.random.default_rng(1)
    for _, bundle, facts in se.pool():
        for box in (box for boxes in facts.layouts for box in boxes):
            picks = {box.pick(rng, bundle) for _ in range(120)}
            assert box.point in picks and len(picks) > 10, bundle.map_w
            for x, z in picks - {box.point}:
                assert _in_box(box, x, z)
                assert po.is_buildable(bundle, x, z, se.START_ROOM)
                assert x % se.START_GRID == 0.0 and z % se.START_GRID == 0.0


def _near_box(box: se.StartBox, x: float, z: float) -> bool:
    """The env's starts are 32-bit floats by now."""
    return np.hypot(x - box.point[0], z - box.point[1]) < 1.0 or any(se._inside(polygon, x, z) for polygon in box.polygons)


def test_games_start_in_different_boxes_and_the_script_goes_for_the_enemys_start():
    env = se.SimEnv(16, 11, ["scripts"], threads=2)
    env.begin()
    tables = {id(b): facts for _, b, facts in se.pool()}
    for g in range(16):
        x0, z0, x1, z1 = env.starts[g]
        # In one of the layouts its map can draw.
        for boxes in tables[id(env.bundle[g])].layouts:
            own = [i for i, box in enumerate(boxes) if _near_box(box, float(x0), float(z0))]
            other = [i for i, box in enumerate(boxes) if _near_box(box, float(x1), float(z1))]
            if own and other and own != other:
                break
        else:
            raise AssertionError(g)


def test_a_map_that_draws_its_boxes_has_each_draw_and_they_all_get_played():
    env = se.SimEnv(1, 3, ["scripts"], threads=1)
    faultline = next(facts for slug, (_, _, facts) in zip(se.TRAINING_MAPS, se.pool()) if slug.startswith("faultline"))
    assert len(faultline.layouts) > 1
    seen = set()
    for _ in range(60):
        env.begin()
        if env.bundle[0] is next(b for slug, (_, b, _) in zip(se.TRAINING_MAPS, se.pool()) if slug.startswith("faultline")):
            x0, z0, x1, z1 = env.starts[0]
            seen.add(next(i for i, boxes in enumerate(faultline.layouts) if any(_near_box(box, float(x0), float(z0)) for box in boxes)))
    assert len(seen) > 1


def test_what_lies_on_a_map_from_the_start_is_there_to_see():
    env = se.SimEnv(1, 3, ["scripts"], threads=1)
    se_pool = se.pool()
    cove = next(i for i, slug in enumerate(se.TRAINING_MAPS) if slug.startswith("frostycove"))
    assert 100 < len(se_pool[cove][2].features) and se_pool[cove][2].features[:, 2].min() >= 1.0
    keep = se_pool[cove]
    se._POOL[:] = [keep]
    try:
        env.begin()
        seen = env.frame(0).features
        assert len(seen) and set(seen["metal"].tolist()) <= set(keep[2].features[:, 2].tolist())
    finally:
        se._POOL.clear()


def test_a_plan_on_a_box_is_kept_and_not_redone():
    env = se.SimEnv(1, 3, ["scripts"], threads=1)
    _, bundle, facts = se.pool()[0]
    boxes = facts.layouts[0]
    assert env.opening.plan(bundle, boxes[0].point, boxes[1].point) is env.opening.plan(bundle, boxes[0].point, boxes[1].point)


def test_every_map_has_a_plan_with_a_solar_beside_each_extractor():
    env = se.SimEnv(1, 3, ["scripts"], threads=1)
    op, cat = env.opening, se.catalog()
    mex_r, solar_r = cat.units[op.mex]["footprint_radius"], cat.units[op.solar]["footprint_radius"]
    for bundle, start, enemy in _sides(env):
        plan = op.plan(bundle, start, enemy)
        mexes = [b for b in plan.builds if b.type == op.mex]
        solars = [b for b in plan.builds if b.type == op.solar]
        assert 3 <= len(mexes) <= 6 and len(solars) == len(mexes), bundle.map_w
        for m, s in zip(mexes, solars):
            assert np.hypot(m.x - s.x, m.z - s.z) <= po.LINK_REACH
        spots = bundle.spot_list
        assert all(np.min(np.hypot(spots[:, 0] - m.x, spots[:, 1] - m.z)) < 1.0 for m in mexes)
        # Nothing is planned on a spot it doesn't use, or on the factory.
        assert np.min(np.hypot(spots[:, 0] - plan.factory[0], spots[:, 1] - plan.factory[1])) > op.factory_radius
        for s in solars:
            assert np.min(np.hypot(spots[:, 0] - s.x, spots[:, 1] - s.z)) >= mex_r + solar_r
            assert np.hypot(s.x - plan.factory[0], s.z - plan.factory[1]) >= op.factory_radius + solar_r


def test_each_side_only_takes_spots_on_its_own_side():
    env = se.SimEnv(1, 3, ["scripts"], threads=1)
    for bundle, start, enemy in _sides(env):
        for b in env.opening.plan(bundle, start, enemy).builds:
            if b.type == env.opening.mex:
                assert np.hypot(b.x - start[0], b.z - start[1]) <= np.hypot(b.x - enemy[0], b.z - enemy[1])


def _play(env: se.SimEnv, decisions: int) -> None:
    env.begin()
    for _ in range(decisions):
        env.step(0.0)
        env.finished_seats()


def test_the_opening_gets_through_its_plan():
    env = se.SimEnv(24, 5, ["scripts"], threads=4)
    _play(env, 200)
    live = [(g, s) for g in range(24) for s in (0, 1) if env.batch.records[g, s, 0]["uid"] == s]
    assert len(live) >= 40
    for g, s in live:
        seat = env.seats[g][s]
        units = env.batch.records[g, s, :env.batch.counts[g, s]]
        own = units[units["player"] == s]
        assert seat.factory_uid >= 0, (g, s)
        assert own[0]["ord_kind"] in (rf.BUILD, rf.ASSIST, -1), (g, s)


def bundle_spot_income(bundle: bd.Bundle, x: float, z: float) -> float:
    return bundle.spot_list[np.argmin(np.hypot(bundle.spot_list[:, 0] - x, bundle.spot_list[:, 1] - z)), 2]


def test_the_commander_finishes_on_the_factory_and_income_shows_the_plan():
    # The old script: the army's raids take extractors early, and this test is about the opening's income, not the raids.
    env = se.SimEnv(16, 7, ["scripts"], threads=4, scripts=("fight", "fight"))
    env.begin()
    for _ in range(420):                                # 3.5 minutes, early enough that few games have ended
        env.step(0.0)
        env.finished_seats()
    checked = 0
    for g in range(16):
        for s in (0, 1):
            seat = env.seats[g][s]
            units = env.batch.records[g, s, :env.batch.counts[g, s]]
            own = units[(units["player"] == s) & ((units["flags"] & rf.BEING_BUILT) == 0)]
            if env.decisions[g] < 400 or not len(own) or own["uid"][0] != s or not (own["uid"] == seat.factory_uid).any():
                continue
            extractors = own[own["def_id"] == se.catalog().units[env.opening.mex]["def_id"]]
            mex = len(extractors)
            planned = sum(b.type == env.opening.mex for b in seat.plan.builds)
            # A stunned extractor earns nothing, and some raiders stun.
            if seat.done and mex == planned and not (extractors["flags"] & rf.PARALYZED).any():
                assert own[0]["ord_kind"] == rf.ASSIST
                base = 6.0 + sum(float(bundle_spot_income(env.bundle[g], b.x, b.z)) for b in seat.plan.builds if b.type == env.opening.mex)
                # Overdrive adds to the first and takes from the second, and each extractor has at least its solar's 2 to draw on.
                assert env.batch.globs[g, s, rf.G["metal_income"]] >= base - 1e-3
                assert 8.0 - 1e-3 <= env.batch.globs[g, s, rf.G["energy_income"]] <= 8.0 + 2.0 * planned + 1e-3
                checked += 1
    assert checked >= 16


def test_the_script_sends_raiders_and_leaves_the_commander_to_build():
    env = se.SimEnv(4, 9, ["scripts"], threads=2)
    _play(env, 60)
    raiders = 0
    for g in range(4):
        for s in (0, 1):
            units = env.batch.records[g, s, :env.batch.counts[g, s]]
            own = units[units["player"] == s]
            rows = env.table.rows(own["def_id"])
            fighters = own[env.table.combat[rows] & ((own["flags"] & rf.BEING_BUILT) == 0)]
            raiders += len(fighters)
            assert np.all(np.isin(fighters["ord_kind"], (rf.FIGHT, rf.MOVE, rf.ATTACK_UNIT, -1)))
            builders = own[env.table.builder[rows] & env.table.mobile[rows]]
            assert np.all(builders["ord_kind"] != rf.FIGHT)
    assert raiders > 0


def test_the_policy_is_given_its_fighters_and_factories_but_no_constructors():
    env = se.SimEnv(2, 11, ["mirror"], threads=2)
    _play(env, 80)
    seen = factories = 0
    for g in range(2):
        for s in (0, 1):
            units = env.batch.records[g, s, :env.batch.counts[g, s]]
            kept, controllable, _, _, _ = fz._select(units, env.table, s, s)
            own = units[kept[:controllable]]
            seen += controllable
            assert np.all(own["player"] == s) and np.all((own["flags"] & rf.BEING_BUILT) == 0)
            rows = env.table.rows(own["def_id"])
            factory = env.table.factory[rows]
            assert np.all(env.table.mobile[rows] & ~env.table.builder[rows] | factory)
            factories += int(factory.sum())
    assert seen > factories > 0


def test_a_policy_with_no_units_yet_still_has_a_frame_to_read():
    env = se.SimEnv(1, 13, ["mirror"], threads=1)
    env.begin()
    units = env.batch.records[0, 0, :env.batch.counts[0, 0]]
    assert len(units) >= 1 and fz._select(units, env.table, 0, 0)[1] == 0


def test_the_same_seed_plays_the_same_game_at_any_thread_count():
    hashes = []
    for threads in (1, 5):
        env = se.SimEnv(6, 21, ["scripts"], threads=threads)
        _play(env, 240)
        hashes.append(env.batch.frame_hash())
    assert hashes[0] == hashes[1]


def test_the_older_start_still_has_no_economy():
    env = se.SimEnv(1, 3, ["mirror"], units=2, threads=1)
    env.begin()
    assert env.seats[0] is None and env.batch.globs[0, 0, rf.G["metal"]] == 0.0 and env.batch.counts[0, 0] == 4


def test_self_play_takes_its_share_of_new_games_only():
    env = se.SimEnv(64, 3, ["fight"], threads=1)
    env.begin()
    assert set(env.mode) == {"fight"} and len(env.policy_seats_to_act()) == 64
    env.self_play_share = 0.5
    assert set(env.mode) == {"fight"}, "games under way stay as they are"
    for g in range(64):
        env._reset(g)
    mirrors = env.mode.count("mirror")
    assert 16 < mirrors < 48 and len(env.policy_seats_to_act()) == 64 + mirrors
    env.self_play_share = 1.0
    for g in range(64):
        env._reset(g)
    assert set(env.mode) == {"mirror"} and len(env.policy_seats_to_act()) == 128


def test_a_finished_game_is_noted_from_the_policys_side():
    env = se.SimEnv(1, 3, ["fight"], threads=1)
    env.begin()
    while not env.done:
        env.step(0.0)
    facts = env.ended[0]
    assert facts["map"] in se.TRAINING_MAPS and facts["result"] in ("won", "lost", "drew") and facts["how"] in ("eliminated", "lead", "limit")
    assert facts["seconds"] > 0.0 and facts["enemy_value"] >= 0.0
    # Ones the game has: the opening picks them from its catalog.
    assert all(facts[k] in se.catalog().index for k in ("own_factory", "enemy_factory"))


def test_nothing_on_either_side_is_no_lead():
    env = _env_with_values(0.0, 0.0)
    assert _steps_to_end(env, se.EARLY_END_DECISIONS + 5)[1] == []
    assert np.all(env.result == 0.0)
