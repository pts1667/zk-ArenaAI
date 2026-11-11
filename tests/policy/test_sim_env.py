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
    from zkbot.policy import bundle as bd, featurize as fz, rawframe as rf
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)


def _env_with_values(a: float, b: float, decide_every: int = 1, lead_share: float = 1.0) -> se.SimEnv:
    """Both sides stay alive at these metal values, whatever the sim does."""
    env = se.SimEnv(1, 3, ["mirror"], units=2, threads=1, decide_every=decide_every)
    env.lead_share = lead_share
    env.begin()
    real = env.batch.step

    def step(*args, **kwargs):
        real(*args, **kwargs)
        env.batch.stats[0, :, :2] = ((a, 5.0), (b, 5.0))
        env.batch.stats[0, :, se.KEEP_STAT] = 5.0       # in the game, whichever stat says so
    env.batch.step = step
    return env


def _steps_to_end(env: se.SimEnv, most: int) -> tuple[int, list]:
    for n in range(1, most + 1):
        env.step(0.0)
        if env.done:
            return n, env.done
    return most + 1, []


def test_shared_maps_are_used_and_play_as_each_env_making_its_own():
    """share_maps makes the maps' data once, for workers forked after to share: the same games as without."""
    shared, own = list(se.pool()), list(se.pool())
    se.share_maps(shared)
    try:
        a = se.SimEnv(4, 21, ["scripts"], threads=2, maps=shared, variants=True)
        b = se.SimEnv(4, 21, ["scripts"], threads=2, maps=own, variants=True)
        assert a.grounds is se._SHARED[id(shared)][1] and a.cpp_data is se._SHARED[id(shared)][2]
        assert b.grounds is not a.grounds and b.cpp_data is not a.cpp_data
        a.begin(), b.begin()
        for _ in range(400):
            a.step(0.0), b.step(0.0)
            assert (a.batch.counts == b.batch.counts).all() and a.batch.records.tobytes() == b.batch.records.tobytes()
    finally:
        del se._SHARED[id(shared)]


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


def test_a_game_drawn_without_the_lead_rule_plays_on_however_far_ahead_and_its_facts_say_so():
    env = _env_with_values(16000.0, 6000.0, lead_share=0.0)
    assert not env.lead_rule[0]
    assert _steps_to_end(env, se.EARLY_END_DECISIONS + 5)[1] == []
    env.time_limit = int(env.decisions[0]) + 1
    env.step(0.0)
    assert env.done and env.result[0] == 0.0 and env.ended[0]["how"] == "limit" and env.ended[0]["lead_rule"] is False
    won = _env_with_values(16000.0, 6000.0)
    _steps_to_end(won, se.EARLY_END_DECISIONS)
    assert won.ended[0]["how"] == "lead" and won.ended[0]["lead_rule"] is True


def test_the_lead_rule_is_drawn_for_each_new_game_at_its_share():
    env = se.SimEnv(64, 3, ["fight"], threads=1)
    env.begin()
    assert env.lead_rule.all(), "whole by default: tests, recording and evals"
    env.lead_share = 0.5
    assert env.lead_rule.all(), "games under way stay as they are"
    for g in range(64):
        env._reset(g)
    assert 16 < int(env.lead_rule.sum()) < 48


@pytest.mark.parametrize("beta", (0.0, 0.6))
def test_a_win_pays_a_share_of_the_time_left_on_top_unfaded_and_the_loser_pays_it(beta):
    """policy_sim_env._speed: at the lead's end at 2 minutes, speed_weight x the share of the time limit left, whatever beta,
    zero-sum; none for a draw."""
    env = _env_with_values(16000.0, 6000.0)
    env.speed_weight = 0.1
    _steps_to_end(env, se.EARLY_END_DECISIONS)
    with_speed = [env.reward(seat, beta) for seat in (0, 1)]
    env.speed_weight = 0.0
    without = [env.reward(seat, beta) for seat in (0, 1)]
    term = 0.1 * (1.0 - se.EARLY_END_DECISIONS / se.TIME_LIMIT_DECISIONS)
    assert with_speed[0] - without[0] == pytest.approx(term) and with_speed[1] - without[1] == pytest.approx(-term)
    assert with_speed[0] == pytest.approx(-with_speed[1])
    draw = _env_with_values(16000.0, 6000.0, lead_share=0.0)
    draw.speed_weight, draw.time_limit = 0.1, 10
    _steps_to_end(draw, 10)
    paid = draw.reward(0, beta)
    draw.speed_weight = 0.0
    assert draw.done and draw.result[0] == 0.0 and draw.reward(0, beta) == paid


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


def test_a_start_point_its_commander_cannot_stand_on_becomes_the_nearest_place_in_the_box_with_room():
    """Esker Creek's start points are on a ridge only spiders climb.  Here, Agalta's first with a ridge made round it."""
    import dataclasses
    _, bundle, facts = se.pool()[0]
    box, amph, spider = facts.layouts[0][0], se.bd.MOVE_CLASSES.index("amph"), se.bd.MOVE_CLASSES.index("spider")
    x, z = box.point
    assert box.at_point(bundle, box.point, amph) == box.point
    passable = bundle.passable.copy()
    i, j = int(z / se.bd.PASS_CELL), int(x / se.bd.PASS_CELL)
    passable[i - 6:i + 7, j - 6:j + 7] = 1 << spider         # about 100 elmos each way
    ridge = dataclasses.replace(bundle, passable=passable)
    moved = box.at_point(ridge, box.point, amph)
    assert _in_box(box, *moved) and po.is_buildable(ridge, *moved, se.START_ROOM)
    assert 100.0 < np.hypot(moved[0] - x, moved[1] - z) < 100.0 + se.START_ROOM + 2 * se.START_GRID      # just off the ridge
    assert box.at_point(ridge, box.point, spider) == box.point and box.at_point(ridge, box.point, None) == box.point
    rng = np.random.default_rng(1)
    assert moved in {box.pick(rng, ridge, amph) for _ in range(20)} and box.picks(rng, ridge, 1, amph) == [moved]


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
    assert env.opening.plan(env.grounds[id(bundle)],boxes[0].point, boxes[1].point) is env.opening.plan(env.grounds[id(bundle)],boxes[0].point, boxes[1].point)


def test_every_map_has_a_plan_with_a_solar_beside_each_extractor():
    env = se.SimEnv(1, 3, ["scripts"], threads=1)
    op, cat = env.opening, se.catalog()
    mex_r, solar_r = cat.units[op.mex]["footprint_radius"], cat.units[op.solar]["footprint_radius"]
    # Footprints are rectangles kept apart along x or along z, so centres are at least the shorter halves apart.
    factory_r = min(op.factory_footing.half_x, op.factory_footing.half_z)
    for bundle, start, enemy in _sides(env):
        plan = op.plan(env.grounds[id(bundle)], start, enemy)
        mexes = [b for b in plan.builds if b.type == op.mex]
        solars = [b for b in plan.builds if b.type == op.solar]
        assert 3 <= len(mexes) <= 6 and len(solars) == len(mexes), bundle.map_w
        for m, s in zip(mexes, solars):
            assert np.hypot(m.x - s.x, m.z - s.z) <= po.LINK_REACH
        spots = bundle.spot_list
        assert all(np.min(np.hypot(spots[:, 0] - m.x, spots[:, 1] - m.z)) < 1.0 for m in mexes)
        # Nothing is planned on a spot it doesn't use, or on the factory.
        assert np.min(np.hypot(spots[:, 0] - plan.factory[0], spots[:, 1] - plan.factory[1])) > factory_r
        for s in solars:
            assert np.min(np.hypot(spots[:, 0] - s.x, spots[:, 1] - s.z)) >= mex_r + solar_r
            assert np.hypot(s.x - plan.factory[0], s.z - plan.factory[1]) >= factory_r + solar_r


def test_each_side_only_takes_spots_on_its_own_side():
    env = se.SimEnv(1, 3, ["scripts"], threads=1)
    for bundle, start, enemy in _sides(env):
        for b in env.opening.plan(env.grounds[id(bundle)],start, enemy).builds:
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
    # Without the economy script, which takes the commander off the factory once the opening's done.
    # The army script raids extractors, which this test has to skip, so it plays more games than it needs.
    env = se.SimEnv(40, 7, ["scripts"], threads=4, economy=False)
    env.begin()
    for _ in range(420):                                # 3.5 minutes, early enough that few games have ended
        env.step(0.0)
        env.finished_seats()
    checked = 0
    for g in range(40):
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


def test_the_policy_is_given_its_fighters_factories_and_constructors():
    env = se.SimEnv(2, 11, ["mirror"], threads=2)
    _play(env, 80)
    seen = factories = constructors = 0
    for g in range(2):
        for s in (0, 1):
            units = env.batch.records[g, s, :env.batch.counts[g, s]]
            kept, controllable, _, _, _ = fz._select(units, env.table, s, s)
            own = units[kept[:controllable]]
            seen += controllable
            assert np.all(own["player"] == s) and np.all((own["flags"] & rf.BEING_BUILT) == 0)
            rows = env.table.rows(own["def_id"])
            factory = env.table.factory[rows]
            constructor = env.table.constructor[rows]
            assert np.all(env.table.mobile[rows] & ~env.table.builder[rows] | factory | constructor)
            factories += int(factory.sum())
            constructors += int(constructor.sum())
    assert seen > factories > 0 and constructors > 0


def test_a_policy_starts_with_its_commander_to_control():
    env = se.SimEnv(1, 13, ["mirror"], threads=1)
    env.begin()
    units = env.batch.records[0, 0, :env.batch.counts[0, 0]]
    kept, controllable, *_ = fz._select(units, env.table, 0, 0)
    assert controllable == 1 and env.table.constructor[env.table.rows(units["def_id"][kept[:1]])][0]


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


def test_aligned_games_all_decide_on_the_same_slots_and_an_ended_one_waits_for_the_next():
    """SimEnv.align: games ending between decision slots (here at their 9th step, every 6 slots a decision) start afresh only on the
    next decision slot, so every live game's own count, which its scripts go by, stays on the same slots as the others'."""
    env = se.SimEnv(3, 4, ["fight"], threads=2)
    env.act_every, env.align, env.diversity, env.time_limit = 6, True, 1.0, 9
    env.begin()
    policy = [s for s in range(env.seat_count) if env._is_policy(s)]
    waited = restarted = 0
    for _ in range(60):
        acting = env.policy_seats_to_act()
        live = [s for s in policy if not env.waiting[s // 2]]
        assert acting == (live if env.slot % 6 == 0 else [])
        assert all(env.decisions[s // 2] % 6 == env.slot % 6 for s in live)
        for seat in acting:
            env.give_orders(seat, [])
        env.step(0.0)
        assert not env.bonus.any() or env.slot % 6 == 0          # counted once a decision
        env.finished_seats()
        waited += int(env.waiting.any())
        restarted += int((env.decisions == 0).any())
    assert waited and restarted        # games ended at slot 9 (mod 6: 3), waited, and started afresh at 12
    assert env.slot == 60 and (env.decisions[~env.waiting] % 6 == env.slot % 6).all()


def test_finished_units_of_rare_types_earn_a_zero_sum_bonus_and_the_first_units_none():
    """policy_sim_env._diversity, after Pluto's: what a side finishes is worth more the rarer its type among what the policy makes,
    nothing for a type it makes as often as most; the units a side starts with are none of it; and the reward stays zero-sum."""
    env = se.SimEnv(4, 9, ["mirror"], threads=2)
    env.diversity = 1.0
    env.begin()
    makeable = np.flatnonzero(env.makeable)
    env.made[makeable[0]] = 10_000.0
    assert env.rarity(makeable[:1])[0] == 0.0 and env.rarity(makeable[1:2])[0] > 0.8
    env.made[:] = 0.0
    env.step(0.0)
    assert not env.bonus.any()          # the commanders, and nothing finished yet
    earned = 0.0
    for _ in range(300):
        for seat in env.policy_seats_to_act():
            env.give_orders(seat, [])
        env.step(0.0)
        for seat in range(0, env.seat_count, 2):
            assert env.reward(seat, 0.3) == pytest.approx(-env.reward(seat + 1, 0.3))
        earned += float(env.bonus.sum())
        env.finished_seats()
    assert earned > 0.0 and env.made.sum() > 0.0
    assert env.bonus.max() <= env.diversity * 50

# Games of several teams ------------------------------------------------------------------------------------------------------------


def _ffa_maps(boxes: int = 4, **facts) -> list:
    """Agalta with `boxes` start boxes of its own, round places a base fits, as a free-for-all map: the training maps have two."""
    import dataclasses
    world_map, bundle, known = se.pool()[0]
    places = []
    for fx, fz in ((0.15, 0.2), (0.85, 0.8), (0.85, 0.2), (0.15, 0.8)):
        x0, z0 = fx * bundle.map_w, fz * bundle.map_h
        near = [(x0 + dx, z0 + dz) for r in range(0, 1600, 64) for dx in (-r, 0, r) for dz in (-r, 0, r)]
        places.append(next((float(x), float(z)) for x, z in near if po.is_buildable(bundle, x, z, se.START_ROOM)))
    starts = [se.StartBox(([[x - 300.0, z - 300.0], [x + 300.0, z - 300.0], [x + 300.0, z + 300.0], [x - 300.0, z + 300.0]],), (x, z))
              for x, z in places[:boxes]]
    modes = {**known.modes, "ffa": boxes, **facts.pop("modes", {})}
    return [(world_map, bundle, dataclasses.replace(known, layouts=[starts], modes=modes, **facts))]


def _held(env: se.SimEnv, values: np.ndarray, keep: np.ndarray) -> None:
    """Game 0's teams at these values, and in the game where `keep` says, whatever the sim does."""
    real = env.batch.step

    def step(*args, **kwargs):
        real(*args, **kwargs)
        n = len(values)
        env.batch.stats[0, :n, 0] = values
        env.batch.stats[0, :n, se.KEEP_STAT] = keep
    env.batch.step = step


def test_kinds_draw_only_maps_that_hold_them_and_a_groups_maps_weigh_as_one():
    """Violet Rampart's three seeds of a number of bases weigh as one map; a map that isn't a 1v1 map isn't drawn for 1v1s."""
    one = _ffa_maps(2, modes={"ffa": 0})[0]
    rampart = _ffa_maps(3, modes={"1v1": False}, group="violet_rampart:3")[0]
    big = _ffa_maps(4)[0]
    env = se.SimEnv(1, 3, ["scripts"], threads=1, maps=[one, rampart, rampart, rampart, big], kinds={"1v1": 1.0, "ffa": 1.0}, teams=4)
    index, p = env._map_draws["ffa"]
    assert index.tolist() == [1, 2, 3, 4] and np.allclose(p, [1 / 6, 1 / 6, 1 / 6, 1 / 2])
    assert env._map_draws["1v1"][0].tolist() == [0, 4]
    assert se.SimEnv(1, 3, ["scripts"], threads=1)._map_draws == {"1v1": None}, "every map, none grouped: drawn as always"
    with pytest.raises(ValueError):
        se.SimEnv(1, 3, ["scripts"], threads=1, kinds={"ffa": 1.0})        # the training maps have two boxes


def test_a_1v1_plays_the_same_in_a_batch_with_room_for_8_teams():
    narrow, wide = se.SimEnv(4, 31, ["fight"], threads=2, variants=True), se.SimEnv(4, 31, ["fight"], threads=2, variants=True, teams=8)
    for env in (narrow, wide):
        env.diversity = 0.05
        env.begin()
    for _ in range(300):
        for env in (narrow, wide):
            for s in env.policy_seats_to_act():
                env.give_orders(s, [])
            env.step(0.0)
        assert narrow.batch.records[:, :2].tobytes() == wide.batch.records[:, :2].tobytes()
        assert [narrow.reward(g * 2 + t, 0.3) for g in range(4) for t in (0, 1)] == [wide.reward(g * 8 + t, 0.3) for g in range(4) for t in (0, 1)]
        assert [(s // 2, s % 2) for s, _, _ in narrow.finished_seats()] == [(s // 8, s % 8) for s, _, _ in wide.finished_seats()]


def test_a_team_game_starts_each_alliance_in_its_own_box_each_team_with_its_own_commander_and_its_allies_in_sight():
    env = se.SimEnv(2, 5, ["mirror"], threads=2, kinds={"teams": 1.0})
    env.begin()
    assert env.slots == 8 and env.seat_count == 16
    layouts = {id(b): facts.layouts for _, b, facts in se.pool()}
    for g in range(2):
        assert env.kind[g] == "teams" and env.n_teams[g] == 4 and env.ally[g].tolist() == [0, 0, 1, 1, -1, -1, -1, -1]
        starts = env.starts[g, :8].reshape(4, 2)
        layout = next(b for b in layouts[id(env.bundle[g])] if any(_near_box(box, *starts[0]) for box in b))
        box_of = [next(i for i, box in enumerate(layout) if _near_box(box, float(x), float(z))) for x, z in starts]
        assert box_of[0] == box_of[1] != box_of[2] == box_of[3]
        assert np.hypot(*(starts[0] - starts[1])) > 0.0 and np.hypot(*(starts[2] - starts[3])) > 0.0
        for t in range(4):
            u = env.batch.records[g, t, :env.batch.counts[g, t]]
            assert u["uid"][0] == t and u["player"][0] == t
            ally = t + 1 if t % 2 == 0 else t - 1
            assert ally in u["player"].tolist() and (u["ally_team"][u["player"] == ally] == t // 2).all()
    assert env.policy_seats() == [0, 1, 2, 3, 8, 9, 10, 11] == env.policy_seats_to_act()


def test_every_team_of_an_alliance_gets_its_alliances_reward_and_the_two_alliances_rewards_sum_to_zero():
    env = se.SimEnv(2, 9, ["scripts"], threads=2, kinds={"teams": 1.0}, variants=True)
    env.diversity, env.speed_weight = 1.0, 0.1
    env.begin()
    for _ in range(300):
        env.step(0.0)
        for g in range(2):
            r = [env.reward(g * env.slots + t, 0.3) for t in range(4)]
            assert r[0] == r[1] == -r[2] == -r[3]
        env.finished_seats()


def test_in_a_free_for_all_an_alliance_out_is_paid_its_loss_at_once_and_the_survivors_share_it_at_the_end():
    """With A alliances: -1 / (A - 1) at once for one out, and the speed term on top; its seat finishes and acts no more while the
    game goes on, and the rest's shaped rewards sum to zero.  At the end the last one in gets 1, and the whole game's results and
    speed terms sum to zero."""
    env = se.SimEnv(1, 3, ["mirror"], threads=1, maps=_ffa_maps(4), kinds={"ffa": 1.0}, teams=4)
    env.speed_weight = 0.1
    values, keep = np.full(4, 5000.0, np.float32), np.ones(4, np.float32)
    _held(env, values, keep)
    env.begin()
    A = int(env.n_allies[0])
    assert env.kind[0] == "ffa" and A in (3, 4) and env.ally[0, :A].tolist() == list(range(A))
    paid = np.zeros(A)
    for _ in range(5):
        env.step(0.0)
        assert not env.done
    keep[A - 1] = 0.0
    env.step(0.0)
    left = 1.0 - float(env.decisions[0]) / env.time_limit
    assert env.done == [(A - 1, "mirror", 0.0)] and not env._replace
    assert env.reward(A - 1, 1.0) == pytest.approx(-(1.0 + 0.1 * left) / (A - 1))
    assert env.outcome[0, A - 1] == -1.0 / (A - 1) and env.ended[A - 1]["result"] == "lost" and env.ended[A - 1]["how"] == "eliminated"
    paid += env.paid[0, :A] + 0.1 * env.speed_paid[0, :A]
    env.finished_seats()
    assert env.policy_seats_to_act() == list(range(A - 1))
    for _ in range(3):
        env.step(0.0)
        assert env.reward(A - 1, 1.0) == 0.0 and not env.done
        assert sum(env.reward(t, 0.0) for t in range(A - 1)) == pytest.approx(0.0, abs=1e-9)
        env.finished_seats()
    keep[1:] = 0.0
    env.step(0.0)
    paid += env.paid[0, :A] + 0.1 * env.speed_paid[0, :A]
    assert env._replace == [0] and env.result[0] == 1.0 and env.outcome[0, :A].tolist() == [1.0] + [-1.0 / (A - 1)] * (A - 1)
    assert sorted(s for s, _, _ in env.done) == list(range(A - 1)) and env.ended[0]["result"] == "won"
    assert paid.sum() == pytest.approx(0.0, abs=1e-12) and paid[0] > 1.0


def test_a_lead_over_the_rest_together_ends_a_free_for_all_with_every_other_alliance_out():
    env = se.SimEnv(1, 3, ["scripts"], threads=1, maps=_ffa_maps(3), kinds={"ffa": 1.0}, teams=3)
    _held(env, np.array([40000.0, 6000.0, 6000.0], np.float32), np.ones(3, np.float32))
    env.begin()
    for _ in range(se.EARLY_END_DECISIONS):
        env.step(0.0)
        if env._replace:
            break
    assert env.decisions[0] == se.EARLY_END_DECISIONS and env.outcome[0, :3].tolist() == [1.0, -0.5, -0.5]
    assert {o["how"] for o in env.outcomes} == {"lead"} and len(env.outcomes) == 3


# Chicken games ---------------------------------------------------------------------------------------------------------------------


def _queen(env: se.SimEnv, state: dict) -> None:
    """Game 0's chicken state as `state` says, whatever the sim's (its "queen": 0 not yet, 1 alive, 2 dead)."""
    real = env.batch.chicken_state
    env.batch.chicken_state = lambda g: {**real(g), **state} if g == 0 else real(g)


def _players(env: se.SimEnv, n: int) -> None:
    """Every chicken game of the environment with n players."""
    env._sizes = lambda kind, facts, boxes: [n]


def test_chicken_difficulties_go_by_zero_k_names_and_a_chicken_game_needs_a_team_or_chicken_map_and_a_slot_for_its_chickens():
    assert se.difficulty_weights(["very-easy:2", "Easy", "3:0.5"]) == {1: 2.0, 2: 1.0, 3: 0.5}
    assert se.difficulty_weights({"Normal": 1.0, "Hard": 0.0}) == {3: 1.0}
    assert set(se.difficulty_weights(se.CHICKEN_DIFFICULTY_WEIGHTS)) == {1, 2, 3}
    with pytest.raises(ValueError):
        se.difficulty_weights(["medium:1"])
    assert se.slots_for({"chicken": 1.0}) == 5 and se.slots_for(se.parse_kinds("1v1:0.5 teams:0.2 ffa:0.2 chicken:0.1")) == 8
    import dataclasses
    facts = se.pool()[0][2]
    ffa = dataclasses.replace(facts, modes={"1v1": False, "teams": 0, "ffa": 4, "chickens": False, "chickens_or_teams": False})
    chickens = dataclasses.replace(facts, modes={"1v1": False, "teams": 0, "ffa": 0, "chickens": True, "chickens_or_teams": True})
    older = dataclasses.replace(facts, modes={"1v1": True, "teams": 2, "ffa": 0, "chickens": False})       # before the key
    assert [se.holds(f, "chicken", 5) for f in (ffa, chickens, older)] == [False, True, True]
    assert not se.holds(chickens, "1v1", 5) and not se.holds(chickens, "chicken", 1)


def test_a_chicken_game_starts_its_players_in_one_box_and_the_chickens_after_them_in_an_alliance_of_their_own_with_no_seat():
    env = se.SimEnv(2, 4, ["mirror"], threads=2, kinds={"chicken": 1.0})
    env.begin()
    assert env.slots == 5
    layouts = {id(b): facts.layouts for _, b, facts in se.pool()}
    for g in range(2):
        n = int(env.n_teams[g])
        c = n - 1
        assert env.kind[g] == "chicken" and 2 <= n <= 5 and env.chickens[g] == c and env.seats[g][c] is None
        assert env.ally[g, :n].tolist() == [0] * c + [1] and env.n_allies[g] == 2 and not env.lead_rule[g]
        assert [env.plays(g, t) for t in range(n)] == ["policy"] * c + ["chickens"]
        assert env.batch.chicken_state(g)["team"] == c and zksim.CHICKEN_DIFFICULTIES[env.difficulty[g]] in se.CHICKEN_DIFFICULTY_WEIGHTS
        own = env.batch.records[g, c, :env.batch.counts[g, c]]
        assert not (own["player"] == c).any(), "the sim spawns them later"
        starts = env.starts[g, :2 * c].reshape(c, 2)
        layout = next(b for b in layouts[id(env.bundle[g])] if any(_near_box(box, *starts[0]) for box in b))
        assert len({next(i for i, box in enumerate(layout) if _near_box(box, float(x), float(z))) for x, z in starts}) == 1
    seats = [g * 5 + t for g in range(2) for t in range(int(env.n_teams[g]) - 1)]
    assert env.policy_seats() == seats == env.policy_seats_to_act()


def test_on_a_chicken_map_the_players_take_its_first_box_which_is_theirs():
    import dataclasses
    world, bundle, facts = se.pool()[0]
    chicken_map = dataclasses.replace(facts, modes={"1v1": False, "teams": 0, "ffa": 0, "chickens": True, "chickens_or_teams": True})
    env = se.SimEnv(8, 4, ["mirror"], threads=2, maps=[(world, bundle, chicken_map)], kinds={"chicken": 1.0})
    env.begin()
    first = [layout[0] for layout in facts.layouts]
    for g in range(8):
        starts = env.starts[g, :2 * (int(env.n_teams[g]) - 1)].reshape(-1, 2)
        assert all(any(_near_box(box, float(x), float(z)) for box in first) for x, z in starts), g


def test_a_chicken_game_is_won_when_the_queen_dies_and_until_then_its_chickens_are_in_with_no_unit_at_all():
    env = se.SimEnv(1, 4, ["mirror"], threads=1, kinds={"chicken": 1.0}, difficulties={"Easy": 1.0})
    env.speed_weight = 0.1
    _players(env, 2)
    _held(env, np.full(3, 5000.0, np.float32), np.array([1.0, 1.0, 0.0], np.float32))
    state = {"queen": 1}
    _queen(env, state)
    env.begin()
    for _ in range(5):
        env.step(0.0)
        assert not env.done and env.reward(0, 0.3) == env.reward(1, 0.3)
    state["queen"] = 2
    env.step(0.0)
    left = 1.0 - float(env.decisions[0]) / env.time_limit
    assert env._replace == [0] and env.outcome[0, :3].tolist() == [1.0, 1.0, -1.0] and env.result[0] == 1.0
    assert sorted((s, won) for s, _, won in env.done) == [(0, 1.0), (1, 1.0)]
    assert env.ended[0]["how"] == "queen" and env.ended[0]["difficulty"] == "Easy" and env.ended[0]["game_kind"] == "chicken"
    assert env.reward(0, 1.0) == env.reward(1, 1.0) == pytest.approx(1.0 + 0.1 * left)
    chickens = env._game_facts(0, 2, False, True)
    assert chickens["variant"] == "chickens" and chickens["own_factory"] == "" and chickens["how"] == "queen"


@pytest.mark.parametrize("out", [True, False])
def test_a_chicken_game_is_lost_when_its_players_are_out_and_drawn_at_the_time_limit(out):
    env = se.SimEnv(1, 4, ["mirror"], threads=1, kinds={"chicken": 1.0})
    _players(env, 2)
    env.time_limit = 6
    keep = np.array([1.0, 1.0, 0.0], np.float32)
    _held(env, np.full(3, 5000.0, np.float32), keep)
    _queen(env, {"queen": 1})
    env.begin()
    if out:
        keep[:2] = 0.0
    while not env._replace:
        env.step(0.0)
    assert env.outcome[0, :3].tolist() == ([-1.0, -1.0, 1.0] if out else [0.0, 0.0, 0.0])
    assert env.decisions[0] == (1 if out else 6) and env.ended[0]["how"] == ("eliminated" if out else "limit")
