"""Several alliances and players in one frame: the view any seat of a 3v3v3, a 5v5 or a free-for-all has."""
from __future__ import annotations

import numpy as np
import pytest

from zkbot.policy import featurize as fz, rawframe as rf

from conftest import BLIP, ENEMY, make_frame, make_units


def teams(**kw):
    """I am player 0 of alliance 0, with player 1 beside me.  Alliances 1 and 2 have two players each."""
    rows = [{"uid": 1, "x": 500.0}, {"uid": 2, "x": 520.0},
            {"uid": 10, "player": 1, "ally_team": 0, "x": 540.0}, {"uid": 11, "player": 1, "ally_team": 0, "x": 560.0},
            {"uid": 20, "side": ENEMY, "player": 2, "ally_team": 1, "x": 800.0}, {"uid": 21, "side": ENEMY, "player": 3, "ally_team": 1, "x": 820.0},
            {"uid": 30, "side": ENEMY, "player": 4, "ally_team": 2, "x": 840.0}, {"uid": 31, "side": ENEMY, "player": 5, "ally_team": 2, "x": 860.0}]
    return make_units(*rows)


def token(t, uid):
    return int(np.flatnonzero(t.uids == uid)[0])


def test_only_my_own_units_get_an_action(small_map, table):
    t = fz.featurize(make_frame(teams()), small_map, table)
    assert t.controllable == 2 and set(t.uids[:2].tolist()) == {1, 2}
    assert t.kind_mask.shape[0] == 2


def test_every_other_alliance_can_be_aimed_at_and_mine_cannot(small_map, table):
    t = fz.featurize(make_frame(teams()), small_map, table)
    assert set(t.uids[t.enemy_tokens].tolist()) == {20, 21, 30, 31}


def test_ids_are_the_viewers(small_map, table):
    t = fz.featurize(make_frame(teams()), small_map, table)
    assert [int(t.unit_ally[token(t, u)]) for u in (1, 10, 20, 30)] == [0, 0, 1, 2]
    assert [int(t.unit_player[token(t, u)]) for u in (1, 10, 20, 21, 30, 31)] == [0, 1, 2, 3, 4, 5]


def test_both_seats_of_a_1v1_see_the_same_numbers(small_map, table):
    units = make_units({"uid": 1, "x": 500.0}, {"uid": 2, "side": ENEMY, "x": 800.0})
    seats = []
    for seat in (0, 1):
        frame = make_frame(units)
        frame.glob[rf.G["player"]], frame.glob[rf.G["ally_team"]] = seat, seat
        t = fz.featurize(frame, small_map, table)
        seats.append((sorted(t.unit_ally.tolist()), sorted(t.unit_player.tolist()), t.controllable))
    assert seats[0] == seats[1] == ([0, 1], [0, 1], 1)


def test_relative_ids_stay_apart_and_unknown_ones_share_a_row():
    ids = np.array([0, 1, 2, 3, -1, 40])
    assert fz.relative_id(ids, 2, 16).tolist() == [1, 2, 0, 3, 16, 16]
    assert fz.relative_id(ids, 0, 16).tolist() == [0, 1, 2, 3, 16, 16]


def test_an_allys_guns_count_as_friendly_and_another_alliances_as_hostile(small_map, table):
    # Everyone stands apart along +x from unit 1, so sector 2 is the one they're all in.
    alone = make_units({"uid": 1, "x": 500.0})
    ally = make_units({"uid": 1, "x": 500.0}, {"uid": 2, "player": 1, "ally_team": 0, "x": 700.0})
    foe = make_units({"uid": 1, "x": 500.0}, {"uid": 2, "side": ENEMY, "player": 2, "ally_team": 2, "x": 700.0})
    hostile, friendly = fz._R + 8 + 2, fz._R + 16 + 2
    a = fz.featurize(make_frame(alone), small_map, table).unit_x[0]
    b = fz.featurize(make_frame(ally), small_map, table).unit_x[0]
    c = fz.featurize(make_frame(foe), small_map, table).unit_x[0]
    assert a[hostile] == a[friendly] == 0.0
    assert b[friendly] > 0.0 and b[hostile] == 0.0
    assert c[hostile] > 0.0 and c[friendly] == 0.0


def test_one_pool_holds_every_alliance_and_the_cap_counts_them_all(small_map, table):
    rows = [{"uid": i + 1, "x": 300.0 + i % 40, "z": 300.0 + i // 40} for i in range(fz.CAP_CONTROLLABLE)]
    rows += [{"uid": 1000 + i, "side": ENEMY, "player": 2 + i % 3, "ally_team": 1 + i % 3, "x": 800.0 + i % 30, "z": 800.0} for i in range(300)]
    # Allies, as near as the enemy and worth as much: what can be seen shooting at us still comes first.
    rows += [{"uid": 3000 + i, "player": 1, "ally_team": 0, "x": 800.0 + i % 30, "z": 800.0} for i in range(300)]
    t = fz.featurize(make_frame(make_units(*rows)), small_map, table)
    units = t.uids[t.uids >= 0]
    assert len(units) == fz.CAP_UNITS
    assert (units >= 3000).sum() == fz.CAP_UNITS - fz.CAP_CONTROLLABLE - 300
    assert ((units >= 1000) & (units < 3000)).sum() == 300
    assert set(t.unit_ally[t.uids < 0].tolist()) <= {0, 1, 2, 3}


def test_a_far_off_enemy_is_not_crowded_out_by_my_own_units_nearby(small_map, table):
    rows = [{"uid": i + 1, "x": 300.0 + i % 40, "z": 300.0 + i // 40} for i in range(400)]
    rows += [{"uid": 1000 + i, "side": ENEMY, "x": 2900.0 + i % 40, "z": 2900.0 + i // 40} for i in range(400)]
    t = fz.featurize(make_frame(make_units(*rows)), small_map, table)
    assert t.controllable == fz.CAP_CONTROLLABLE
    assert len(t.enemy_tokens) == fz.CAP_UNITS - fz.CAP_CONTROLLABLE


def test_in_a_full_pool_sight_comes_before_radar_and_radar_before_the_rest(small_map, table):
    # Worth the same, and the blips and my allies nearer: only what is known of them can put the seen ones first.
    room = fz.CAP_UNITS - 1
    rows = [{"uid": 1, "x": 300.0}]
    rows += [{"uid": 3000 + i, "player": 1, "ally_team": 0, "x": 310.0} for i in range(room)]
    rows += [{"uid": 2000 + i, "side": BLIP, "x": 320.0} for i in range(room)]
    rows += [{"uid": 1000 + i, "side": ENEMY, "x": 2900.0} for i in range(room // 2)]
    t = fz.featurize(make_frame(make_units(*rows)), small_map, table)
    units = t.uids[t.uids >= 0]
    assert ((units >= 1000) & (units < 2000)).sum() == room // 2
    assert ((units >= 2000) & (units < 3000)).sum() == room - room // 2
    assert (units >= 3000).sum() == 0


def aiming(target, *others, ord_x=0.0, ord_z=0.0):
    return make_units({"uid": 1, "x": 500.0, "z": 500.0, "ord_kind": rf.ATTACK_UNIT, "ord_target": target, "ord_x": ord_x, "ord_z": ord_z}, *others)


def test_an_attacker_knows_where_its_target_is(small_map, table):
    t = fz.featurize(make_frame(aiming(7, {"uid": 7, "side": ENEMY, "x": 800.0, "z": 900.0})), small_map, table)
    row = t.unit_x[0]
    assert row[fz._O + 17] == pytest.approx(300.0 / small_map.map_w)
    assert row[fz._O + 18] == pytest.approx(400.0 / small_map.map_h)
    assert row[fz._O + 19] == pytest.approx(500.0 / max(small_map.map_w, small_map.map_h))
    assert row[fz._O + 20] == 1.0 and row[fz._O + 21] == 1.0


def test_a_target_on_radar_has_a_direction_but_is_not_in_sight(small_map, table):
    t = fz.featurize(make_frame(aiming(7, {"uid": 7, "side": BLIP, "x": 800.0})), small_map, table)
    assert t.unit_x[0, fz._O + 17] > 0.0 and t.unit_x[0, fz._O + 21] == 0.0


def test_a_target_gone_has_no_direction(small_map, table):
    t = fz.featurize(make_frame(aiming(7, {"uid": 8, "side": ENEMY, "x": 800.0}, ord_x=900.0, ord_z=900.0)), small_map, table)
    assert (t.unit_x[0, fz._O + 17:fz._O + 20] == 0.0).all()
    assert t.unit_x[0, fz._O + 20] == 1.0 and t.unit_x[0, fz._O + 21] == 0.0


def test_a_target_past_the_pool_still_has_a_direction(small_map, table):
    # Enemies the pool has no room for: the first own units are worth the most, and one of them aims at the last-ranked enemy.
    rows = [{"uid": 1, "x": 500.0, "z": 500.0, "name": "cloakriot", "ord_kind": rf.ATTACK_UNIT, "ord_target": 5000}]
    rows += [{"uid": 1000 + i, "side": ENEMY, "x": 800.0 + i % 40, "z": 800.0} for i in range(fz.CAP_UNITS)]
    rows += [{"uid": 5000, "side": ENEMY, "x": 2900.0, "z": 2900.0}]
    t = fz.featurize(make_frame(make_units(*rows)), small_map, table)
    assert 5000 not in t.uids.tolist()
    assert t.unit_x[0, fz._O + 19] > 0.0 and t.unit_x[0, fz._O + 21] == 1.0


def test_there_are_never_too_many_summaries(small_map, table):
    rows = [{"uid": i + 1, "x": 300.0 + i % 40, "z": 300.0} for i in range(fz.CAP_UNITS)]
    rows += [{"uid": 2000 + i, "side": ENEMY, "player": i % 15 + 1, "ally_team": i % 15 + 1, "x": 2000.0, "name": ("cloakraid", "cloakcon", "staticmex")[i % 3]} for i in range(600)]
    t = fz.featurize(make_frame(make_units(*rows)), small_map, table)
    assert (t.tok_kind == fz.TK_SUMMARY).sum() <= fz.CAP_SUMMARIES


def test_the_hidden_say_whose_they_are_to_the_critic_alone(small_map, table):
    units = make_units({"uid": 1}, {"uid": 2, "side": "hidden", "player": 4, "ally_team": 2, "x": 2000.0, "z": 2000.0})
    t = fz.featurize(make_frame(units), small_map, table, privileged=True)
    assert t.hidden_ally.tolist() == [2] and t.hidden_player.tolist() == [4]
    assert not len(fz.featurize(make_frame(units), small_map, table).hidden_ally)


@pytest.fixture(scope="module")
def actor(table, encoder_path):
    """The actor on latents, untrained."""
    torch = pytest.importorskip("torch")
    import policy_latent as pl

    torch.manual_seed(5)
    return pl.make(len(table.names) + 1, encoder_path).eval()


def _kinds(actor, t: fz.Tensors) -> np.ndarray:
    """Each controllable unit's kind probabilities, with nothing before this frame."""
    import torch

    import policy_model as pm

    b = pm.collate([t]).to("cpu")
    with torch.no_grad():
        units, reg_h, *_ = actor.decide(b, *actor.initial_state(1, 0, "cpu"))
        return torch.softmax(actor.heads(b, units, reg_h)[0], -1)[0, :t.controllable].numpy()


def test_the_actor_decides_on_a_frame_with_three_alliances(small_map, table, actor):
    t = fz.featurize(make_frame(teams()), small_map, table)
    kinds = _kinds(actor, t)
    assert kinds.shape == (2, rf.KIND_COUNT) and (kinds[~t.kind_mask] < 1e-6).all()


def test_which_enemy_a_unit_aims_at_reaches_the_actor(small_map, table, actor):
    enemies = ({"uid": 7, "side": ENEMY, "x": 800.0}, {"uid": 8, "side": ENEMY, "x": 300.0, "z": 1400.0})
    kinds = []
    for target in (7, 8):
        units = aiming(target, *enemies)
        units["ord_age"] = 1000.0       # past the commit lock, or KEEP is all it may do
        kinds.append(_kinds(actor, fz.featurize(make_frame(units), small_map, table)))
    assert not np.allclose(kinds[0], kinds[1])


def test_who_owns_a_unit_reaches_the_actor(small_map, table, actor):
    base = _kinds(actor, fz.featurize(make_frame(teams()), small_map, table))
    swapped = teams()
    swapped["ally_team"][swapped["uid"] == 30] = 3
    other = _kinds(actor, fz.featurize(make_frame(swapped), small_map, table))
    assert not np.allclose(base, other)
