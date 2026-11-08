"""Schema, shapes, masks and the live producer, on frames made by hand."""
from __future__ import annotations

import copy
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from zkbot import units as unit_data
from zkbot.policy import bundle as bd, featurize as fz, rawframe as rf
from zkbot.policy.bundle import Bundle

from conftest import BLIP, ENEMY, HIDDEN, MEMORY, make_frame, make_units

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def run(units, small_map, table, **kw):
    return fz.featurize(make_frame(units), small_map, table, **kw)


def test_header_is_generated_from_the_schema():
    assert (ROOT / "sim/include/zksim/rawframe.hpp").read_text("utf8") == rf.cpp_header()


def test_unit_record_has_no_padding():
    assert rf.UNIT_DTYPE.itemsize == 80
    assert sum(np.dtype(t).itemsize for _, t in rf.UNIT_FIELDS) == 80


def test_frames_survive_a_file(tmp_path):
    a = rf.Frame(make_units({"uid": 5}, {"uid": 6, "side": ENEMY}), np.zeros(2, rf.SITE_DTYPE), np.arange(len(rf.GLOBAL_FIELDS), dtype=np.float32))
    b = rf.Frame(make_units({"uid": 7}), np.zeros(0, rf.SITE_DTYPE), np.ones(len(rf.GLOBAL_FIELDS), np.float32))
    path = tmp_path / "f.npz"
    rf.save_frames(path, [a, b])
    back = rf.load_frames(path)
    assert [f.units["uid"].tolist() for f in back] == [[5, 6], [7]]
    assert back[0].glob[3] == 3 and len(back[0].sites) == 2 and len(back[1].sites) == 0


def test_bundle_survives_a_file_and_finds_its_way(small_map, tmp_path):
    path = tmp_path / "b.npz"
    small_map.save(path)
    back = Bundle.load(path)
    assert back.regions == small_map.regions >= 2
    assert np.array_equal(back.walk, small_map.walk, equal_nan=True)
    west, east = int(small_map.region_at(300.0, 1280.0)), int(small_map.region_at(2300.0, 1280.0))
    assert west != east
    kbot = 0
    assert np.isfinite(small_map.walk[kbot, west, east]) and small_map.walk[kbot, west, east] > 0
    assert small_map.walk[kbot, west, west] == 0


def _features(*rows):
    out = np.zeros(len(rows), rf.FEATURE_DTYPE)
    for i, (x, z, metal) in enumerate(rows):
        out[i] = (i + 100, x, z, metal, 7)
    return out


def _with_features(units, features, small_map, table):
    frame = make_frame(units)
    return fz.featurize(rf.Frame(frame.units, frame.sites, frame.glob, features), small_map, table)


def test_frames_keep_their_features_in_a_file(tmp_path):
    frame = make_frame(make_units({"uid": 1}))
    path = tmp_path / "f.npz"
    rf.save_frames(path, [rf.Frame(frame.units, frame.sites, frame.glob, _features((10.0, 20.0, 30.0))), frame])
    a, b = rf.load_frames(path)
    assert a.features["metal"].tolist() == [30.0] and not len(b.features)


def test_a_frame_from_before_features_reads_as_having_none():
    assert all(not len(f.features) for f in rf.load_frames(ROOT / "tests" / "policy" / "data" / "golden_frames.npz"))


def test_features_get_tokens_before_the_global_one(small_map, table):
    units = make_units({"uid": 1, "x": 500.0, "z": 500.0}, {"uid": 2, "side": ENEMY, "x": 900.0, "z": 500.0})
    bare = _with_features(units, _features(), small_map, table)
    t = _with_features(units, _features((520.0, 500.0, 50.0), (2000.0, 500.0, 400.0)), small_map, table)
    assert t.feat_x.shape == (2, fz.FEATURE_DIM) and t.feat_ids.tolist() == [100, 101]
    assert len(t.tok_kind) == len(bare.tok_kind) + 2
    assert t.tok_kind[-3:].tolist() == [fz.TK_FEATURE, fz.TK_FEATURE, fz.TK_GLOBAL]
    assert t.feat_x[0, 2] == pytest.approx(np.log1p(50.0) / 8.0) and t.feat_x[1, 0] > t.feat_x[0, 0]
    assert np.isfinite(t.feat_x).all() and t.feat_x[0, 4] < t.feat_x[1, 4]
    # Everything the units already had stays as it was.
    assert np.array_equal(t.unit_x, bare.unit_x) and np.array_equal(t.ptr_enemy, bare.ptr_enemy)


def test_the_cap_keeps_most_metal_for_the_walk_and_regions_count_the_rest(small_map, table):
    units = make_units({"uid": 1, "x": 500.0, "z": 500.0})
    many = [(600.0, 500.0, 1.0)] * (fz.CAP_FEATURES + 5) + [(650.0, 500.0, 500.0)]
    features = _features(*many)
    t = _with_features(units, features, small_map, table)
    assert len(t.feat_x) == fz.CAP_FEATURES and 100 + fz.CAP_FEATURES + 5 in t.feat_ids
    region = int(small_map.region_at(600.0, 500.0))
    assert t.reg_x[region, 33] == pytest.approx(np.log1p(sum(m for *_, m in many)) / 10.0, rel=1e-3)
    assert t.reg_x[:, 33].sum() == t.reg_x[region, 33]
    # Which record came first can't matter.
    again = _with_features(units, features[::-1], small_map, table)
    assert np.array_equal(t.feat_ids, again.feat_ids) and np.array_equal(t.feat_x, again.feat_x)


def test_tensor_shapes_and_token_order(small_map, table):
    units = make_units({"name": "cloakraid", "uid": 1}, {"name": "cloakcon", "uid": 2}, {"name": "staticmex", "uid": 3},
                       {"name": "cloakriot", "side": ENEMY, "uid": 4, "x": 800.0},
                       {"name": None, "side": BLIP, "uid": 5, "x": 900.0, "hp_frac": np.nan},
                       {"name": "staticmex", "side": MEMORY, "uid": 6, "x": 1000.0, "ord_age": 300.0})
    t = run(units, small_map, table)
    assert t.controllable == 1
    assert sorted(t.uids.tolist()) == [1, 2, 3, 4, 5, 6]
    assert t.unit_x.shape == (6, fz.UNIT_DIM) and t.unit_name.shape == (6,)
    assert t.reg_x.shape == (small_map.regions, fz.REGION_DIM)
    n = 6 + small_map.regions + 0 + 1
    assert t.tok_kind.shape == (n,)
    assert t.controllable == 1 and t.uids[0] == 1
    kinds = {int(uid): int(k) for uid, k in zip(t.uids, t.tok_kind)}
    assert kinds == {1: fz.TK_SEEN, 2: fz.TK_SEEN, 3: fz.TK_SEEN, 4: fz.TK_SEEN, 5: fz.TK_BLIP, 6: fz.TK_MEMORY}
    assert t.unit_ally.shape == t.unit_player.shape == (6,)
    assert t.tok_kind[-1] == fz.TK_GLOBAL
    assert t.kind_mask.shape == (1, rf.KIND_COUNT) and t.ptr_region.shape == (1, small_map.regions)
    assert t.ptr_enemy.shape == (3, 1, 2)
    assert np.isfinite(t.unit_x).all() and np.isfinite(t.reg_x).all() and np.isfinite(t.glob_x).all()


def test_a_blip_without_identity_has_no_type_stats(small_map, table):
    t = run(make_units({"uid": 1}, {"name": None, "side": BLIP, "uid": 2, "hp_frac": np.nan, "blip_top_speed": 90.0}), small_map, table)
    blip = t.unit_x[1]
    assert not blip[:fz.TYPE_DIM].any()
    assert t.unit_name[1] == table.unknown
    assert blip[fz._I + 16] == pytest.approx(90.0 / 150.0)


def test_caps_summarise_what_they_drop(small_map, table):
    many = [{"uid": i + 1, "x": 300.0 + i % 40, "z": 300.0 + i // 40} for i in range(fz.CAP_CONTROLLABLE + 22)]
    room = fz.CAP_UNITS - fz.CAP_CONTROLLABLE
    foes = [{"uid": 1000 + i, "side": ENEMY, "x": 800.0 + i % 30} for i in range(room + 16)]
    t = run(make_units(*many, *foes), small_map, table)
    assert t.controllable == fz.CAP_CONTROLLABLE
    assert (t.uids >= 1000).sum() == room     # what's seen of the enemy comes before the rest of our own
    summaries = np.flatnonzero(t.tok_kind == fz.TK_SUMMARY)
    assert len(summaries) == 2
    assert t.uids[summaries].tolist() == [-1, -1]
    # The count of the 22 own units left over comes back out of the summary token, which says whose they are.
    assert np.expm1(t.unit_x[summaries[0], fz._S] * 5.0) == pytest.approx(22, abs=0.01)
    assert t.unit_ally[summaries].tolist() == [0, 1] and t.unit_player[summaries[0]] == 0


def test_sorted_tokens_do_not_depend_on_record_order(small_map, table):
    rows = [{"uid": i + 1, "x": 300.0 + 7 * i, "name": n} for i, n in enumerate(("cloakraid", "cloakskirm", "cloakriot", "cloakassault"))]
    a = run(make_units(*rows), small_map, table)
    b = run(make_units(*rows[::-1]), small_map, table)
    assert a.uids.tolist() == b.uids.tolist()
    assert np.array_equal(a.unit_x, b.unit_x)


def test_rays_see_walls_and_enemies(small_map, table):
    # The map's edge is 100 elmos west of this unit.  Sector 0 is +z, sector 2 +x, sector 6 -x.
    t = run(make_units({"uid": 1, "x": 100.0, "z": 400.0}, {"uid": 2, "name": "cloakriot", "side": ENEMY, "x": 400.0, "z": 400.0}), small_map, table)
    free = t.unit_x[0, fz._R:fz._R + 8]
    assert free[2] == pytest.approx(1.0)
    assert free[6] < 0.5
    enemy_dps = t.unit_x[0, fz._R + 8:fz._R + 16]
    assert enemy_dps[2] > 0 and enemy_dps[6] == 0 and enemy_dps[0] == 0
    assert not t.unit_x[0, fz._R + 16:fz._R + 24].any()


# Masks: one hand-made frame for each row of 02-policy.md section 4.1.
def kinds_of(units, small_map, table, **kw):
    t = run(units, small_map, table, **kw)
    return t, {k for k in range(rf.KIND_COUNT) if t.kind_mask[0, k]}


def test_a_fresh_unit_may_keep_move_fight_stop(small_map, table):
    _, kinds = kinds_of(make_units({"uid": 1}), small_map, table)
    assert kinds == {rf.KEEP, rf.MOVE, rf.FIGHT, rf.STOP}


def test_an_idle_unit_cannot_keep(small_map, table):
    _, kinds = kinds_of(make_units({"uid": 1, "flags": rf.IN_LOS | rf.TYPE_KNOWN | rf.IDLE}), small_map, table)
    assert rf.KEEP not in kinds and rf.MOVE in kinds


def test_commit_lock_allows_only_keep(small_map, table):
    _, kinds = kinds_of(make_units({"uid": 1, "ord_kind": rf.MOVE, "ord_age": 30.0, "ord_x": 900.0, "ord_z": 900.0}), small_map, table)
    assert kinds == {rf.KEEP}


@pytest.mark.parametrize("change", [{"ord_age": 90.0}, {"ord_age": 30.0, "flags": rf.IN_LOS | rf.TYPE_KNOWN | rf.DAMAGED}, {"ord_age": 30.0, "flags": rf.IN_LOS | rf.TYPE_KNOWN | rf.IDLE}])
def test_commit_lock_ends(small_map, table, change):
    _, kinds = kinds_of(make_units({"uid": 1, "ord_kind": rf.MOVE, "ord_x": 900.0, "ord_z": 900.0, **change}), small_map, table)
    assert rf.MOVE in kinds


def test_commit_lock_ends_when_the_target_is_gone(small_map, table):
    locked = {"uid": 1, "ord_kind": rf.ATTACK_UNIT, "ord_age": 30.0, "ord_target": 2}
    _, kinds = kinds_of(make_units(locked, {"uid": 2, "side": ENEMY, "x": 600.0}), small_map, table)
    assert kinds == {rf.KEEP}
    _, kinds = kinds_of(make_units(locked, {"uid": 3, "side": ENEMY, "x": 600.0}), small_map, table)
    assert rf.MOVE in kinds


def test_attack_needs_a_target_in_los_it_can_hit(small_map, table):
    t, kinds = kinds_of(make_units({"uid": 1}, {"uid": 2, "side": ENEMY, "x": 700.0}), small_map, table)
    assert rf.ATTACK_UNIT in kinds and t.ptr_enemy[0, 0, 0]
    # A blip isn't in LOS.
    t, kinds = kinds_of(make_units({"uid": 1}, {"uid": 2, "name": None, "side": BLIP, "x": 700.0, "hp_frac": np.nan}), small_map, table)
    assert rf.ATTACK_UNIT not in kinds and not t.ptr_enemy[0].any()
    # An anti-air weapon can't hit a ground unit, and can hit a gunship; a ground-only weapon is the reverse.
    assert rf.ATTACK_UNIT not in kinds_of(make_units({"uid": 1, "name": "cloakaa"}, {"uid": 2, "side": ENEMY, "x": 700.0}), small_map, table)[1]
    gunship = {"uid": 2, "name": "gunshipraid", "side": ENEMY, "x": 700.0}
    assert rf.ATTACK_UNIT in kinds_of(make_units({"uid": 1, "name": "cloakaa"}, gunship), small_map, table)[1]
    i = next(i for i, n in enumerate(table.names) if table.combat[i] and table.mobile[i] and not table.air[i] and table.can_hit_ground[i] and not table.can_hit_air[i])
    assert rf.ATTACK_UNIT not in kinds_of(make_units({"uid": 1, "name": table.names[i]}, gunship), small_map, table)[1]


def test_attack_ground_needs_a_target_within_reach(small_map, table):
    near = make_units({"uid": 1, "x": 500.0}, {"uid": 2, "name": None, "side": BLIP, "x": 700.0, "hp_frac": np.nan})
    far = make_units({"uid": 1, "x": 200.0, "z": 200.0}, {"uid": 2, "name": None, "side": BLIP, "x": 2400.0, "z": 2400.0, "hp_frac": np.nan})
    assert rf.ATTACK_GROUND in kinds_of(near, small_map, table)[1]
    assert rf.ATTACK_GROUND not in kinds_of(far, small_map, table)[1]


def test_abilities_come_from_tags(small_map, table):
    charged = rf.IN_LOS | rf.TYPE_KNOWN | rf.JUMP_READY
    assert rf.JUMP in kinds_of(make_units({"uid": 1, "name": "jumpraid", "flags": charged}), small_map, table)[1]
    assert rf.JUMP not in kinds_of(make_units({"uid": 1, "name": "cloakraid", "flags": charged}), small_map, table)[1]
    assert rf.CLOAK_ON in kinds_of(make_units({"uid": 1, "name": "cloaksnipe"}), small_map, table)[1]
    cloaked = make_units({"uid": 1, "name": "cloaksnipe", "flags": rf.IN_LOS | rf.TYPE_KNOWN | rf.CLOAKED})
    _, kinds = kinds_of(cloaked, small_map, table)
    assert rf.CLOAK_OFF in kinds and rf.CLOAK_ON not in kinds


def test_a_jump_must_be_charged_and_the_policy_can_see_that(small_map, table):
    charged = make_units({"uid": 1, "name": "jumpraid", "flags": rf.IN_LOS | rf.TYPE_KNOWN | rf.JUMP_READY})
    waiting = make_units({"uid": 1, "name": "jumpraid"})
    t, kinds = kinds_of(charged, small_map, table)
    assert rf.JUMP in kinds and t.unit_x[0, fz._I + 18] == 1.0
    t, kinds = kinds_of(waiting, small_map, table)
    assert rf.JUMP not in kinds and t.unit_x[0, fz._I + 18] == 0.0 and not t.ptr_jump.any()


def test_a_jump_can_cross_what_the_unit_cannot_walk_but_only_within_reach(small_map, table):
    """Nothing's walkable from where it stands, so a walk mask would leave it its own region."""
    walled = copy.copy(small_map)
    walled.walk = np.full_like(small_map.walk, np.inf)
    centres = small_map.centre[:, :2]
    off_centre = np.sqrt(small_map.area / np.pi) * 0.5
    reach = float(table.jump_range[table.names.index("jumpraid")])
    # Just inside what a jump reaches of the region next to the first.
    next_door = 1 + int(np.argmin(np.hypot(*(centres[1:] - centres[0]).T)))
    line = centres[next_door] - centres[0]
    at = centres[next_door] - line / np.hypot(*line) * (reach + off_centre[next_door] - 10.0)
    u = make_units({"uid": 1, "name": "jumpraid", "x": float(at[0]), "z": float(at[1]), "flags": rf.IN_LOS | rf.TYPE_KNOWN | rf.JUMP_READY})
    t = run(u, walled, table)
    assert t.ptr_region[0].sum() == 1
    assert t.ptr_jump[0, next_door] and not t.ptr_region[0, next_door]
    away = np.hypot(*(centres - at).T)
    assert (away[t.ptr_jump[0]] <= reach + off_centre[t.ptr_jump[0]] + 1e-3).all() and not t.ptr_jump[0].all()


def test_a_jump_is_not_offered_where_it_could_not_land(small_map, table):
    blocked = copy.copy(small_map)
    blocked.passable = small_map.passable.copy()
    kbot = bd.MOVE_CLASSES.index("kbot")
    c = small_map.centre[1]
    blocked.passable[int(c[1] / bd.PASS_CELL), int(c[0] / bd.PASS_CELL)] &= 255 ^ (1 << kbot)
    at = small_map.centre[0]
    u = make_units({"uid": 1, "name": "jumpraid", "x": float(at[0]), "z": float(at[1]), "flags": rf.IN_LOS | rf.TYPE_KNOWN | rf.JUMP_READY})
    assert not run(u, blocked, table).ptr_jump[0, 1]


def test_who_jumps_agrees_with_the_sim(table):
    cat = json.loads((ROOT / "sim/data/catalog.json").read_text("utf8"))
    ranges = {u["name"]: u["jump_range"] for u in cat["units"]}
    for row, name in enumerate(table.names):
        if table.jump[row] and table.mobile[row] and name in ranges:
            assert ranges[name] > 0.0 and table.jump_range[row] == ranges[name], name
    assert table.jump_range[table.names.index("jumpsumo")] == 360.0


def test_a_live_jumper_is_charged_by_its_rules_param():
    class Client:
        def __init__(self, value):
            self.value = value

        def query(self, q, timeout=10.0):
            assert q["type"] == "unit_rules_param" and q["key"] == "jumpReload"
            return {"floatValue": self.value}

    class Brain:
        def __init__(self, client):
            self.client = client

    pyro, glaive = {"defId": unit_data.id_of("jumpraid")}, {"defId": unit_data.id_of("cloakraid")}
    assert rf._jump_charged(Brain(Client(1.0)), 5, pyro) and not rf._jump_charged(Brain(Client(0.4)), 5, pyro)
    assert not rf._jump_charged(Brain(Client(1.0)), 5, glaive)
    assert rf._jump_charged(object(), 5, pyro)      # no game behind it


def test_manual_fire_needs_range_and_reload(small_map, table):
    name = next(n for n in table.names if table.manualfire[table.names.index(n)] and table.combat[table.names.index(n)]
                and table.mobile[table.names.index(n)] and not table.air[table.names.index(n)])
    row = table.names.index(name)
    reach = float(table.manual_range[row])
    near = {"uid": 2, "side": ENEMY, "x": 500.0 + reach * 0.5}
    far = {"uid": 2, "side": ENEMY, "x": 500.0 + reach * 2.0}
    assert rf.MANUAL_FIRE in kinds_of(make_units({"uid": 1, "name": name}, near), small_map, table)[1]
    assert rf.MANUAL_FIRE not in kinds_of(make_units({"uid": 1, "name": name}, far), small_map, table)[1]


def test_stage_2_masks_off_construction(small_map, table):
    assert not run(make_units({"uid": 1}), small_map, table).kind_mask[:, 10:].any()
    assert run(make_units({"uid": 1}), small_map, table, enabled_kinds=(0, 1)).kind_mask[0].tolist()[:2] == [True, True]
    assert not run(make_units({"uid": 1}), small_map, table, enabled_kinds=(0, 1)).kind_mask[0, 2:].any()


def test_only_finished_ground_combat_units_are_controllable(small_map, table):
    units = make_units({"uid": 1, "name": "cloakcon"}, {"uid": 2, "name": "staticmex"}, {"uid": 3, "name": "gunshipraid"},
                       {"uid": 4, "flags": rf.IN_LOS | rf.TYPE_KNOWN | rf.BEING_BUILT}, {"uid": 5, "flags": rf.IN_LOS | rf.TYPE_KNOWN | rf.RESERVED},
                       {"uid": 6})
    t = run(units, small_map, table)
    assert t.controllable == 1 and t.uids[0] == 6


def test_every_unit_of_every_type_has_a_legal_kind_and_target(small_map, table):
    combat = [n for i, n in enumerate(table.names) if table.combat[i] and table.mobile[i] and not table.air[i]]
    # Waterborne classes stand on land here: their own region must still be legal.
    units = make_units(*[{"uid": i + 1, "name": n, "x": 300.0 + i % 10 * 40, "z": 300.0 + i // 10 * 40} for i, n in enumerate(combat[:128])],
                       {"uid": 5000, "side": ENEMY, "x": 800.0, "z": 800.0})
    t = run(units, small_map, table)
    assert t.kind_mask.any(axis=1).all()
    assert t.ptr_region.any(axis=1).all()
    for k, slot in ((rf.ATTACK_UNIT, 0), (rf.ATTACK_GROUND, 1), (rf.MANUAL_FIRE, 2)):
        assert t.ptr_enemy[slot][t.kind_mask[:, k]].any(axis=1).all()


def test_hidden_units_only_ever_reach_the_critic(small_map, table):
    units = make_units({"uid": 1}, {"uid": 2, "side": HIDDEN, "x": 2000.0, "z": 2000.0})
    actor = run(units, small_map, table)
    assert 2 not in actor.uids.tolist() and not len(actor.hidden_x)
    critic = run(units, small_map, table, privileged=True)
    assert 2 not in critic.uids.tolist()
    assert len(critic.hidden_x) == 1 and critic.hidden_x[0, fz._I + 0] > 0.5
    assert np.array_equal(actor.unit_x, critic.unit_x)


# The live producer, over a stand-in for the host (bots/zkbot/host.py).
class FakeHost:
    def __init__(self):
        self.unit_order = {1: ("fight", 900.0, 910.0, None, 870), 2: ("attack", 0.0, 0.0, 20, 840)}
        self.units = {1: SimpleNamespace(name="cloakraid", is_mobile=True, is_finished=True, is_builder=False, uid=1, alive_frac=0.5),
                      2: SimpleNamespace(name="cloaksnipe", is_mobile=True, is_finished=True, is_builder=False, uid=2, alive_frac=1.0),
                      3: SimpleNamespace(name="cloakcon", is_mobile=True, is_finished=True, is_builder=True, uid=3, alive_frac=1.0)}
        self.commander_id = None
        self.team_id, self.ally_team_id = 3, 1
        self.base_pos = (100.0, 200.0)
        self.map = SimpleNamespace(width=2560.0, height=2560.0, spots=[SimpleNamespace(x=400.0, z=500.0, income=2.5, status="ours")])
        self.blip_tracks = {30: {"smax": 88.0}}
        self.last_enemy_seen_frame = 880
        self.enemy_structures = {40: (SimpleNamespace(def_id=7, x=1.0, y=2.0, z=3.0, health=50.0, max_health=100.0, build_progress=1.0, is_being_built=False), 600),
                                 20: (SimpleNamespace(def_id=7, x=1.0, y=2.0, z=3.0, health=50.0, max_health=100.0, build_progress=1.0, is_being_built=False), 600)}

    def metal(self):
        return {"storage": 10500.0}

    def energy(self):
        return {"storage": 500.0}

    def metal_current(self):
        return 70.0

    def metal_income(self):
        return 4.0

    def energy_current(self):
        return 30.0

    def energy_income(self):
        return 6.0

    @staticmethod
    def effective_storage(resource):
        raw = resource["storage"]
        return raw - 10000.0 if raw > 10000.0 else raw


def test_from_host_reads_what_is_left_of_a_feature_in_sight():
    obs = {"frame": 900, "units": {}, "enemies": {}, "radarBlips": {},
           "features": {"9": {"defId": 12, "pos": [100.0, 5.0, 200.0], "reclaimLeft": 0.5, "definition": {"containedResources": {"Metal": 60.0, "Energy": 0.0}}},
                        "10": {"defId": 13, "pos": [300.0, 5.0, 400.0], "definition": {"containedResources": {"Metal": 20.0}}}}}
    f = rf.from_host(FakeHost(), obs, rf.LiveMemory(), region_of=lambda x, z: 4)
    by_id = {int(r["feat_id"]): r for r in f.features}
    assert (by_id[9]["x"], by_id[9]["z"], by_id[9]["metal"], by_id[9]["def_id"]) == (100.0, 200.0, 30.0, 12)
    assert by_id[10]["metal"] == 20.0


def test_from_host_reads_everything_the_schema_asks_for():
    obs = {
        "frame": 900,
        "units": {"1": {"defId": 5, "pos": [10.0, 20.0, 30.0], "vel": [1.0, 0.0, -1.0], "health": 50.0, "maxHealth": 100.0, "buildProgress": 1.0, "experience": 0.25},
                  "2": {"defId": 6, "pos": [0, 0, 0], "vel": [0, 0, 0], "health": 10.0, "maxHealth": 10.0, "buildProgress": 1.0, "isCloaked": True},
                  "3": {"defId": 7, "pos": [0, 0, 0], "vel": [0, 0, 0], "health": 10.0, "maxHealth": 10.0, "buildProgress": 0.5, "isBeingBuilt": True, "isParalyzed": True}},
        "enemies": {"20": {"defId": 8, "allyTeam": 0, "pos": [40.0, 0.0, 50.0], "vel": [0.5, 0, 0.5], "health": 5.0, "maxHealth": 20.0, "buildProgress": 1.0}},
        "radarBlips": {"30": {"allyTeam": 0, "pos": [60.0, 0.0, 70.0], "vel": [2.0, 0, 0.0], "isRadarBlip": True}},
    }
    memory = rf.LiveMemory()
    memory.health = {1: 60.0, 2: 10.0}
    memory.note_events({"frame": 860, "events": [{"topic": 7, "unitId": 1}]})    # before its last order: no longer idle
    memory.note_events({"frame": 885, "events": [{"topic": 7, "unitId": 3}, {"topic": 9, "unitId": 2}]})
    f = rf.from_host(FakeHost(), obs, memory, region_of=lambda x, z: 4)
    u = {int(r["uid"]): r for r in f.units}
    assert set(u) == {1, 2, 3, 20, 30, 40}
    assert (u[1]["x"], u[1]["y"], u[1]["z"], u[1]["vx"], u[1]["vz"]) == (10.0, 20.0, 30.0, 1.0, -1.0)
    assert u[1]["hp_frac"] == 0.5 and u[1]["experience"] == 0.25
    assert u[1]["flags"] & rf.DAMAGED and not u[1]["flags"] & rf.IDLE
    assert u[1]["ord_kind"] == rf.FIGHT and u[1]["ord_age"] == 30 and (u[1]["ord_x"], u[1]["ord_z"]) == (900.0, 910.0) and u[1]["ord_target"] == -1
    assert u[2]["ord_kind"] == rf.ATTACK_UNIT and u[2]["ord_target"] == 20 and u[2]["flags"] & rf.CLOAKED and not u[2]["flags"] & rf.RESERVED
    assert not u[2]["flags"] & rf.DAMAGED
    assert u[3]["flags"] & rf.IDLE and u[3]["flags"] & rf.BEING_BUILT and u[3]["flags"] & rf.PARALYZED and u[3]["ord_kind"] == -1
    assert u[20]["knowledge"] == rf.KNOW_LOS and u[20]["flags"] & rf.IN_LOS and u[20]["hp_frac"] == 0.25 and u[20]["vx"] == 0.5
    assert u[30]["knowledge"] == rf.KNOW_RADAR and u[30]["def_id"] == -1 and np.isnan(u[30]["hp_frac"]) and u[30]["blip_top_speed"] == 88.0
    assert not u[30]["flags"] & (rf.IN_LOS | rf.TYPE_KNOWN)
    # A remembered structure is a ghost only while it isn't in sight.
    assert u[40]["knowledge"] == rf.KNOW_MEMORY and u[40]["ord_age"] == 300 and u[40]["flags"] & rf.TYPE_KNOWN
    assert (u[1]["player"], u[1]["ally_team"]) == (3, 1)
    # No player for a hostile unit in the observation yet: its alliance stands in, and a structure from before keeps the last one seen.
    assert (u[20]["player"], u[20]["ally_team"]) == (0, 0) == (u[30]["player"], u[30]["ally_team"]) == (u[40]["player"], u[40]["ally_team"])
    g = f.glob
    assert (g[rf.G["player"]], g[rf.G["ally_team"]]) == (3, 1)
    assert g[rf.G["frame"]] == 900 and g[rf.G["metal_storage"]] == 500 and g[rf.G["energy_storage"]] == 500
    assert g[rf.G["frames_since_contact"]] == 20 and g[rf.G["idle_builders"]] == 1 and g[rf.G["commander_uid"]] == -1
    assert f.sites["status"].tolist() == [rf.SITE_OURS] and f.sites["region_id"].tolist() == [4]
    assert f.idle_uids.tolist() == [3]
