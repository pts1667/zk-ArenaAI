"""The actions of the units that do more than walk and shoot: what's legal, what decoding makes of them, and what the sim does when told."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zkbot import catalog, units
from zkbot.policy import actions, featurize as fz, rawframe as rf

from conftest import BLIP, ENEMY, MEMORY, make_frame, make_units

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sim" / "py"), str(ROOT / "tools")]

READY = rf.IN_LOS | rf.TYPE_KNOWN | rf.JUMP_READY
PLAIN = rf.IN_LOS | rf.TYPE_KNOWN


def run(units, small_map, table):
    return fz.featurize(make_frame(units), small_map, table)


def legal(t, i=0):
    return {k for k in range(rf.KIND_COUNT) if t.kind_mask[i, k]}


# What's legal --------------------------------------------------------------------------------------------------------------------------------------------


def test_order_memory_keeps_its_width_when_the_policy_gains_kinds():
    """Frames on disk and encoders trained on them have 16 slots."""
    assert rf.ORDER_SLOTS == 16 and rf.KIND_COUNT > rf.ORDER_SLOTS
    assert fz.UNIT_DIM == 127 and fz.FEATURE_VERSION == 5


def test_the_facts_come_from_the_generated_data(table):
    row = table.names.index
    assert table.throw_radius[row("amphlaunch")] == 160.0 and table.throw_reach[row("amphlaunch")] == 620.0
    assert table.switches[row("jumpsumo")] and table.switches[row("turretimpulse")] and not table.switches[row("cloakraid")]
    assert table.teleports[row("amphtele")] and table.beacon[row("tele_beacon")] and not table.beacon[row("amphtele")]
    made = lambda maker: {table.names[r] for r in table.producible[table.produces[row(maker)]]}  # noqa: E731
    assert made("staticmissilesilo") == {"tacnuke", "seismic", "empmissile", "napalmmissile", "missileslow"}
    assert made("factorycloak") == set(catalog.roster("factorycloak")) and "cloakraid" in made("factorycloak")
    assert "cloakraid" not in made("factoryshield") and table.acts_in_place[row("factorycloak")]
    assert table.acts_in_place[row("turretimpulse")] and table.acts_in_place[row("staticmissilesilo")] and not table.acts_in_place[row("turretriot")]


def test_a_factory_chooses_what_to_build_among_its_roster_but_not_what_it_builds_already(small_map, table):
    x, z = (float(v) for v in small_map.centre[0])
    roster = set(catalog.roster("factorycloak"))
    t = run(make_units({"uid": 1, "name": "factorycloak", "x": x, "z": z}), small_map, table)
    assert t.controllable == 1 and legal(t) == {rf.KEEP, rf.PRODUCE}
    assert {table.names[r] for r in table.producible[t.ptr_produce[0]]} == roster
    making = run(make_units({"uid": 1, "name": "factorycloak", "x": x, "z": z, "ord_kind": rf.PRODUCE, "ord_target": units.id_of("cloakraid"), "ord_age": 900.0}), small_map, table)
    assert {table.names[r] for r in table.producible[making.ptr_produce[0]]} == roster - {"cloakraid"}
    # What it makes is a type, not a unit it aims at, even when a uid happens to have that number.
    aimed = make_units({"uid": 1, "name": "factorycloak", "x": x, "z": z, "ord_kind": rf.PRODUCE, "ord_target": units.id_of("cloakraid"), "ord_age": 900.0},
                       {"uid": units.id_of("cloakraid"), "side": ENEMY, "x": x + 100.0, "z": z})
    t = run(aimed, small_map, table)
    assert t.unit_x[0, fz._O + 20] == 0.0
    unfinished = run(make_units({"uid": 1, "name": "factorycloak", "x": x, "z": z, "flags": PLAIN | rf.BEING_BUILT}), small_map, table)
    assert unfinished.controllable == 0


def test_missiles_and_the_trinity_are_the_policys_to_aim(small_map, table):
    """At what's on the ground in sight or on radar, not what's only remembered, which isn't a token."""
    x, z = (float(v) for v in small_map.centre[0])
    assert table.missile[table.names.index("tacnuke")] and table.launcher[table.names.index("staticnuke")]
    others = ({"uid": 10, "name": "staticmex", "side": BLIP, "x": x + 800.0, "z": z},
              {"uid": 11, "name": "energyfusion", "side": MEMORY, "x": x + 900.0, "z": z})
    aims = lambda t, k: {int(t.uids[t.enemy_tokens[j]]) for j in np.flatnonzero(t.ptr_enemy[k, 0])}  # noqa: E731
    missile = run(make_units({"uid": 1, "name": "tacnuke", "x": x, "z": z}, *others), small_map, table)
    assert missile.controllable == 1 and legal(missile) == {rf.KEEP, rf.ATTACK_UNIT} and aims(missile, 0) == {10}
    nuke = run(make_units({"uid": 1, "name": "staticnuke", "x": x, "z": z}, *others), small_map, table)
    assert nuke.controllable == 1 and legal(nuke) == {rf.KEEP, rf.MANUAL_FIRE} and aims(nuke, 2) == {10}


def test_a_throw_needs_a_ready_thrower_with_someone_beside_it(small_map, table):
    x, z = (float(v) for v in small_map.centre[0])
    friend = {"uid": 2, "x": x + 50.0, "z": z}
    assert rf.THROW in legal(run(make_units({"uid": 1, "name": "amphlaunch", "x": x, "z": z, "flags": READY}, friend), small_map, table))
    assert rf.THROW not in legal(run(make_units({"uid": 1, "name": "amphlaunch", "x": x, "z": z}, friend), small_map, table))
    alone = run(make_units({"uid": 1, "name": "amphlaunch", "x": x, "z": z, "flags": READY}, {"uid": 2, "x": x + 400.0, "z": z}), small_map, table)
    assert rf.THROW not in legal(alone) and not alone.ptr_jump[0].any()


def test_a_throw_reaches_only_so_far_and_a_jump_is_not_what_it_does(small_map, table):
    x, z = (float(v) for v in small_map.centre[0])
    t = run(make_units({"uid": 1, "name": "amphlaunch", "x": x, "z": z, "flags": READY}, {"uid": 2, "x": x + 50.0, "z": z}), small_map, table)
    away = np.hypot(small_map.centre[:, 0] - x, small_map.centre[:, 1] - z)
    off_centre = np.sqrt(small_map.area / np.pi) * 0.5
    assert t.ptr_jump[0].any() and (away[t.ptr_jump[0]] <= 620.0 + off_centre[t.ptr_jump[0]] + 1e-3).all()
    assert rf.JUMP not in legal(t)


def test_something_that_cannot_walk_is_given_orders_only_for_what_it_can_do(small_map, table):
    x, z = (float(v) for v in small_map.centre[0])
    t = run(make_units({"uid": 1, "name": "turretimpulse", "x": x, "z": z}), small_map, table)
    assert t.controllable == 1 and legal(t) == {rf.KEEP, rf.STOP, rf.PULL}
    t = run(make_units({"uid": 1, "name": "staticmissilesilo", "x": x, "z": z}), small_map, table)
    assert t.controllable == 1 and legal(t) == {rf.KEEP, rf.PRODUCE} and t.ptr_produce[0].sum() == 5
    t = run(make_units({"uid": 1, "name": "turretriot", "x": x, "z": z}), small_map, table)
    assert t.controllable == 0


def test_a_turret_does_not_change_where_the_army_is(small_map, table):
    """North and which units come first go by the army's centre, and frames made so far had no turret in it."""
    x, z = (float(v) for v in small_map.centre[0])
    walkers = ({"uid": 1, "x": x, "z": z}, {"uid": 2, "x": x + 60.0, "z": z}, {"uid": 3, "side": ENEMY, "x": x, "z": z + 900.0})
    alone = run(make_units(*walkers), small_map, table)
    with_turret = run(make_units(*walkers, {"uid": 4, "name": "turretimpulse", "x": x + 1500.0, "z": z - 500.0}), small_map, table)
    assert np.allclose(alone.north, with_turret.north)


def test_push_and_pull_go_by_which_it_is_doing_and_the_policy_can_see_that(small_map, table):
    x, z = (float(v) for v in small_map.centre[0])
    pushing = run(make_units({"uid": 1, "name": "jumpsumo", "x": x, "z": z}), small_map, table)
    pulling = run(make_units({"uid": 1, "name": "jumpsumo", "x": x, "z": z, "flags": PLAIN | rf.PULLING}), small_map, table)
    assert rf.PULL in legal(pushing) and rf.PUSH not in legal(pushing) and pushing.unit_x[0, fz._O + rf.RESERVED_KIND] == 0.0
    assert rf.PUSH in legal(pulling) and rf.PULL not in legal(pulling) and pulling.unit_x[0, fz._O + rf.RESERVED_KIND] == 1.0
    assert not ({rf.PUSH, rf.PULL} & legal(run(make_units({"uid": 1, "x": x, "z": z}), small_map, table)))


def test_a_beacon_is_placed_by_a_teleporter_and_waited_at_by_others(small_map, table):
    x, z = (float(v) for v in small_map.centre[0])
    bx, bz = (float(v) for v in small_map.centre[1])
    djinn, raider = {"uid": 1, "name": "amphtele", "x": x, "z": z}, {"uid": 2, "x": x + 30.0, "z": z}
    t = run(make_units(djinn, raider), small_map, table)
    assert rf.PLACE_BEACON in legal(t, 0) and rf.PLACE_BEACON not in legal(t, 1) and rf.WAIT_AT_BEACON not in legal(t, 1)
    t = run(make_units(djinn, raider, {"uid": 3, "name": "tele_beacon", "x": bx, "z": bz}), small_map, table)
    assert rf.WAIT_AT_BEACON not in legal(t, 0) and rf.WAIT_AT_BEACON in legal(t, 1)
    assert t.ptr_wait[1, 1] and t.ptr_wait[1].sum() == 1 and t.uids[t.own_beacons].tolist() == [3] and t.beacon_regions.tolist() == [1]


def test_a_beacon_that_is_not_ours_or_not_finished_is_nowhere_to_wait(small_map, table):
    x, z = (float(v) for v in small_map.centre[0])
    bx, bz = (float(v) for v in small_map.centre[1])
    raider = {"uid": 2, "x": x, "z": z}
    assert rf.WAIT_AT_BEACON not in legal(run(make_units(raider, {"uid": 3, "name": "tele_beacon", "side": ENEMY, "x": bx, "z": bz}), small_map, table))
    building = {"uid": 3, "name": "tele_beacon", "x": bx, "z": bz, "flags": PLAIN | rf.BEING_BUILT}
    assert rf.WAIT_AT_BEACON not in legal(run(make_units(raider, building), small_map, table))


def test_kinds_left_out_are_not_legal(small_map, table):
    u = make_units({"uid": 1, "name": "jumpsumo"})
    t = fz.featurize(make_frame(u), small_map, table, enabled_kinds=(rf.KEEP, rf.MOVE))
    assert legal(t) == {rf.KEEP, rf.MOVE}


# Decoding ------------------------------------------------------------------------------------------------------------------------------------------------


def decide(t, **chosen):
    c = t.controllable
    base = {k: np.full(c, -1, np.int64) for k in ("region", "enemy", "fine", "produce")}
    base["kind"] = np.full(c, rf.KEEP, np.int64)
    for name, values in chosen.items():
        base[name][: len(values)] = values
    return SimpleNamespace(**base)


def test_decoding_each_new_kind(small_map, table):
    x, z = (float(v) for v in small_map.centre[0])
    bx, bz = (float(v) for v in small_map.centre[1])
    units = make_units({"uid": 1, "name": "amphlaunch", "x": x, "z": z, "flags": READY}, {"uid": 2, "name": "amphtele", "x": x, "z": z},
                       {"uid": 3, "name": "jumpsumo", "x": x, "z": z}, {"uid": 4, "name": "turretimpulse", "x": x, "z": z, "flags": PLAIN | rf.PULLING},
                       {"uid": 5, "x": x, "z": z}, {"uid": 6, "name": "staticmissilesilo", "x": x, "z": z}, {"uid": 7, "name": "tele_beacon", "x": bx, "z": bz})
    t = run(units, small_map, table)
    at = {int(t.uids[i]): i for i in range(t.controllable)}
    assert set(at) == {1, 2, 3, 4, 5, 6}
    kind, region, fine, produce = [np.full(6, v, np.int64) for v in (rf.KEEP, -1, -1, -1)]
    for uid, k, r, f, p in ((1, rf.THROW, 1, 0, -1), (2, rf.PLACE_BEACON, 1, 3, -1), (3, rf.PULL, -1, -1, -1), (4, rf.PUSH, -1, -1, -1), (5, rf.WAIT_AT_BEACON, 1, -1, -1), (6, rf.PRODUCE, -1, -1, 2)):
        i = at[uid]
        assert t.kind_mask[i, k], uid
        kind[i], region[i], fine[i], produce[i] = k, r, f, p
    x_all, z_all = actions.world_positions(t, small_map)
    orders = {o.uid: o for o in actions.decode(decide(t, kind=kind, region=region, fine=fine, produce=produce), t, small_map, x_all, z_all)}
    assert orders[1].kind == rf.THROW
    want = fz.fine_point(t, small_map, 1, 0)
    # This region's past its reach: the aim is short of it, on the way there.
    assert np.hypot(want[0] - x, want[1] - z) > 620.0 and orders[1].x == pytest.approx(x + 620.0 * actions.THROW_SHORT) and orders[1].z == pytest.approx(z)
    assert (orders[2].kind, orders[2].x, orders[2].z) == (rf.PLACE_BEACON, *fz.fine_point(t, small_map, 1, 3))
    assert orders[3].kind == rf.PULL and orders[4].kind == rf.PUSH
    assert orders[5].kind == rf.WAIT_AT_BEACON and orders[5].target == 7
    assert orders[6].kind == rf.PRODUCE and orders[6].target == table.producible[2]


def test_a_throw_past_its_reach_goes_as_far_as_it_can_the_same_way(small_map, table):
    """Otherwise it would walk off, without those it meant to throw."""
    x, z = (float(v) for v in small_map.centre[0])
    away = np.hypot(small_map.centre[:, 0] - x, small_map.centre[:, 1] - z)
    region = int(np.argmax(away))
    t = run(make_units({"uid": 1, "name": "amphlaunch", "x": x, "z": z, "flags": READY}, {"uid": 2, "x": x + 50.0, "z": z}), small_map, table)
    i = int(np.flatnonzero(t.uids[:t.controllable] == 1)[0])
    kind, chosen = [np.full(t.controllable, v, np.int64) for v in (rf.KEEP, -1)]
    fine = chosen.copy()
    kind[i], chosen[i], fine[i] = rf.THROW, region, 0
    ux, uz = actions.world_positions(t, small_map)
    order = next(o for o in actions.decode(decide(t, kind=kind, region=chosen, fine=fine), t, small_map, ux, uz) if o.uid == 1)
    assert away[region] > 620.0
    assert np.hypot(order.x - ux[i], order.z - uz[i]) == pytest.approx(620.0 * actions.THROW_SHORT, abs=0.1)
    to = small_map.centre[region] - np.array((ux[i], uz[i]))
    assert abs((order.x - ux[i]) * to[1] - (order.z - uz[i]) * to[0]) < 1.0 * float(np.hypot(*to))     # the same line, to an elmo


# Torch ---------------------------------------------------------------------------------------------------------------------------------------------------


def _actor(table, encoder_path):
    """An untrained actor that is never done while there's a unit to tell something, so every unit and kind gets chosen."""
    torch = pytest.importorskip("torch")
    import policy_latent as pl

    torch.manual_seed(0)
    actor = pl.make(len(table.names) + 1, encoder_path).eval()
    with torch.no_grad():
        actor.head_done.bias[0] = -20.0
    return actor


def test_the_actor_chooses_from_what_is_legal_and_says_what_to_produce(small_map, table, encoder_path):
    torch = pytest.importorskip("torch")
    import policy_model as pm

    x, z = (float(v) for v in small_map.centre[0])
    u = make_units({"uid": 1, "name": "staticmissilesilo", "x": x, "z": z}, {"uid": 2, "name": "amphlaunch", "x": x + 40.0, "z": z, "flags": READY}, {"uid": 3, "x": x + 60.0, "z": z})
    t = run(u, small_map, table)
    actor = _actor(table, encoder_path)
    b = pm.collate([t] * 60).to("cpu")
    with torch.no_grad():
        ctx, *_ = actor.decide(b, *actor.initial_state(60, 0, "cpu"))
        a = pm.sample_actions(actor, b, ctx)
    seen = set()
    for row in range(60):
        picked = a.anchor[row] >= 0
        anchor, kind, produce, region, fine = (v[row][picked].numpy() for v in (a.anchor, a.kind, a.produce, a.region, a.fine))
        assert sorted(anchor) == [u for u in range(t.controllable) if t.kind_mask[u].sum() > t.kind_mask[u, rf.KEEP]]
        seen |= {int(k) for k in kind}
        for i, k in zip(anchor, kind):
            assert t.kind_mask[i, k] and k != rf.KEEP
        for i, k, p, r, f in zip(anchor, kind, produce, region, fine):
            assert (p >= 0) == (k == rf.PRODUCE)
            assert (r >= 0) == (k in fz.KINDS_WITH_REGION)
            assert (f >= 0) == (k in fz.KINDS_WITH_FINE)
            if k == rf.THROW:
                assert t.ptr_jump[i, r]
    assert {rf.PRODUCE, rf.THROW} <= seen


def test_sampling_and_evaluating_agree_on_the_new_head(small_map, table, encoder_path):
    torch = pytest.importorskip("torch")
    import policy_model as pm

    x, z = (float(v) for v in small_map.centre[0])
    bx, bz = (float(v) for v in small_map.centre[1])
    u = make_units({"uid": 1, "name": "staticmissilesilo", "x": x, "z": z}, {"uid": 2, "name": "amphlaunch", "x": x + 40.0, "z": z, "flags": READY}, {"uid": 3, "x": x + 60.0, "z": z},
                   {"uid": 4, "name": "amphtele", "x": x + 20.0, "z": z}, {"uid": 5, "name": "tele_beacon", "x": bx, "z": bz}, {"uid": 6, "name": "jumpsumo", "x": x, "z": z + 20.0})
    t = run(u, small_map, table)
    actor = _actor(table, encoder_path)
    b = pm.collate([t] * 8).to("cpu")
    with torch.no_grad():
        ctx, *_ = actor.decide(b, *actor.initial_state(8, 0, "cpu"))
        a = pm.sample_actions(actor, b, ctx)
        logp, entropy = pm.evaluate_actions(actor, b, ctx, a)
    assert torch.allclose(a.logp, logp, atol=1e-3)
    uses = a.kind == rf.PRODUCE
    assert uses.any() and ((a.produce >= 0) == uses).all() and "produce" in entropy
    assert torch.isin(a.kind, torch.tensor(fz.KINDS_WITH_REGION))[a.region >= 0].all()


# In the sim ----------------------------------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim():
    try:
        import map_gen as mg
        import policy_sim_env as se
    except OSError:
        pytest.skip("needs the built sim library")
    return mg, se


@pytest.fixture(scope="module")
def land(sim, tmp_path_factory):
    mg, se = sim
    out = tmp_path_factory.mktemp("maps")
    made = mg.make(1, mg.Rules(waters=(("none", 1.0),), sides=(8,), side_weights=(1.0,), square=1.0))
    mg.write_entry(made.map, "made", made.layouts, made.features, out, made.style)
    return se.load_pool(out)


class Game:
    """One sim game with these units standing together for each side, all told by the policy's own path: masks, decoding, sim_orders."""

    def __init__(self, se, land, names, table, seed=1):
        self.se, self.table = se, table

        class Fixed(se.SimEnv):
            def _start_armies(env, g, world_map, bundle, x0, z0, x1, z1):
                cat = se.catalog()
                types = np.array([cat.index[n] for n in names], np.int32)
                kbot = np.full(len(names), se.bd.MOVE_CLASSES.index("kbot"), np.int32)
                ax, az = se.place(bundle, env.rng, kbot, x0, z0)
                bx, bz = se.place(bundle, env.rng, kbot, x1, z1)
                # Together: a throw takes who's beside the thrower.  The first stands where it was put, on ground its class can stand on.
                beside = 30.0 * np.arange(len(names), dtype=np.float32)
                ax, az, bx, bz = ax[0] + beside, np.full(len(names), az[0]), bx[0] + beside, np.full(len(names), bz[0])
                env.batch.reset(g, world_map, seed, np.tile(types, 2), np.repeat([0, 1], len(names)), np.concatenate((ax, bx)), np.concatenate((az, bz)),
                                base=(x0, z0, x1, z1), sites=bundle.sites())
                env.batch.economy(g, bundle.spot_list, metal=50000.0, energy=50000.0)
                env.scripts[g] = (None, None)

        self.env = Fixed(1, seed, ["mirror"], units=len(names), threads=1, maps=land)
        self.env.begin()
        self.bundle = self.env.bundle_of(0)
        self.names = list(names)

    def tensors(self, seat=0):
        return fz.featurize(self.env.frame(seat), self.bundle, self.table, enabled_kinds=self.se.KINDS)

    def own(self, seat=0):
        f = self.env.frame(seat).units
        return f[f["player"] == seat]

    def tell(self, chosen):
        """chosen: unit-type name to (kind, region or None, produce or None).  The rest keep what they have."""
        t = self.tensors()
        kind, region, fine, produce = [np.full(t.controllable, v, np.int64) for v in (rf.KEEP, -1, -1, -1)]
        for i in range(t.controllable):
            name = self.table.names[t.unit_name[i]]
            if name in chosen:
                k, r, p = chosen[name]
                assert t.kind_mask[i, k], (name, k)
                kind[i], produce[i] = k, -1 if p is None else p
                if k in fz.KINDS_WITH_REGION:
                    mask = {rf.THROW: t.ptr_jump, rf.WAIT_AT_BEACON: t.ptr_wait}.get(k, t.ptr_region)[i]
                    region[i] = int(np.flatnonzero(mask)[0]) if r is None else r
                    assert mask[region[i]]
                if k in fz.KINDS_WITH_FINE:
                    fine[i] = 0
        x, z = actions.world_positions(t, self.bundle)
        orders = actions.decode(decide(t, kind=kind, region=region, fine=fine, produce=produce), t, self.bundle, x, z)
        self.env.give_orders(0, orders)
        self.env.give_orders(1, [])
        self.env.step(1.0)
        return t, orders

    def wait(self, decisions):
        for _ in range(decisions):
            self.env.give_orders(0, [])
            self.env.give_orders(1, [])
            self.env.step(1.0)

    def of(self, name, seat=0):
        rows = self.table.rows(self.own(seat)["def_id"])
        return self.own(seat)[rows == self.table.names.index(name)]


def farthest(game, name, mask_of):
    t = game.tensors()
    i = next(i for i in range(t.controllable) if game.table.names[t.unit_name[i]] == name)
    x, z = actions.world_positions(t, game.bundle)
    away = np.hypot(game.bundle.centre[:, 0] - x[i], game.bundle.centre[:, 1] - z[i])
    return int(np.argmax(np.where(mask_of(t)[i], away, -1.0)))


def test_the_sim_throws_who_the_policy_says(sim, land, table):
    _, se = sim
    game = Game(se, land, ["amphlaunch", "cloakraid", "cloakraid"], table)
    game.wait(2)
    thrower = game.of("amphlaunch")
    assert thrower["flags"][0] & rf.JUMP_READY
    before = game.of("cloakraid")
    region = farthest(game, "amphlaunch", lambda t: t.ptr_jump)
    target = game.bundle.centre[region]
    game.tell({"amphlaunch": (rf.THROW, region, None)})
    thrown, flown = False, 0.0
    for _ in range(40):
        game.wait(1)
        now = game.of("cloakraid")
        flown = max(flown, float(np.hypot(now["x"] - before["x"], now["z"] - before["z"]).max()))
        thrown = thrown or not (game.of("amphlaunch")["flags"][0] & rf.JUMP_READY)
    assert thrown, "it never fired"
    assert flown > 150.0, f"the raiders only went {flown:.0f} elmos towards {target}"


def test_the_sim_switches_between_push_and_pull(sim, land, table):
    _, se = sim
    game = Game(se, land, ["turretimpulse", "jumpsumo"], table)
    game.tell({"turretimpulse": (rf.PULL, None, None), "jumpsumo": (rf.PULL, None, None)})
    assert game.of("turretimpulse")["flags"][0] & rf.PULLING and game.of("jumpsumo")["flags"][0] & rf.PULLING
    t = game.tensors()
    assert all(rf.PUSH in legal(t, i) and rf.PULL not in legal(t, i) for i in range(t.controllable))
    game.tell({"turretimpulse": (rf.PUSH, None, None)})
    assert not game.of("turretimpulse")["flags"][0] & rf.PULLING and game.of("jumpsumo")["flags"][0] & rf.PULLING


def test_the_enemy_does_not_see_which_it_is_doing(sim, land, table):
    _, se = sim
    game = Game(se, land, ["turretimpulse"], table)
    game.tell({"turretimpulse": (rf.PULL, None, None)})
    seen = game.env.frame(1).units
    assert not (seen[seen["player"] == 0]["flags"] & rf.PULLING).any()


def test_the_sim_teleports_who_waits_at_the_beacon_the_policy_placed(sim, land, table):
    """The Djinn stays where it is and leaves a beacon a region away.  The raider walks there, and comes back in one go."""
    _, se = sim
    game = Game(se, land, ["amphtele", "cloakraid"], table)
    game.wait(1)
    djinn = game.of("amphtele")
    t = game.tensors()
    i = next(i for i in range(t.controllable) if table.names[t.unit_name[i]] == "amphtele")
    away = np.hypot(game.bundle.centre[:, 0] - djinn["x"][0], game.bundle.centre[:, 1] - djinn["z"][0])
    near = np.where(t.ptr_region[i] & (away > 300.0), away, np.inf)
    region = int(np.argmin(near))
    game.tell({"amphtele": (rf.PLACE_BEACON, region, None)})
    game.wait(25)
    beacon = game.of("tele_beacon")
    assert len(beacon) == 1, "no beacon after 13 s"
    _, orders = game.tell({"cloakraid": (rf.WAIT_AT_BEACON, int(game.bundle.region_at(beacon["x"], beacon["z"])[0]), None)})
    assert orders[-1].target == int(beacon["uid"][0])
    before, went, came = game.of("cloakraid"), 0.0, False
    for _ in range(240):
        game.wait(1)
        now = game.of("cloakraid")
        step = float(np.hypot(now["x"][0] - before["x"][0], now["z"][0] - before["z"][0]))
        beside = float(np.hypot(now["x"][0] - djinn["x"][0], now["z"][0] - djinn["z"][0]))
        went = max(went, beside)
        before = now
        if step > 150.0 and beside < 200.0:
            came = True
            break
    assert came, f"it got {went:.0f} elmos from the Djinn and never came back at once"


def test_the_sim_makes_the_missile_the_policy_chose(sim, land, table):
    _, se = sim
    game = Game(se, land, ["staticmissilesilo"], table)
    which = int(np.flatnonzero(np.array(table.names)[table.producible] == "tacnuke")[0])
    game.tell({"staticmissilesilo": (rf.PRODUCE, None, which)})
    game.wait(125)
    made = game.of("tacnuke")
    assert len(made) >= 1 and not len(game.of("seismic"))


def test_the_sim_builds_what_the_policy_chose_and_the_frame_says_so(sim, land, table):
    _, se = sim
    game = Game(se, land, ["factorycloak"], table)
    which = int(np.flatnonzero(np.array(table.names)[table.producible] == "cloakskirm")[0])
    t, orders = game.tell({"factorycloak": (rf.PRODUCE, None, which)})
    assert [o.kind for o in orders] == [rf.PRODUCE]
    game.wait(200)
    assert len(game.of("cloakskirm")) >= 1 and not len(game.of("cloakraid"))
    factory = game.of("factorycloak")[0]
    assert factory["ord_kind"] == rf.PRODUCE and factory["ord_target"] == units.id_of("cloakskirm")
    after = game.tensors()
    i = next(i for i in range(after.controllable) if table.names[after.unit_name[i]] == "factorycloak")
    assert not after.ptr_produce[i, which] and after.ptr_produce[i].any()


def test_every_kind_the_policy_has_is_one_the_sim_obeys(sim):
    _, se = sim
    orders = se.sim_orders([actions.Order(3, k, 1.0, 2.0, 4) for k in (rf.THROW, rf.PUSH, rf.PULL, rf.PLACE_BEACON, rf.WAIT_AT_BEACON)])
    zksim = se.zksim
    assert orders["kind"].tolist() == [zksim.ORDER_MANUAL_FIRE, zksim.ORDER_ON_OFF, zksim.ORDER_ON_OFF, zksim.ORDER_PLACE_BEACON, zksim.ORDER_WAIT_AT_BEACON]
    assert orders["target"].tolist() == [4, 1, 0, 4, 4]
    table = se.tables.table()
    row = int(table.producible[0])
    produced = se.sim_orders([actions.Order(3, rf.PRODUCE, target=row)])
    assert produced["kind"][0] == zksim.ORDER_PRODUCE and produced["target"][0] == se.catalog().index[table.names[row]]
    # Those with the same numbers on both sides.
    for k, sim_kind in ((rf.MOVE, zksim.ORDER_MOVE), (rf.FIGHT, zksim.ORDER_FIGHT), (rf.ATTACK_UNIT, zksim.ORDER_ATTACK), (rf.MANUAL_FIRE, zksim.ORDER_MANUAL_FIRE), (rf.JUMP, zksim.ORDER_JUMP)):
        assert se.sim_orders([actions.Order(1, k)])["kind"][0] == sim_kind
