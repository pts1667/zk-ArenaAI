"""What units do on their own: a fight order's search and leash, idle units, hold position, fire states, the tactical AI, overkill and bait
prevention, shooting at radar, and overheating.  Measured against a game in 01-simulator.md section 8.12."""
from __future__ import annotations

import pathlib
import pytest
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402

CAT = zksim.Catalog()
FLAT = zksim.Map.flat()
X, Z = 1000.0, 2000.0
RAIDER, SKIRMISHER, RIOT, ASSAULT, LANCE, STARDUST = "cloakraid", "cloakskirm", "cloakriot", "cloakassault", "hoverarty", "turretriot"
RANGE = CAT.units[CAT.index[RAIDER]]["max_range"]
TOUGH = 1000000.0


def world(seed: int = 1) -> zksim.World:
    return zksim.World(CAT, FLAT, seed=seed)


def dummy(w: zksim.World, name: str, dx: float, dz: float, hp: float | None = None) -> int:
    """Takes what it's given and does nothing."""
    u = w.add(name, 1, X + dx, Z + dz)
    w.set_fire_state(u, 0)
    w.set_move_state(u, 0)
    w.set_unit_ai(u, False)
    if hp:
        w.set_hp(u, hp)
    return u


def seconds(w: zksim.World, s: float) -> None:
    w.step(int(s * 30 / zksim.TICK_FRAMES))


def fight_past(state: int, off_line: float, at: float = 600.0):
    """A raider fights along +z, past a weak target to one side."""
    w = world()
    r = w.add(RAIDER, 0, X, Z)
    w.set_move_state(r, state)
    t = dummy(w, "staticradar", off_line, at)
    w.ready_weapons()
    w.fight(r, X, Z + 2000)
    widest = 0.0
    for _ in range(int(30 * 6)):
        w.step(1)
        widest = max(widest, abs(w.get(r).x - X))
    return w, r, t, widest


def test_fight_goes_for_what_is_within_its_range_plus_100():
    w, r, t, widest = fight_past(1, RANGE + 60)
    assert not w.get(t).alive and widest > 50.0, widest


def test_fight_ignores_what_is_further_off_the_line():
    w, r, t, widest = fight_past(1, RANGE + 160)
    assert w.get(t).alive and widest < 1.0, widest


def test_fight_on_hold_position_only_shoots_what_it_can_reach_and_does_not_leave_the_line():
    w, r, t, widest = fight_past(0, RANGE - 35)
    assert not w.get(t).alive and widest < 30.0, widest
    w, r, t, widest = fight_past(0, RANGE + 60)
    assert w.get(t).alive and widest < 1.0, widest


def test_fight_returns_to_its_line_and_finishes():
    w, r, t, _ = fight_past(1, RANGE + 60)
    s = w.get(r)
    assert abs(s.x - X) < 60.0 and abs(s.z - Z - 2000) < 20.0 and s.order == zksim.ORDER_NONE, (s.x - X, s.z - Z, s.order)


def test_fight_gives_up_at_the_leash():
    w = world()
    r = w.add(RAIDER, 0, X, Z)
    scout = dummy(w, "vehscout", 200, 300, TOUGH)
    w.ready_weapons()
    w.fight(r, X, Z + 2000)
    seconds(w, 1)
    assert w.get(r).chasing == scout or w.get(r).x == X
    w.set_move_state(scout, 1)
    w.move(scout, X + 2500, Z + 300)
    widest, chased_until = 0.0, 0.0
    for tick in range(6 * 12):
        w.step(1)
        s = w.get(r)
        widest = max(widest, s.x - X)
        if s.chasing == scout:
            chased_until = w.get(scout).x - X
    assert widest > 100.0, "it did go after it"
    assert 500.0 + RANGE - 60.0 < chased_until < 500.0 + RANGE + 200.0, chased_until
    seconds(w, 20)
    assert abs(w.get(r).x - X) < 60.0, "and came back"


def idle_near(state: int, gap: float, fire_state: int = 2, after_order: bool = False):
    w = world()
    r = w.add(RAIDER, 0, X, Z)
    w.set_move_state(r, state)
    w.set_fire_state(r, fire_state)
    if after_order:
        w.move(r, X + 5.0, Z)
        seconds(w, 3)
    t = dummy(w, "staticradar", gap, 0)
    w.ready_weapons()
    return w, r, t


def test_idle_goes_for_what_is_within_its_range_plus_200():
    w, r, t = idle_near(1, RANGE + 150)
    seconds(w, 6)
    assert not w.get(t).alive and w.get(r).x - X > 100.0


def test_idle_leaves_alone_what_is_further():
    w, r, t = idle_near(1, RANGE + 250)
    seconds(w, 6)
    assert w.get(t).alive and w.get(r).x == X


def test_idle_on_hold_position_stays_put():
    w, r, t = idle_near(0, RANGE + 150)
    seconds(w, 6)
    assert w.get(t).alive and w.get(r).x == X


def test_idle_goes_back_to_where_it_was():
    w, r, t = idle_near(1, RANGE + 150, after_order=True)
    start = w.get(r).x
    seconds(w, 3)
    assert w.get(r).x - start > 100.0
    seconds(w, 8)
    assert not w.get(t).alive and abs(w.get(r).x - start) < 30.0, w.get(r).x - start


def test_hold_fire_shoots_at_nothing():
    w = world()
    g = w.add(ASSAULT, 0, X, Z)
    w.set_move_state(g, 0)
    w.set_fire_state(g, 0)
    e = dummy(w, RAIDER, 150, 0, TOUGH)
    w.ready_weapons()
    seconds(w, 5)
    assert w.get(e).damage_taken == 0.0


def test_return_fire_waits_to_be_shot_at():
    for provoked in (False, True):
        w = world()
        g = w.add(ASSAULT, 0, X, Z)
        w.set_move_state(g, 0)
        w.set_fire_state(g, 1)
        e = w.add(RAIDER, 1, X + 150, Z)
        w.set_move_state(e, 0)
        w.set_fire_state(e, 2 if provoked else 0)
        w.set_unit_ai(e, False)
        w.set_hp(e, TOUGH)
        w.ready_weapons()
        seconds(w, 5)
        assert (w.get(e).damage_taken > 0.0) == provoked, provoked


def test_hold_fire_still_obeys_an_attack_order():
    w = world()
    g = w.add(ASSAULT, 0, X, Z)
    w.set_fire_state(g, 0)
    e = dummy(w, RAIDER, 150, 0, TOUGH)
    w.ready_weapons()
    w.attack(g, e)
    seconds(w, 3)
    assert w.get(e).damage_taken > 0.0


def test_an_idle_raider_runs_from_a_riot_and_not_from_another_raider():
    w = world()
    r = w.add(RAIDER, 0, X, Z)
    w.move(r, X, Z + 5.0)
    seconds(w, 3)
    o = dummy(w, RIOT, 700, 0, TOUGH)
    w.set_move_state(o, 1)
    w.move(o, X - 1500, Z)
    start = w.get(r).x
    ran = False
    for _ in range(6 * 10):
        w.step(1)
        ran = ran or bool(w.get(r).dodging)
        assert w.get(r).alive
    assert ran and w.get(r).x < start - 50.0, w.get(r).x - start


def test_a_skirmisher_keeps_away_from_an_assault_walking_at_it():
    w = world()
    k = w.add(SKIRMISHER, 0, X, Z)
    a = dummy(w, ASSAULT, 800, 0, TOUGH)
    w.set_move_state(a, 1)
    w.ready_weapons()
    w.fight(k, X + 1500, Z)
    w.move(a, X - 1500, Z)
    reach = CAT.units[CAT.index[ASSAULT]]["max_range"]
    seconds(w, 3)
    for _ in range(6 * 20):
        w.step(1)
        assert w.get(a).x - w.get(k).x > reach, "it never lets the assault reach it"
    assert w.get(a).damage_taken > 0.0, "and shoots it all the while"


def test_a_skirmisher_steps_back_from_a_turret_while_it_reloads():
    w = world()
    k = w.add(SKIRMISHER, 0, X, Z)
    dummy(w, "turretlaser", 700, 0, TOUGH)
    w.ready_weapons()
    w.fight(k, X + 1500, Z)
    seconds(w, 4)
    xs = []
    for _ in range(6 * 16):
        w.step(1)
        xs.append(w.get(k).x - X)
    assert max(xs) - min(xs) > 30.0, (min(xs), max(xs))
    assert 700.0 - max(xs) > 400.0, "it comes in only to its reach"


def test_turning_the_tactical_ai_off_stops_the_stepping():
    w = world()
    k = w.add(SKIRMISHER, 0, X, Z)
    w.set_unit_ai(k, False)
    dummy(w, "turretlaser", 700, 0, TOUGH)
    w.ready_weapons()
    w.fight(k, X + 1500, Z)
    seconds(w, 4)
    for _ in range(6 * 6):
        w.step(1)
        assert not w.get(k).dodging


def shots(name: str, n: int, target: str, gap: float, secs: float, spotter: bool = False) -> int:
    w = world()
    dummy(w, target, 0, gap)
    if spotter:
        w.add("staticradar", 0, X, Z - 100)
    guns = [w.add(name, 0, X + i * 60 - (n - 1) * 30, Z) for i in range(n)]
    for g in guns:
        w.set_move_state(g, 0)
        w.set_unit_ai(g, False)
    w.ready_weapons()
    seconds(w, secs)
    return sum(w.get(g).shots_fired for g in guns)


def test_overkill_prevention_lets_one_shell_go_at_what_it_kills():
    assert shots("tankriot", 5, RAIDER, 250, 4) == 1


def test_overkill_prevention_leaves_alone_what_takes_many():
    assert shots("tankriot", 5, ASSAULT, 250, 4) >= 10


def test_overkill_prevention_that_only_counts_buildings():
    assert shots("cloakarty", 4, "staticradar", 700, 4, spotter=True) == 1
    w = world()
    w.add("staticradar", 0, X, Z - 100)
    t = dummy(w, "cloakskirm", 0, 700)
    guns = [w.add("cloakarty", 0, X + i * 60, Z) for i in range(4)]
    w.ready_weapons()
    w.step(4)
    assert sum(w.get(g).shots_fired for g in guns) == 4, "against a unit they all fire"
    assert w.get(t).alive


def test_overkill_is_off_unless_firing_at_will():
    w = world()
    guns = [w.add("tankriot", 0, X + i * 60, Z) for i in range(3)]
    t = w.add(RAIDER, 1, X, Z + 240)
    w.set_fire_state(t, 0)
    w.set_unit_ai(t, False)
    for g in guns:
        w.set_move_state(g, 0)
        w.set_fire_state(g, 0)
        w.attack(g, t)
    w.ready_weapons()
    w.step(2)
    assert sum(w.get(g).shots_fired for g in guns) >= 3


def test_bait_is_left_alone_unless_ordered():
    w = world()
    g = w.add(LANCE, 0, X, Z)
    solar = dummy(w, "energysolar", 500, 0, TOUGH)
    w.ready_weapons()
    seconds(w, 5)
    assert w.get(solar).damage_taken == 0.0 and w.get(g).shots_fired == 0
    w.attack(g, solar)
    seconds(w, 2)
    assert w.get(solar).damage_taken > 0.0


def test_a_lance_shoots_what_is_not_bait():
    w = world()
    w.add(LANCE, 0, X, Z)
    e = dummy(w, RAIDER, 500, 0, TOUGH)
    w.ready_weapons()
    seconds(w, 5)
    assert w.get(e).damage_taken > 0.0


def radar_shots(target: str, seen: bool, seed: int, gun: str = LANCE, gap: float = 900.0, secs: float = 72.0) -> float:
    w = world(seed)
    w.add("staticradar", 0, X - 900, Z)
    g = w.add(gun, 0, X, Z)
    w.set_move_state(g, 0)
    w.set_fire_state(g, 0)
    w.set_unit_ai(g, False)
    e = dummy(w, target, gap, 0, TOUGH)
    if seen:
        s = w.add(RAIDER, 0, X + gap - 30, Z)
        w.set_fire_state(s, 0)
        w.step(9)
        w.kill(s)
        w.step(1)
    else:
        w.step(6)
    w.ready_weapons()
    w.set_fire_state(g, 2)
    seconds(w, secs)
    assert w.get(g).shots_fired >= 3 or gun != LANCE, w.get(g).shots_fired
    return w.get(e).damage_taken


def test_a_lance_misses_a_radar_dot_that_moves_or_was_never_seen():
    for seed in range(6):
        assert radar_shots(ASSAULT, False, seed) == 0.0
        assert radar_shots("turretlaser", False, seed) == 0.0


def test_a_lance_hits_a_building_that_has_been_seen():
    assert radar_shots("turretlaser", True, 1) >= 3 * 3000.0


def test_splash_can_still_catch_a_radar_dot():
    lost = [radar_shots(ASSAULT, False, seed, gun="cloakarty", gap=850.0, secs=20.0) for seed in range(6)]
    seen = [radar_shots("turretlaser", True, seed, gun="cloakarty", gap=850.0, secs=20.0) for seed in range(6)]
    assert sum(lost) < sum(seen), (lost, seen)


def test_targeting_goes_by_what_a_side_can_see():
    w = world()
    g = w.add("cloakarty", 0, X, Z)       # reach 860, sight 660
    w.set_move_state(g, 0)
    w.set_unit_ai(g, False)
    e = dummy(w, "turretlaser", 800, 0, TOUGH)
    w.ready_weapons()
    seconds(w, 8)
    assert w.get(g).shots_fired == 0, "nothing shows it"
    w.add("staticradar", 0, X + 500, Z)
    seconds(w, 8)
    assert w.get(g).shots_fired > 0 and w.get(e).damage_taken > 0.0


def test_a_stardust_slows_as_it_heats_and_cools_when_it_stops():
    w = world()
    g = w.add(STARDUST, 0, X, Z)
    unit = CAT.units[CAT.index[STARDUST]]
    assert w.get(g).heat == unit["heat_initial"] == 1.0
    w.ready_weapons()
    seconds(w, 8)
    assert w.get(g).heat == 0.0 or w.get(g).heat < 1.0
    t = dummy(w, ASSAULT, 150, 0, TOUGH)
    seconds(w, 20)
    assert w.get(g).heat > 0.85
    fired = w.get(g).shots_fired
    seconds(w, 10)
    per_second = (w.get(g).shots_fired - fired) / 10.0
    assert 4.0 < per_second < 6.0, f"half of 10 a second, is {per_second}"
    w.kill(t)
    seconds(w, 7)
    assert w.get(g).heat < 0.1


def test_scripted_worlds_do_nothing_of_their_own():
    w = zksim.World(CAT, FLAT, seed=1, behaviour=zksim.SCRIPTED)
    r = w.add(RAIDER, 0, X, Z)
    dummy(w, "staticradar", RANGE + 100, 0)
    seconds(w, 4)
    assert w.get(r).x == X and w.get(r).chasing == -1


# ---------- Bombs ----------

BOMBS = ("amphbomb", "cloakbomb", "jumpbomb", "shieldbomb")
LIMPET = "amphbomb"


def _bomb(w: zksim.World, name: str = LIMPET, fire_at_will: bool = False) -> int:
    b = w.add(name, 0, X, Z)
    if fire_at_will:
        w.set_fire_state(b, 2)
    return b


def _walking(w: zksim.World, dz: float, to_dz: float) -> int:
    t = w.add("cloakassault", 1, X, Z + dz)
    w.set_fire_state(t, 0)
    w.set_unit_ai(t, False)
    w.move(t, X, Z + to_dz)
    return t


def test_bombs_have_a_kamikaze_distance():
    have = {u["name"]: u["kamikaze_distance"] for u in CAT.units if u["kamikaze_distance"] > 0.0}
    # The defs' numbers and 25 more: UnitDef.cpp.
    assert have["amphbomb"] == 145.0 and have["cloakbomb"] == 110.0 and have["jumpbomb"] == 50.0 and have["shieldbomb"] == 105.0
    assert have["gunshipbomb"] == 85.0 and "cloakraid" not in have
    assert CAT.units[CAT.index["subscout"]]["flags"] & zksim.KAMIKAZE_BLIND
    assert not CAT.units[CAT.index[LIMPET]]["flags"] & zksim.KAMIKAZE_BLIND


@pytest.mark.parametrize("name", BOMBS)
def test_a_bomb_told_to_attack_walks_up_and_goes_off_in_reach(name):
    w = world()
    b = _bomb(w, name)
    t = dummy(w, "cloakassault", 0.0, 200.0, TOUGH)     # in sight of them all: an order at what it can't see lapses
    reach = CAT.units[CAT.index[name]]["kamikaze_distance"]
    w.attack(b, t)
    last = 200.0
    for _ in range(600):
        w.step(1)
        if not w.get(b).alive:
            break
        last = w.get(t).z - w.get(b).z
    assert not w.get(b).alive
    # Checked every 15 frames, so it may be some way inside its reach by then.
    assert reach * 0.4 < last < reach + 2.0 * 15.0 * CAT.units[CAT.index[name]]["speed"] / 30.0
    if name != "cloakbomb":
        assert w.get(t).hp < TOUGH


def test_on_hold_fire_a_bomb_lets_what_comes_near_walk_past():
    w = world()
    b = _bomb(w)
    t = _walking(w, 500.0, -500.0)
    seconds(w, 12.0)
    assert w.get(b).alive and w.get(t).hp == CAT.units[CAT.index["cloakassault"]]["hp"]


def test_on_fire_at_will_a_bomb_goes_off_when_something_comes_into_reach():
    w = world()
    b = _bomb(w, fire_at_will=True)
    t = _walking(w, 500.0, -500.0)
    gap = 500.0
    for _ in range(300):
        w.step(1)
        if not w.get(b).alive:
            break
        gap = w.get(t).z - w.get(b).z
    assert not w.get(b).alive
    assert 100.0 < gap < 160.0


def test_on_fire_at_will_it_doesnt_matter_which_way_the_enemy_is_going():
    """A game's Limpet went off for a Knight standing still and for one walking away (01-simulator.md 8.13)."""
    for target in (lambda w: _walking(w, 100.0, 2000.0), lambda w: dummy(w, "cloakassault", 0.0, 100.0, TOUGH)):
        w = world()
        b = _bomb(w, fire_at_will=True)
        target(w)
        seconds(w, 1.0)
        assert not w.get(b).alive


def test_on_fire_at_will_a_bomb_on_a_fight_order_goes_off_passing_something_close():
    w = world()
    b = _bomb(w, fire_at_will=True)
    dummy(w, "cloakassault", 60.0, 300.0, TOUGH)
    w.fight(b, X, Z + 800.0)
    seconds(w, 4.0)
    assert not w.get(b).alive


def test_an_attack_order_at_what_cant_be_seen_lapses_short_of_a_bomb_reaching_it():
    w = world()
    b = _bomb(w)
    t = dummy(w, "cloakassault", 0.0, 400.0, TOUGH)
    w.attack(b, t)
    seconds(w, 8.0)
    s = w.get(b)
    assert s.alive and s.z - Z < 160.0      # Sight is 240.  A game's stopped after 73.


@pytest.mark.parametrize("fire_at_will", (False, True))
def test_a_bomb_on_a_fight_order_doesnt_chase(fire_at_will):
    w = world()
    b = _bomb(w, fire_at_will=fire_at_will)
    dummy(w, "cloakassault", 200.0, 300.0, TOUGH)
    w.fight(b, X, Z + 800.0)
    seconds(w, 4.0)
    s = w.get(b)
    assert s.alive and abs(s.x - X) < 5.0 and s.z > Z + 100.0


def test_a_bomb_needs_to_see_what_sets_it_off_unless_its_def_says_not():
    reach = CAT.units[CAT.index[LIMPET]]["kamikaze_distance"]
    shows_at = CAT.units[CAT.index["cloakheavyraid"]]["decloak_distance"]
    assert shows_at < 0.8 * reach, "this test needs a cloaked unit that can get inside its reach unseen"
    w = world()
    t = w.add("cloakheavyraid", 1, X, Z + 500.0)
    w.set_fire_state(t, 0)
    w.set_unit_ai(t, False)
    w.cloak(t, True)
    seconds(w, 6.0)
    b = _bomb(w, fire_at_will=True)
    w.move(t, X, Z - 500.0)
    gap = 500.0
    for _ in range(300):
        w.step(1)
        if not w.get(b).alive:
            break
        gap = w.get(t).z - w.get(b).z
    assert not w.get(b).alive and gap < 0.8 * reach


def test_a_self_destruct_is_at_once_and_leaves_a_heap():
    w = world()
    b = _bomb(w)
    t = dummy(w, "cloakassault", 0.0, 80.0, TOUGH)
    w.self_destruct(b)
    w.step(1)
    assert not w.get(b).alive
    assert w.get(t).hp < TOUGH
    u = CAT.units[CAT.index[LIMPET]]
    assert [float(f["metal"]) for f in w.features(1)] == [pytest.approx(u["heap_metal"])]     # as the target saw it


def test_self_destruct_is_only_for_bombs():
    w = world()
    u = w.add(RAIDER, 0, X, Z)
    w.self_destruct(u)
    w.step(1)
    assert w.get(u).alive


def test_a_stunned_bomb_doesnt_go_off():
    w = world()
    b = _bomb(w)
    t = dummy(w, "cloakassault", 0.0, 60.0, TOUGH)
    for imp in [w.add("cloakbomb", 1, X, Z - 40.0) for _ in range(4)]:
        w.self_destruct(imp)
    w.step(1)
    assert w.get(b).para > 1.0, "this test needs it stunned"
    w.attack(b, t)
    w.self_destruct(b)
    w.step(6)
    assert w.get(b).alive


def test_limpets_blast_slows_enemies_and_spares_friends():
    free = zksim.World(CAT, FLAT, seed=1)
    k = free.add("cloakassault", 1, X, Z + 60.0)
    free.set_fire_state(k, 0)
    free.set_unit_ai(k, False)
    free.move(k, X, Z + 3000.0)
    free.step(12)
    z0 = free.get(k).z
    free.step(6)
    free_speed = free.get(k).z - z0

    w = world()
    b = _bomb(w)
    k = w.add("cloakassault", 1, X, Z + 60.0)
    w.set_fire_state(k, 0)
    w.set_unit_ai(k, False)
    friend = w.add("cloakassault", 0, X + 60.0, Z)
    w.set_fire_state(friend, 0)
    w.set_unit_ai(friend, False)
    w.move(k, X, Z + 3000.0)
    w.step(12)
    w.self_destruct(b)
    w.step(1)
    z0 = w.get(k).z
    w.step(6)
    assert w.get(k).z - z0 < 0.8 * free_speed
    assert w.get(k).hp < CAT.units[CAT.index["cloakassault"]]["hp"]
    assert w.get(friend).hp == CAT.units[CAT.index["cloakassault"]]["hp"]


def test_snitchs_blast_hurts_friends_too_and_imps_stuns():
    w = world()
    b = _bomb(w, "shieldbomb")
    enemy = dummy(w, "cloakassault", 0.0, 60.0, TOUGH)
    friend = w.add("cloakassault", 0, X + 60.0, Z)
    full = w.get(friend).hp
    w.set_fire_state(friend, 0)
    w.self_destruct(b)
    w.step(1)
    assert w.get(enemy).hp < TOUGH and w.get(friend).hp < full

    w = world()
    b = _bomb(w, "cloakbomb")
    enemy = dummy(w, "cloakassault", 0.0, 60.0)
    hp = w.get(enemy).hp
    w.self_destruct(b)
    w.step(1)
    s = w.get(enemy)
    assert s.para > 0.0 and s.hp == pytest.approx(hp, abs=1.0)
