"""What shields stop, what charging them costs and how they share charge: CheckShieldCollisions, TraceRayShields,
unit_shield_charge.lua and unit_shield_link.lua.  Measured against a game (01-simulator.md, "Validation")."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "py"))
import zksim  # noqa: E402


def scripted_world(*args, **kwargs):
    """Units that do only what they're told and see everything, so a test measures one thing.  test_unit_ai.py is for the rest."""
    kwargs.setdefault("behaviour", zksim.SCRIPTED)
    return zksim.World(*args, **kwargs)


CAT = zksim.Catalog()
FLAT = zksim.Map.flat()
ASPIS = "shieldshield"          # radius 350, unarmed, charging costs energy
SMALL = "shieldcon"             # radius 80, charging is free
UNLINKED = "statictempshield"
BEAM = "turretheavylaser"       # range 620, instant
SPLASH = "turretriot"           # range 410, blast 48
ARTILLERY = "cloakarty"         # range 860, ballistic
Z = 2000.0


def _weapon(name: str, mount: int = 0) -> dict:
    return CAT.weapons[CAT.units[CAT.index[name]]["mount_weapon"][mount]]


def _shield_weapon(name: str) -> dict:
    return next(CAT.weapons[m] for m in CAT.units[CAT.index[name]]["mount_weapon"] if CAT.weapons[m]["kind"] == 3)


def _target(attacker: str) -> str:
    """Unarmed and immobile, so that it neither shoots back nor walks off, with no shield of its own."""
    can_hit = CAT.units[CAT.index[attacker]]["mount_only"][0]
    options = [u for u in CAT.units if not u["flags"] & ~zksim.LEAVES_GHOST and u["hp"] > 2000.0 and u["armor_class"] == 2 and u["category"] & can_hit
               and not any(CAT.weapons[m]["kind"] == 3 for m in u["mount_weapon"])]
    return min(options, key=lambda u: u["hp"])["name"]


def _duel(shooter: str, target_x: float, target_z: float = Z, seed: int = 3, ticks: int = 40, world_map: zksim.Map = FLAT):
    """A shooter at x 1000 told to shoot a target: returns the world, the target and a function that plays it out."""
    w = scripted_world(CAT, world_map, seed=seed)
    s = w.add(shooter, 0, 1000.0, Z)
    t = w.add(_target(shooter), 1, target_x, target_z)

    def play() -> float:
        w.ready_weapons()
        w.attack(s, t)
        w.step(ticks)
        return w.get(t).damage_taken
    return w, s, t, play


def _economy(energy: float = 1000.0) -> zksim.World:
    w = scripted_world(CAT, FLAT)
    w.economy(np.zeros((0, 3), np.float32), 0.0, energy, 0.0, 0.0, 100000.0)
    return w


def test_a_shot_stops_at_a_bubble_its_path_crosses_though_its_target_is_beyond():
    w, _, _, play = _duel(BEAM, 1500.0, ticks=3)    # one shot, which is all that a small one can pay for
    sh = w.add(SMALL, 1, 1300.0, Z)
    full = w.get(sh).shield
    assert play() == 0.0
    assert w.get(sh).shield < full


def test_a_shot_whose_path_misses_a_bubble_is_not_stopped_by_it():
    w, _, _, play = _duel(BEAM, 1500.0, Z + 250.0)
    sh = w.add(SMALL, 1, 1250.0, Z - 100.0)
    full = w.get(sh).shield
    assert play() > 0.0
    assert w.get(sh).shield == pytest.approx(full)


def test_what_is_inside_a_bubble_is_covered():
    w, _, _, play = _duel(BEAM, 1500.0, ticks=3)
    w.add(SMALL, 1, 1510.0, Z)
    assert play() == 0.0


def test_a_shooter_inside_fires_out_past_it():
    w, _, _, play = _duel(BEAM, 1500.0)
    sh = w.add(SMALL, 1, 1040.0, Z)
    full = w.get(sh).shield
    assert play() > 0.0
    assert w.get(sh).shield == pytest.approx(full)


def test_a_shooter_inside_a_hostile_bubble_still_hits_what_is_in_there_with_it():
    w, _, _, play = _duel(BEAM, 1100.0)
    w.add(ASPIS, 1, 1050.0, Z)
    assert play() > 0.0


def test_a_sides_own_shield_lets_its_shots_through():
    w, _, _, play = _duel(BEAM, 1500.0)
    sh = w.add(SMALL, 0, 1300.0, Z)
    full = w.get(sh).shield
    assert play() > 0.0
    assert w.get(sh).shield == pytest.approx(full)


def test_the_first_bubble_along_the_path_pays():
    w, _, _, play = _duel(BEAM, 1500.0, ticks=3)
    far = w.add(SMALL, 1, 1400.0, Z)
    near = w.add(SMALL, 1, 1200.0, Z)
    full = w.get(far).shield
    assert play() == 0.0
    assert w.get(near).shield < full and w.get(far).shield == pytest.approx(full)


def test_a_bubble_that_cannot_pay_lets_it_through_to_the_next():
    w, _, _, play = _duel(BEAM, 1500.0, ticks=3)    # before the one it took is charged back
    far = w.add(SMALL, 1, 1400.0, Z)
    near = w.add(SMALL, 1, 1200.0, Z)
    w.set_shield(near, 0.0)
    full = w.get(far).shield
    assert play() == 0.0
    assert w.get(far).shield < full


def test_with_nothing_left_a_shot_gets_through():
    w, _, _, play = _duel(BEAM, 1500.0)
    sh = w.add(SMALL, 1, 1300.0, Z)
    w.set_shield(sh, 0.0)
    assert play() > 0.0


def test_a_weapon_that_ignores_shields_does():
    name = next(u["name"] for u in CAT.units if u["flags"] & zksim.ARMED and u["mount_weapon"] and CAT.weapons[u["mount_weapon"][0]]["intercepted_by"] == 0
                and CAT.weapons[u["mount_weapon"][0]]["range"] >= 300.0 and CAT.weapons[u["mount_weapon"][0]]["damage"][0] > 0.0)
    w, _, _, play = _duel(name, 1000.0 + CAT.weapons[CAT.units[CAT.index[name]]["mount_weapon"][0]]["range"] * 0.8)
    sh = w.add(SMALL, 1, 1000.0 + CAT.weapons[CAT.units[CAT.index[name]]["mount_weapon"][0]]["range"] * 0.4, Z)
    full = w.get(sh).shield
    assert play() > 0.0
    assert w.get(sh).shield == pytest.approx(full)


def test_a_stopped_shot_splashes_what_is_near_the_surface():
    """A shot at the shield's near edge, with something just inside it."""
    w, _, t, play = _duel(SPLASH, 1225.0, ticks=4)
    w.add(SMALL, 1, 1300.0, Z)
    assert play() > 0.0


def test_a_stopped_shot_does_not_splash_what_is_deep_inside():
    w, _, _, play = _duel(SPLASH, 1350.0, ticks=4)
    w.add(SMALL, 1, 1300.0, Z)
    assert play() == 0.0


def test_a_whole_beam_takes_what_the_weapon_does_in_all_its_frames():
    """armordefs.lua: shield_damage is 35 a frame for this one, and a shot is 850."""
    assert _weapon(BEAM)["damage"][4] == pytest.approx(850.1, abs=0.1)
    w, _, _, play = _duel(BEAM, 1500.0, ticks=3)
    sh = w.add(SMALL, 1, 1300.0, Z)
    full = w.get(sh).shield
    assert play() == 0.0
    regen = _shield_weapon(SMALL)["shield_regen"] * 3 * zksim.TICK_FRAMES
    assert w.get(sh).shield == pytest.approx(min(full, full - 850.1 + regen), abs=1.0)


def test_a_beam_gets_through_for_what_a_shield_cannot_pay_for():
    w, _, _, play = _duel(BEAM, 1500.0, ticks=1)
    sh = w.add(SMALL, 1, 1300.0, Z)
    w.set_shield(sh, 212.5)     # a quarter of a shot
    through = play()
    assert through == pytest.approx(0.75 * _weapon(BEAM)["damage"][0], rel=0.1)
    assert w.get(sh).shield < 10.0


def test_a_beam_that_empties_one_shield_goes_on_to_the_next():
    w, _, _, play = _duel(BEAM, 1500.0, ticks=1)
    near = w.add(SMALL, 1, 1200.0, Z)
    far = w.add(SMALL, 1, 1400.0, Z)
    w.set_shield(near, 425.0)
    w.set_shield(far, 425.0 + 100.0)
    assert play() == 0.0
    assert w.get(near).shield < 10.0 and w.get(far).shield == pytest.approx(100.0, abs=10.0)


def test_a_shell_is_stopped_whole_or_not_at_all():
    w, _, _, play = _duel("tankassault", 1340.0, ticks=12)
    sh = w.add(SMALL, 1, 1170.0, Z)
    cost = _weapon("tankassault")["damage"][4]
    w.set_shield(sh, cost * 0.9)
    assert play() > 0.0
    assert w.get(sh).shield > cost * 0.9      # nothing taken, and it's charging


def test_what_a_shot_costs_does_not_fall_with_range():
    def cost(weapon_range_share: float) -> float:
        w, _, _, play = _duel("tankassault", 1000.0 + 360.0 * weapon_range_share, ticks=12)
        sh = w.add(SMALL, 1, 1000.0 + 360.0 * weapon_range_share - 10.0, Z)
        start = w.get(sh).shield
        play()
        return start - w.get(sh).shield
    assert cost(0.5) == pytest.approx(cost(0.9), abs=20.0)


def test_a_ballistic_shot_flies_over_a_small_bubble_half_way():
    w, _, _, play = _duel(ARTILLERY, 1800.0, seed=5, ticks=300)
    sh = w.add(SMALL, 1, 1400.0, Z)
    full = w.get(sh).shield
    play()
    assert w.get(sh).shield == pytest.approx(full)


def test_a_ballistic_shot_comes_down_into_a_bubble_round_its_target():
    w, _, t, play = _duel(ARTILLERY, 1800.0, seed=5, ticks=300)
    w.add(ASPIS, 1, 1800.0, Z)
    assert play() == 0.0


def test_a_ballistic_shot_cannot_clear_a_big_bubble_half_way():
    w, _, t, play = _duel(ARTILLERY, 1800.0, seed=5, ticks=300)
    aspis = w.add(ASPIS, 1, 1400.0, Z)
    w.set_shield(aspis, _shield_weapon(ASPIS)["shield_power"])
    assert play() == 0.0


def test_a_shield_up_on_high_ground_is_not_in_the_way_of_a_shot_below_it():
    height = np.zeros((256, 256), np.float32)
    height[115:136, 71:92] = 400.0      # round where the shield stands, on the line of fire
    hill = zksim.Map(height, np.ones((len(zksim.MOVE_CLASSES), 256, 256), np.float32), 16.0, "hill")
    w, _, _, play = _duel(BEAM, 1500.0, world_map=hill)
    sh = w.add(SMALL, 1, 1300.0, Z)
    full = w.get(sh).shield
    assert play() > 0.0
    assert w.get(sh).shield == pytest.approx(full)


def test_charging_costs_the_listed_energy():
    w = _economy(energy=1000.0)
    u = w.add(ASPIS, 0, 1000.0, Z)
    w.set_shield(u, 0.0)
    ticks = 30
    w.step(ticks)
    seconds = ticks * zksim.TICK_FRAMES / 30.0
    weapon = _shield_weapon(ASPIS)
    assert w.get(u).shield == pytest.approx(weapon["shield_regen"] * 30.0 * seconds, rel=1e-3)
    assert w.resources(0)["energy"] == pytest.approx(1000.0 - weapon["shield_energy"] * 30.0 * seconds, abs=0.1)
    assert weapon["shield_energy"] * 30.0 == pytest.approx(12.0)


def test_a_full_shield_costs_nothing():
    w = _economy(energy=1000.0)
    w.add(ASPIS, 0, 1000.0, Z)
    w.step(30)
    assert w.resources(0)["energy"] == pytest.approx(1000.0)


def test_only_what_fills_it_is_paid_for():
    w = _economy(energy=1000.0)
    u = w.add(ASPIS, 0, 1000.0, Z)
    weapon = _shield_weapon(ASPIS)
    a_tenth_of_a_step = 0.1 * weapon["shield_regen"] * zksim.TICK_FRAMES
    w.set_shield(u, weapon["shield_power"] - a_tenth_of_a_step)
    w.step(1)
    assert w.get(u).shield == pytest.approx(weapon["shield_power"])
    assert w.resources(0)["energy"] == pytest.approx(1000.0 - 0.1 * weapon["shield_energy"] * zksim.TICK_FRAMES, abs=1e-2)


def test_without_energy_it_does_not_charge():
    w = _economy(energy=0.0)
    u = w.add(ASPIS, 0, 1000.0, Z)
    w.set_shield(u, 100.0)
    w.step(30)
    assert w.get(u).shield == pytest.approx(100.0)


def test_a_shield_that_needs_no_energy_charges_without_any():
    w = _economy(energy=0.0)
    u = w.add(SMALL, 0, 1000.0, Z)
    w.set_shield(u, 100.0)
    w.step(30)
    assert w.get(u).shield > 100.0


def test_without_an_economy_charging_is_free():
    w = scripted_world(CAT, FLAT)
    u = w.add(ASPIS, 0, 1000.0, Z)
    w.set_shield(u, 100.0)
    w.step(30)
    assert w.get(u).shield > 100.0


def test_an_unfinished_shield_does_not_charge():
    w = _economy(energy=1000.0)
    u = w.add(ASPIS, 0, 1000.0, Z, build=0.5)
    w.set_shield(u, 100.0)
    w.step(30)
    assert w.get(u).shield == pytest.approx(100.0)
    assert w.resources(0)["energy"] == pytest.approx(1000.0)


def _pair(a: str, b: str, gap: float, side_b: int = 0, charge_a: float | None = None, charge_b: float = 0.0, seed: int = 1):
    w = scripted_world(CAT, FLAT, seed=seed)
    ua = w.add(a, 0, 1000.0, Z)
    ub = w.add(b, side_b, 1000.0 + gap, Z)
    w.set_shield(ua, _shield_weapon(a)["shield_power"] if charge_a is None else charge_a)
    w.set_shield(ub, charge_b)
    return w, ua, ub


def test_touching_shields_even_out():
    w, a, b = _pair(ASPIS, ASPIS, 400.0, charge_a=3000.0)
    w.step(120)   # 20 s, and the difference halves about every 2.3 s
    ca, cb = w.get(a).shield, w.get(b).shield
    assert abs(ca - cb) < 100.0, (ca, cb)
    assert ca > 1000.0 and cb > 1000.0


def test_sharing_neither_makes_nor_loses_charge():
    w, a, b = _pair(ASPIS, ASPIS, 400.0, charge_a=2000.0)
    w.step(12)     # 2 s, too little to fill either
    regen = _shield_weapon(ASPIS)["shield_regen"] * 12 * zksim.TICK_FRAMES
    assert w.get(a).shield + w.get(b).shield == pytest.approx(2000.0 + 2.0 * regen, rel=1e-4)
    assert w.get(b).shield > regen


def test_shields_that_do_not_touch_share_nothing():
    w, a, b = _pair(ASPIS, ASPIS, 750.0, charge_a=2000.0)
    w.step(12)
    assert w.get(b).shield == pytest.approx(_shield_weapon(ASPIS)["shield_regen"] * 12 * zksim.TICK_FRAMES, rel=1e-3)


def test_enemies_share_nothing():
    w, a, b = _pair(ASPIS, ASPIS, 400.0, side_b=1, charge_a=2000.0)
    w.step(12)
    assert w.get(b).shield == pytest.approx(_shield_weapon(ASPIS)["shield_regen"] * 12 * zksim.TICK_FRAMES, rel=1e-3)


def test_an_unlinked_shield_shares_nothing():
    w, a, b = _pair(ASPIS, UNLINKED, 400.0, charge_a=2000.0)
    w.step(12)
    assert w.get(b).shield == pytest.approx(_shield_weapon(UNLINKED)["shield_regen"] * 12 * zksim.TICK_FRAMES, rel=1e-3)


def test_a_higher_rank_gives_but_starts_no_transfers():
    """Only the small one picks a partner, so about half the flow two of a rank make."""
    def gained(other: str) -> float:
        total = 0.0
        for seed in range(1, 9):
            w, a, b = _pair(ASPIS, other, 300.0, charge_a=3000.0, seed=seed)
            w.step(12)
            total += w.get(b).shield - _shield_weapon(other)["shield_regen"] * 12 * zksim.TICK_FRAMES
        return total / 8.0
    assert 0.35 < gained(SMALL) / gained(ASPIS) < 0.65


def test_it_leaves_room_to_charge():
    weapon = _shield_weapon(SMALL)
    w, a, b = _pair(ASPIS, SMALL, 200.0)
    w.set_shield(b, weapon["shield_power"] - weapon["shield_regen"] * 30.0)
    before = w.get(a).shield
    w.step(1)
    assert w.get(a).shield >= before    # nothing given to a shield within a second's charge of full


def test_a_dead_shield_stops_taking_part():
    w, a, b = _pair(ASPIS, ASPIS, 400.0, charge_a=3000.0)
    w.step(6)
    w.kill(a)
    w.step(1)
    charge = w.get(b).shield
    w.step(6)
    assert w.get(b).shield == pytest.approx(charge + _shield_weapon(ASPIS)["shield_regen"] * 6 * zksim.TICK_FRAMES, rel=1e-3)


def test_no_shields_on_the_field_costs_no_randomness():
    """The sim's own random numbers aren't drawn for sharing unless there's sharing to do, so games without shields replay."""
    def play() -> tuple[float, float]:
        w = scripted_world(CAT, FLAT, seed=9)
        a = w.add("cloakraid", 0, 1000.0, Z)
        b = w.add("cloakraid", 1, 1100.0, Z)
        w.step(30)
        return w.get(a).hp, w.get(b).hp
    assert play() == play()
