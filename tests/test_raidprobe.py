"""The raiders-only opening, and squads that keep looking instead of parking."""
import collections
import math
import pathlib
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bots"))

from zkbot import raidprobe, tactics, units  # noqa: E402
from zkbot.models import UnitInfo  # noqa: E402
from zkbot.raidprobe import RaidProbeMixin  # noqa: E402

BASE, ENEMY = (4800.0, 4800.0), (300.0, 300.0)


def unit(uid, name, x, z):
    d = units.by_name(name)
    hp = float(d.get("health") or 100.0)
    return UnitInfo(uid, 0, name, d, x, 0.0, z, hp, hp, 1.0, False)


class Bot(RaidProbeMixin, tactics.TacticsMixin):
    def armored_heavies(self):
        return 0

    def __init__(self, t=60.0, spots=(), raiders=0, seen=None):
        self.t, self.frame = t, int(t * 30)
        self.map = NS(spots=[NS(x=x, z=z, status=s) for x, z, s in spots], width=5120, height=5120)
        self.base_pos, self.enemy_base = BASE, ENEMY
        self.enemies, self.radar, self.enemy_structures = {}, {}, {}
        self.units = {i: unit(i, "cloakraid", 2000.0, 2000.0) for i in range(1, raiders + 1)}
        self.raid_blacklist = {}
        self._seen = collections.Counter(seen or {})

    def frame_time(self):
        return self.t

    def dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def our_role_counts(self):
        return {"raider": len(self.units)}

    def seen_enemy_roles(self):
        return self._seen

    def is_dangerous(self, e):
        return e.is_combat or "dyntrainer" in e.name

    def enemy_near(self, pos, radius):
        return next((e for e in self.enemies.values() if self.dist(e.pos, pos) < radius), None)

    def threat_at(self, pos):
        return 0.0

    def _clamp(self, p):
        return min(5056.0, max(64.0, p[0])), min(5056.0, max(64.0, p[1]))

    def add(self, *es):
        for e in es:
            self.enemies[e.uid] = e
        return self


@pytest.fixture
def flags(monkeypatch):
    def use(*names):
        for mod in (raidprobe, tactics):
            monkeypatch.setattr(mod, "FLAGS", set(names))
    return use


def test_flag_off_keeps_the_normal_mix(flags):
    flags()
    assert Bot(t=60.0).raid_ramp() == 1.0


def test_opening_is_raiders_only_then_phases_in(flags):
    flags("raidfirst")
    assert Bot(t=120.0).raid_ramp() == 0.0
    assert Bot(t=299.0).raid_ramp() == 0.0
    assert Bot(t=450.0).raid_ramp() == pytest.approx(0.5)
    assert Bot(t=600.0).raid_ramp() == 1.0


def test_fleas_do_not_end_the_opening(flags):
    flags("raidfirst")
    bot = Bot(t=90.0).add(*(unit(i, "spiderscout", 2000.0, 2000.0) for i in range(100, 112)))
    assert bot.enemy_real_raiders() == 0 and bot.raid_ramp() == 0.0


def test_raiders_only_end_it_when_they_outnumber_ours(flags):
    flags("raidfirst")
    many = [unit(i, "cloakraid", 2000.0, 2000.0) for i in range(100, 106)]
    assert Bot(t=90.0, raiders=2).add(*many).raid_ramp() == 1.0
    assert Bot(t=90.0, raiders=8).add(*many).raid_ramp() == 0.0


def test_slow_units_end_the_opening(flags):
    flags("raidfirst")
    assert Bot(t=90.0, seen={"riot": 2}).raid_ramp() == 0.0
    assert Bot(t=90.0, seen={"riot": 2, "assault": 1}).raid_ramp() == 1.0


def test_only_a_turret_in_range_defends_a_target(flags):
    flags("raidprobe")
    bot = Bot().add(unit(50, "turretlaser", 1000.0, 1000.0))
    assert bot.raid_defense_value((1000.0, 1400.0)) > 0.0      # 400 away, range 460
    assert bot.raid_defense_value((1000.0, 1510.0)) == 0.0     # dead zone
    flags()
    assert bot.raid_defense_value((1000.0, 1400.0)) == 0.0     # without it, a turret in view is missed


def test_anti_air_turrets_do_not_defend_against_raiders(flags):
    flags("raidprobe")
    bot = Bot().add(unit(50, "turretaalaser", 1000.0, 1000.0))
    assert bot.raid_defense_value((1000.0, 1100.0)) == 0.0
    assert bot.known_defenses() == []


def test_remembered_turrets_count_the_same_way(flags):
    flags("raidprobe")
    bot = Bot()
    bot.enemy_structures[50] = (unit(50, "turretlaser", 1000.0, 1000.0), 0)
    assert bot.raid_defense_value((1000.0, 1400.0)) > 0.0
    assert bot.raid_defense_value((1000.0, 1510.0)) == 0.0
    flags()
    assert bot.raid_defense_value((1000.0, 1510.0)) > 0.0      # the old radius of 520


def test_squads_accept_more_with_the_flag(flags):
    bot = Bot(raiders=4)
    flags()
    assert bot.raid_accept_value([1, 2, 3, 4]) == pytest.approx(130.0)
    flags("raidprobe")
    assert bot.raid_accept_value([1, 2, 3, 4]) == pytest.approx(260.0)


def test_empty_target_means_there_and_nothing_to_shoot():
    bot = Bot()
    at = (1000.0, 1000.0)
    assert bot.target_is_empty(at, (1100.0, 1000.0))
    assert not bot.target_is_empty(at, (1400.0, 1000.0))     # not there yet
    bot.enemy_structures[7] = (unit(7, "staticmex", 1000.0, 1100.0), 0)
    assert not bot.target_is_empty(at, (1100.0, 1000.0))     # an extractor to kill


def test_moves_on_inward_past_checked_points(flags):
    flags("raidprobe")
    spots = [(1500.0, 1500.0, "free"), (800.0, 800.0, "free"), (4000.0, 4000.0, "free"), (1000.0, 1500.0, "ours")]
    bot = Bot(t=120.0, spots=spots, raiders=4)
    squad = [1, 2, 3, 4]
    center = (2000.0, 2000.0)
    assert bot.probe_target(squad, center, []) == (1500.0, 1500.0)   # our half's spot is never listed
    bot.note_points_checked((1500.0, 1500.0))
    assert bot.probe_target(squad, (1500.0, 1500.0), []) == (800.0, 800.0)
    bot.note_points_checked((800.0, 800.0))
    assert bot.probe_target(squad, (800.0, 800.0), []) == ENEMY


def test_every_point_checked_still_gives_a_target(flags):
    flags("raidprobe")
    bot = Bot(t=120.0, spots=[(800.0, 800.0, "free")], raiders=4)
    bot.note_points_checked((800.0, 800.0))
    bot.frame += 300
    bot.note_points_checked(ENEMY)
    # Nothing unchecked: the longest unseen point, not a wait at the last one.
    assert bot.probe_target([1, 2, 3, 4], ENEMY, []) == (800.0, 800.0)


def test_all_guarded_means_poke_at_the_edge(flags):
    flags("raidprobe")
    towers = [unit(50 + i, "turretheavylaser", x, z) for i, (x, z) in enumerate([(800.0, 800.0), (300.0, 300.0)])]
    bot = Bot(t=120.0, spots=[(800.0, 800.0, "free")], raiders=2).add(*towers)
    center = (3000.0, 3000.0)
    target = bot.probe_target([1, 2], center, [])
    assert all(not bot.turret_covers(t, target) for t in towers)
    near = min(bot.dist(t.pos, target) for t in towers)
    assert near < max(bot.turret_reach(t) for t in towers) + raidprobe.STAND_OFF + 1.0


def test_route_bends_around_coverage_and_clears_it(flags):
    flags("raidprobe")
    tower = unit(50, "turretlaser", 2000.0, 2000.0)
    bot = Bot().add(tower)
    start, dest = (2000.0, 3000.0), (2000.0, 1000.0)
    wp = bot.coverage_waypoint(start, dest)
    assert wp is not None and bot.dist(wp, tower.pos) > bot.turret_reach(tower)
    assert bot.coverage_waypoint((3000.0, 3000.0), (3000.0, 1000.0)) is None      # path misses it


def test_route_ignores_a_turret_the_squad_accepted(flags):
    flags("raidprobe")
    bot = Bot().add(unit(50, "turretlaser", 2000.0, 2000.0))
    assert bot.coverage_waypoint((2000.0, 3000.0), (2000.0, 2100.0)) is None      # covers the target
    assert bot.coverage_waypoint((2000.0, 2100.0), (2000.0, 1000.0)) is None      # already inside


def test_stage_moves_forward_unless_the_enemy_holds_it(flags):
    flags("raidprobe")
    home = (4000.0, 4000.0)
    fwd = Bot().forward_stage(home)
    assert Bot().dist(fwd, ENEMY) < Bot().dist(home, ENEMY)
    assert Bot().add(unit(50, "cloakraid", *fwd)).forward_stage(home) == home


def test_stage_never_moves_back_from_a_rally_already_forward(flags):
    flags("raidprobe")
    ahead = (1000.0, 1000.0)
    assert Bot().forward_stage(ahead) == ahead


class Raid(Bot):
    """A squad's view of things: orders and log kept, not sent."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.orders, self.lines, self.retreating = [], [], set()

    def log(self, msg):
        self.lines.append(msg)

    def order_move(self, uid, pos):
        self.orders.append(("move", uid, pos))

    def order_fight(self, uid, pos):
        self.orders.append(("fight", uid, pos))

    def order_attack(self, uid, target):
        self.orders.append(("attack", uid, target))

    def safe_retreat_point(self, u):
        return BASE

    def rally_point(self):
        return (4000.0, 4000.0)


def test_see_turrets_counts_one_in_view_within_the_old_radius(flags):
    flags("seeturrets")
    bot = Bot().add(unit(50, "turretlaser", 1000.0, 1000.0))
    assert bot.raid_defense_value((1000.0, 1400.0)) == pytest.approx(135.0)
    assert bot.raid_defense_value((1000.0, 1600.0)) == 0.0
    assert bot.raid_defense_value((1000.0, 1400.0)) == bot.raid_defense_value((1000.0, 1450.0))


def test_squad_needs_twice_the_defence(flags):
    flags("hitrun", "raidprobe")
    bot = Bot(raiders=5).add(unit(50, "turretlaser", 1000.0, 1000.0))
    defended = bot.raid_defense_value((1000.0, 1100.0))
    assert defended == pytest.approx(135.0)
    assert bot.raid_accept_value([1, 2, 3, 4]) < defended        # 260 is not twice 135
    assert bot.raid_accept_value([1, 2, 3, 4, 5]) >= defended    # 325 is


def test_a_riot_rules_out_any_target(flags):
    flags("hitrun", "raidprobe")
    bot = Bot(raiders=40).add(unit(50, "cloakriot", 1000.0, 1000.0))
    assert bot.raid_defense_value((1000.0, 1300.0)) >= raidprobe.RIOT_VETO
    assert bot.raid_defense_value((1000.0, 1300.0)) > bot.raid_accept_value(list(range(1, 41)))
    assert bot.raid_defense_value((1000.0, 1600.0)) == 0.0


def test_a_riot_just_out_of_sight_still_counts(flags):
    flags("hitrun", "raidprobe")
    bot = Bot(t=100.0)
    bot.enemy_memory = {50: (unit(50, "cloakriot", 1000.0, 1000.0), bot.frame - 100)}
    assert bot.raid_defense_value((1000.0, 1300.0)) >= raidprobe.RIOT_VETO
    bot.enemy_memory = {50: (unit(50, "cloakriot", 1000.0, 1000.0), bot.frame - raidprobe.RIOT_MEMORY - 1)}
    assert bot.raid_defense_value((1000.0, 1300.0)) == 0.0


def test_nowhere_to_go_but_riots_means_no_target(flags):
    flags("hitrun", "raidprobe")
    riots = [unit(50, "cloakriot", 800.0, 800.0), unit(51, "cloakriot", 300.0, 300.0)]
    bot = Raid(t=120.0, spots=[(800.0, 800.0, "free")], raiders=4).add(*riots)
    assert bot.probe_target([1, 2, 3, 4], (3000.0, 3000.0), []) is None


def test_runs_until_at_the_target_then_shoots_what_cannot_shoot_back(flags):
    flags("hitrun", "raidprobe")
    mex = unit(60, "staticmex", 1000.0, 1000.0)
    bot = Raid(raiders=4).add(mex)
    assert bot.raid_mode((3000.0, 3000.0), mex.pos) is None
    assert bot.raid_mode((1100.0, 1000.0), mex.pos) == 60
    assert bot.hitrun_order(bot.units[1], mex.pos, 60) == "raid_strike"
    assert bot.hitrun_order(bot.units[1], mex.pos, None) == "raid_run"
    assert [o[0] for o in bot.orders] == ["attack", "move"]


def test_constructors_before_extractors(flags):
    flags("hitrun", "raidprobe")
    bot = Raid().add(unit(60, "staticmex", 1000.0, 1000.0), unit(61, "cloakcon", 1300.0, 1000.0))
    assert bot.strike_target((1000.0, 1000.0)) == 61


def test_armed_unit_at_the_target_gets_fought(flags):
    flags("hitrun", "raidprobe")
    bot = Raid().add(unit(60, "staticmex", 1000.0, 1000.0), unit(70, "cloakraid", 1100.0, 1000.0))
    assert bot.raid_mode((1000.0, 1000.0), (1000.0, 1000.0)) == "fight"


def test_target_done_when_the_extractor_is_gone(flags):
    flags("hitrun", "raidprobe")
    mex = unit(60, "staticmex", 1000.0, 1000.0)
    bot = Raid().add(mex)
    assert not bot.target_is_empty(mex.pos, mex.pos)
    bot.enemies.clear()
    assert bot.target_is_empty(mex.pos, mex.pos)


def test_an_armed_unit_alone_does_not_hold_a_raid_at_a_finished_target(flags):
    flags("hitrun", "raidprobe")
    bot = Raid().add(unit(70, "cloakraid", 1100.0, 1000.0))
    assert bot.target_is_empty((1000.0, 1000.0), (1000.0, 1000.0))


def test_squad_breaks_off_once_when_outweighed(flags):
    flags("hitrun", "raidprobe")
    bot = Raid(raiders=4).add(unit(50, "cloakassault", 1000.0, 1000.0))
    sq, members = {}, [1, 2, 3, 4]
    assert bot.squad_breaks_off(sq, members, (1000.0, 1100.0))
    assert bot.squad_running(sq) and not bot.squad_breaks_off(sq, members, (1000.0, 1100.0))
    bot.frame += raidprobe.BREAK_FRAMES
    assert not bot.squad_running(sq) and bot.squad_breaks_off(sq, members, (1000.0, 1100.0))


def test_hurt_raiders_peel_off_large_squads_only(flags):
    flags("hitrun")
    bot = Raid(raiders=6)
    bot.units[1].health = 50.0
    trail = [(100.0, 100.0), (500.0, 500.0)]
    assert bot.peel_hurt({"members": {1, 2, 3, 4, 5}, "trail": trail}, [1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]
    sq = {"members": set(range(1, 7)), "trail": trail}
    assert bot.peel_hurt(sq, [1, 2, 3, 4, 5, 6]) == [2, 3, 4, 5, 6]
    assert 1 not in sq["members"] and bot.raid_regen[1]["path"] == [(500.0, 500.0), (100.0, 100.0)]


def test_healing_raider_goes_back_the_way_it_came_then_wanders(flags):
    flags("hitrun")
    bot = Raid(t=200.0, spots=[(2500.0, 2500.0, "free"), (3000.0, 3000.0, "ours")], raiders=1)
    u = bot.units[1]
    u.health = 50.0
    bot.raid_regen = {1: {"path": [(1900.0, 1900.0), (1500.0, 1500.0)], "since": bot.frame, "wander": None}}
    assert bot.beh_raider_regen(u, None) == "raider_regen"
    assert bot.orders[-1][2] == (1500.0, 1500.0)                      # the first point is already reached
    u.x, u.z = 1500.0, 1500.0
    bot.beh_raider_regen(u, None)
    assert bot.orders[-1][2] == (2500.0, 2500.0)                      # route done: wander, and never to our spots


def test_healing_raider_rejoins_at_full_health_or_after_too_long(flags):
    flags("hitrun")
    bot = Raid(t=200.0, raiders=2)
    bot.units[1].health = bot.units[1].max_health
    bot.units[2].health = 50.0
    bot.raid_regen = {1: {"path": [], "since": bot.frame, "wander": None},
                      2: {"path": [], "since": bot.frame - raidprobe.REGEN_TIMEOUT - 1, "wander": None}}
    assert bot.beh_raider_regen(bot.units[1], None) is None
    assert bot.beh_raider_regen(bot.units[2], None) is None
    assert bot.raid_regen == {}


def test_healing_raider_runs_from_enemies_and_does_not_wander_into_them(flags):
    flags("hitrun")
    bot = Raid(t=200.0, spots=[(2500.0, 2500.0, "free")], raiders=1)
    u = bot.units[1]
    u.health = 50.0
    bot.raid_regen = {1: {"path": [], "since": bot.frame, "wander": None}}
    bot.add(unit(70, "cloakraid", 2100.0, 2000.0))
    bot.beh_raider_regen(u, None)
    assert bot.orders[-1][2] == BASE
    bot.enemies.clear()
    bot.add(unit(70, "cloakraid", 2500.0, 2600.0))
    assert bot.wander_point(u.pos) is None


def test_spearhead_is_the_first_raiders_and_the_dead_are_not_replaced(flags):
    flags("spearhead")
    bot = Raid(raiders=5)
    assert bot.update_spearhead([3, 1]) == {3, 1}
    assert bot.update_spearhead([3, 1, 2, 4, 5]) == {3, 1, 2}
    del bot.units[1]
    assert bot.update_spearhead([3, 2, 4, 5]) == {3, 2}


def test_no_spearhead_without_the_flag(flags):
    flags()
    assert Raid(raiders=5).update_spearhead([1, 2, 3]) == set()
