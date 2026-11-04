"""What the third spectated game asked for: smoother orders, constructors that finish
things and reclaim, a connected grid, a quicker army with assaults in front, and a
turret before a far extractor."""
import math
import pathlib
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bots"))

from zkbot import attackline, build_defense, buildfocus, military, pylongrid, reclaim, smoothorders, units  # noqa: E402
from zkbot.attackline import AttackLineMixin  # noqa: E402
from zkbot.build_defense import DefenseBuildMixin  # noqa: E402
from zkbot.buildfocus import BuildFocusMixin  # noqa: E402
from zkbot.economy import EconomyMixin  # noqa: E402
from zkbot.models import UnitInfo  # noqa: E402
from zkbot.pylongrid import PylonGridMixin  # noqa: E402
from zkbot.reclaim import ReclaimMixin  # noqa: E402
from zkbot.smoothorders import SmoothOrdersMixin  # noqa: E402

BASE, ENEMY = (4800.0, 4800.0), (300.0, 300.0)


def unit(uid, name, x, z, progress=1.0):
    d = units.by_name(name)
    hp = float(d.get("health") or 100.0)
    return UnitInfo(uid, int(d.get("id") or 0), name, d, x, 0.0, z, hp * progress, hp, progress, progress < 1.0)


class Base:
    """The parts of the bot these mixins lean on."""
    commander_id = 1
    hazards = None
    enemy_base = ENEMY
    base_pos = BASE

    def __init__(self, *own, frame=3000, foes=()):
        self.units = {u.uid: u for u in own}
        self.foes, self.frame = list(foes), frame
        self.orders, self.lines = [], []
        self.tasks, self.builder_task, self.builder_idle = {}, {}, {}
        self.builder_job_frame, self.builder_fallback_until = {}, {}
        self.m_cur, self.m_inc = 0.0, 30.0
        self.threats = set()

    def dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def queue(self, order):
        self.orders.append(order)

    def log(self, msg):
        self.lines.append(msg)

    def frame_time(self):
        return self.frame / 30.0

    def metal_current(self):
        return self.m_cur

    def metal_income(self):
        return self.m_inc

    def metal(self):
        return {"storage": 1000.0}

    def effective_storage(self, resource):
        return resource["storage"]

    def enemy_near(self, pos, radius):
        return next((e for e in self.foes if self.dist(e.pos, pos) < radius), None)

    def threat_in_radius(self, pos, radius):
        return 100.0 if any(self.dist(t, pos) < radius for t in self.threats) else 0.0

    def site_safe_for_builder(self, pos, kind=""):
        return self.threat_in_radius(pos, 100.0) == 0.0

    def path_threat(self, a, b):
        return 0.0

    def retreat_builder(self, u, danger):
        self.orders.append(("retreat", u.uid))


# ------------------------------------------------------------------ smoothorders

class Orders(SmoothOrdersMixin, military.MilitaryMixin, Base):
    pass


@pytest.fixture
def smooth(monkeypatch):
    monkeypatch.setattr(smoothorders, "FLAGS", {"smoothorders"})
    monkeypatch.setattr(military, "FLAGS", {"smoothorders"})


def walker(frame=3000):
    bot = Orders(unit(5, "cloakraid", 0.0, 0.0), frame=frame)
    bot.unit_order = {}
    return bot


def test_far_destination_may_drift_without_a_new_order(smooth):
    bot = walker()
    bot.order_move(5, (4000.0, 0.0))
    bot.frame += 30
    bot.order_move(5, (4000.0, 500.0))                   # 12% of the way
    assert len(bot.orders) == 1


def test_a_reaction_changes_destination_at_once(smooth):
    bot = walker()
    bot.order_move(5, (4000.0, 0.0))
    bot.frame += 1
    bot.order_source = "beh_retreat"
    bot.order_move(5, (0.0, 4000.0))
    assert len(bot.orders) == 2


def test_a_plan_does_not_redirect_a_unit_on_its_way(smooth):
    bot = walker()
    bot.order_source = "beh_raider_squad"
    bot.order_move(5, (4000.0, 0.0))
    bot.frame += 30
    bot.order_source = "beh_mex_patrol"
    bot.order_fight(5, (0.0, 4000.0))
    assert len(bot.orders) == 1
    bot.frame += smoothorders.COMMIT_FRAMES
    bot.order_fight(5, (0.0, 4000.0))
    assert len(bot.orders) == 2


def test_a_plan_may_send_on_a_unit_that_has_arrived(smooth):
    bot = walker()
    bot.order_source = "beh_raider_squad"
    bot.order_move(5, (100.0, 0.0))
    bot.frame += 30
    bot.order_move(5, (4000.0, 0.0))
    assert len(bot.orders) == 2


def test_a_plan_may_redirect_a_unit_that_was_shooting(smooth):
    bot = walker()
    bot.order_attack(5, 99)
    bot.frame += 1
    bot.order_source = "beh_attack"
    bot.order_fight(5, (4000.0, 0.0))
    assert len(bot.orders) == 2


def test_a_plan_leaves_a_unit_shooting_a_living_target_for_a_moment(smooth):
    bot = walker()
    bot.enemies = {99: object()}
    bot.order_attack(5, 99)
    bot.frame += 1
    bot.order_source = "beh_attack"
    bot.order_fight(5, (4000.0, 0.0))
    assert len(bot.orders) == 1
    bot.frame += smoothorders.ATTACK_COMMIT_FRAMES
    bot.order_fight(5, (4000.0, 0.0))
    assert len(bot.orders) == 2


def test_released_units_take_the_next_order_at_once(smooth):
    bot = walker()
    bot.order_source = "beh_raider_squad"
    bot.order_move(5, (4000.0, 0.0))
    bot.frame += 1
    bot.release_orders([5])
    bot.order_move(5, (0.0, 4000.0))
    assert len(bot.orders) == 2


def test_orders_soon_after_another_are_counted(smooth):
    bot = walker()
    bot.order_move(5, (4000.0, 0.0))
    bot.frame += 30
    bot.order_source = "beh_retreat"
    bot.order_move(5, (0.0, 4000.0))
    assert bot.order_counts["soon"] == 1


def test_close_destination_is_still_exact(smooth):
    bot = walker()
    bot.order_move(5, (300.0, 0.0))
    bot.frame += 30
    bot.order_move(5, (300.0, 200.0))
    assert len(bot.orders) == 2


def test_unchanged_order_repeats_slowly(smooth):
    bot = walker()
    bot.order_fight(5, (4000.0, 0.0))
    bot.frame += 300                                     # the old code sent one at 90
    bot.order_fight(5, (4000.0, 0.0))
    assert len(bot.orders) == 1
    bot.frame += smoothorders.REFRESH_FRAMES
    bot.order_fight(5, (4000.0, 0.0))
    assert len(bot.orders) == 2


def test_move_and_fight_to_one_place_do_not_flip_quickly(smooth):
    bot = walker()
    bot.order_move(5, (4000.0, 0.0))
    bot.frame += 15
    bot.order_fight(5, (4000.0, 0.0))
    assert len(bot.orders) == 1
    bot.frame += smoothorders.SWITCH_FRAMES
    bot.order_fight(5, (4000.0, 0.0))
    assert bot.orders[-1]["type"] == "fight"


def test_attack_goes_out_right_after_a_walk(smooth):
    bot = walker()
    bot.order_move(5, (4000.0, 0.0))
    bot.order_attack(5, 99)
    assert bot.orders[-1] == {"type": "attack", "unitId": 5, "targetId": 99}


def test_flag_off_keeps_the_old_repeat(monkeypatch):
    monkeypatch.setattr(smoothorders, "FLAGS", set())
    monkeypatch.setattr(military, "FLAGS", set())
    bot = walker()
    bot.order_move(5, (4000.0, 0.0))
    bot.frame += 90
    bot.order_move(5, (4000.0, 0.0))
    assert len(bot.orders) == 2


def test_order_reasons_are_counted(smooth):
    bot = walker()
    bot.order_move(5, (4000.0, 0.0))
    bot.frame += 450
    bot.order_move(5, (4000.0, 0.0))
    bot.frame += 1
    bot.order_move(5, (0.0, 4000.0))
    bot.frame += 100
    bot.order_fight(5, (0.0, 4000.0))
    assert bot.order_counts == {"first": 1, "repeat": 1, "jump": 1, "soon": 1, "switch": 1}


# ------------------------------------------------------------------ unstick, finishfirst

class Crew(BuildFocusMixin, EconomyMixin, Base):
    pass


@pytest.fixture
def focus(monkeypatch):
    monkeypatch.setattr(buildfocus, "FLAGS", {"unstick", "finishfirst"})


def solar_job(bot, key="energy:1:1", assignee=2):
    t = bot.add_task(key, "energy", units.id_of("energysolar"), 100.0, 100.0, 1000.0)
    t.assignees.add(assignee)
    bot.builder_task[assignee] = key
    return t


def test_guard_of_a_finished_building_ends_and_frees_the_constructor(focus):
    solar = unit(9, "energysolar", 100.0, 100.0, progress=0.5)
    bot = Crew(unit(2, "cloakcon", 0.0, 0.0), solar)
    solar_job(bot)
    bot.note_guard(2, 9, "energy:1:1")
    bot.release_finished_guards()
    assert bot.builder_task == {2: "energy:1:1"}          # not done yet
    solar.build_progress, solar.is_being_built = 1.0, False
    bot.release_finished_guards()
    assert bot.builder_task == {} and bot.tasks == {} and bot.builder_idle[2]


def test_guard_of_a_lost_building_frees_the_constructor_and_keeps_the_job(focus):
    bot = Crew(unit(2, "cloakcon", 0.0, 0.0))
    t = solar_job(bot)
    bot.note_guard(2, 9, t.key)
    bot.release_finished_guards()
    assert bot.builder_task == {} and t.key in bot.tasks and not t.assignees


def test_constructor_given_other_work_keeps_it(focus):
    bot = Crew(unit(2, "cloakcon", 0.0, 0.0))
    t = solar_job(bot)
    bot.note_guard(2, 9, t.key)
    bot.builder_task[2] = "mex:3"
    bot.release_finished_guards()
    assert bot.builder_task == {2: "mex:3"} and bot.builder_guards == {}


def test_guards_are_not_tracked_without_a_flag(monkeypatch):
    monkeypatch.setattr(buildfocus, "FLAGS", set())
    bot = Crew(unit(2, "cloakcon", 0.0, 0.0))
    bot.note_guard(2, 9, "x")
    assert not bot.__dict__.get("builder_guards")


def frames(n):
    return [unit(20 + i, "energysolar", 100.0 + 100.0 * i, 0.0, progress=0.5) for i in range(n)]


def waiting_jobs(bot, n, kind="mex"):
    for i in range(n):
        bot.add_task(f"{kind}:{i}", kind, units.id_of("staticmex"), 500.0 * i, 0.0, 1500.0)


def test_waiting_jobs_justify_constructors(focus):
    bot = Crew(frame=6000)
    waiting_jobs(bot, 4)
    assert bot.constructors_for_waiting_jobs() == 2


def test_the_opening_stays_lean_whatever_is_waiting(focus):
    bot = Crew(frame=1800)
    waiting_jobs(bot, 9)
    assert bot.constructors_for_waiting_jobs() == 0


def test_jobs_with_a_constructor_or_in_danger_do_not_count(focus):
    bot = Crew(frame=6000)
    waiting_jobs(bot, 6)
    bot.tasks["mex:0"].assignees.add(2)
    bot.tasks["mex:1"].unplaceable = True
    bot.threats = {(1000.0, 0.0), (1500.0, 0.0), (2000.0, 0.0), (2500.0, 0.0)}
    assert bot.constructors_for_waiting_jobs() == 0


def test_constructors_for_jobs_are_capped(focus):
    bot = Crew(frame=6000)
    waiting_jobs(bot, 60)
    assert bot.constructors_for_waiting_jobs() == buildfocus.MAX_FOR_JOBS


def test_limit_grows_with_income(focus):
    bot = Crew()
    bot.m_inc = 5.0
    assert bot.frame_limit() == 2
    bot.m_inc = 45.0
    assert bot.frame_limit() == 4
    bot.m_inc = 999.0
    assert bot.frame_limit() == buildfocus.MAX_FRAMES


def test_helps_instead_of_starting_another(focus):
    bot = Crew(unit(2, "cloakcon", 0.0, 0.0), *frames(3))
    assert bot.focus_on_frames(bot.units[2], 1000.0)
    assert bot.orders == [{"type": "guard", "unitId": 2, "targetId": 20}]
    assert bot.frame < bot.builder_fallback_until[2]


def test_starts_freely_under_the_limit_with_metal_banked_or_when_urgent(focus):
    bot = Crew(unit(2, "cloakcon", 0.0, 0.0), *frames(1))
    assert not bot.focus_on_frames(bot.units[2], 1000.0)
    bot = Crew(unit(2, "cloakcon", 0.0, 0.0), *frames(5))
    assert not bot.focus_on_frames(bot.units[2], buildfocus.URGENT_PRIORITY)
    bot.m_cur = 500.0
    assert not bot.focus_on_frames(bot.units[2], 1000.0)


def test_does_not_walk_across_the_map_to_help(focus):
    far = [unit(20 + i, "energysolar", 4000.0, 100.0 * i, progress=0.5) for i in range(3)]
    bot = Crew(unit(2, "cloakcon", 0.0, 0.0), *far)
    assert not bot.focus_on_frames(bot.units[2], 1000.0)


def test_not_too_many_helpers_on_one(focus):
    bot = Crew(unit(2, "cloakcon", 0.0, 0.0), unit(21, "energysolar", 100.0, 0.0, progress=0.5),
               unit(22, "energysolar", 900.0, 0.0, progress=0.5))
    bot.builder_guards = {100 + i: (21, None) for i in range(buildfocus.MAX_HELPERS)}
    assert bot.frame_to_help(bot.units[2], bot.unfinished_structures()).uid == 22


def test_helper_flees(focus):
    bot = Crew(unit(2, "cloakcon", 0.0, 0.0), *frames(3), foes=[unit(90, "cloakraid", 100.0, 0.0)])
    bot.focus_on_frames(bot.units[2], 1000.0)
    bot.release_finished_guards()
    assert bot.orders[-1] == ("retreat", 2)


def test_helper_returns_to_the_board_when_done(focus):
    bot = Crew(unit(2, "cloakcon", 0.0, 0.0), *frames(3))
    bot.focus_on_frames(bot.units[2], 1000.0)
    bot.units[20].build_progress, bot.units[20].is_being_built = 1.0, False
    bot.release_finished_guards()
    assert 2 not in bot.builder_fallback_until and bot.builder_idle[2]


# ------------------------------------------------------------------ reclaimcons

class Salvage(ReclaimMixin, EconomyMixin, Base):
    def __init__(self, *own, wrecks=(), want=2, have=None, **kw):
        super().__init__(*own, **kw)
        self._feature_cache = {str(i): {"pos": [x, 0.0, z], "definition": {"containedResources": {"Metal": m}}}
                               for i, (x, z, m) in enumerate(wrecks)}
        self.want, self.have = want, have

    def builder_count(self):
        return self.have if self.have is not None else sum(1 for u in self.units.values() if u.is_builder)

    def desired_builder_count(self):
        return self.want


@pytest.fixture
def salvage(monkeypatch):
    monkeypatch.setattr(reclaim, "FLAGS", {"reclaimcons"})


def crew(n):
    return [unit(1, "dyntrainer_support_base", *BASE)] + [unit(10 + i, "cloakcon", 4000.0, 4000.0 + i) for i in range(n)]


HOME_WRECKS = [(4000.0, 4100.0, 400.0), (4100.0, 4100.0, 400.0)]


def test_off_without_the_flag(monkeypatch):
    monkeypatch.setattr(reclaim, "FLAGS", set())
    bot = Salvage(*crew(3), wrecks=HOME_WRECKS, want=9)
    bot.reclaim_step()
    assert bot.orders == []


def test_at_least_two_for_700_metal(salvage):
    bot = Salvage(*crew(4), wrecks=HOME_WRECKS, want=9)
    bot.reclaim_step()
    assert [o["type"] for o in bot.orders] == ["reclaim", "reclaim"]
    assert set(bot.reclaim_jobs) <= {10, 11, 12, 13}


def test_one_when_there_are_only_a_couple_of_constructors(salvage):
    bot = Salvage(*crew(2), wrecks=HOME_WRECKS, want=9)
    bot.reclaim_step()
    assert len(bot.reclaim_jobs) == 1


def test_never_the_commander(salvage):
    bot = Salvage(*crew(0), wrecks=HOME_WRECKS, want=0)
    bot.reclaim_step()
    assert bot.orders == []


def test_too_little_metal_needs_no_crew(salvage):
    bot = Salvage(*crew(3), wrecks=[(4000.0, 4100.0, 300.0)], want=9)
    bot.reclaim_step()
    assert bot.orders == []


def test_surplus_constructors_go_for_less(salvage):
    bot = Salvage(*crew(4), wrecks=[(4000.0, 4100.0, 150.0), (4100.0, 4100.0, 150.0)], want=3)
    bot.reclaim_step()
    assert len(bot.reclaim_jobs) == 2


def test_unsafe_and_enemy_side_wrecks_do_not_count(salvage):
    bot = Salvage(*crew(3), wrecks=[(4000.0, 4100.0, 800.0)], want=9)
    bot.threats = {(4000.0, 4100.0)}
    assert bot.safe_wrecks() == []
    bot = Salvage(*crew(3), wrecks=[(400.0, 400.0, 800.0)], want=9)
    assert bot.safe_wrecks() == []
    bot = Salvage(*crew(3), wrecks=[(4000.0, 4100.0, 800.0)], want=9, foes=[unit(90, "cloakraid", 4000.0, 4200.0)])
    assert bot.safe_wrecks() == []


def test_full_storage_needs_no_more_metal(salvage):
    bot = Salvage(*crew(3), wrecks=HOME_WRECKS, want=9)
    bot.m_cur = 900.0
    bot.reclaim_step()
    assert bot.orders == []


def test_idle_constructor_goes_before_one_at_an_extractor(salvage):
    bot = Salvage(unit(10, "cloakcon", 4000.0, 4000.0), unit(11, "cloakcon", 3000.0, 3000.0), wrecks=HOME_WRECKS, want=9)
    bot.add_task("mex:1", "mex", units.id_of("staticmex"), 0.0, 0.0, 1500.0).assignees.add(10)
    bot.builder_task[10] = "mex:1"
    bot.reclaim_step()
    assert set(bot.reclaim_jobs) == {11} and bot.builder_task == {10: "mex:1"}


def test_taken_off_a_job_leaves_the_job_open(salvage):
    bot = Salvage(unit(10, "cloakcon", 4000.0, 4000.0), wrecks=HOME_WRECKS, want=9)
    t = bot.add_task("energy:1", "energy", units.id_of("energysolar"), 0.0, 0.0, 1000.0)
    t.assignees.add(10)
    bot.builder_task[10] = t.key
    bot.reclaim_step()
    assert 10 in bot.reclaim_jobs and not t.assignees and bot.builder_task == {}


def test_moves_on_when_the_wreck_is_gone_and_stays_off_the_board(salvage):
    bot = Salvage(*crew(1), wrecks=HOME_WRECKS, want=9)
    bot.reclaim_step()
    first = bot.reclaim_jobs[10][0]
    del bot._feature_cache[str(first)]
    bot.frame += reclaim.STEP_FRAMES
    bot.m_cur = 0.0
    bot.reclaim_step()
    assert bot.reclaim_jobs[10][0] != first and bot.frame < bot.builder_fallback_until[10]


def test_crew_stays_as_the_wrecks_run_down_then_goes_back(salvage):
    bot = Salvage(*crew(3), wrecks=HOME_WRECKS, want=9)
    bot.reclaim_step()
    n = len(bot.reclaim_jobs)
    bot._feature_cache["0"]["definition"]["containedResources"]["Metal"] = 100.0
    bot.frame += reclaim.STEP_FRAMES
    bot.reclaim_step()
    assert len(bot.reclaim_jobs) == n
    bot._feature_cache = {}
    bot.frame += reclaim.STEP_FRAMES
    bot.reclaim_step()
    assert bot.reclaim_jobs == {} and all(bot.builder_idle[u] for u in (10, 11, 12) if u in bot.builder_idle)


def test_flees_from_an_enemy(salvage):
    bot = Salvage(*crew(1), wrecks=HOME_WRECKS, want=9)
    bot.reclaim_step()
    bot.foes = [unit(90, "cloakraid", 4000.0, 4050.0)]
    bot.frame += reclaim.STEP_FRAMES
    bot.reclaim_step()
    assert ("retreat", 10) in bot.orders and 10 not in bot.reclaim_jobs


def test_wreck_that_takes_too_long_is_given_up(salvage):
    bot = Salvage(*crew(1), wrecks=HOME_WRECKS, want=9)
    bot.reclaim_step()
    first = bot.reclaim_jobs[10][0]
    bot.frame += reclaim.JOB_FRAMES + 1
    bot.reclaim_step()
    assert bot.reclaim_jobs[10][0] != first and first in bot.reclaim_bad


# ------------------------------------------------------------------ pylongrid

class Grid(PylonGridMixin, EconomyMixin, Base):
    od_metal, od_energy, od_waste = 2.0, 40.0, 0.0
    map = NS(spots=[])


def home_grid():
    """A generator touching an extractor."""
    return [unit(2, "energysolar", 0.0, 0.0), unit(3, "staticmex", 100.0, 0.0)]


def test_link_rule_is_the_sum_of_ranges():
    near = Grid(unit(2, "energysolar", 0.0, 0.0), unit(3, "staticmex", 149.0, 0.0)).grid_components()
    far = Grid(unit(2, "energysolar", 0.0, 0.0), unit(3, "staticmex", 151.0, 0.0)).grid_components()
    assert len(near) == 1 and len(far) == 2


def test_chain_links_through_each_member():
    bot = Grid(unit(2, "energysolar", 0.0, 0.0), unit(3, "energysolar", 190.0, 0.0), unit(4, "staticmex", 300.0, 0.0))
    assert len(bot.grid_components()) == 1


def test_unfinished_buildings_do_not_link():
    bot = Grid(unit(2, "energysolar", 0.0, 0.0), unit(3, "energypylon", 400.0, 0.0, progress=0.5),
               unit(4, "staticmex", 800.0, 0.0))
    assert len(bot.grid_components()) == 2


def test_main_grid_is_the_one_with_the_generators():
    bot = Grid(*home_grid(), unit(4, "staticmex", 3000.0, 0.0), unit(5, "staticmex", 3050.0, 0.0))
    assert {u.uid for u in bot.main_grid(bot.grid_components())} == {2, 3}
    assert Grid(unit(4, "staticmex", 0.0, 0.0)).main_grid(Grid(unit(4, "staticmex", 0.0, 0.0)).grid_components()) is None


def test_builds_a_pylon_that_really_reaches_both_sides():
    bot = Grid(*home_grid(), unit(4, "staticmex", 900.0, 0.0))
    bot.ensure_grid_pylons()
    t = bot.tasks[next(iter(bot.tasks))]
    assert t.kind == "pylon"
    assert bot.dist((t.x, t.z), (100.0, 0.0)) < 550.0 and bot.dist((t.x, t.z), (900.0, 0.0)) < 550.0
    bot.units[6] = unit(6, "energypylon", t.x, t.z)
    assert len(bot.grid_components()) == 1


def test_far_extractor_gets_the_chain_started_from_the_grid():
    bot = Grid(*home_grid(), unit(4, "staticmex", 1700.0, 0.0))
    bot.od_energy = 200.0
    bot.ensure_grid_pylons()
    t = bot.tasks[next(iter(bot.tasks))]
    assert bot.dist((t.x, t.z), (100.0, 0.0)) < 550.0 and t.x > 100.0
    bot.tasks.clear()
    bot.units[6] = unit(6, "energypylon", t.x, t.z)
    bot.frame += pylongrid.STEP_FRAMES
    bot.ensure_grid_pylons()
    t2 = bot.tasks[next(iter(bot.tasks))]
    assert t2.x > t.x and bot.dist((t2.x, t2.z), (t.x, t.z)) < 1000.0
    bot.units[7] = unit(7, "energypylon", t2.x, t2.z)
    assert len(bot.grid_components()) == 1


def test_the_sooner_it_pays_the_more_urgent():
    low = Grid(*home_grid(), unit(4, "staticmex", 900.0, 0.0))
    low.ensure_grid_pylons()
    high = Grid(*home_grid(), unit(4, "staticmex", 900.0, 0.0))
    high.od_energy = 400.0
    high.ensure_grid_pylons()
    assert next(iter(high.tasks.values())).priority > next(iter(low.tasks.values())).priority >= pylongrid.PRIORITY


def test_no_spare_energy_is_no_reason():
    bot = Grid(*home_grid(), unit(4, "staticmex", 900.0, 0.0))
    bot.od_energy = 0.0
    bot.ensure_grid_pylons()
    assert bot.tasks == {}


def test_long_chain_for_one_extractor_does_not_pay():
    bot = Grid(*home_grid(), unit(4, "staticmex", 6000.0, 0.0))
    bot.ensure_grid_pylons()
    assert bot.tasks == {}


def test_more_extractors_gain_more():
    bot = Grid(*home_grid())
    one, two = [unit(4, "staticmex", 900.0, 0.0)], [unit(4, "staticmex", 900.0, 0.0), unit(5, "staticmex", 950.0, 0.0)]
    grid = list(bot.units.values())
    assert bot.overdrive_gain(grid, two) > bot.overdrive_gain(grid, one) > 0.0


def test_one_pylon_at_a_time():
    bot = Grid(*home_grid(), unit(4, "staticmex", 900.0, 0.0), unit(5, "staticmex", 0.0, 900.0))
    bot.ensure_grid_pylons()
    bot.frame += pylongrid.STEP_FRAMES
    bot.ensure_grid_pylons()
    assert len(bot.tasks) == 1
    bot.tasks.clear()
    bot.units[6] = unit(6, "energypylon", 500.0, 0.0, progress=0.5)
    bot.frame += pylongrid.STEP_FRAMES
    bot.ensure_grid_pylons()
    assert bot.tasks == {}


def test_not_in_danger():
    bot = Grid(*home_grid(), unit(4, "staticmex", 900.0, 0.0))
    bot.threats = {(500.0, 0.0)}
    bot.ensure_grid_pylons()
    assert bot.tasks == {}


def test_placement_must_keep_the_link():
    bot = Grid(*home_grid())
    assert bot.pylon_links(600.0, 0.0) and not bot.pylon_links(700.0, 0.0)


# ------------------------------------------------------------------ fastgather, assaultfirst

class Army(AttackLineMixin, Base):
    pass


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(attackline, "FLAGS", {"fastgather", "assaultfirst"})


def test_main_target_is_kept_until_time_is_up(monkeypatch):
    monkeypatch.setattr(attackline, "FLAGS", {"stickytarget"})
    bot = Army()
    assert bot.kept_attack_target() is None
    bot.note_attack_target((100.0, 100.0))
    bot.frame += attackline.STICK_FRAMES - 1
    assert bot.kept_attack_target() == (100.0, 100.0)
    bot.note_attack_target((100.0, 100.0))                # the same again doesn't extend it
    bot.frame += 1
    assert bot.kept_attack_target() is None


def test_main_target_found_empty_is_not_kept(monkeypatch):
    monkeypatch.setattr(attackline, "FLAGS", {"stickytarget"})
    bot = Army()
    bot.note_attack_target((100.0, 100.0))
    bot.attack_blacklist = {(0, 0): bot.frame + 3600}
    assert bot.kept_attack_target() is None


def test_main_target_is_not_kept_without_the_flag(monkeypatch):
    monkeypatch.setattr(attackline, "FLAGS", set())
    bot = Army()
    bot.note_attack_target((100.0, 100.0))
    assert bot.kept_attack_target() is None


def test_gather_terms_only_change_with_the_flag(fast, monkeypatch):
    assert Army().gather_terms() == (attackline.GATHER_RADIUS, attackline.GATHER_SHARE)
    monkeypatch.setattr(attackline, "FLAGS", set())
    assert Army().gather_terms() == (700.0, 0.6)
    assert Army().regroup_allowed(True)


def test_regroup_has_a_time_limit_and_a_rest(fast):
    bot = Army()
    assert bot.regroup_allowed(True)
    bot.frame += attackline.REGROUP_FRAMES - 1
    assert bot.regroup_allowed(True)
    bot.frame += 2
    assert not bot.regroup_allowed(True)                  # attacking again
    bot.frame += attackline.REGROUP_REST
    assert bot.regroup_allowed(True)


def test_not_wanted_means_no_regroup_and_uses_no_time(fast):
    bot = Army()
    assert not bot.regroup_allowed(False)
    assert "regroup_until" not in bot.__dict__


def test_meeting_point_is_on_the_way_to_the_target(fast):
    bot = Army()
    assert bot.meeting_point((0.0, 0.0), (1000.0, 0.0)) == pytest.approx((300.0, 0.0))
    assert bot.meeting_point((0.0, 0.0), (9000.0, 0.0)) == pytest.approx((attackline.MEET_MAX, 0.0))
    assert bot.meeting_point((5.0, 5.0), None) == (5.0, 5.0)


def test_squads_meet_between_them_and_on_towards_the_target(fast):
    p = Army().squad_meeting_point([(0.0, 1000.0), (0.0, -1000.0)], (3000.0, 0.0))
    assert p[1] == pytest.approx(0.0) and 0.0 < p[0] < 3000.0 - attackline.SQUAD_STANDOFF


def test_squads_too_close_back_off_to_the_standoff(fast):
    p = Army().squad_meeting_point([(2800.0, 0.0)], (3000.0, 0.0))
    assert p == pytest.approx((3000.0 - attackline.SQUAD_STANDOFF, 0.0))


TARGET = (0.0, 0.0)


def battle(*rest, assaults=(1000.0, 1100.0, 1200.0)):
    front = [unit(10 + i, "cloakassault", d, 0.0) for i, d in enumerate(assaults)]
    return Army(*front, *rest)


def hold(bot, uid):
    return bot.hold_behind_front(bot.units[uid], list(bot.units), (1000.0, 0.0), TARGET)


def test_assaults_are_not_held(fast):
    bot = battle()
    assert hold(bot, 10) is None


def test_riot_ahead_of_the_assaults_waits_behind_them(fast):
    bot = battle(unit(20, "cloakriot", 500.0, 0.0))
    point = hold(bot, 20)
    assert point[0] == pytest.approx(1100.0 + attackline.RIOT_BEHIND)


def test_skirmisher_stays_further_back_than_a_riot(fast):
    bot = battle(unit(20, "cloakriot", 500.0, 0.0), unit(21, "cloakskirm", 500.0, 0.0))
    assert hold(bot, 21)[0] > hold(bot, 20)[0]


def test_unit_already_behind_carries_on(fast):
    bot = battle(unit(20, "cloakskirm", 1700.0, 0.0))
    assert hold(bot, 20) is None


def test_hold_point_stays_put_until_the_front_has_moved_on(fast):
    bot = battle(unit(20, "cloakriot", 500.0, 0.0))
    first = hold(bot, 20)
    bot.frame += 30
    for uid, d in ((10, 900.0), (11, 1000.0), (12, 1100.0)):
        bot.units[uid].x = d
    assert hold(bot, 20) == first                         # 100 elmos on
    bot.frame += 30
    for uid, d in ((10, 600.0), (11, 700.0), (12, 800.0)):
        bot.units[uid].x = d
    assert hold(bot, 20)[0] < first[0]


def test_no_front_means_no_holding(fast):
    bot = Army(unit(20, "cloakskirm", 500.0, 0.0))
    assert hold(bot, 20) is None


def test_riots_lead_when_there_are_no_assaults(fast):
    bot = Army(unit(20, "cloakriot", 1000.0, 0.0), unit(21, "cloakriot", 1000.0, 50.0), unit(22, "cloakskirm", 500.0, 0.0))
    assert hold(bot, 20) is None and hold(bot, 22) is not None


def test_far_from_any_fight_nobody_waits(fast):
    bot = Army(unit(10, "cloakassault", 5000.0, 0.0), unit(11, "cloakassault", 5100.0, 0.0), unit(20, "cloakriot", 4000.0, 0.0))
    assert bot.hold_behind_front(bot.units[20], list(bot.units), (5000.0, 0.0), TARGET) is None


def test_an_enemy_on_the_march_makes_the_order_matter(fast):
    bot = Army(unit(10, "cloakassault", 5000.0, 0.0), unit(11, "cloakassault", 5100.0, 0.0), unit(20, "cloakriot", 4000.0, 0.0),
               foes=[unit(90, "cloakraid", 4500.0, 0.0)])
    assert bot.hold_behind_front(bot.units[20], list(bot.units), (5000.0, 0.0), TARGET) is not None


def test_new_target_replaces_the_hold(fast):
    bot = battle(unit(20, "cloakriot", 500.0, 0.0))
    hold(bot, 20)
    bot.frame += 30
    assert bot.hold_behind_front(bot.units[20], list(bot.units), (1000.0, 0.0), (500.0, 1000.0)) is not None
    assert bot.attack_holds[20][2] == (500.0, 1000.0)


# ------------------------------------------------------------------ mexllt

class Mex(DefenseBuildMixin, EconomyMixin, Base):
    map = NS(center=(2500.0, 2500.0), spots=[])
    extractors = 6

    def mex_count(self):
        return self.extractors


@pytest.fixture
def llt(monkeypatch):
    monkeypatch.setattr(build_defense, "FLAGS", {"mexllt"})


SPOT = NS(index=7, x=3000.0, z=3000.0)
LATE = int(build_defense.MEXLLT_AFTER * 30) + 30


def test_off_without_the_flag(monkeypatch):
    monkeypatch.setattr(build_defense, "FLAGS", set())
    assert not Mex(frame=LATE).mex_waits_for_turret(SPOT, 1500.0)


def test_before_2_30_the_extractor_just_goes_ahead(llt):
    bot = Mex(frame=int(build_defense.MEXLLT_AFTER * 30) - 30)
    assert not bot.mex_waits_for_turret(SPOT, 1500.0) and bot.tasks == {}


def test_turret_first_after_2_30(llt):
    bot = Mex(frame=LATE)
    assert bot.mex_waits_for_turret(SPOT, 1500.0)
    t = bot.tasks["mexllt:7"]
    assert t.def_id == units.id_of("turretlaser") and t.priority == 1500.0
    assert bot.dist((t.x, t.z), (SPOT.x, SPOT.z)) >= 70.0
    assert bot.dist((t.x, t.z), ENEMY) < bot.dist((SPOT.x, SPOT.z), ENEMY)


def test_waits_while_the_turret_goes_up_then_the_extractor_follows(llt):
    bot = Mex(unit(30, "turretlaser", 3050.0, 3050.0, progress=0.5), frame=LATE)
    assert bot.mex_waits_for_turret(SPOT, 1500.0)
    bot.units[30].build_progress, bot.units[30].is_being_built = 1.0, False
    assert not bot.mex_waits_for_turret(SPOT, 1500.0)


def test_covered_spot_needs_no_turret(llt):
    bot = Mex(unit(30, "turretlaser", 3100.0, 3100.0), frame=LATE)
    assert not bot.mex_waits_for_turret(SPOT, 1500.0) and bot.tasks == {}


def test_far_turret_and_anti_air_do_not_cover_it(llt):
    bot = Mex(unit(30, "turretlaser", 4000.0, 4000.0), unit(31, "turretaalaser", 3050.0, 3050.0), frame=LATE)
    assert bot.mex_waits_for_turret(SPOT, 1500.0)


def test_gives_up_and_builds_the_extractor_anyway(llt):
    bot = Mex(frame=LATE)
    bot.mex_waits_for_turret(SPOT, 1500.0)
    bot.frame += build_defense.MEXLLT_GIVE_UP + 1
    assert not bot.mex_waits_for_turret(SPOT, 1500.0) and bot.tasks == {}
    bot.frame += 30
    assert not bot.mex_waits_for_turret(SPOT, 1500.0)     # doesn't ask again straight away


def test_unstarted_extractor_job_waits_for_the_turret_too(llt):
    bot = Mex(frame=LATE)
    bot.map = NS(center=(2500.0, 2500.0), spots=[NS(index=0, x=3000.0, z=3000.0, status="building", task_id="mex:0")])
    bot.frame = LATE - 900
    bot.add_task("mex:0", "mex", units.id_of("staticmex"), 3000.0, 3000.0, 2300.0, spot_index=0)
    bot.frame = LATE
    bot.hold_unstarted_mexes()
    assert list(bot.tasks) == ["mexllt:0"] and bot.map.spots[0].status == "free"
    assert bot.tasks["mexllt:0"].priority == 2300.0


def test_extractor_job_from_after_the_rule_began_is_not_dropped_again(llt):
    bot = Mex(frame=LATE)
    bot.map = NS(center=(2500.0, 2500.0), spots=[NS(index=0, x=3000.0, z=3000.0, status="building", task_id="mex:0")])
    bot.add_task("mex:0", "mex", units.id_of("staticmex"), 3000.0, 3000.0, 1500.0, spot_index=0)
    bot.frame += build_defense.MEXLLT_SKIP + 1
    bot.hold_unstarted_mexes()
    assert list(bot.tasks) == ["mex:0"]


def test_with_few_extractors_income_comes_first(llt):
    bot = Mex(frame=LATE)
    bot.extractors = build_defense.MEXLLT_MIN_MEX - 1
    assert not bot.mex_waits_for_turret(SPOT, 1500.0) and bot.tasks == {}


def test_extractor_job_under_way_is_left_alone(llt):
    bot = Mex(unit(2, "cloakcon", 2900.0, 2900.0), frame=LATE)
    bot.map = NS(center=(2500.0, 2500.0), spots=[NS(index=0, x=3000.0, z=3000.0, status="building", task_id="mex:0")])
    t = bot.add_task("mex:0", "mex", units.id_of("staticmex"), 3000.0, 3000.0, 2300.0, spot_index=0)
    t.assignees.add(2)
    bot.hold_unstarted_mexes()
    assert list(bot.tasks) == ["mex:0"]
    t.assignees.clear()
    bot.units[3] = unit(3, "staticmex", 3000.0, 3000.0, progress=0.5)
    bot.hold_unstarted_mexes()
    assert list(bot.tasks) == ["mex:0"]


def test_constructor_on_the_way_keeps_the_turret_job(llt):
    bot = Mex(frame=LATE)
    bot.mex_waits_for_turret(SPOT, 1500.0)
    bot.tasks["mexllt:7"].assignees.add(2)
    bot.frame += build_defense.MEXLLT_GIVE_UP + 1
    assert bot.mex_waits_for_turret(SPOT, 1500.0) and "mexllt:7" in bot.tasks


def test_only_a_couple_waiting_at_once(llt):
    bot = Mex(frame=LATE)
    for i in range(4):
        assert bot.mex_waits_for_turret(NS(index=i, x=3000.0, z=2000.0 + 400.0 * i), 1500.0)
    assert sum(t.kind == "mexllt" for t in bot.tasks.values()) == build_defense.MEXLLT_PENDING
