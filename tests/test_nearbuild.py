"""A small raid shouldn't recall the army, least of all with help about to appear."""
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bots"))

from zkbot import nearbuild, threat, units  # noqa: E402
from zkbot.models import UnitInfo  # noqa: E402
from zkbot.nearbuild import NearBuildMixin  # noqa: E402

BASE = (4800.0, 4800.0)
FLEA_AT = (4000.0, 4000.0)


def unit(uid, name, x, z, progress=1.0):
    d = units.by_name(name)
    hp = float(d.get("health") or 100.0)
    return UnitInfo(uid, 0, name, d, x, 0.0, z, hp * progress, hp, progress, progress < 1.0)


class Bot(NearBuildMixin):
    spearhead = frozenset()
    commander_id = 1

    def patrol_help_power(self, pos):
        return 0.0

    def __init__(self, *own, foes=(), threat_pos=None, frame=3000):
        self.units = {u.uid: u for u in own}
        self.foes = list(foes)
        self.frame, self.threat_pos, self.threat_frame = frame, threat_pos, frame
        self.orders, self.lines, self.warnings = [], [], []

    def dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def local_balance(self, pos, radius):
        ours = sum(threat.combat_power(u) for u in self.units.values()
                   if u.is_combat and not u.is_being_built and self.dist(u.pos, pos) <= radius)
        return ours, sum(threat.combat_power(e) for e in self.foes if self.dist(e.pos, pos) <= radius)

    def update_raid_warnings(self):
        return self.warnings

    def queue(self, order):
        self.orders.append(order)

    def log(self, msg):
        self.lines.append(msg)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr(nearbuild, "FLAGS", {"nearbuild"})


def factory():
    return unit(2, "factorycloak", *BASE)


def test_off_without_the_flag(monkeypatch):
    monkeypatch.setattr(nearbuild, "FLAGS", set())
    assert not Bot(factory(), threat_pos=FLEA_AT).minor_threat(FLEA_AT)
    bot = Bot(factory(), unit(3, "cloakraid", *BASE, progress=0.9), threat_pos=FLEA_AT)
    bot.update_factory_priority()
    assert bot.orders == []


def test_glaive_nearly_done_answers_a_flea(flag_on):
    flea = unit(90, "spiderscout", *FLEA_AT)
    bot = Bot(factory(), unit(3, "cloakraid", *BASE, progress=0.8), foes=[flea], threat_pos=FLEA_AT)
    assert bot.minor_threat(FLEA_AT)


def test_glaive_just_started_does_not(flag_on):
    flea = unit(90, "spiderscout", *FLEA_AT)
    bot = Bot(factory(), unit(3, "cloakraid", *BASE, progress=0.1), foes=[flea], threat_pos=FLEA_AT)
    assert not bot.minor_threat(FLEA_AT)


def test_a_real_attack_still_recalls(flag_on):
    foes = [unit(90 + i, "cloakassault", *FLEA_AT) for i in range(5)]
    bot = Bot(factory(), unit(3, "cloakraid", *BASE, progress=0.9), foes=foes, threat_pos=FLEA_AT)
    assert not bot.minor_threat(FLEA_AT)


def test_commander_nearby_counts(flag_on):
    flea = unit(90, "spiderscout", *FLEA_AT)
    near = Bot(unit(1, "dyntrainer_support_base", 4200.0, 4000.0), foes=[flea], threat_pos=FLEA_AT)
    far = Bot(unit(1, "dyntrainer_support_base", 5000.0, 5000.0), foes=[flea], threat_pos=FLEA_AT)
    assert near.minor_threat(FLEA_AT) and not far.minor_threat(FLEA_AT)


def test_spearhead_is_not_counted_as_help(flag_on):
    flea = unit(90, "spiderscout", *FLEA_AT)
    bot = Bot(unit(3, "cloakraid", 4100.0, 4000.0), foes=[flea], threat_pos=FLEA_AT)
    assert bot.minor_threat(FLEA_AT)
    bot.spearhead = {3}
    assert not bot.minor_threat(FLEA_AT)


def test_seconds_left_from_build_time(flag_on):
    glaive = unit(3, "cloakraid", *BASE, progress=0.5)
    assert Bot().build_seconds_left(factory(), glaive) == pytest.approx(3.25)      # 65 at 10 per second, half done


def test_metal_stall_shows_in_seconds_left(flag_on):
    glaive = unit(3, "cloakraid", *BASE, progress=0.5)
    bot = Bot()
    bot.build_seconds_left(factory(), glaive)
    bot.frame += 30
    assert bot.build_seconds_left(factory(), glaive) == math.inf                   # no progress in a second
    glaive.build_progress = 0.55
    bot.frame += 30
    assert bot.build_seconds_left(factory(), glaive) == pytest.approx(9.0)         # 0.45 left at 0.05 a second


def test_nanoframe_elsewhere_is_not_the_factorys(flag_on):
    assert Bot(factory(), unit(3, "cloakraid", 1000.0, 1000.0, progress=0.9)).factory_builds() == []


def test_constructor_in_production_is_no_help(flag_on):
    assert Bot(factory(), unit(3, "cloakcon", *BASE, progress=0.9)).factory_builds() == []


def test_priority_goes_up_with_a_threat_and_back_after(flag_on):
    bot = Bot(factory(), unit(3, "cloakraid", *BASE, progress=0.5), threat_pos=FLEA_AT)
    bot.update_factory_priority()
    bot.update_factory_priority()
    assert [(o["unitId"], o["cmdId"], o["params"]) for o in bot.orders] == [(2, 34220, [2.0])]
    bot.frame += nearbuild.RECENT_FRAMES
    bot.update_factory_priority()
    assert bot.orders[-1]["params"] == [1.0] and bot.boosted_factories == set()


def test_raid_warning_also_raises_priority(flag_on):
    bot = Bot(factory(), unit(3, "cloakraid", *BASE, progress=0.5))
    bot.warnings = [{"pos": FLEA_AT, "power": 60.0}]
    bot.update_factory_priority()
    assert bot.orders[0]["params"] == [2.0]


def test_no_order_for_a_lost_factory(flag_on):
    bot = Bot(factory(), unit(3, "cloakraid", *BASE, progress=0.5), threat_pos=FLEA_AT)
    bot.update_factory_priority()
    del bot.units[2]
    bot.orders.clear()
    bot.update_factory_priority()
    assert bot.orders == [] and bot.boosted_factories == set()
