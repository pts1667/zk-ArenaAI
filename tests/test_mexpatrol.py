"""A few raiders walk our extractors and answer threats, and do nothing else."""
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bots"))

from zkbot import mexpatrol, raidprobe, tactics, units  # noqa: E402
from zkbot.mexpatrol import MexPatrolMixin  # noqa: E402
from zkbot.models import UnitInfo  # noqa: E402

BASE = (4800.0, 4800.0)


def unit(uid, name, x, z):
    d = units.by_name(name)
    hp = float(d.get("health") or 100.0)
    return UnitInfo(uid, 0, name, d, x, 0.0, z, hp, hp, 1.0, False)


class Bot(MexPatrolMixin, tactics.TacticsMixin):
    def __init__(self, *own, t=120.0, foes=()):
        self.t, self.frame = t, int(t * 30)
        self.units = {u.uid: u for u in own}
        self.enemies = {e.uid: e for e in foes}
        self.radar, self.enemy_structures, self.retreating = {}, {}, set()
        self.base_pos, self.warnings, self.orders = BASE, [], []

    def frame_time(self):
        return self.t

    def dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def is_dangerous(self, e):
        return e.is_combat or "dyntrainer" in e.name

    def enemy_near(self, pos, radius):
        return next((e for e in self.enemies.values() if self.is_dangerous(e) and self.dist(e.pos, pos) < radius), None)

    def update_raid_warnings(self):
        return self.warnings

    def order_fight(self, uid, pos):
        self.orders.append((uid, pos))


@pytest.fixture(autouse=True)
def flags(monkeypatch):
    for mod in (mexpatrol, tactics, raidprobe):
        monkeypatch.setattr(mod, "FLAGS", {"mexpatrol", "hitrun", "raidprobe"})


def raiders(n, start=10):
    return [unit(start + i, "cloakraid", 4000.0 - i * 100.0, 4000.0) for i in range(n)]


def test_off_without_the_flag(monkeypatch):
    monkeypatch.setattr(mexpatrol, "FLAGS", set())
    bot = Bot(*raiders(10))
    assert bot.update_mex_patrol(list(bot.units), set()) == set()
    assert not bot.patrol_short()
    assert bot.beh_mex_patrol(bot.units[10], None) is None


def test_about_a_fifth_of_raiders_and_at_least_two():
    bot = Bot(*raiders(5))
    assert len(bot.update_mex_patrol(list(bot.units), set())) == 2
    bot = Bot(*raiders(20))
    assert len(bot.update_mex_patrol(list(bot.units), set())) == 4


def test_spearhead_is_never_taken():
    bot = Bot(*raiders(5))
    assert bot.update_mex_patrol(list(bot.units), {10, 11, 12}) == {13, 14}
    bot = Bot(*raiders(3))
    assert bot.update_mex_patrol(list(bot.units), {10, 11, 12}) == set()


def test_membership_lasts_and_gaps_are_filled():
    bot = Bot(*raiders(6))
    first = bot.update_mex_patrol(list(bot.units), set())
    assert bot.update_mex_patrol(list(bot.units), set()) == first
    gone = min(first)
    del bot.units[gone]
    second = bot.update_mex_patrol(list(bot.units), set())
    assert len(second) == 2 and first - {gone} <= second


def test_production_covers_the_minimum_before_it_is_due():
    bot = Bot(*raiders(3), t=30.0)
    assert not bot.patrol_short()
    bot.t = 60.0
    assert bot.patrol_short()
    bot.mex_patrol = {10, 11}
    assert not bot.patrol_short()


def test_answers_a_flea_at_an_extractor():
    flea = unit(90, "spiderscout", 3000.0, 3000.0)
    bot = Bot(unit(1, "staticmex", 3100.0, 3000.0), *raiders(2), foes=[flea])
    bot.mex_patrol = {10, 11}
    assert bot.beh_mex_patrol(bot.units[10], None) == "patrol_respond"
    assert bot.orders[-1] == (10, flea.pos)


def test_answers_a_raid_warning():
    bot = Bot(unit(1, "staticmex", 3100.0, 3000.0), *raiders(2))
    bot.mex_patrol = {10, 11}
    bot.warnings = [{"pos": (3200.0, 3200.0), "power": 60.0}]
    assert bot.beh_mex_patrol(bot.units[10], None) == "patrol_respond"


def test_enemy_away_from_the_economy_is_not_its_business():
    bot = Bot(unit(1, "staticmex", 3100.0, 3000.0), *raiders(2), foes=[unit(90, "cloakraid", 300.0, 300.0)])
    bot.mex_patrol = {10, 11}
    assert bot.beh_mex_patrol(bot.units[10], None) == "patrol"


def test_does_not_throw_itself_at_a_riot():
    riot = unit(90, "cloakriot", 3000.0, 3000.0)
    bot = Bot(unit(1, "staticmex", 3100.0, 3000.0), unit(2, "staticmex", 3900.0, 4000.0), *raiders(2), foes=[riot])
    bot.mex_patrol = {10, 11}
    assert bot.beh_mex_patrol(bot.units[10], None) == "patrol"
    assert bot.orders[-1] == (10, (3900.0, 4000.0))      # not the extractor under the riot


def test_walks_on_to_the_extractor_longest_unvisited():
    a, b = unit(1, "staticmex", 3000.0, 3000.0), unit(2, "staticmex", 3500.0, 3000.0)
    bot = Bot(a, b, *raiders(1))
    bot.mex_patrol = {10}
    bot.units[10].x, bot.units[10].z = a.pos
    bot.beh_mex_patrol(bot.units[10], None)
    assert bot.orders[-1] == (10, b.pos)
    bot.frame += 300
    bot.units[10].x, bot.units[10].z = b.pos
    bot.beh_mex_patrol(bot.units[10], None)
    assert bot.orders[-1] == (10, a.pos)


def test_two_patrols_take_different_extractors():
    a, b = unit(1, "staticmex", 3000.0, 3000.0), unit(2, "staticmex", 3000.0, 3400.0)
    bot = Bot(a, b, *raiders(2))
    bot.mex_patrol = {10, 11}
    bot.beh_mex_patrol(bot.units[10], None)
    bot.beh_mex_patrol(bot.units[11], None)
    assert {bot.orders[0][1], bot.orders[1][1]} == {a.pos, b.pos}


def test_no_extractors_means_stay_home():
    bot = Bot(*raiders(1))
    bot.mex_patrol = {10}
    bot.beh_mex_patrol(bot.units[10], None)
    assert bot.orders[-1] == (10, BASE)
