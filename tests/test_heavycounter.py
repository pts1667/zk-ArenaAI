"""Armored heavies such as the Crab: outrange them or overwhelm them, and keep raiders away."""
import math
import pathlib
import sys
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bots"))

from zkbot import heavycounter, raiding, snipers, tactics, units  # noqa: E402
from zkbot.heavycounter import HeavyCounterMixin, is_armored_heavy  # noqa: E402
from zkbot.models import UnitInfo  # noqa: E402
from zkbot.raiding import RaidMixin  # noqa: E402
from zkbot.snipers import PhantomMixin  # noqa: E402

CRAB_AT = (1000.0, 1000.0)


def unit(uid, name, x, z):
    d = units.by_name(name)
    hp = float(d.get("health") or 100.0)
    return UnitInfo(uid, 0, name, d, x, 0.0, z, hp, hp, 1.0, False)


class Bot(HeavyCounterMixin, RaidMixin, PhantomMixin):
    factory_name = "factorycloak"

    def __init__(self, *own, foes=(), frame=9000, income=30.0):
        self.units = {u.uid: u for u in own}
        self.foes = {e.uid: e for e in foes}
        self.frame, self.income = frame, income
        self.map = NS(width=6000.0, height=6000.0)

    def contacts(self):
        return self.foes

    def dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def metal_income(self):
        return self.income

    def frame_time(self):
        return self.frame / 30.0

    def seen_enemy_roles(self):
        return {"riot": 0, "assault": 0}

    def enemy_slow_ball(self):
        return None

    def log(self, msg):
        pass


@pytest.fixture(autouse=True)
def flag_on(monkeypatch):
    for mod in (heavycounter, raiding, snipers, tactics):
        monkeypatch.setattr(mod, "FLAGS", {"armorcounter", "hitrun"})


def crab(uid=90):
    return unit(uid, "spidercrabe", *CRAB_AT)


def test_the_data_singles_out_the_crab():
    assert is_armored_heavy("spidercrabe")
    for other in ("amphassault", "spiderskirm", "cloakassault", "tankheavyassault", "striderdante", "jumpsumo"):
        assert not is_armored_heavy(other), other


def test_off_without_the_flag(monkeypatch):
    monkeypatch.setattr(heavycounter, "FLAGS", set())
    bot = Bot(foes=[crab()])
    assert bot.armored_heavies() == 0
    assert bot.armored_counter_mix({"raid": 0.6}) == {"raid": 0.6}
    assert bot.armored_phantom_quota() == 0


def test_stays_counted_after_it_goes_out_of_sight():
    bot = Bot(foes=[crab()])
    assert bot.armored_heavies() == 1
    bot.foes = {}
    bot.frame += 3600                                     # long past the 60 s enemy memory
    assert bot.armored_heavies() == 1
    bot.frame += heavycounter.SEEN_FRAMES
    assert bot.armored_heavies() == 0


def test_production_turns_to_knights_and_slings():
    bot = Bot(foes=[crab()])
    mix = bot.armored_counter_mix({"raid": 0.62, "skirm": 0.3, "riot": 0.1, "assault": 0.1, "arty": 0.0, "aa": 0.07})
    assert mix["assault"] > mix["arty"] > mix["raid"] and mix["aa"] == 0.07


def test_nothing_changes_with_no_crab_around():
    desired = {"raid": 0.62}
    assert Bot(foes=[unit(90, "amphassault", *CRAB_AT)]).armored_counter_mix(desired) is desired


def test_phantoms_come_early_and_as_income_allows():
    bot = Bot(foes=[crab(), crab(91)], frame=3600, income=30.0)    # 2:00, before the usual 10:00
    assert bot.phantom_quota() == 2
    bot.income = 60.0
    assert bot.phantom_quota() == 3                       # one more than there are crabs
    bot.income = 10.0
    assert bot.phantom_quota() == 0


def ball(armored=True):
    return {"pos": CRAB_AT, "value": 1600.0, "range": 600.0, "count": 1, "speed": 40.0, "armored": armored}


def test_raiders_do_not_join_against_it():
    assert Bot().ball_joiners([1, 2, 3], ball()) == set()


def test_raiders_still_join_against_other_balls():
    bot = Bot(unit(1, "cloakraid", 0.0, 0.0))
    assert bot.ball_joiners([1], ball(armored=False)) == {1}


def test_raiders_alone_never_commit():
    bot = Bot(*[unit(i, "cloakraid", 2000.0, 2000.0) for i in range(60)])
    assert bot.ball_engagement(ball(), list(bot.units), (2000.0, 2000.0))[0] == "gather"


def test_enough_knights_commit():
    bot = Bot(*[unit(i, "cloakassault", 2000.0, 2000.0) for i in range(4)])
    assert bot.ball_engagement(ball(), list(bot.units), (2000.0, 2000.0)) == ("commit", CRAB_AT)
    bot = Bot(*[unit(i, "cloakassault", 2000.0, 2000.0) for i in range(3)])
    assert bot.ball_engagement(ball(), list(bot.units), (2000.0, 2000.0))[0] == "gather"


def test_sling_shells_from_its_own_range():
    bot = Bot()
    sling = unit(1, "cloakarty", 3000.0, 1000.0)
    x, z = bot.shell_point(sling, ball())
    assert z == pytest.approx(1000.0) and 600.0 < x - 1000.0 < 860.0


def test_units_it_outranges_do_not_shell():
    bot = Bot()
    assert bot.shell_point(unit(1, "cloakassault", 3000.0, 1000.0), ball()) is None
    assert bot.shell_point(unit(1, "cloakarty", 3000.0, 1000.0), ball(armored=False)) is None


def test_committed_knights_walk_in_but_others_may_still_back_off():
    bot = Bot()
    plan = NS(ball=NS(ball=ball()), posture=NS(attack_target=CRAB_AT))
    assert bot.closes_with_armored_ball(unit(1, "cloakassault", 0.0, 0.0), plan)
    assert not bot.closes_with_armored_ball(unit(2, "cloakriot", 0.0, 0.0), plan)
    plan.posture.attack_target = (5.0, 5.0)
    assert not bot.closes_with_armored_ball(unit(1, "cloakassault", 0.0, 0.0), plan)


def test_phantom_is_queued_whatever_the_production_counter_says(monkeypatch):
    from zkbot import production
    from zkbot.production import ProductionMixin

    class Factory(ProductionMixin, Bot):
        production_counter = 1                            # odd: the old rule never queued one
        army, hunt_commander, threat_pos, last_composition_log_frame = [], False, None, 0
        threat_frame = 0

        def builder_count(self):
            return 9

        def desired_builder_count(self):
            return 1

        def should_build_gremlin_scout(self):
            return False

    monkeypatch.setattr(production, "FLAGS", {"armorcounter"})
    bot = Factory(foes=[crab()])
    assert bot.next_unit_for_factory(unit(2, "factorycloak", 0.0, 0.0)) == units.id_of("cloaksnipe")
