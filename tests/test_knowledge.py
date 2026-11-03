"""Offline checks of the knowledge layer (no engine needed).

  python -m pytest tests -q

They pin down facts the bot relies on, so a data refresh after a balance
patch (tools/extract_weapons.py, tools/extract_abilities.py) or a refactor
that breaks them fails here instead of in a 20-minute game.
"""
from __future__ import annotations

import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "bots"))

from zkbot import catalog, units  # noqa: E402

GROUND_FACTORIES = ("factorycloak", "factoryshield", "factoryveh", "factorytank",
                    "factoryhover", "factoryamph", "factoryspider", "factoryjump")


def test_every_ground_factory_has_builder_raider_and_mainline_roles():
    for f in GROUND_FACTORIES:
        roles = {catalog.role(n) for n in catalog.roster(f)}
        assert "builder" in roles, f
        assert roles & {"raid", "scout"}, f
        assert roles & {"riot", "assault", "skirm"}, f


def test_factory_unit_never_falls_back_to_nothing_for_core_roles():
    for f in GROUND_FACTORIES:
        for role in ("builder", "raid"):
            assert units.factory_unit(f, role) is not None, (f, role)


def test_inferred_roles_mostly_agree_with_hand_roles():
    agree = total = 0
    for f in GROUND_FACTORIES:
        for n in catalog.roster(f):
            hand = units.role_of(units.by_name(n))
            if hand in ("combat", "", "other", "support", "heavy"):
                continue
            total += 1
            agree += catalog.infer_role(n) == hand
    assert total > 50
    assert agree / total >= 0.9


def test_abilities_capture_special_weapons():
    dante = catalog.manual_weapon("striderdante")
    assert dante and "fire" in dante["effects"] and dante["range"] > 300
    reef = catalog.manual_weapon("shipcarrier")
    assert reef and "disarm" in reef["effects"]
    assert "shield" in catalog.tags("shieldshield")
    assert "area_cloak" in catalog.tags("cloakjammer")
    assert "jump" in catalog.tags("jumpsumo")
    assert catalog.roles("striderdante")[:2] in (["riot", "assault"], ["assault", "riot"])


def test_known_raiders_are_raiders():
    for n in ("cloakraid", "shieldraid", "vehraid", "tankraid", "hoverraid", "amphraid", "jumpraid"):
        assert catalog.role(n) == "raid", n


def test_flags_are_exact_tokens(monkeypatch):
    monkeypatch.setenv("ZKBOT_FLAGS", "noemergency, oldmicro")
    monkeypatch.setenv("ZKBOT_PARAMS", "resign.low_share=0.25,bogus=3")
    from zkbot import params

    importlib.reload(params)
    assert "emergency" not in params.FLAGS
    assert "noemergency" in params.FLAGS and "oldmicro" in params.FLAGS
    assert params.P("resign.low_share", 0.30, 0.1, 0.6) == 0.25
    assert params.P("x.clamped", 5.0, 0.0, 1.0) == 1.0
    assert "UNKNOWN" in params.describe()
    monkeypatch.delenv("ZKBOT_FLAGS")
    monkeypatch.delenv("ZKBOT_PARAMS")
    importlib.reload(params)


def test_brain_composes():
    from zkbot.brain import Brain
    from zkbot.behaviours import UNIT_BEHAVIOURS

    for name in UNIT_BEHAVIOURS:
        assert hasattr(Brain, name), name
    for name in ("plan_army", "plan_raids", "plan_posture", "economy_step", "ensure_mex_tasks",
                 "ensure_defense_tasks", "engagement_step", "resign_step"):
        assert hasattr(Brain, name), name
