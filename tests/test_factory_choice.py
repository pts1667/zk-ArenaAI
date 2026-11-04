"""Lava and void read as water in the heights, but hovers can't use them."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bots"))

from zkbot import mapdata  # noqa: E402
from zkbot.mapdata import MapData  # noqa: E402


@pytest.fixture
def safefac(monkeypatch):
    monkeypatch.setattr(mapdata, "flag", lambda name: name == "safefac")


def _map(water=0.5, damage=None):
    m = MapData()
    features = {} if damage is None else {"waterDamage": damage, "waterIsHarmful": damage > 0.0}
    m.set_map_features(features)
    m.water_fraction = water
    return m


def test_water_map_favours_hover(safefac):
    assert _map(water=0.5).favours_hover(ground=50.0)
    assert _map(water=0.5, damage=0.0).favours_hover(ground=50.0)
    assert _map(water=0.0).favours_hover(ground=-20.0)


def test_dry_map_does_not(safefac):
    assert not _map(water=0.1).favours_hover(ground=50.0)


@pytest.mark.parametrize("damage", [7.5, 50.0, 175.0, 1000.0, 50000.0])
def test_lava_and_void_do_not(safefac, damage):
    assert not _map(water=0.6, damage=damage).favours_hover(ground=50.0)
    assert not _map(water=0.6, damage=damage).favours_hover(ground=-20.0)


def test_without_the_flag_nothing_changes(monkeypatch):
    monkeypatch.setattr(mapdata, "flag", lambda name: False)
    assert _map(water=0.6, damage=175.0).favours_hover(ground=50.0)
    assert not _map(water=0.1, damage=175.0).favours_hover(ground=50.0)
