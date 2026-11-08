"""The live terrain build, on a map split by a lava channel with a bridge-less far side, and movement classes."""
import base64
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bots"))

from zkbot.hazards import Hazards, move_class  # noqa: E402
from zkbot.mapdata import MapData  # noqa: E402
from zkbot.terrain import slope_from_corners  # noqa: E402

N = 320
WEST, EAST = (500.0, 1280.0), (2100.0, 1280.0)


def _heights():
    """Banks gentle enough for a hover, so only the lava keeps walkers out."""
    x = np.abs(np.arange(N) - N // 2).astype(np.float32)
    row = np.clip(-10.0 + (x - 20.0) * 50.0 / 30.0, -10.0, 40.0)
    return np.tile(row, (N, 1)).astype(np.float32)


def _encode(a):
    return {"width": a.shape[1], "height": a.shape[0], "data_b64": base64.b64encode(a.astype("<f4").tobytes()).decode()}


class FakeClient:
    def __init__(self, level=-12345.0, fail=None):
        self.level, self.fail = level, fail

    def query(self, q, timeout=10.0):
        if q["type"] == self.fail:
            raise RuntimeError("boom")
        if q["type"] == "height_map":
            return _encode(_heights())
        if q["type"] == "slope_map":
            corners = np.pad(_heights(), ((0, 1), (0, 1)), mode="edge")
            return _encode(slope_from_corners(corners))
        assert q["type"] == "game_rules_param" and q["key"] == "lavaLevel"
        return {"floatValue": q["defaultFloat"] if self.level == -12345.0 else self.level, "stringValue": ""}


def _map(damage=50.0):
    m = MapData()
    m.set_map_features({"waterDamage": damage,
                        "spots": [{"pos": [*WEST[:1], 0.0, WEST[1]], "income": 2.0}]})
    return m


def test_move_classes():
    assert move_class("cloakcon") == "kbot"
    assert move_class("dynassault0") == "kbot"
    assert move_class("hovercon") == "hover"
    assert move_class("spidercon") == "spider"
    assert move_class("tankcon") == move_class("vehcon") == "tank"
    assert move_class("shipcon") == "ship"
    assert move_class("gunshipcon") is None and move_class("planecon") is None



def test_build_from_live_queries():
    lines = []
    z = Hazards.build(FakeClient(), _map(), lines.append)
    assert z is not None and z.terrain.liquid == "lava" and not z.terrain.hover_safe
    assert not z.terrain.can_reach("hover", WEST, EAST)
    assert "terrain: lava" in lines[0] and "build=" in lines[0]


def test_lava_level_param_means_hover_is_safe():
    z = Hazards.build(FakeClient(level=1.0), _map(), lambda _m: None)
    assert z.terrain.hover_safe and z.terrain.can_reach("hover", WEST, EAST)
    assert not z.terrain.can_reach("kbot", WEST, EAST)


def test_a_failed_query_leaves_the_bot_without_it():
    lines = []
    assert Hazards.build(FakeClient(fail="slope_map"), _map(), lines.append) is None
    assert "terrain: none" in lines[0]
