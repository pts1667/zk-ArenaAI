"""Builders and lava: a map split by a lava channel, with a bridge-less far side."""
import base64
import pathlib
import sys
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bots"))

from zkbot.hazards import Hazards, move_class  # noqa: E402
from zkbot.mapdata import MapData  # noqa: E402
from zkbot.terrain import TerrainMap, slope_from_corners  # noqa: E402

N = 320
WEST, EAST = (500.0, 1280.0), (2100.0, 1280.0)
RIVER = (1280.0, 1280.0)


def _heights():
    """Banks gentle enough for a hover, so only the lava keeps walkers out."""
    x = np.abs(np.arange(N) - N // 2).astype(np.float32)
    row = np.clip(-10.0 + (x - 20.0) * 50.0 / 30.0, -10.0, 40.0)
    return np.tile(row, (N, 1)).astype(np.float32)


def _hazards(damage=50.0):
    return Hazards(TerrainMap(_heights(), spots=[(*WEST, 2.0), (*EAST, 2.0)], water_damage=damage))


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
    m.set_map_features({"waterDamage": damage, "waterIsHarmful": damage > 0.0,
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


def test_walkers_stay_on_their_side_of_the_lava():
    z = _hazards()
    assert z.can_walk("cloakcon", WEST, (700.0, 1500.0))
    assert not z.can_walk("cloakcon", WEST, EAST)
    assert not z.can_walk("hovercon", WEST, EAST)
    assert not z.can_walk("amphcon", WEST, EAST)


def test_flyers_go_anywhere():
    assert _hazards().can_walk("gunshipcon", WEST, EAST)
    assert _hazards().can_walk("gunshipcon", WEST, RIVER)


def test_target_in_the_lava_is_out_of_reach():
    assert not _hazards().can_walk("cloakcon", WEST, RIVER)


def test_unit_stranded_on_odd_ground_is_not_stuck_for_good():
    assert _hazards().can_walk("cloakcon", (1180.0, 1280.0), WEST)


def test_only_ground_clear_of_the_liquid_takes_a_building():
    z = _hazards()
    assert z.site_bad(*RIVER)
    assert not z.terrain.hazard_at(1070.0, 1280.0) and z.site_bad(1070.0, 1280.0)   # dry, but its footprint isn't
    assert not z.site_bad(*WEST)


def test_spots_out_of_every_builders_reach_are_dropped():
    z = _hazards()
    near, far = NS(x=WEST[0], z=WEST[1]), NS(x=EAST[0], z=EAST[1])
    walker = NS(name="cloakcon", pos=WEST)
    assert z.spots_in_reach([near, far], [walker]) == [near]
    # A builder on the far side, or one that flies, brings it back.
    assert z.spots_in_reach([near, far], [walker, NS(name="cloakcon", pos=EAST)]) == [near, far]
    assert z.spots_in_reach([near, far], [NS(name="gunshipcon", pos=WEST)]) == [near, far]
    assert z.spots_in_reach([near, far], []) == []


def test_skipped_spots_are_logged_when_the_count_changes():
    lines = []
    z = Hazards(_hazards().terrain, lines.append)
    spots = [NS(x=EAST[0], z=EAST[1])]
    walker = [NS(name="cloakcon", pos=WEST)]
    z.spots_in_reach(spots, walker)
    z.spots_in_reach(spots, walker)
    assert len(lines) == 1 and "1 of 1" in lines[0]


def test_build_from_live_queries():
    lines = []
    z = Hazards.build(FakeClient(), _map(), lines.append)
    assert z is not None and z.terrain.liquid == "lava" and not z.terrain.hover_safe
    assert not z.can_walk("hovercon", WEST, EAST)
    assert "avoidlava: lava" in lines[0] and "build=" in lines[0]


def test_lava_level_param_means_hover_is_safe():
    z = Hazards.build(FakeClient(level=1.0), _map(), lambda _m: None)
    assert z.terrain.hover_safe and z.can_walk("hovercon", WEST, EAST)
    assert not z.can_walk("cloakcon", WEST, EAST)


def test_a_failed_query_leaves_the_bot_without_it():
    lines = []
    assert Hazards.build(FakeClient(fail="slope_map"), _map(), lines.append) is None
    assert "no terrain" in lines[0]
