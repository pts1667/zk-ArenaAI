"""Terrain regions on synthetic maps with a known right answer (no engine)."""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "bots"))

from zkbot import terrain as t  # noqa: E402

N = 320                       # squares: a 2560-elmo map


def flat(h=50.0, n=N):
    return np.full((n, n), h, np.float32)


def two_halves_cliff(gap_rows=None):
    """Low west, plateau east, a sheer cliff between; optional ramp rows."""
    h = flat(50.0)
    h[:, N // 2:] = 250.0
    if gap_rows:
        z0, z1 = gap_rows
        for x in range(N // 2 - 40, N // 2 + 40):
            h[z0:z1, x] = 50.0 + 200.0 * (x - (N // 2 - 40)) / 80.0
    return h


def channel(depth, bank=30, half=30, ground=40.0):
    """Ground either side of a north-south channel, with walkable sloped banks."""
    h = flat(ground)
    x = np.abs(np.arange(N) - N // 2).astype(np.float32)
    h[:, :] = np.where(x < half, depth, np.where(x < half + bank, depth + (ground - depth) * (x - half) / bank, ground))
    return h


W = N * t.SQUARE
WEST, EAST = (W * 0.2, W * 0.5), (W * 0.8, W * 0.5)
SPOTS = [(*WEST, 2.0), (*EAST, 2.0)]


def test_max_slope_matches_engine_clamp():
    assert t.MOVE_CLASSES["kbot"].max_slope == pytest.approx(1 - math.cos(math.radians(54)))
    assert t.MOVE_CLASSES["tank"].max_slope == pytest.approx(1 - math.cos(math.radians(27)))
    # 70 degrees clamps to 60: all terrain, since slope never exceeds 1.
    assert t.MOVE_CLASSES["spider"].max_slope == pytest.approx(1.0)


def test_slope_from_corners_flat_and_incline():
    assert np.allclose(t.slope_from_corners(np.full((9, 9), 7.0)), 0.0)
    # A constant incline: every face normal has the same y.
    z, x = np.mgrid[0:9, 0:9]
    s = t.slope_from_corners(x * 8.0)
    assert np.allclose(s, 1 - 1 / math.sqrt(2), atol=1e-5)


def test_cliff_separates_regions_for_bots_not_spiders():
    m = t.TerrainMap(two_halves_cliff(), spots=SPOTS)
    a, b = m.region_at(*WEST), m.region_at(*EAST)
    assert a.rid != b.rid
    assert not m.can_reach("kbot", WEST, EAST)
    assert m.can_reach("spider", WEST, EAST)
    assert m.route(WEST, EAST, "kbot") is None
    assert m.route(WEST, EAST, "spider") is not None


def test_ramp_joins_two_levels():
    m = t.TerrainMap(two_halves_cliff((N // 2 - 16, N // 2 + 16)), spots=SPOTS, cluster_radius=1500.0)
    assert m.can_reach("kbot", WEST, EAST)
    assert len(m.levels) == 2 and len(m.ramps) == 1
    r = m.ramps[0]
    assert m.levels[r.low].height == pytest.approx(50.0, abs=5)
    assert m.levels[r.high].height == pytest.approx(250.0, abs=5)
    assert r.rise == pytest.approx(200.0, abs=10)
    assert abs(r.x - W * 0.5) < 96 and abs(r.z - W * 0.5) < 96
    assert 192 <= r.width <= 320 and 500 <= r.length <= 750
    assert r.tanks
    east, west = m.region_at(*EAST), m.region_at(*WEST)
    assert "plateau" in east.tags and "basin" in west.tags
    assert (r.low_region, r.high_region) == (west.rid, east.rid)
    cost, gates = m.route(WEST, EAST, "kbot")
    assert any(g.ramp == r.rid and "ramp" in g.tags for g in gates)
    assert cost >= 0.99 * math.hypot(WEST[0] - EAST[0], WEST[1] - EAST[1])


def test_slope_to_a_small_bump_goes_nowhere():
    """A hill with a little flat top is no level, so its slope is no ramp."""
    h = flat(50.0)
    zz, xx = np.ogrid[:N, :N]
    r = np.sqrt((xx - N // 2) ** 2 + (zz - N // 2) ** 2)
    h = np.maximum(h, np.clip(150.0 - (r - 8.0) * 3.0, 50.0, 150.0)).astype(np.float32)
    m = t.TerrainMap(h, spots=SPOTS)
    assert len(m.levels) == 1 and not m.ramps


def test_cliff_without_a_slope_is_no_ramp():
    m = t.TerrainMap(two_halves_cliff(), spots=SPOTS)
    assert len(m.levels) == 2 and not m.ramps


def test_small_step_is_no_ramp():
    x = np.clip((np.arange(N) - N // 2) / 40.0, 0.0, 1.0)
    h = np.tile(50.0 + 20.0 * x, (N, 1)).astype(np.float32)   # 20 elmos: under RAMP_RISE
    assert not t.TerrainMap(h, spots=SPOTS).ramps


def test_two_separate_ramps_and_stairs():
    h = two_halves_cliff()
    for z0, z1 in ((60, 92), (228, 260)):
        for x in range(N // 2 - 40, N // 2 + 40):
            h[z0:z1, x] = 50.0 + 200.0 * (x - (N // 2 - 40)) / 80.0
    assert len(t.TerrainMap(h, spots=SPOTS).ramps) == 2
    # Three terraces: each ramp joins neighbours, none spans the whole climb.
    n = 480
    x = np.arange(n, dtype=np.float32)
    steps = 50.0 + 100.0 * np.clip((x - 150) / 30.0, 0, 1) + 100.0 * np.clip((x - 320) / 30.0, 0, 1)
    m = t.TerrainMap(np.tile(steps, (n, 1)), spots=[(300.0, 1900.0, 2.0)])
    assert len(m.levels) == 3 and len(m.ramps) == 2
    assert all(r.rise == pytest.approx(100.0, abs=10) for r in m.ramps)


def test_one_hillside_with_small_breaks_is_one_ramp():
    """Rocks on one long slope leave a way around them: still one ramp."""
    h = two_halves_cliff()
    for x in range(N // 2 - 40, N // 2 + 40):
        h[100:230, x] = 50.0 + 200.0 * (x - (N // 2 - 40)) / 80.0
    h[130:150, N // 2 - 8:N // 2 + 8] += 80.0
    h[180:200, N // 2 - 8:N // 2 + 8] += 80.0
    m = t.TerrainMap(h, spots=SPOTS)
    assert len(m.ramps) == 1
    assert m.ramps[0].width >= 900


def test_both_ends_of_a_long_ridge_are_separate_ramps():
    """Two ways up on one connected slope, far apart, stay two ramps."""
    h = two_halves_cliff()
    for z0, z1 in ((40, 80), (240, 280)):
        for x in range(N // 2 - 40, N // 2 + 40):
            h[z0:z1, x] = 50.0 + 200.0 * (x - (N // 2 - 40)) / 80.0
    # A low bench joins the feet of both, so they share one slope area.
    h[40:280, N // 2 - 40:N // 2 - 36] = 58.0
    m = t.TerrainMap(h, spots=SPOTS)
    assert len(m.ramps) == 2


def test_large_level_without_metal_gets_a_region():
    m = t.TerrainMap(two_halves_cliff((N // 2 - 16, N // 2 + 16)), spots=[(*WEST, 2.0)])
    assert m.region_at(*EAST).rid != m.region_at(*WEST).rid
    assert m.region_at(*EAST).level != m.region_at(*WEST).level


def test_clusters_do_not_span_a_ramp():
    h = two_halves_cliff((N // 2 - 16, N // 2 + 16))
    near = [(W / 2 - 500, W / 2, 2.0), (W / 2 + 500, W / 2, 2.0)]
    m = t.TerrainMap(h, spots=near, cluster_radius=2000.0)
    assert m.region_at(*near[0][:2]).rid != m.region_at(*near[1][:2]).rid


def test_narrow_pass_keeps_full_resolution_reachability():
    """A gap under two work cells wide still connects on the full grid."""
    h = flat(50.0)
    h[:, N // 2 - 2:N // 2 + 2] = 500.0   # a wall
    h[N // 2 - 3:N // 2 + 3, :] = 50.0    # a 48-elmo gap
    m = t.TerrainMap(h, spots=SPOTS)
    assert m.can_reach("kbot", WEST, EAST)
    _cost, gates = m.route(WEST, EAST, "kbot")
    assert any("choke" in g.tags for g in gates)


def _compound(gate=20, n=480, ring=(120, 360), wall=8):
    """A walled square in open ground, with a gate in the middle of each side."""
    h = np.full((n, n), 50.0, np.float32)
    a, b = ring
    mid = n // 2
    for k in range(wall):
        h[a + k, a:b] = h[b - k, a:b] = h[a:b, a + k] = h[a:b, b - k] = 400.0
    half = gate // 2
    h[a:a + wall, mid - half:mid + half] = 50.0
    h[b - wall + 1:b + 1, mid - half:mid + half] = 50.0
    h[mid - half:mid + half, a:a + wall] = 50.0
    h[mid - half:mid + half, b - wall + 1:b + 1] = 50.0
    return h


def test_walled_compound_has_a_choke_at_each_of_its_four_gates():
    n = 480
    size = n * t.SQUARE
    c = size / 2
    inner = (c, c, 2.0)
    outer = [(300.0, 300.0, 2.0), (size - 300.0, 300.0, 2.0), (300.0, size - 300.0, 2.0), (size - 300.0, size - 300.0, 2.0)]
    m = t.TerrainMap(_compound(), spots=[inner, *outer])
    a, b = 120 * t.SQUARE, 360 * t.SQUARE
    gates = [(c, a), (c, b), (a, c), (b, c)]
    for gx, gz in gates:
        assert any(math.hypot(k.x - gx, k.z - gz) < 100 for k in m.chokes), (gx, gz, m.chokes)
    assert len(m.chokes) == 4
    assert all(100 <= k.width <= 240 for k in m.chokes)
    # The inside is its own region, and every way in goes through a choke gate.
    inside = m.region_at(c, c)
    assert m.region_at(50.0, 50.0).rid != inside.rid
    for o in inside.neighbors:
        e = m.edges[(min(o, inside.rid), max(o, inside.rid))]
        assert all("choke" in g.tags for g in e.gates.get("kbot", ()))


def test_no_gate_through_a_sealed_wall_with_walkable_flanks():
    """A wall with a flat top and low walkable banks, one real gap at its far end.
    The banks must count as the side they stand on, or a false gate crosses it."""
    n = 480
    h = np.full((n, n), 50.0, np.float32)
    for k in range(4):
        h[:, 230 + k] = 50.0 + 3.0 * k
        h[:, 249 - k] = 50.0 + 3.0 * k
    h[:, 234:246] = 300.0
    h[100:120, 230:250] = 50.0
    m = t.TerrainMap(h, spots=[(1000.0, 3000.0, 2.0), (2800.0, 3000.0, 2.0)])
    left, right = m.region_at(1000.0, 3000.0).rid, m.region_at(2800.0, 3000.0).rid
    assert left != right
    assert (min(left, right), max(left, right)) not in m.edges or all(
        abs(g.z - 880.0) < 200.0 for g in m.edges[(min(left, right), max(left, right))].gates["kbot"])
    routed = m.route((1000.0, 3000.0), (2800.0, 3000.0))
    assert routed is not None
    assert any(abs(g.z - 880.0) < 200.0 for g in routed[1])
    assert routed[0] > 3000.0


def test_route_starts_from_the_side_of_the_wall_it_is_on():
    """A point on the far flank belongs to the far side, whichever region is nearest."""
    n = 480
    h = np.full((n, n), 50.0, np.float32)
    for k in range(4):
        h[:, 230 + k] = 50.0 + 3.0 * k
        h[:, 249 - k] = 50.0 + 3.0 * k
    h[:, 234:246] = 300.0
    h[100:120, 230:250] = 50.0
    m = t.TerrainMap(h, spots=[(1000.0, 3000.0, 2.0), (2800.0, 3000.0, 2.0)])
    assert m.region_for("kbot", 248 * 8.0, 3000.0) == m.region_at(256 * 8.0, 3000.0).rid
    assert m.region_for("kbot", 231 * 8.0, 3000.0) == m.region_at(224 * 8.0, 3000.0).rid
    routed = m.route((248 * 8.0, 3000.0), (1000.0, 3000.0))
    assert routed is not None and any(abs(g.z - 880.0) < 200.0 for g in routed[1])


def test_gap_beside_a_small_rock_is_no_choke():
    n = 480
    h = np.full((n, n), 50.0, np.float32)
    h[200:230, 200:210] = 400.0   # two rocks in the open, 80 elmos apart
    h[240:270, 200:210] = 400.0
    assert not t.TerrainMap(h, spots=[(300.0, 300.0, 2.0)]).chokes


def test_wide_opening_is_no_choke():
    assert not t.TerrainMap(_compound(gate=60), spots=[(1920.0, 1920.0, 2.0)]).chokes


def test_strait_at_sea_is_a_ship_choke():
    n = 480
    h = np.full((n, n), -60.0, np.float32)
    h[:, 230:250] = 60.0          # a land wall across the sea ...
    h[225:245, 230:250] = -60.0   # ... with a strait through it
    m = t.TerrainMap(h, spots=[(500.0, 1900.0, 2.0)])
    assert [k.cls for k in m.chokes] == ["ship"]
    k = m.chokes[0]
    assert abs(k.x - 240 * 8) < 100 and abs(k.z - 235 * 8) < 100


def test_noisy_plateau_is_still_a_level():
    """Small bumps above the tight flatness limit must not dissolve a level."""
    rng = np.random.default_rng(1)
    h = two_halves_cliff((N // 2 - 16, N // 2 + 16))
    h = h + rng.normal(0.0, 0.35, h.shape).astype(np.float32)
    m = t.TerrainMap(h, spots=SPOTS)
    assert len(m.levels) == 2 and len(m.ramps) == 1


def test_gentle_climb_between_flats_is_still_a_ramp():
    """Flat by the normal limit, but it climbs a level's worth: gradient 0.15."""
    n = 480
    x = np.arange(n, dtype=np.float32)
    row = 50.0 + 100.0 * np.clip((x - 180) / 83.0, 0, 1)
    m = t.TerrainMap(np.tile(row, (n, 1)), spots=[(300.0, 1900.0, 2.0)])
    assert len(m.levels) == 2 and len(m.ramps) == 1


def test_regions_meeting_at_a_corner_are_no_choke():
    q = W / 4
    spots = [(q, q, 2.0), (3 * q, q, 2.0), (q, 3 * q, 2.0), (3 * q, 3 * q, 2.0)]
    m = t.TerrainMap(flat(), spots=spots, cluster_radius=1000.0)
    assert len(m.regions) == 4
    assert not any("choke" in g.tags for e in m.edges.values() for g in e.gates["kbot"])


def test_small_metalless_terrace_joins_a_neighbour():
    h = flat(50.0)
    h[100:130, 100:130] = 200.0     # 240 elmos across: a level would need 320
    m = t.TerrainMap(h, spots=SPOTS, cluster_radius=1500.0)
    assert len(m.regions) == 2


def test_label_runs_matches_a_plain_flood_fill():
    rng = np.random.default_rng(3)
    for share in (0.3, 0.6, 0.8):
        mask = rng.random((60, 83)) < share
        comp, sizes = t.label_runs(mask)
        assert ((comp >= 0) == mask).all() and sizes.sum() == mask.sum()
        # Neighbours agree, and each label is one connected piece.
        right = mask[:, :-1] & mask[:, 1:]
        down = mask[:-1, :] & mask[1:, :]
        assert (comp[:, :-1][right] == comp[:, 1:][right]).all()
        assert (comp[:-1, :][down] == comp[1:, :][down]).all()
        seen = np.zeros(mask.shape, bool)
        pieces = 0
        for start in zip(*np.nonzero(mask)):
            if seen[start]:
                continue
            pieces += 1
            stack = [start]
            seen[start] = True
            while stack:
                i, j = stack.pop()
                for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if 0 <= a < 60 and 0 <= b < 83 and mask[a, b] and not seen[a, b]:
                        seen[a, b] = True
                        stack.append((a, b))
        assert pieces == len(sizes)


def test_steep_ramp_blocks_tanks_only():
    # 200 up over 640 elmos is a gradient of 0.31: under every limit.
    gentle = t.TerrainMap(two_halves_cliff((N // 2 - 16, N // 2 + 16)), spots=SPOTS)
    assert gentle.can_reach("tank", WEST, EAST)
    assert gentle.ramps[0].tanks
    # 200 up over 256 elmos is a gradient of 0.78, an engine slope of about
    # 0.21: above the tank limit (0.109), below the bot limit (0.412).
    h = flat(50.0)
    h[:, N // 2:] = 250.0
    span = 16
    for x in range(N // 2 - span, N // 2 + span):
        h[:, x] = 50.0 + 200.0 * (x - (N // 2 - span)) / (2 * span)
    steep = t.TerrainMap(h, spots=SPOTS)
    assert steep.can_reach("kbot", WEST, EAST)
    assert not steep.can_reach("tank", WEST, EAST)
    assert len(steep.ramps) == 1 and not steep.ramps[0].tanks


def _islands():
    """Deep sea with two islands, plus one metal spot on the seabed."""
    h = flat(-100.0)
    zz, xx = np.ogrid[:N, :N]
    for cx, cz in ((N // 4, N // 2), (3 * N // 4, N // 2)):
        r = np.sqrt((xx - cx) ** 2 + (zz - cz) ** 2)
        # A flat top and a beach a hover can climb, falling away to the seabed.
        h = np.maximum(h, np.clip(30.0 - (r - 30.0) * 2.0, -100.0, 30.0))
    return h.astype(np.float32)


IS_A, IS_B = (W * 0.25, W * 0.5), (W * 0.75, W * 0.5)
OFFSHORE = ((W * 0.5, W * 0.1), (W * 0.5, W * 0.9))
ISLAND_SPOTS = [(*IS_A, 2.0), (*IS_B, 2.0), (W * 0.5, W * 0.5, 2.0)]


def test_water_map_islands_and_sea():
    m = t.TerrainMap(_islands(), spots=ISLAND_SPOTS)
    a, b, s = m.region_at(*IS_A), m.region_at(*IS_B), m.region_at(W * 0.5, W * 0.5)
    assert len({a.rid, b.rid, s.rid}) == 3
    assert a.domain == t.LAND and s.domain == t.SEA
    assert "sea" in s.tags and s.spots == [2]
    assert "coast" in a.tags
    assert not m.can_reach("kbot", IS_A, IS_B)
    assert not m.can_reach("tank", IS_A, IS_B)
    assert m.can_reach("hover", IS_A, IS_B)
    assert m.can_reach("amph", IS_A, IS_B)
    assert not m.can_reach("ship", IS_A, IS_B)   # ships stay off the land
    assert m.route(IS_A, IS_B, "kbot") is None
    cost, gates = m.route(IS_A, IS_B, "hover")
    assert any("shore" in g.tags for g in gates)


def test_water_without_metal_still_gets_regions():
    m = t.TerrainMap(flat(-80.0), spots=[])
    assert m.regions and all(r.domain == t.SEA for r in m.regions)
    assert m.region_at(10.0, 10.0) is not None
    assert m.can_reach("ship", (100.0, 100.0), (W - 100.0, W - 100.0))
    assert not m.can_reach("kbot", (100.0, 100.0), (W - 100.0, W - 100.0))


def test_sea_gets_regions_even_with_many_metal_clusters():
    """Lots of clusters must not use up the regions that open water needs."""
    n = 1024
    h = np.full((n, n), 40.0, np.float32)
    x = np.arange(n, dtype=np.float32)
    h[:, :] = np.clip(40.0 - (x - 512.0) * 2.0, -80.0, 40.0)   # west land, east deep sea
    spots = [(100.0 + 200.0 * i, 100.0 + 200.0 * j, 2.0) for i in range(18) for j in range(40)]
    spots = [sp for sp in spots if sp[0] < 3600.0]
    m = t.TerrainMap(h, spots=spots, cluster_radius=100.0)
    assert sum(1 for r in m.regions if r.domain == t.LAND) > 64
    east = m.region_at(7500.0, 4000.0)
    assert east.domain == t.SEA and "sea" in east.tags


def test_shallow_water_is_walkable_land_domain():
    m = t.TerrainMap(channel(-10.0), spots=SPOTS)   # a ford, under the 22-elmo bot limit
    assert m.can_reach("kbot", WEST, EAST)
    assert not m.can_reach("ship", WEST, EAST)
    assert m.speed["kbot"][m.h // 2, m.w // 2] < m.speed["kbot"][m.h // 2, 2]   # wading is slow
    assert m.domain[m.h // 2, m.w // 2] == t.LAND


def test_deep_channel_cuts_bots_but_not_amphibious():
    m = t.TerrainMap(channel(-60.0), spots=SPOTS)
    assert not m.can_reach("kbot", WEST, EAST)
    assert m.can_reach("amph", WEST, EAST)
    assert m.can_reach("hover", WEST, EAST)


@pytest.mark.parametrize("shift", [37.5, -120.0])
def test_water_level_is_only_a_shift(shift):
    """The same terrain with the water plane moved is the same map."""
    base = t.TerrainMap(_islands(), spots=ISLAND_SPOTS)
    moved = t.TerrainMap(_islands() + shift, spots=ISLAND_SPOTS, water_level=shift)
    assert np.array_equal(base.label, moved.label)
    assert np.array_equal(base.domain, moved.domain)
    for name in t.MOVE_CLASSES:
        assert np.array_equal(base.comp[name] >= 0, moved.comp[name] >= 0)


def test_raising_water_floods_low_ground():
    """A lowland below the raised plane stops being walkable."""
    h = channel(-5.0, ground=10.0)
    assert t.TerrainMap(h, spots=SPOTS).can_reach("kbot", WEST, EAST)
    assert not t.TerrainMap(h, spots=SPOTS, water_level=30.0).can_reach("kbot", WEST, EAST)


def test_lava_stops_every_class_and_makes_no_sea():
    h = _islands()
    m = t.TerrainMap(h, spots=ISLAND_SPOTS, water_damage=50.0)
    assert m.liquid == "lava"
    for cls in ("hover", "amph", "kbot", "spider"):
        assert not m.can_reach(cls, IS_A, IS_B)
    assert not m.can_reach("ship", *OFFSHORE)
    assert not any(r.domain == t.SEA for r in m.regions)
    assert m.route(IS_A, IS_B, "hover") is None
    # A shallow lava ford, which harmless water would let bots wade.
    ford = channel(-10.0)
    assert t.TerrainMap(ford, spots=SPOTS).can_reach("kbot", WEST, EAST)
    assert not t.TerrainMap(ford, spots=SPOTS, water_damage=50.0).can_reach("kbot", WEST, EAST)


def test_hover_cross_lava_only_where_the_game_spares_them():
    h = _islands()
    m = t.TerrainMap(h, spots=ISLAND_SPOTS, water_damage=50.0, hover_safe=True)
    assert m.can_reach("hover", IS_A, IS_B)
    assert m.route(IS_A, IS_B, "hover") is not None
    assert any(r.domain == t.SEA for r in m.regions)
    assert not m.can_reach("amph", IS_A, IS_B) and not m.can_reach("ship", *OFFSHORE)


def test_gadget_lava_with_no_engine_damage_is_still_lava():
    m = t.TerrainMap(_islands(), spots=ISLAND_SPOTS, hover_safe=True)
    assert m.liquid == "lava"
    assert not m.can_reach("amph", IS_A, IS_B)
    assert m.can_reach("hover", IS_A, IS_B)


def test_void_stops_every_class_even_hover_on_a_lava_map():
    h = _islands()
    assert t.TerrainMap(h, spots=ISLAND_SPOTS, water_damage=999.0).liquid == "lava"
    m = t.TerrainMap(h, spots=ISLAND_SPOTS, water_damage=5e4, hover_safe=True)
    assert m.liquid == "void"
    assert not m.can_reach("hover", IS_A, IS_B)
    assert not any(r.domain == t.SEA for r in m.regions)


def test_ground_just_above_a_deadly_plane_is_off_limits():
    """The Hole destroys anything below y = 20, on dry ground too."""
    h = channel(10.0, ground=60.0)
    assert t.TerrainMap(h, spots=SPOTS).can_reach("kbot", WEST, EAST)
    m = t.TerrainMap(h, spots=SPOTS, water_damage=5e4)
    assert not m.can_reach("kbot", WEST, EAST)
    assert m.hazard_at(N * t.SQUARE / 2, 500.0) and not m.hazard_at(*WEST)


def test_low_shelf_above_lava_stays_walkable():
    """Only the plane itself burns: Carabanossi has real ground at 10-20 elmos."""
    m = t.TerrainMap(channel(10.0, ground=60.0), spots=SPOTS, water_damage=50.0)
    assert m.can_reach("kbot", WEST, EAST)
    assert not m.hazard_at(N * t.SQUARE / 2, 500.0)


def test_land_beside_a_void_is_not_shallow():
    """Regions own the void cells nearest them, but those say nothing about the land."""
    m = t.TerrainMap(_islands(), spots=ISLAND_SPOTS, water_damage=5e4)
    assert all("shallow" not in r.tags and r.water_share < 0.3 for r in m.regions)
    assert m.region_at(N * t.SQUARE / 2, N * t.SQUARE / 2) is not None


def test_water_is_not_a_hazard():
    m = t.TerrainMap(_islands(), spots=ISLAND_SPOTS)
    assert m.liquid == "water"
    assert m.can_reach("hover", IS_A, IS_B) and m.can_reach("ship", *OFFSHORE)
    assert not m.hazard_at(N * t.SQUARE / 2, N * t.SQUARE / 2)


def test_bridge_over_lava_is_a_choke():
    n = 480
    h = np.full((n, n), 60.0, np.float32)
    h[:, 225:255] = -60.0
    h[225:245, 225:255] = 60.0
    m = t.TerrainMap(h, spots=[(800.0, 1900.0, 2.0), (3000.0, 1900.0, 2.0)], water_damage=175.0)
    assert m.liquid == "lava"
    assert len(m.chokes) == 1 and m.chokes[0].cls == "kbot"
    assert abs(m.chokes[0].x - 240 * 8) < 100 and abs(m.chokes[0].z - 235 * 8) < 100
    routed = m.route((800.0, 1900.0), (3000.0, 1900.0))
    assert routed is not None and any("choke" in g.tags for g in routed[1])


def test_every_cell_has_a_region():
    m = t.TerrainMap(two_halves_cliff(), spots=SPOTS)
    assert (m.label >= 0).all()
    assert m.region_at(-5.0, 1e9) is not None


def test_spot_on_a_cliff_still_seeds_a_region():
    h = two_halves_cliff()
    on_cliff = (N // 2 * t.SQUARE, W * 0.5, 2.0)
    m = t.TerrainMap(h, spots=[on_cliff])
    assert any(r.spots == [0] for r in m.regions)


def test_large_open_map_gets_open_regions_without_metal():
    m = t.TerrainMap(flat(n=1024), spots=[(100.0, 100.0, 2.0)])
    assert len(m.regions) > 1
    # Nothing is left far from a region centre.
    assert np.nanmax(np.where(np.isfinite(m.centre_dist), m.centre_dist, np.nan)) < m.cluster_radius * 4


def test_nearby_spots_over_a_cliff_do_not_merge():
    h = two_halves_cliff()
    near = [(W / 2 - 100, W / 2, 2.0), (W / 2 + 100, W / 2, 2.0)]
    assert len(t.TerrainMap(h, spots=near).regions) >= 2
    joined = t.TerrainMap(flat(), spots=near)
    assert sorted(r.spots for r in joined.regions if r.spots) == [[0, 1]]
