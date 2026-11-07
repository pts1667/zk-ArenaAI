"""Build terrain regions offline from map archives, and draw or check them.

  python tools/terrain_map.py "TitanDuel 2.2" "Crescent Bay" [--out results/terrain]
  python tools/terrain_map.py --all [--no-png] [--check-graph]

Reads the .smf inside each map's .sd7/.sdz (via 7-Zip), honouring the height
overrides in mapinfo.lua and .smd, so no game is needed.  Metal spots come from
what a game on the map reported (sim/data/map_facts.json, by tools/sim_engine.py facts), or else
the game's list, the map's layout file or the metal map; these only approximate
what LuaRules/Gadgets/mex_spot_finder.lua does in a game.  Maps that generate
their terrain in-game hold flat placeholder heights: they are flagged
`generated` and get no picture.

--check-graph compares routes between metal spots on the region graph with
reachability on the full 16-elmo grid, which is the ground truth.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bots"))
from zkbot import terrain as tr  # noqa: E402

GAME_DIR = pathlib.Path(os.environ.get("ZK_GAME_DIR", ROOT.parent / "Zero-K"))
MAP_FACTS = ROOT / "sim" / "data" / "map_facts.json"
MAPS_DIR = pathlib.Path(os.environ.get("ZK_MAPS_DIR", pathlib.Path.home() / "Downloads" / "zero-k-portable" / "maps"))
SMF_HEADER = "<16s2i5i2f7i"     # magic, version, id, mapx, mapy, square, texel, tile, min, max, 6 ptrs, extra count


@dataclass
class MapFile:
    name: str
    corners: np.ndarray
    metal: np.ndarray
    water_damage: float = 0.0     # per second, as mapinfo says it
    hover_safe: bool = False      # Zero-K's map_lava gadget runs here
    spots: list[tuple[float, float, float]] = field(default_factory=list)
    spot_source: str = "metalmap"
    blocked_types: int = 0
    generated: bool = False    # placeholder heights: the map builds its terrain in-game

    @property
    def center(self) -> np.ndarray:
        c = self.corners
        return ((c[:-1, :-1] + c[:-1, 1:] + c[1:, :-1] + c[1:, 1:]) * 0.25).astype(np.float32)


def seven_zip() -> str:
    found = shutil.which("7z") or shutil.which("7za")
    if found:
        return found
    default = pathlib.Path("C:/Program Files/7-Zip/7z.exe")
    if default.exists():
        return str(default)
    raise SystemExit("7-Zip not found")


def strip_lua_comments(text: str) -> str:
    return re.sub(r"--[^\n]*", "", text)


def lua_block(text: str, key: str) -> str:
    """The balanced {...} after `key =`, or ''."""
    m = re.search(rf"\b{key}\s*=\s*\{{", text, re.I)
    if not m:
        return ""
    depth, i = 1, m.end()
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end():i - 1]


def arithmetic(text: str) -> float | None:
    """A plain number expression such as 360*2 or 1000-22; anything else is None."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            x, y = ev(n.left), ev(n.right)
            return {ast.Add: x + y, ast.Sub: x - y, ast.Mult: x * y}.get(type(n.op), x / y if y else 0.0)
        raise ValueError
    try:
        return ev(ast.parse(text.strip(), mode="eval"))
    except (ValueError, SyntaxError):
        return None


def lua_number(block: str, key: str) -> float | None:
    m = re.search(rf"\b{key}\s*=\s*([-+*/().\d\s eE]+?)\s*[,;}}\n]", block + "\n", re.I)
    return arithmetic(m.group(1)) if m else None


def parse_layout(text: str) -> list[tuple[float, float, float]]:
    """Spot tables of map_metal_layout.lua: {x = .., z = .., metal = ..}."""
    out = []
    for m in re.finditer(r"\{([^{}]*)\}", strip_lua_comments(text)):
        x, z = lua_number(m.group(1), "x"), lua_number(m.group(1), "z")
        if x is not None and z is not None:
            out.append((x, z, lua_number(m.group(1), "metal") or 2.0))
    return out


def spots_from_metalmap(metal: np.ndarray) -> list[tuple[float, float, float]]:
    """Each connected blob of the metal map is one spot."""
    h, w = metal.shape
    seen = np.zeros(metal.shape, bool)
    out = []
    for i, j in zip(*np.nonzero(metal > 0)):
        if seen[i, j]:
            continue
        stack = [(int(i), int(j))]
        seen[i, j] = True
        cells = []
        while stack:
            a, b = stack.pop()
            cells.append((a, b))
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    c, d = a + da, b + db
                    if 0 <= c < h and 0 <= d < w and not seen[c, d] and metal[c, d] > 0:
                        seen[c, d] = True
                        stack.append((c, d))
        if len(cells) >= 4:
            zs = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            out.append(((sum(xs) / len(xs) + 0.5) * 16.0, (sum(zs) / len(zs) + 0.5) * 16.0, 2.0))
    return out


# mapinfo gives damage per second; the engine holds it per slow update (MapInfo.cpp).
DAMAGE_PER_SLOW_UPDATE = 15 / 30


def _reported_spots(name: str) -> list[tuple[float, float, float]]:
    if not MAP_FACTS.exists():
        return []
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return [(x, z, metal) for x, z, metal in json.loads(MAP_FACTS.read_text("utf8")).get(slug, {}).get("spots", [])]


def load_map(path: pathlib.Path, name: str = "") -> MapFile:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="zkmap", dir=os.environ.get("CLAUDE_JOB_DIR") and
                                        pathlib.Path(os.environ["CLAUDE_JOB_DIR"]) / "tmp"))
    try:
        subprocess.run([seven_zip(), "x", "-y", f"-o{tmp}", str(path), "*.smf", "*.smd", "*.lua", "-r"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        smfs = list(tmp.rglob("*.smf"))
        if not smfs:
            raise ValueError("no .smf in archive")
        raw = smfs[0].read_bytes()
        magic, _v, _id, mapx, mapy, _sq, _tx, _tl, lo, hi, hptr, typ, _t, _m, mptr, _f, _e = struct.unpack_from(SMF_HEADER, raw)
        if not magic.startswith(b"spring map"):
            raise ValueError("not an SMF file")
        info = next(iter(tmp.rglob("mapinfo.lua")), None)
        text = strip_lua_comments(info.read_text("utf8", "replace")) if info else ""
        smf = lua_block(text, "smf")
        # Precedence as the engine's map helper: mapinfo.lua, then .smd, then the file.
        smd = next(iter(tmp.rglob("*.smd")), None)
        smd_text = smd.read_text("utf8", "replace") if smd else ""
        for key, cur in (("minheight", "lo"), ("maxheight", "hi")):
            value = lua_number(smf, key)
            if value is None:
                value = lua_number(smd_text, key)
            if value is not None:
                lo, hi = (value, hi) if cur == "lo" else (lo, value)
        heights = np.frombuffer(raw, "<u2", (mapy + 1) * (mapx + 1), hptr).reshape(mapy + 1, mapx + 1)
        corners = lo + heights.astype(np.float32) * ((hi - lo) / 65536.0)
        metal = np.frombuffer(raw, "u1", (mapy // 2) * (mapx // 2), mptr).reshape(mapy // 2, mapx // 2)
        water = lua_block(text, "water")
        damage = lua_number(water, "damage")
        if damage is None:
            damage = lua_number(smd_text, "waterdamage") or 0.0
        out = MapFile(path.stem, corners, metal, water_damage=damage,
                      generated=int(heights.max()) == int(heights.min()))
        terrain_types = lua_block(text, "terrainTypes")
        out.blocked_types = len(re.findall(r"(?:tank|kbot|hover|ship)\s*=\s*0(?:\.0*)?\s*[,}\n]", terrain_types, re.I))
        # Precedence as mex_spot_finder.lua: the game's list, then the map's.
        # map_lava.lua removes itself unless one of these exists.
        out.hover_safe = any(p.name == "lava_config.lua" for p in tmp.rglob("*.lua")) or any(
            (GAME_DIR / "LuaRules" / "Configs" / "LavaConf" / f"{n}.lua").exists() for n in (name, path.stem) if n)
        game_side = [GAME_DIR / "LuaRules" / "Configs" / "MetalSpots" / f"{n}.lua" for n in (name, path.stem) if n]
        listed = next((f for f in game_side if f.exists()), None)
        if listed:   # may only override metal values, and list no spots
            out.spots = parse_layout(listed.read_text("utf8", "replace"))
            out.spot_source = "game"
        if not out.spots:
            layouts = [p for p in tmp.rglob("*.lua") if p.name in ("map_metal_layout.lua", "map_resource_spot_layout.lua")]
            if layouts:
                out.spots = parse_layout(layouts[0].read_text("utf8", "replace"))
                out.spot_source = "layout"
            # Some layouts build their list in code, which a text scan half reads.
            blobs = spots_from_metalmap(metal)
            if len(blobs) > len(out.spots):
                out.spots = blobs
                out.spot_source = "metalmap"
        # The game treats a metal map with so few spots as metal everywhere.
        if out.spot_source == "metalmap" and len(out.spots) < 6:
            out.spots = []
        # Better than all of those, for a map that has played.  Faultline's layout builds half its spots in code, and maps' metal values vary.
        reported = _reported_spots(name)
        if reported:
            out.spots = reported
            out.spot_source = "engine"
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def find_maps(names: list[str]) -> list[tuple[str, pathlib.Path]]:
    catalog = json.loads((ROOT / "maps.json").read_text("utf8"))
    files = {p.name.lower(): p for p in MAPS_DIR.glob("*.sd[7z]")}
    known = {m["File"].lower(): m["Name"] for m in catalog}
    if not names:
        return [(known.get(k, p.stem), p) for k, p in sorted(files.items())]
    out = []
    squeeze = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    for n in names:
        hits = [(known.get(k, p.stem), p) for k, p in files.items()
                if squeeze(n) in squeeze(known.get(k, p.stem)) or squeeze(n) in squeeze(k)]
        if not hits:
            raise SystemExit(f"no map matches {n!r} in {MAPS_DIR}")
        out.append(min(hits, key=lambda h: len(h[0])))
    return out


def build(mf: MapFile, water_level: float = 0.0) -> tr.TerrainMap:
    # The slope uses the real corner heights, as the engine does.
    return tr.TerrainMap(mf.center, tr.slope_from_corners(mf.corners), spots=mf.spots,
                         water_level=water_level, water_damage=mf.water_damage * DAMAGE_PER_SLOW_UPDATE,
                         hover_safe=mf.hover_safe)


def graph_errors(mf: MapFile, m: tr.TerrainMap) -> dict[str, tuple[int, int]]:
    """Spot pairs per class that the 16-elmo grid connects but the region graph
    does not.  Routes stay inside one passable area by construction, so the
    graph can't invent a way between areas; the first count is always 0."""
    out = {}
    for name in tr.MOVE_CLASSES:
        where = []
        for sp in mf.spots:
            found = m.area_of(name, sp[0], sp[1])
            where.append(None if found is None else (found[0], m.region_for(name, found[1], found[2])))
        linked: dict[tuple[int, int, int], bool] = {}
        phantom = missing = 0
        for i in range(len(where)):
            for j in range(i + 1, len(where)):
                a, b = where[i], where[j]
                if a is None or b is None:
                    continue
                if a[0] != b[0]:
                    continue      # route() refuses these by construction
                if a[1] == b[1]:
                    continue
                key = (min(a[1], b[1]), max(a[1], b[1]), a[0])
                if key not in linked:
                    linked[key] = m._graph_cost(key[0], key[1], name, key[2]) is not None
                missing += not linked[key]
        out[name] = (phantom, missing)
    return out


# Each gate kind has its own colour and marker shape, so it reads without colour too.
SURFACE = "#fcfcfb"
LIQUID_COLOUR = {"water": (0x10, 0x42, 0x81), "lava": (0xb5, 0x3a, 0x0e), "void": (0x24, 0x10, 0x3e)}
LIQUID_WORD = {"water": "under water", "lava": "lava", "void": "void"}
INK = "#0b0b0b"
GATE_STYLE = {
    "open": ("#2a78d6", "o", "Gate"),
    "choke": ("#eb6834", "D", "Choke (narrow, wider both sides)"),
    "ramp": ("#1baf7a", "^", "Ramp between levels"),
    "shore": ("#6250d6", "s", "Hover or sea link"),
    "cliff": ("#eda100", "x", "Spiders only"),
}


def hillshade(height: np.ndarray, cell: float) -> np.ndarray:
    dz, dx = np.gradient(height.astype(np.float64), cell)
    norm = np.sqrt(dx * dx + dz * dz + 1.0)
    light = (-dx - dz + 1.0) / (norm * np.sqrt(3.0))
    return np.clip(0.55 + 0.75 * light, 0.0, 1.0)


def render(name: str, mf: MapFile, m: tr.TerrainMap, out: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = (0, m.w * m.cell, m.h * m.cell, 0)
    fig, ax = plt.subplots(figsize=(11, 11 * m.h / m.w + 0.8), facecolor=SURFACE)
    ax.imshow(hillshade(m.height, m.cell), cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    # One-hue depth ramp; ground below the plane, darker with depth.
    depth = np.clip(-m.height / max(40.0, float(-m.height.min())), 0.0, 1.0)
    rgba = np.zeros((m.h, m.w, 4))
    rgba[..., :3] = np.array(LIQUID_COLOUR[m.liquid]) / 255.0
    rgba[..., 3] = np.where(m.height < 0.0, 0.35 + 0.5 * depth, 0.0)
    ax.imshow(rgba, extent=ext, interpolation="nearest")
    # Ground bots can't walk.
    rgba = np.zeros((m.h, m.w, 4))
    rgba[..., :3] = 0.0
    rgba[..., 3] = np.where(m.domain == tr.WALL, 0.45, 0.0)
    ax.imshow(rgba, extent=ext, interpolation="nearest")
    # Region borders.
    xs = (np.arange(m.w + 1)) * m.cell
    zs = (np.arange(m.h + 1)) * m.cell
    segs = []
    # No borders inside lava or void: nothing stands there to be in a region.
    for i, j in zip(*np.nonzero((m.label[:, :-1] != m.label[:, 1:]) & ~(m.burning[:, :-1] & m.burning[:, 1:]))):
        segs.append(((xs[j + 1], zs[i]), (xs[j + 1], zs[i + 1])))
    for i, j in zip(*np.nonzero((m.label[:-1, :] != m.label[1:, :]) & ~(m.burning[:-1, :] & m.burning[1:, :]))):
        segs.append(((xs[j], zs[i + 1]), (xs[j + 1], zs[i + 1])))
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors="#fcfcfb", linewidths=2.2, zorder=3))
    ax.add_collection(LineCollection(segs, colors="#0b0b0b", linewidths=0.9, zorder=3))
    if mf.spots:
        ax.scatter([s[0] for s in mf.spots], [s[1] for s in mf.spots], s=22, c="#fcfcfb", edgecolors=INK,
                   linewidths=0.8, zorder=4, label="Metal spot")
    # Gates: bots' gates, then what only other classes can cross.
    drawn = {k: [] for k in GATE_STYLE}
    drawn["ramp"] = list(m.ramps)
    drawn["choke"] = list(m.chokes)
    for e in m.edges.values():
        # Bots paddling along a beach is not a crossing worth a mark.
        kbot = [g for g in e.gates.get("kbot", ()) if "shore" not in g.tags]
        for g in kbot:
            if "ramp" not in g.tags and g.choke is None:   # those are drawn themselves
                drawn["choke" if "choke" in g.tags else "open"].append(g)
        if not kbot:
            shore = next((e.best_gate(c) for c in ("hover", "amph", "ship") if e.gates.get(c)), None)
            if shore is not None:
                drawn["shore"].append(shore)
            elif "spider" in e.gates:
                drawn["cliff"].append(e.best_gate("spider"))
    for kind, gs in drawn.items():
        if not gs:
            continue
        colour, marker, label = GATE_STYLE[kind]
        ax.scatter([g.x for g in gs], [g.z for g in gs], s=62, marker=marker, c=colour,
                   edgecolors=None if marker == "x" else SURFACE, linewidths=2.0 if marker == "x" else 1.2,
                   zorder=5, label=f"{label} ({len(gs)})")
    for rp in m.ramps:
        ax.annotate(f"+{rp.rise:.0f}", (rp.x, rp.z), xytext=(7, 5), textcoords="offset points", fontsize=7,
                    color=INK, zorder=6)
    for r in m.regions:
        short = {"plateau": "high", "terrace": "mid", "basin": "low", "rough": "rough", "island": "island", "sea": "sea",
                 "shallow": "shallow"}
        tag = ",".join(short[t] for t in sorted(r.tags) if t in short)
        ax.annotate(f"{r.rid}\n{tag}" if tag else str(r.rid), (r.x, r.z), ha="center", va="center", fontsize=8,
                    color=INK, zorder=6, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
    s = m.summary()
    ax.set_xlim(0, m.w * m.cell)
    ax.set_ylim(m.h * m.cell, 0)
    ax.set_title(f"{name}: {s['land']} land and {s['sea']} sea regions, {s['edges']} links; "
                 f"{s['levels']} levels, {s['ramps']} ramps; {s['walls']:.0%} too steep for bots, {s['water']:.0%} {LIQUID_WORD[m.liquid]}",
                 color=INK, fontsize=11, loc="left")
    ax.tick_params(colors="#52514e", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#bdbcb6")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.035), ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "terrain"))
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--check-graph", action="store_true")
    ap.add_argument("--water-level", type=float, default=0.0)
    ap.add_argument("--table", default="", help="write one row per map to this CSV")
    a = ap.parse_args(argv)
    if not a.all and not a.maps:
        ap.error("name some maps or pass --all")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    catalog = {m["File"].lower(): m for m in json.loads((ROOT / "maps.json").read_text("utf8"))}
    rows = []
    failed = 0
    for name, path in find_maps([] if a.all else a.maps):
        t0 = time.time()
        try:
            mf = load_map(path, name)
            t1 = time.time()
            m = build(mf, a.water_level)
            t2 = time.time()
            s = m.summary()
            ref = catalog.get(path.name.lower(), {})
            row = {"map": name, "elmos": f"{int(m.width)}x{int(m.depth)}", "cell": int(m.cell), "secs": round(t2 - t1, 2),
                   "spots": len(mf.spots), "ref_spots": ref.get("MetalSpots", ""), "src": mf.spot_source,
                   "water": round(s["water"], 3), "ref_water": ref.get("WaterShare", ""), "liquid": s["liquid"], "hover_safe": mf.hover_safe, "dmg": mf.water_damage,
                   "walls": round(s["walls"], 3), "land": s["land"], "sea": s["sea"], "edges": s["edges"],
                   "chokes": s["chokes"], "levels": s["levels"], "ramps": s["ramps"], "blocked_types": mf.blocked_types,
                   "generated": int(mf.generated)}
            if a.check_graph:
                errs = graph_errors(mf, m)
                row["phantom"] = sum(v[0] for v in errs.values())
                row["missing"] = sum(v[1] for v in errs.values())
                row["bad"] = ",".join(f"{k}:{v[0]}/{v[1]}" for k, v in errs.items() if v[0] or v[1])
            rows.append(row)
            print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
            if not a.no_png and not mf.generated:
                render(name, mf, m, out / (re.sub(r"[^A-Za-z0-9._-]+", "_", name) + ".png"))
        except Exception as e:  # keep scanning; a bad map must not stop the survey
            failed += 1
            print(f"map={name} ERROR {type(e).__name__}: {e}", flush=True)
    if a.table and rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.table, "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} maps ok, {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
