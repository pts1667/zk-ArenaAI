"""A run's dashboard: a page on localhost with the pipeline's log as it comes, each stage's results, and replays of test games.

    ~/zkml/bin/python tools/policy_dashboard.py --root /mnt/d/zk-learned/runs/full-1      # any run, during it or after
    (tools/policy_pipeline.py serves it itself while it runs: its --port)

From Windows, http://localhost:PORT reaches it in WSL, which forwards localhost.  The page (tools/dashboard/) is HTMX: the log
comes as server-sent events, and the tables are fragments rendered here (tools/dashboard/render.py).  Charts and the replay
viewer read JSON:

  GET /api/run                      the stages: status, times, exit codes, their results' files; the pipeline's arguments
  GET /api/metrics/STAGE?keys=&max= a training's log (jsonl rows), thinned to at most max rows (the last one always kept), only
                                    the keys asked for if any; with every numeric key there is
  GET /api/eval/STAGE               a test's result (policy_clone_eval.py's JSON); for ppo, its tests (tests.jsonl)
  GET /api/replays                  every replay's index line (policy_replay.py), with its url, oldest first
  GET /replay/DIR/FILE              a replay: gzipped JSON, policy_replay.py describes it
  GET /api/units                    each unit def by name: human name, description, icon type and size, air, mobile
  GET /api/map/SLUG                 a map's name, kind ("real" or "made"), size in elmos, liquid, and its picture's url
  GET /map/SLUG.png                 the picture: the minimap in the map's archive (.sd7, .sdz), or for a made map its heights shaded
  GET /icon/TYPE.png                a unit icon type's picture (Zero-K's icons/, the .dds ones turned the right way up)

Events (/events): `log` (a line, rendered; its id is the line's number, so that a reconnect carries on where it was), `stages`
(pipeline.json changed), `results` (a stage's results changed; data: the stage), `replays` (a replay was written).

Pictures are made once and kept in --cache.  Only numpy and the standard library: a training's Python, without torch.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import queue
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import zipfile
import zlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB = ROOT / "tools" / "dashboard"
sys.path.insert(0, str(ROOT / "tools"))
from dashboard import render  # noqa: E402

DEFAULT_MAPS = "/mnt/d/zk-learned/maps"
DEFAULT_ARCHIVES = "/mnt/c/Users/Thoma/Downloads/zero-k-portable/maps"
DEFAULT_CACHE = "/mnt/d/zk-learned/dashboard-cache"
DEFAULT_ZERO_K = str(ROOT.parent / "Zero-K")
BACKLOG = 5000          # log lines a page gets when it opens
PRELOAD = 400           # of each log already there when the server starts
LINE_CHARS = 4000
POLL = 1.0              # seconds between looks at the run's files
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml", ".ico": "image/x-icon", ".woff2": "font/woff2"}


# Pictures ------------------------------------------------------------------------------------------------------------------------------------------

def png(pixels: np.ndarray) -> bytes:
    """An RGB or RGBA uint8 picture as a PNG, every row filtered by the one above."""
    h, w, c = pixels.shape
    raw = np.ascontiguousarray(pixels, np.uint8)
    up = raw.copy()
    up[1:] -= raw[:-1]                  # wraps, as PNG's Up filter does
    rows = np.concatenate([np.full((h, 1), 2, np.uint8), up.reshape(h, w * c)], axis=1)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    head = struct.pack(">IIBBBBB", w, h, 8, {3: 2, 4: 6}[c], 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)) + chunk(b"IEND", b"")


def _rgb565(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.int32)
    return np.stack([((c >> 11) & 31) * 255 // 31, ((c >> 5) & 63) * 255 // 63, (c & 31) * 255 // 31], -1)


def _dxt_colors(blocks: np.ndarray, dxt1: bool) -> tuple[np.ndarray, np.ndarray]:
    """S3TC colour blocks (n x 8 bytes): each block's 16 pixels' RGB, and an alpha that's only DXT1's punch-through."""
    c0 = blocks[:, 0].astype(np.uint16) | (blocks[:, 1].astype(np.uint16) << 8)
    c1 = blocks[:, 2].astype(np.uint16) | (blocks[:, 3].astype(np.uint16) << 8)
    p0, p1 = _rgb565(c0), _rgb565(c1)
    four = (c0 > c1) | (not dxt1)
    p2 = np.where(four[:, None], (2 * p0 + p1) // 3, (p0 + p1) // 2)
    p3 = np.where(four[:, None], (p0 + 2 * p1) // 3, 0)
    palette = np.stack([p0, p1, p2, p3], 1)
    alphas = np.full((len(blocks), 4), 255, np.int32)
    alphas[:, 3] = np.where(four, 255, 0)
    bits = blocks[:, 4:8].copy().view("<u4")[:, 0]
    index = ((bits[:, None] >> (2 * np.arange(16, dtype=np.uint32))) & 3).astype(np.int64)
    rgb = np.take_along_axis(palette, index[..., None], 1)
    return rgb, np.take_along_axis(alphas, index, 1)


def _dxt5_alpha(blocks: np.ndarray) -> np.ndarray:
    a0, a1 = blocks[:, 0].astype(np.int32), blocks[:, 1].astype(np.int32)
    eight = a0 > a1
    k = np.arange(1, 7)
    table = np.zeros((len(blocks), 8), np.int32)
    table[:, 0], table[:, 1] = a0, a1
    table[:, 2:8] = np.where(eight[:, None], ((7 - k) * a0[:, None] + k * a1[:, None]) // 7, 0)
    six = ((5 - k[:4]) * a0[:, None] + k[:4] * a1[:, None]) // 5
    table[~eight, 2:6] = six[~eight]
    table[~eight, 6], table[~eight, 7] = 0, 255
    bits = np.zeros(len(blocks), np.uint64)
    for i in range(6):
        bits |= blocks[:, 2 + i].astype(np.uint64) << np.uint64(8 * i)
    index = ((bits[:, None] >> (3 * np.arange(16, dtype=np.uint64))) & np.uint64(7)).astype(np.int64)
    return np.take_along_axis(table, index, 1)


def decode_dxt(data: bytes, width: int, height: int, kind: str) -> np.ndarray:
    """A DXT1, DXT3 or DXT5 picture's top mip level as RGBA, rows in the order they're stored."""
    bw, bh = max(1, (width + 3) // 4), max(1, (height + 3) // 4)
    size = 8 if kind == "DXT1" else 16
    blocks = np.frombuffer(data, np.uint8, bw * bh * size).reshape(bw * bh, size)
    if kind == "DXT1":
        rgb, alpha = _dxt_colors(blocks, True)
    else:
        rgb, _ = _dxt_colors(blocks[:, 8:], False)
        if kind == "DXT3":
            bits = blocks[:, :8].copy().view("<u8")[:, 0]
            alpha = (((bits[:, None] >> (4 * np.arange(16, dtype=np.uint64))) & np.uint64(15)) * 17).astype(np.int32)
        else:
            alpha = _dxt5_alpha(blocks[:, :8])
    pixels = np.concatenate([rgb, alpha[..., None]], -1).reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 4)
    return pixels[:height, :width].astype(np.uint8)


def decode_dds(data: bytes) -> np.ndarray:
    """A .dds file's top level as RGBA, rows top first as stored (Zero-K's icons are stored bottom row first)."""
    if data[:4] != b"DDS ":
        raise ValueError("not a DDS file")
    height, width = struct.unpack_from("<II", data, 12)
    flags, four, bits = struct.unpack_from("<I4sI", data, 80)
    body = data[128:]
    if four in (b"DXT1", b"DXT3", b"DXT5"):
        return decode_dxt(body, width, height, four.decode())
    if flags & 0x40 and bits in (24, 32):        # uncompressed RGB(A), by its masks
        masks = struct.unpack_from("<4I", data, 92)
        step = bits // 8
        words = np.frombuffer(body, np.uint8, width * height * step).reshape(-1, step)
        value = np.zeros(len(words), np.uint32)
        for i in range(step):
            value |= words[:, i].astype(np.uint32) << np.uint32(8 * i)
        out = np.empty((len(words), 4), np.uint8)
        for c, mask in enumerate(masks):
            if mask == 0:
                out[:, c] = 255
                continue
            shift = (mask & -mask).bit_length() - 1
            top = mask >> shift
            out[:, c] = ((value & np.uint32(mask)) >> np.uint32(shift)).astype(np.uint32) * 255 // top
        if not (flags & 0x1):
            out[:, 3] = 255
        return out.reshape(height, width, 4)
    raise ValueError(f"DDS format {four!r} with {bits} bits isn't read here")


def shade(height: np.ndarray, cell: float, liquid: str = "water") -> np.ndarray:
    """A map's heights as a picture: ground coloured by height and lit from the north-west, the sea (below 0) by its depth."""
    h = height.astype(np.float32)
    dz, dx = np.gradient(h, cell)
    light = np.array([-1.0, -1.0, 1.4])
    light /= np.linalg.norm(light)
    lit = np.clip((-dx * light[0] - dz * light[1] + light[2]) / np.sqrt(dx * dx + dz * dz + 1.0), 0.0, 1.0)
    land = h[h >= 0]
    top = float(np.percentile(land, 99)) if land.size else 1.0
    t = np.clip(h / max(top, 1.0), 0.0, 1.0)[..., None]
    low, high = np.array([92, 104, 70], np.float32), np.array([176, 160, 132], np.float32)
    ground = (low * (1 - t) + high * t) * (0.45 + 0.75 * lit[..., None])
    sea_shallow, sea_deep = (np.array([214, 96, 40]), np.array([120, 30, 10])) if liquid == "lava" else (np.array([66, 112, 150]), np.array([24, 44, 78]))
    d = np.clip(-h / 200.0, 0.0, 1.0)[..., None]
    sea = sea_shallow * (1 - d) + sea_deep * d
    rgb = np.where((h < 0)[..., None], sea * (0.85 + 0.15 * lit[..., None]), ground)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _normal(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class Assets:
    """Map pictures, unit icons and the units' names, made when first asked for and kept in `cache`."""

    def __init__(self, maps: pathlib.Path, archives: pathlib.Path, zero_k: pathlib.Path, cache: pathlib.Path):
        self.maps, self.archives, self.zero_k, self.cache = maps, archives, zero_k, cache
        self.lock = threading.Lock()
        self._units: dict | None = None
        self._icons: dict | None = None

    def facts(self, slug: str) -> dict | None:
        path = self.maps / f"{slug}.facts.json"
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", slug) or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf8"))

    def map_info(self, slug: str) -> dict | None:
        facts = self.facts(slug)
        if facts is None:
            return None
        with np.load(self.maps / f"{slug}.npz") as d:
            rows, cols = d["height"].shape
            cell = float(d["cell"])
        return {"slug": slug, "name": facts.get("name", slug), "kind": facts.get("kind", "real"), "width": cols * cell, "height": rows * cell,
                "liquid": facts.get("liquid", "water"), "water": facts.get("water", 0.0), "image": f"/map/{slug}.png",
                "archive": self._archive(facts.get("name", "")) is not None if facts.get("kind") == "real" else False}

    def _archive(self, name: str) -> pathlib.Path | None:
        want = _normal(name)
        if not want or not self.archives.is_dir():
            return None
        for p in self.archives.iterdir():
            if p.suffix.lower() in (".sd7", ".sdz") and _normal(p.stem) == want:
                return p
        return None

    def map_png(self, slug: str) -> bytes | None:
        out = self.cache / "maps" / f"{slug}.png"
        if out.exists():
            return out.read_bytes()
        facts = self.facts(slug)
        if facts is None:
            return None
        pixels = None
        archive = self._archive(facts.get("name", "")) if facts.get("kind") == "real" else None
        if archive is not None:
            try:
                pixels = minimap(archive)
            except (OSError, ValueError, subprocess.SubprocessError) as e:
                print(f"dashboard: no minimap from {archive}: {e}", file=sys.stderr, flush=True)
        if pixels is None:
            with np.load(self.maps / f"{slug}.npz") as d:
                pixels = shade(d["height"], float(d["cell"]), facts.get("liquid", "water"))
        data = png(pixels)
        _save(out, data)
        return data

    def icons(self) -> dict[str, dict]:
        """Zero-K's icon types: bitmap and size."""
        if self._icons is None:
            text = _read(self.zero_k / "LuaUI" / "Configs" / "icontypes.lua")
            out = {}
            for name, body in re.findall(r"(\w+)\s*=\s*\{([^{}]*)\}", text):
                bitmap = re.search(r"""^\s*bitmap\s*=\s*['"]([^'"]+)['"]""", body, re.M)       # the line not commented out
                size = re.search(r"\bsize\s*=\s*([\d.]+)", body)
                if bitmap:
                    out[name] = {"bitmap": bitmap.group(1), "size": float(size.group(1)) if size else 1.0}
            self._icons = out
        return self._icons

    def units(self) -> dict[str, dict]:
        """Every unit def of the sim's catalog by name, with what Zero-K's unit files say of it."""
        if self._units is None:
            lua = {}
            for p in sorted((self.zero_k / "units").glob("*.lua")):
                text = _read(p)
                m = re.search(r"return\s*\{\s*([A-Za-z0-9_]+)\s*=\s*\{", text)
                if not m:
                    continue
                field = lambda key: (lambda f: f.group(1) if f else None)(re.search(rf"\b{key}\s*=\s*\[\[(.*?)\]\]", text, re.S))
                speed = re.search(r"\bspeed\s*=\s*([\d.]+)", text)
                lua[m.group(1)] = {"human": field("name"), "description": field("description"), "icon": field("iconType"),
                                   "air": bool(re.search(r"\bcanFly\s*=\s*true", text)), "mobile": bool(speed and float(speed.group(1)) > 0)}
            icons = self.icons()
            catalog = json.loads((ROOT / "sim" / "data" / "catalog.json").read_text("utf8"))["units"]
            out = {}
            for u in catalog:
                name = u["name"]
                info = lua.get(name)
                if info is None:
                    commander = "com" in name or name.startswith("dyn")
                    info = {"human": "Commander" if commander else name, "description": name, "icon": "commander1" if commander else None,
                            "air": False, "mobile": commander}
                icon = info["icon"] if info["icon"] in icons else "default"
                out[name] = {**info, "icon": icon, "icon_size": icons.get(icon, {}).get("size", 1.3), "cost": u.get("cost")}
            self._units = out
        return self._units

    def icon_png(self, kind: str) -> bytes | None:
        if not re.fullmatch(r"\w+", kind):
            return None
        out = self.cache / "icons" / f"{kind}.png"
        if out.exists():
            return out.read_bytes()
        entry = self.icons().get(kind) or self.icons().get("default")
        path = self.zero_k / entry["bitmap"] if entry else None
        if path is None or not path.exists():
            path = self.zero_k / "icons" / f"{kind}.dds"
            if not path.exists():
                return None
        data = path.read_bytes()
        if path.suffix.lower() == ".dds":
            data = png(decode_dds(data)[::-1])       # stored bottom row first
        _save(out, data)
        return data


def minimap(archive: pathlib.Path) -> np.ndarray:
    """The 1024 x 1024 minimap a Spring map file (.smf) holds, from the archive it's in: DXT1, the map's north at the top."""
    if archive.suffix.lower() == ".sdz":
        with zipfile.ZipFile(archive) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".smf"))
            with z.open(name) as f:
                return _smf_minimap(f.read)
    listing = subprocess.run(["7z", "l", "-slt", "-ba", str(archive)], capture_output=True, text=True, timeout=120, check=True).stdout
    names = [line[7:] for line in listing.splitlines() if line.startswith("Path = ") and line.lower().endswith(".smf")]
    if not names:
        raise ValueError("no .smf in it")
    with subprocess.Popen(["7z", "e", "-so", str(archive), names[0]], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        try:
            return _smf_minimap(p.stdout.read)
        finally:
            p.kill()


def _smf_minimap(read) -> np.ndarray:
    def take(n: int) -> bytes:
        parts, got = [], 0
        while got < n:
            part = read(n - got)
            if not part:
                raise ValueError("the map file ends early")
            parts.append(part)
            got += len(part)
        return b"".join(parts)
    head = take(80)
    if not head.startswith(b"spring map file"):
        raise ValueError("not a Spring map file")
    at = struct.unpack_from("<i", head, 64)[0]
    take(at - 80)
    return decode_dxt(take(1024 * 1024 // 2), 1024, 1024, "DXT1")[..., :3]


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf8", errors="replace") if path.exists() else ""


def _save(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


# The run ---------------------------------------------------------------------------------------------------------------------------------------------

def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def read_jsonl(path: pathlib.Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass        # a line being written
    return rows


class Run:
    """What a pipeline's directory says: its stages (pipeline.json), their logs (logs/), results and replays."""

    def __init__(self, root: pathlib.Path):
        self.root = root

    def pipeline(self) -> dict:
        path = self.root / "pipeline.json"
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {"done": [], "runs": [], "plan": []}

    def files(self, stage: str, data: dict | None = None) -> dict[str, pathlib.Path]:
        """A stage's results, by kind: `metrics` (jsonl, a row a report), `eval` (one JSON), `tests` (jsonl), `manifest` (JSON)."""
        root = self.root
        if stage == "frames":
            frames = (data or self.pipeline()).get("args", {}).get("frames")
            return {"manifest": (pathlib.Path(frames) if frames and frames != "None" else root / "frames") / "manifest.json"}
        if stage == "encoder":
            return {"metrics": root / "encoder" / "metrics.jsonl"}
        if stage == "clone" or stage.startswith("dagger-"):
            return {"metrics": root / stage / "log.jsonl"}
        if stage.startswith("record-"):
            return {"eval": root / "evals" / f"{stage}.json"}
        if stage == "eval":
            return {"eval": root / "evals" / "final.json"}
        if stage == "ppo":
            return {"metrics": root / "ppo" / "metrics.jsonl", "tests": root / "ppo" / "tests.jsonl"}
        return {}

    def stages(self) -> dict:
        data = self.pipeline()
        done, runs, current = set(data.get("done", [])), data.get("runs", []), data.get("current")
        alive = bool(current and _alive(current.get("pid")))
        running = current.get("stage") if alive else None
        killed = current.get("stage") if current and not alive else None     # the pipeline ended in it, without noting how
        out = []
        for name in data.get("plan", []):
            mine = [r for r in runs if r.get("stage") == name]
            last = mine[-1] if mine else None
            if name == running:
                status = "running"
            elif name == killed:
                status = "interrupted"
            elif name in done:
                status = "done"
            elif last is not None:
                status = "failed" if last.get("exit") else "stopped"
            else:
                status = "pending"
            row = {"name": name, "status": status, "attempts": len(mine), "began": last.get("began") if last else None,
                   "seconds": last.get("seconds") if last else None, "exit": last.get("exit") if last else None,
                   "results": {k: str(p) for k, p in self.files(name, data).items() if p.exists()}}
            if status == "running":
                row["began"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current["began"]))
                row["seconds"] = round(time.time() - current["began"])
            out.append(row)
        return {"root": str(self.root), "stages": out, "running": running, "pipeline_alive": alive,
                "args": data.get("args", {}), "now": time.strftime("%Y-%m-%d %H:%M:%S")}

    def metrics(self, stage: str, keys: list[str] | None, most: int) -> dict:
        files = self.files(stage)
        rows = read_jsonl(files["metrics"]) if "metrics" in files else []
        numeric = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        if most > 0 and len(rows) > most:
            stride = math.ceil(len(rows) / most)
            rows = rows[::stride] + ([rows[-1]] if (len(rows) - 1) % stride else [])
        if keys:
            rows = [{k: r[k] for k in keys if k in r} for r in rows]
        return {"stage": stage, "keys": numeric, "rows": rows}

    def evaluation(self, stage: str) -> dict | list | None:
        files = self.files(stage)
        if "eval" in files and files["eval"].exists():
            try:
                return json.loads(files["eval"].read_text())
            except json.JSONDecodeError:
                return None
        if "tests" in files:
            return read_jsonl(files["tests"])
        return None

    def summary(self, stage: str) -> dict:
        """What a stage's results come to, for its card: the last row of its metrics, its eval without the games, the manifest."""
        files = self.files(stage)
        out: dict = {"stage": stage}
        if "metrics" in files and files["metrics"].exists():
            rows = read_jsonl(files["metrics"])
            out["rows"] = len(rows)
            out["last"] = rows[-1] if rows else None
            out["first"] = rows[0] if rows else None
        if "eval" in files and files["eval"].exists():
            e = self.evaluation(stage)
            out["eval"] = {k: v for k, v in e.items() if k != "records"} if isinstance(e, dict) else None
        if "tests" in files and files["tests"].exists():
            tests = read_jsonl(files["tests"])
            out["tests"] = [{"iteration": t.get("iteration"), "decisions": t.get("decisions"), "opponent": t.get("opponent"),
                             "games": len(t.get("games", [])),
                             "won": sum(g.get("result") == "won" for g in t.get("games", [])),
                             "drew": sum(g.get("result") == "drew" for g in t.get("games", []))} for t in tests]
        if "manifest" in files and files["manifest"].exists():
            try:
                m = json.loads(files["manifest"].read_text())
                out["manifest"] = {k: v for k, v in m.items() if not isinstance(v, (dict, list))}
            except json.JSONDecodeError:
                pass
        return out

    def replays(self) -> list[dict]:
        out = []
        for index in sorted((self.root / "replays").glob("*/index.jsonl")):
            for line in read_jsonl(index):
                if (index.parent / line.get("file", "")).exists():
                    out.append({**line, "dir": index.parent.name, "url": f"/replay/{index.parent.name}/{line['file']}"})
        out.sort(key=lambda r: r.get("written", ""))
        return out


# Log lines and events -----------------------------------------------------------------------------------------------------------------------------

class Hub:
    """The log's lines, numbered, the last BACKLOG of them kept, and every page's queue of events."""

    def __init__(self):
        self.lock = threading.Lock()
        self.lines: collections.deque = collections.deque(maxlen=BACKLOG)
        self.count = 0
        self.clients: list[queue.Queue] = []

    def line(self, stage: str, text: str, when: float | None = None, live: bool = True) -> None:
        with self.lock:
            self.count += 1
            entry = {"id": self.count, "stage": stage, "text": text[:LINE_CHARS], "time": when or time.time(), "live": live}
            self.lines.append(entry)
            for q in self.clients:
                q.put(("log", entry))

    def event(self, kind: str, data: str = "") -> None:
        with self.lock:
            for q in self.clients:
                q.put((kind, data))

    def join(self, after: int) -> tuple[queue.Queue, list[dict]]:
        q: queue.Queue = queue.Queue()
        with self.lock:
            self.clients.append(q)
            return q, [e for e in self.lines if e["id"] > after]

    def leave(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)


class Watcher:
    """Looks at the run's files every POLL seconds: new log lines become lines of the hub, changed results and replays events."""

    def __init__(self, run: Run, hub: Hub):
        self.run, self.hub = run, hub
        self.offsets: dict[pathlib.Path, int] = {}
        self.partial: dict[pathlib.Path, str] = {}
        self.seen: dict[str, tuple] = {}

    def preload(self) -> None:
        """The end of each log already there, the pipeline's first and then the stages' in their order."""
        logs = self.run.root / "logs"
        plan = self.run.pipeline().get("plan", [])
        order = ["pipeline"] + plan
        paths = sorted(logs.glob("*.out"), key=lambda p: (order.index(p.stem) if p.stem in order else len(order), p.name)) if logs.is_dir() else []
        for p in paths:
            size = p.stat().st_size
            with p.open("rb") as f:
                f.seek(max(0, size - 400 * PRELOAD))
                text = f.read().decode("utf8", errors="replace")
            for line in _split(text)[-PRELOAD:]:
                self.hub.line(p.stem, line, p.stat().st_mtime, live=False)
            self.offsets[p] = size
        self._changed()

    def loop(self) -> None:
        while True:
            try:
                self._logs()
                self._changed()
            except Exception as e:       # a file half written, or gone: look again next time
                print(f"dashboard: {e!r}", file=sys.stderr, flush=True)
            time.sleep(POLL)

    def _logs(self) -> None:
        logs = self.run.root / "logs"
        if not logs.is_dir():
            return
        for p in sorted(logs.glob("*.out")):
            size = p.stat().st_size
            at = self.offsets.get(p, 0)
            if size < at:
                at = 0
            if size == at:
                continue
            with p.open("rb") as f:
                f.seek(at)
                data = f.read(size - at)
            self.offsets[p] = size
            text = self.partial.pop(p, "") + data.decode("utf8", errors="replace")
            *whole, rest = re.split(r"\r?\n", text)
            if rest:
                self.partial[p] = rest
            for line in whole:
                for part in line.split("\r"):
                    if part.strip():
                        self.hub.line(p.stem, part)

    def _changed(self) -> None:
        def stamp(path: pathlib.Path):
            try:
                s = path.stat()
                return (s.st_mtime_ns, s.st_size)
            except OSError:
                return None
        data = self.run.pipeline()
        checks = {"stages": (stamp(self.run.root / "pipeline.json"),)}
        for stage in data.get("plan", []):
            checks[f"results:{stage}"] = tuple(stamp(p) for p in self.run.files(stage, data).values())
        checks["replays"] = tuple(stamp(p) for p in sorted((self.run.root / "replays").glob("*/index.jsonl")))
        for key, value in checks.items():
            if self.seen.get(key) != value:
                first = key not in self.seen
                self.seen[key] = value
                if not first:
                    kind, _, stage = key.partition(":")
                    self.hub.event(kind, stage)


def _split(text: str) -> list[str]:
    return [part for line in re.split(r"\r?\n", text) for part in line.split("\r") if part.strip()]


# The server -------------------------------------------------------------------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "zk-dashboard"
    protocol_version = "HTTP/1.1"
    app: "Dashboard"

    def log_message(self, fmt, *args) -> None:      # quiet: the run's log is what matters
        pass

    def _send(self, body: bytes, kind: str, status: int = 200, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, value) -> None:
        self._send(json.dumps(value, separators=(",", ":")).encode(), "application/json")

    def _html(self, text: str) -> None:
        self._send(text.encode(), "text/html; charset=utf-8")

    def _missing(self, what: str = "not found") -> None:
        self._send(what.encode(), "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        path, query = urllib.parse.unquote(url.path), urllib.parse.parse_qs(url.query)
        app = self.app
        try:
            if path == "/events":
                return self._events()
            if path in ("/", "/index.html"):
                return self._file(WEB / "index.html")
            if path.startswith("/static/"):
                return self._file(WEB / "static" / path[len("/static/"):])
            if path == "/api/run":
                return self._json(app.run.stages())
            if path.startswith("/api/metrics/"):
                keys = [k for k in query.get("keys", [""])[0].split(",") if k]
                return self._json(app.run.metrics(path.rsplit("/", 1)[1], keys or None, int(query.get("max", ["0"])[0] or 0)))
            if path.startswith("/api/eval/"):
                value = app.run.evaluation(path.rsplit("/", 1)[1])
                return self._json(value) if value is not None else self._missing()
            if path == "/api/replays":
                return self._json(app.run.replays())
            if path == "/api/units":
                return self._json(app.assets.units())
            if path.startswith("/api/map/"):
                info = app.assets.map_info(path.rsplit("/", 1)[1])
                return self._json(info) if info else self._missing()
            if path.startswith("/replay/"):
                return self._replay(path[len("/replay/"):])
            if path.startswith("/map/") and path.endswith(".png"):
                with app.assets.lock:
                    data = app.assets.map_png(path[len("/map/"):-len(".png")])
                return self._send(data, "image/png", extra={"Cache-Control": "max-age=86400"}) if data else self._missing()
            if path.startswith("/icon/") and path.endswith(".png"):
                data = app.assets.icon_png(path[len("/icon/"):-len(".png")])
                return self._send(data, "image/png", extra={"Cache-Control": "max-age=86400"}) if data else self._missing()
            if path.startswith("/fragment/"):
                return self._fragment(path[len("/fragment/"):], query)
            return self._missing()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"dashboard: {self.path}: {e!r}", file=sys.stderr, flush=True)
            try:
                self._send(f"error: {e!r}".encode(), "text/plain; charset=utf-8", HTTPStatus.INTERNAL_SERVER_ERROR)
            except OSError:
                pass

    def _fragment(self, name: str, query: dict) -> None:
        run = self.app.run
        if name == "stages":
            return self._html(render.stages(run.stages()))
        if name == "results":
            state = run.stages()
            return self._html(render.results(state, {s["name"]: run.summary(s["name"]) for s in state["stages"]}))
        if name.startswith("results/"):
            stage = name.split("/", 1)[1]
            return self._html(render.stage_result(stage, run.summary(stage)))
        if name == "replays":
            return self._html(render.replays(run.replays()))
        return self._missing()

    def _file(self, path: pathlib.Path) -> None:
        path = path.resolve()
        if WEB.resolve() not in path.parents or not path.is_file():
            return self._missing()
        self._send(path.read_bytes(), TYPES.get(path.suffix.lower(), "application/octet-stream"))

    def _replay(self, rest: str) -> None:
        parts = rest.split("/")
        if len(parts) != 2 or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", p) for p in parts) or not parts[1].endswith(".json.gz"):
            return self._missing()
        path = self.app.run.root / "replays" / parts[0] / parts[1]
        if not path.is_file():
            return self._missing()
        self._send(path.read_bytes(), "application/json", extra={"Content-Encoding": "gzip", "Cache-Control": "max-age=86400"})

    def _events(self) -> None:
        after = self.headers.get("Last-Event-ID")
        q, backlog = self.app.hub.join(int(after) if after and after.isdigit() else 0)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(b"retry: 2000\n\n")
            for entry in backlog:
                self._event("log", render.log_line(entry), entry["id"])
            self.wfile.flush()
            while True:
                try:
                    kind, data = q.get(timeout=15.0)
                except queue.Empty:
                    self.wfile.write(b": still here\n\n")
                    self.wfile.flush()
                    continue
                if kind == "log":
                    self._event("log", render.log_line(data), data["id"])
                else:
                    self._event(kind, data)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.app.hub.leave(q)

    def _event(self, kind: str, data: str, ident: int | None = None) -> None:
        lines = [f"event: {kind}"] + ([f"id: {ident}"] if ident is not None else []) + [f"data: {d}" for d in (data.split("\n") or [""])]
        self.wfile.write(("\n".join(lines) + "\n\n").encode())


class Dashboard:
    def __init__(self, root: pathlib.Path, host: str = "127.0.0.1", port: int = 8765, maps: str = DEFAULT_MAPS,
                 archives: str = DEFAULT_ARCHIVES, zero_k: str = DEFAULT_ZERO_K, cache: str = DEFAULT_CACHE):
        self.run = Run(pathlib.Path(root))
        self.hub = Hub()
        self.assets = Assets(pathlib.Path(maps), pathlib.Path(archives), pathlib.Path(zero_k), pathlib.Path(cache))
        handler = type("BoundHandler", (Handler,), {"app": self})
        self.server = ThreadingHTTPServer((host, port), handler)
        self.server.daemon_threads = True
        self.watcher = Watcher(self.run, self.hub)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"

    def start(self) -> "Dashboard":
        """Serves on threads of its own, which end with the process."""
        self.watcher.preload()
        threading.Thread(target=self.watcher.loop, daemon=True, name="dashboard-watch").start()
        threading.Thread(target=self.server.serve_forever, daemon=True, name="dashboard-http").start()
        return self


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, required=True, help="the run's directory (policy_pipeline.py's --root)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--maps", default=DEFAULT_MAPS, help="the map pool, for the maps' names, sizes and heights")
    ap.add_argument("--map-archives", default=DEFAULT_ARCHIVES, help="the maps' archives (.sd7, .sdz), for their minimaps")
    ap.add_argument("--zero-k", default=DEFAULT_ZERO_K, help="Zero-K's repository, for unit icons and names")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="where pictures are kept once made")
    args = ap.parse_args()
    board = Dashboard(args.root.resolve(), args.host, args.port, args.maps, args.map_archives, args.zero_k, args.cache).start()
    print(f"dashboard for {args.root} at {board.url}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
