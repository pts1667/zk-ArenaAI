"""The dashboard's HTML fragments (tools/policy_dashboard.py serves them): each takes plain data and gives HTML.

The page (index.html, static/) swaps these in with HTMX.  Charts and the replay viewer are drawn by static/*.js from the JSON
endpoints; a card leaves a `hx-preserve` box for them, so that refreshing a card's numbers keeps its charts.
"""
from __future__ import annotations

import html
import math
import re
import time

esc = html.escape

KINDS = {"frames": "Recorded frames", "encoder": "Frame encoder", "zindex": "z index", "clone": "Behaviour cloning",
         "test": "Closed-loop test", "ppo": "PPO", "other": ""}
STATUS = {"done": ("✓", "done"), "running": ("●", "running"), "failed": ("✕", "failed"), "stopped": ("■", "stopped"), "interrupted": ("■", "interrupted"),
          "pending": ("○", "pending")}
PROBLEM = re.compile(r"traceback|error|exception|failed|killed|out of memory|nan\b", re.I)
WARNING = re.compile(r"warn", re.I)


def kind(stage: str) -> str:
    if stage == "frames":
        return "frames"
    if stage == "encoder":
        return "encoder"
    if stage.startswith("zindex"):
        return "zindex"
    if stage == "clone" or stage.startswith("dagger-"):
        return "clone"
    if stage.startswith("record-") or stage == "eval":
        return "test"
    if stage == "ppo":
        return "ppo"
    return "other"


def describe(stage: str) -> str:
    n = stage.rsplit("-", 1)[-1]
    if stage == "frames":
        return "frames recorded from script games"
    if stage == "encoder":
        return "frame autoencoder"
    if stage == "zindex":
        return "z index of the recorded games"
    if stage.startswith("zindex-"):
        return f"z index of round {n}'s games"
    if stage == "clone":
        return "behaviour cloning of the scripts"
    if stage.startswith("dagger-"):
        return f"cloning on the actor's own games, round {n}"
    if stage.startswith("record-"):
        return f"64 games vs the army script, recorded for round {n}"
    if stage == "eval":
        return "final test: 64 games vs the army script"
    if stage == "ppo":
        return "reinforcement learning"
    return ""


# Numbers ---------------------------------------------------------------------------------------------------------------------------

def num(v, places: int | None = None) -> str:
    """A number for reading: compact past 100K, about three significant figures below."""
    if v is None:
        return "–"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return esc(str(v))
    if math.isnan(v) or math.isinf(v):
        return str(v)
    a = abs(v)
    if places is not None:
        return f"{v:,.{places}f}"
    if a >= 1e9:
        return f"{v / 1e9:.2f}B"
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e5:
        return f"{v / 1e3:.0f}K"
    if isinstance(v, int) or (a >= 100 and float(v).is_integer()):
        return f"{int(v):,}"
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.001 or a == 0:
        return f"{v:.3g}"
    return f"{v:.1e}"


def pct(v) -> str:
    return "–" if not isinstance(v, (int, float)) else f"{v * 100:.1f}%"


def duration(seconds) -> str:
    if seconds is None:
        return ""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def clock(seconds) -> str:
    if not isinstance(seconds, (int, float)):
        return "–"
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _f(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def tile(label: str, value: str, sub: str = "", cls: str = "") -> str:
    return (f'<div class="tile {cls}"><div class="t-label">{esc(label)}</div><div class="t-value">{value}</div>'
            + (f'<div class="t-sub">{sub}</div>' if sub else "") + "</div>")


def wash(v, lo: float = 0.0, hi: float = 1.0) -> str:
    """A cell's background strength for a value: an inline custom property the stylesheet turns into a tint."""
    if not _f(v):
        return ""
    t = max(0.0, min(1.0, (v - lo) / (hi - lo) if hi > lo else 0.0))
    return f' style="--w:{t * 100:.0f}%"'


def log_button(stage: str) -> str:
    return f'<button type="button" class="logf" data-stage="{esc(stage)}" title="Show only this stage\'s log">log</button>'


# The log ------------------------------------------------------------------------------------------------------------------------------

def log_line(entry: dict) -> str:
    """One line of the log, as the `log` event carries it: one line of HTML."""
    when = time.strftime("%H:%M:%S", time.localtime(entry["time"]))
    day = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry["time"]))
    text = entry["text"]
    cls = "line"
    if PROBLEM.search(text):
        cls += " err"
    elif WARNING.search(text):
        cls += " warn"
    if not entry.get("live", True):
        cls += " old"
    stage = esc(entry["stage"])
    return (f'<div class="{cls}" data-stage="{stage}" data-id="{int(entry["id"])}"><span class="when" title="{day}">{when}</span>'
            f'<span class="stage">{stage}</span><span class="text">{esc(text)}</span></div>')


# Stages ------------------------------------------------------------------------------------------------------------------------------

def _status(s: dict) -> str:
    mark, word = STATUS.get(s["status"], ("?", s["status"]))
    if s["status"] == "failed" and s.get("exit") is not None:
        word += f" ({s['exit']})"
    return f'<span class="status st-{esc(s["status"])}"><b>{mark}</b> {esc(word)}</span>'


def _elapsed(seconds) -> str:
    return f'<span class="elapsed" data-seconds="{int(seconds or 0)}">{duration(seconds)}</span>'


def run_status(state: dict) -> str:
    rows = state.get("stages", [])
    done = sum(r["status"] == "done" for r in rows)
    running = next((r for r in rows if r["status"] == "running"), None)
    failed = [r for r in rows if r["status"] == "failed"]
    if running:
        pill = (f'<span class="pill running" data-state="running" data-stage="{esc(running["name"])}"><i class="dot"></i>'
                f'{esc(running["name"])} running · {_elapsed(running["seconds"])}</span>')
    elif rows and done == len(rows):
        pill = '<span class="pill done" data-state="done"><b>✓</b> finished</span>'
    elif failed:
        pill = f'<span class="pill failed" data-state="failed"><b>✕</b> {esc(failed[-1]["name"])} failed</span>'
    elif any(r["status"] == "interrupted" for r in rows):
        cut = next(r for r in rows if r["status"] == "interrupted")
        pill = f'<span class="pill failed" data-state="failed"><b>■</b> {esc(cut["name"])} interrupted</span>'
    elif done:
        pill = '<span class="pill idle" data-state="idle"><b>■</b> not running</span>'
    else:
        pill = '<span class="pill idle" data-state="idle"><b>○</b> waiting</span>'
    began = next((r["began"] for r in rows if r.get("began")), None)
    total = sum(r.get("seconds") or 0 for r in rows)
    meta = f'{done} of {len(rows)} stages done'
    if began:
        meta += f' · began {esc(str(began)[:16])} · {duration(total)} of stages'
    return (f'<div id="run-status" class="run-status" hx-swap-oob="true">{pill}'
            f'<span class="meta">{meta}</span><span class="root mono" title="the run\'s directory">{esc(state.get("root", ""))}</span></div>')


def stages(state: dict) -> str:
    rows = state.get("stages", [])
    longest = max([r.get("seconds") or 0 for r in rows] + [1])
    body = []
    for s in rows:
        name, st = s["name"], s["status"]
        secs = s.get("seconds")
        width = 0 if secs is None else max(1.5, 100 * secs / longest)
        took = _elapsed(secs) if st == "running" else duration(secs)
        tries = f'<span class="tries" title="attempts">×{s["attempts"]}</span>' if s.get("attempts", 0) > 1 else ""
        link = (f'<a href="#result-{esc(name)}" class="stage-link">{esc(name)}</a>' if s.get("results")
                else f'<span class="stage-name">{esc(name)}</span>')
        began = esc(str(s["began"])[5:16]) if s.get("began") else ""
        body.append(f'<tr class="row-{esc(st)}" data-stage="{esc(name)}"><td>{link}{tries}<div class="sub" title="{esc(describe(name))}">{esc(describe(name))}</div></td>'
                    f'<td>{_status(s)}</td><td class="mono dim">{began}</td><td class="num">{took}</td>'
                    f'<td class="barcell"><span class="dbar"><i class="b-{esc(st)}" style="width:{width:.1f}%"></i></span></td>'
                    f'<td>{log_button(name)}</td></tr>')
    args = state.get("args") or {}
    details = ""
    if args:
        items = "".join(f"<dt>{esc(str(k))}</dt><dd class='mono'>{esc(str(v))}</dd>" for k, v in args.items())
        details = f'<details class="args"><summary>pipeline arguments ({len(args)})</summary><dl>{items}</dl></details>'
    table = ('<table class="stages"><thead><tr><th>Stage</th><th>Status</th><th>Began</th><th class="num">Took</th><th></th><th></th>'
             f'</tr></thead><tbody>{"".join(body)}</tbody></table>') if rows else '<p class="empty">No pipeline.json yet.</p>'
    return run_status(state) + table + details + f'<div class="updated dim">checked {esc(state.get("now", ""))}</div>'


# Results -----------------------------------------------------------------------------------------------------------------------------

def _charts(stage: str, x: str) -> str:
    return f'<div class="charts" id="charts-{esc(stage)}" data-stage="{esc(stage)}" data-kind="{kind(stage)}" data-x="{x}" hx-preserve="true"></div>'


def _frames_body(summary: dict) -> str:
    m = summary.get("manifest") or {}
    if not m:
        return '<p class="empty">The manifest has no counts.</p>'
    tiles = [tile(k.replace("_", " "), num(v) if _f(v) else f'<span class="mono small" title="{esc(str(v))}">{esc(str(v))}</span>') for k, v in m.items()]
    return f'<div class="tiles">{"".join(tiles)}</div>'


def _probe_table(last: dict, first: dict) -> str:
    """r2_<thing>, r2_real_<thing>, r2_made_<thing>, r2_wet_<thing>: how well a linear probe reads each thing from z."""
    groups = ("real", "made", "wet")
    things = [k[3:] for k in last if k.startswith("r2_") and not any(k.startswith(f"r2_{g}_") for g in groups)]
    if not things:
        return ""
    cols = [g for g in groups if any(f"r2_{g}_{t}" in last for t in things)]
    head = "<tr><th>probe R²</th><th class='num'>all</th><th class='num'>Δ</th>" + "".join(f"<th class='num'>{g}</th>" for g in cols) + "</tr>"
    rows = []
    for t in things:
        v = last.get(f"r2_{t}")
        d = v - first[f"r2_{t}"] if _f(v) and _f(first.get(f"r2_{t}")) else None
        delta = "" if d is None else f'<span class="{"up" if d >= 0 else "down"}">{"+" if d >= 0 else ""}{d:.2f}</span>'
        cells = "".join(f'<td class="num heat"{wash(last.get(f"r2_{g}_{t}"))}>{_r2(last.get(f"r2_{g}_{t}"))}</td>' for g in cols)
        rows.append(f'<tr><td>{esc(t.replace("_", " "))}</td><td class="num heat"{wash(v)}>{_r2(v)}</td><td class="num small">{delta}</td>{cells}</tr>')
    return f'<div class="tablebox"><table class="grid">{head}{"".join(rows)}</table></div>'


def _r2(v) -> str:
    if not _f(v):
        return "–"
    return f'<span class="neg">{v:.2f}</span>' if v < 0 else f"{v:.2f}"


def _recon_table(last: dict) -> str:
    """recon_r2_<x>, recon_map_r2_<x>, found_r2_<x>, found_map_r2_<x>, where_<x>, how_much_<x>: the decoder's maps, by channel."""
    things = [k[len("recon_r2_"):] for k in last if k.startswith("recon_r2_")]
    if not things:
        return ""
    cols = [("recon_r2_", "recon"), ("recon_map_r2_", "map"), ("found_r2_", "found"), ("where_", "where"), ("how_much_", "how much")]
    cols = [(p, n) for p, n in cols if any(f"{p}{t}" in last for t in things)]
    head = "<tr><th>decoded channel R²</th>" + "".join(f"<th class='num'>{n}</th>" for _, n in cols) + "</tr>"
    rows = "".join(f'<tr><td>{esc(t.replace("_", " "))}</td>'
                   + "".join(f'<td class="num heat"{wash(last.get(p + t))}>{_r2(last.get(p + t))}</td>' for p, _ in cols) + "</tr>"
                   for t in things)
    return f'<div class="tablebox"><table class="grid">{head}{rows}</table></div>'


def _encoder_body(summary: dict) -> str:
    last, first = summary.get("last") or {}, summary.get("first") or {}
    tiles = [tile("steps", num(last.get("step")), f'{summary.get("rows", 0)} reports'),
             tile("loss", num(last.get("loss")), f'from {num(first.get("loss"))}'),
             tile("recon", num(last.get("recon")), f'from {num(first.get("recon"))}')]
    for key, label in (("recon_r2", "recon R²"), ("where_top_0.05", "where top 5%"), ("where_any_0.05", "where any 5%"),
                       ("what_type", "what: type"), ("what_side", "what: side"), ("kl_nats", "KL (nats)"),
                       ("where_nats", "where (nats)"), ("what_nats", "what (nats)"), ("effective_rank", "effective rank")):
        if key in last:
            tiles.append(tile(label, num(last[key])))
    if _f(last.get("seconds")):
        tiles.append(tile("training time", duration(last["seconds"])))
    tables = _probe_table(last, first) + _recon_table(last)
    return f'<div class="tiles">{"".join(tiles)}</div>' + (f'<div class="two">{tables}</div>' if tables else "")


def _clone_body(summary: dict) -> str:
    last, first = summary.get("last") or {}, summary.get("first") or {}
    tiles = [tile("steps", num(last.get("step")), f'{summary.get("rows", 0)} checks'),
             tile("loss", num(last.get("loss")), f'from {num(first.get("loss"))}')]
    for key, label in (("ordered_bce", "ordered BCE"), ("decisions_a_second", "decisions / s"), ("missed_share", "missed share"),
                       ("cut_share", "cut share"), ("lr", "learning rate")):
        if key in last:
            tiles.append(tile(label, num(last[key])))
    measures = [("kind_right", "kind right"), ("anchor_right", "anchor right"), ("done_right", "done right"), ("ordered_right", "ordered right"),
                ("stops_early", "stops early")]
    zu = any(k.startswith("held_zu_") for k in last)
    head = "<tr><th>held out</th><th class='num'>z known</th><th class='num'>Δ</th>" + ("<th class='num'>z unknown</th>" if zu else "") + "</tr>"
    rows = []
    for key, label in measures:
        v = last.get(f"held_{key}")
        if v is None:
            continue
        f = first.get(f"held_{key}")
        d = v - f if _f(v) and _f(f) else None
        delta = "" if d is None else f'<span class="{"up" if d >= 0 else "down"}">{"+" if d >= 0 else ""}{d * 100:.1f}</span>'
        rows.append(f'<tr><td>{label}</td><td class="num heat"{wash(v)}>{pct(v)}</td><td class="num small">{delta}</td>'
                    + (f'<td class="num heat"{wash(last.get(f"held_zu_{key}"))}>{pct(last.get(f"held_zu_{key}"))}</td>' if zu else "") + "</tr>")
    main = f'<div class="tablebox"><table class="grid">{head}{"".join(rows)}</table></div>' if rows else ""
    kinds = sorted({k[len("held_right_"):] for k in last if k.startswith("held_right_")} | {k[len("held_nll_"):] for k in last if k.startswith("held_nll_")},
                   key=lambda k: -(last.get(f"held_n_{k}") or 0))
    per = ""
    if kinds:
        head = ("<tr><th>order kind</th><th class='num'>held-out n</th><th class='num'>right</th><th class='num'>NLL</th>"
                + ("<th class='num'>right, z unk.</th><th class='num'>NLL, z unk.</th>" if zu else "") + "</tr>")
        body = "".join(f'<tr><td>{esc(k)}</td><td class="num dim">{num(last.get(f"held_n_{k}"))}</td>'
                       f'<td class="num heat"{wash(last.get(f"held_right_{k}"))}>{pct(last.get(f"held_right_{k}"))}</td>'
                       f'<td class="num">{num(last.get(f"held_nll_{k}"))}</td>'
                       + (f'<td class="num heat"{wash(last.get(f"held_zu_right_{k}"))}>{pct(last.get(f"held_zu_right_{k}"))}</td>'
                          f'<td class="num">{num(last.get(f"held_zu_nll_{k}"))}</td>' if zu else "") + "</tr>" for k in kinds)
        per = f'<div class="tablebox"><table class="grid">{head}{body}</table></div>'
    return f'<div class="tiles">{"".join(tiles)}</div><div class="two">{main}{per}</div>'


def wdl(win, draw, loss, games) -> str:
    """Won / drawn / lost as one segmented bar with its counts."""
    win, draw, loss = (v if _f(v) else 0.0 for v in (win, draw, loss))
    n = int(games or 0)
    counts = [round(r * n) for r in (win, draw, loss)]
    segs = "".join(f'<i class="seg {c}" style="flex-grow:{r:.4f}" title="{w} {k} ({r * 100:.1f}%)"></i>'
                   for c, w, r, k in (("w", "won", win, counts[0]), ("d", "drew", draw, counts[1]), ("l", "lost", loss, counts[2])) if r > 0)
    return (f'<div class="segbar">{segs}</div><div class="seglegend"><span><i class="sw w"></i>won {counts[0]}</span>'
            f'<span><i class="sw d"></i>drew {counts[1]}</span><span><i class="sw l"></i>lost {counts[2]}</span></div>')


def _test_body(stage: str, summary: dict) -> str:
    e = summary.get("eval") or {}
    if not e:
        return '<p class="empty">No result yet.</p>'
    games = e.get("games")
    how = e.get("how") or {}
    hero = (f'<div class="hero"><div class="t-label">win rate vs army script</div><div class="hero-value">{pct(e.get("win_rate"))}</div>'
            f'<div class="t-sub">{num(games)} games · draws {pct(e.get("draw_rate"))}</div>{wdl(e.get("win_rate"), e.get("draw_rate"), e.get("loss_rate"), games)}'
            + (f'<div class="how">ended by ' + " · ".join(f"{esc(str(k))} {num(v)}" for k, v in how.items()) + "</div>" if how else "")
            + "</div>")
    minutes = sorted({int(m) for k in e for m in re.findall(r"^actor_\w+?_(\d+)$", k)})
    metrics = [("income", "metal income"), ("extractors", "extractors"), ("value", "army value")]
    econ = ""
    if minutes:
        head = "<tr><th>minute</th>" + "".join(f'<th colspan="2">{label}</th>' for _, label in metrics) + "</tr>"
        top = {m: max([e.get(f"{s}_{m}_{t}") or 0 for s in ("actor", "script") for t in minutes] + [1e-9]) for m, _ in metrics}
        rows = []
        for t in minutes:
            cells = []
            for m, _ in metrics:
                a, s = e.get(f"actor_{m}_{t}"), e.get(f"script_{m}_{t}")
                bars = "".join(f'<div class="pair"><span class="pbar {c}" style="width:{(v or 0) / top[m] * 100:.1f}%"></span><span class="pv">{num(v)}</span></div>'
                               for c, v in (("a", a), ("s", s)))
                ratio = f"{a / s:.2f}×" if _f(a) and _f(s) and s else "–"
                cells.append(f'<td class="pairs">{bars}</td><td class="num ratio" title="policy ÷ script">{ratio}</td>')
            reached = e.get(f"reached_{t}")
            rows.append(f'<tr><td class="num">{t}<div class="sub">{num(reached)} games</div></td>{"".join(cells)}</tr>')
        econ = (f'<div class="econ"><div class="econ-head"><span class="t-label">economy, mean over games still going</span>'
                f'<span class="keys"><span><i class="sw a"></i>policy</span><span><i class="sw s"></i>army script</span><span class="dim">ratio = policy ÷ script</span></span></div>'
                f'<table class="grid econ-table">{head}{"".join(rows)}</table></div>')
    meta_keys = ("actor", "temperature", "z", "policy_economy", "decisions", "seconds", "kept_games")
    meta = "".join(f"<div><dt>{esc(k.replace('_', ' '))}</dt><dd class='mono'>{num(e[k]) if _f(e[k]) else esc(str(e[k]))}</dd></div>" for k in meta_keys if k in e)
    breakdown = f'<div class="breakdown" id="breakdown-{esc(stage)}" data-stage="{esc(stage)}" hx-preserve="true"></div>'
    return f'<div class="test-top">{hero}{econ}</div>{breakdown}<dl class="meta inline">{meta}</dl>'


def _ppo_body(summary: dict) -> str:
    last = summary.get("last") or {}
    tests = summary.get("tests") or []
    tiles = [tile("iteration", num(last.get("iteration")), f'{num(last.get("total_decisions"))} decisions')]
    if _f(last.get("game_minutes")):
        tiles.append(tile("game time played", f'{last["game_minutes"] / 60:,.0f} h'))
    if tests:
        t = tests[-1]
        g = t.get("games") or 0
        tiles.append(tile("last test win rate", pct(t["won"] / g) if g else "–",
                          f'it {num(t.get("iteration"))} · {g} games vs {esc(str(t.get("opponent") or "?"))}', "accent"))
    for key, label in (("kl", "KL"), ("kl_clone", "KL to clone"), ("explained_variance", "explained variance"), ("clip_fraction", "clip fraction"),
                       ("overall_rate", "decisions / s"), ("self_play_share", "self-play share"), ("beta", "beta"), ("lead_share", "lead rule share")):
        if key in last:
            tiles.append(tile(label, num(last[key])))
    table = ""
    if tests:
        kinds = any(t.get("kinds") for t in tests)      # the games say their kind: the columns are the 1v1s, the rest beside
        body = []
        for t in reversed(tests[-200:]):
            g = t.get("games") or 0
            rate = t["won"] / g if g else 0
            body.append(f'<tr><td class="num">{num(t.get("iteration"))}</td><td class="num dim">{num(t.get("decisions"))}</td>'
                        f'<td>{esc(str(t.get("opponent") or ""))}</td><td class="num">{g}</td><td class="num">{t["won"]}</td><td class="num">{t["drew"]}</td>'
                        f'<td class="num wbar"><span style="width:{rate * 100:.0f}%"></span>{pct(rate) if g else "–"}</td>'
                        + (f'<td class="small">{kind_tally(t.get("kinds") or {})}</td>' if kinds else "") + "</tr>")
        table = ('<div class="tablebox tall"><table class="grid"><tr><th class="num">iteration</th><th class="num">decisions</th><th>opponent</th>'
                 '<th class="num">games</th><th class="num">won</th><th class="num">drew</th><th class="num">win rate</th>'
                 + ('<th title="won of played, by kind of game">by kind</th>' if kinds else "") + f'</tr>{"".join(body)}</table></div>')
    return f'<div class="tiles">{"".join(tiles)}</div>' + (f'<div class="ppo-tests"><div class="t-label">tests during PPO ({len(tests)})</div>{table}</div>' if table else "")


def kind_tally(kinds: dict) -> str:
    """Won of played, by kind of game, the 1v1s first: "1v1 12/32 · ffa 3/8"."""
    return " · ".join(f'{esc(str(k))} {c.get("won", 0)}/{c.get("games", 0)}' for k, c in sorted(kinds.items(), key=lambda kc: (kc[0] != "1v1", kc[0])))


def _card(stage: str, summary: dict) -> str:
    k = kind(stage)
    if k == "frames":
        body, x = _frames_body(summary), None
    elif k == "encoder":
        body, x = _encoder_body(summary), "step"
    elif k == "clone":
        body, x = _clone_body(summary), "step"
    elif k == "test":
        body, x = _test_body(stage, summary), None
    elif k == "ppo":
        body, x = _ppo_body(summary), "iteration"
    else:
        body, x = f'<pre>{esc(repr({a: b for a, b in summary.items() if a != "stage"}))[:3000]}</pre>', None
    charts = _charts(stage, x) if x and summary.get("rows") else ""
    return (f'<article class="card k-{k}" id="result-{esc(stage)}" data-stage="{esc(stage)}">'
            f'<header class="card-head"><h3>{esc(stage)}</h3><span class="kind">{esc(KINDS.get(k, ""))}</span>'
            f'<span class="desc">{esc(describe(stage))}</span><span class="spacer"></span>{log_button(stage)}</header>'
            f'<div class="card-body">{body}</div>{charts}</article>')


def stage_result(stage: str, summary: dict) -> str:
    return _card(stage, summary)


def headline(state: dict, summaries: dict[str, dict]) -> str:
    """The closed-loop tests side by side, and PPO's latest: what the run comes to."""
    cols = []
    for s in state.get("stages", []):
        name = s["name"]
        summary = summaries.get(name) or {}
        if kind(name) == "test" and summary.get("eval"):
            e = summary["eval"]
            cols.append((name, e.get("win_rate"), e.get("draw_rate"), e.get("games"),
                         {m: (e.get(f"actor_{m}_10"), e.get(f"script_{m}_10")) for m in ("income", "extractors", "value")}))
        elif name == "ppo" and summary.get("tests"):
            t = summary["tests"][-1]
            g = t.get("games") or 0
            cols.append((f'ppo, it {t.get("iteration")}', t["won"] / g if g else None, t["drew"] / g if g else None, g, {}))
    if not cols:
        return '<div id="headline"></div>'

    def ratio(pair):
        a, s = pair
        return f"{a / s:.2f}×" if _f(a) and _f(s) and s else "–"
    head = "<tr><th></th>" + "".join(f"<th class='num'>{esc(c[0])}</th>" for c in cols) + "</tr>"
    win = "<tr><td>win rate</td>" + "".join(f'<td class="num big">{pct(c[1])}<span class="wbar inline"><span style="width:{(c[1] or 0) * 100:.0f}%"></span></span></td>' for c in cols) + "</tr>"
    draw = "<tr><td>draw rate</td>" + "".join(f'<td class="num">{pct(c[2])}</td>' for c in cols) + "</tr>"
    games = "<tr><td>games</td>" + "".join(f'<td class="num dim">{num(c[3])}</td>' for c in cols) + "</tr>"
    econ = "".join(f'<tr><td>{label} at 10 min, policy ÷ script</td>' + "".join(f'<td class="num">{ratio(c[4][m]) if c[4] else ""}</td>' for c in cols) + "</tr>"
                   for m, label in (("income", "metal income"), ("extractors", "extractors"), ("value", "army value")))
    return (f'<div id="headline" class="headline"><div class="t-label">against the army script, test by test</div>'
            f'<table class="grid hl">{head}{win}{draw}{games}{econ}</table></div>')


def results(state: dict, summaries: dict[str, dict]) -> str:
    """Every stage with results, the latest first, under the tests' headline."""
    order = [s["name"] for s in state.get("stages", [])]
    names = [n for n in order if n in summaries] + [n for n in summaries if n not in order]
    cards = [_card(n, summaries[n]) for n in reversed(names) if len(summaries[n]) > 1 and kind(n) != "zindex"]
    if not cards:
        return headline(state, summaries) + '<p class="empty">No results yet: each stage shows its numbers here when it writes them.</p>'
    return headline(state, summaries) + "".join(cards)


# Replays -----------------------------------------------------------------------------------------------------------------------------

def factory_name(name: str) -> str:
    return (name or "?").removeprefix("factory").removeprefix("plate") or name


def teams_of(r: dict) -> list[dict]:
    """A replay's teams, from its index line: format 3's "teams", or format 2's two "sides", each its own alliance."""
    if r.get("teams") is not None:
        return r["teams"]
    return [{**s, "ally": i, "plays": "policy" if s.get("name") == "policy" else "script"} for i, s in enumerate(r.get("sides") or [])]


def alliances(teams: list[dict]) -> list[tuple[int, list[int]]]:
    """Each alliance's allyteam and its teams, in the order of their allyteams: the viewer's alliance A, B, ..."""
    by: dict[int, list[int]] = {}
    for i, t in enumerate(teams):
        by.setdefault(t.get("ally", i), []).append(i)
    return sorted(by.items())


def kind_name(r: dict) -> str:
    """What kind of game a replay is, as the viewer names it (static/replay.js `kindName`): "1v1", "2v2", "4-way free-for-all"."""
    teams = teams_of(r)
    chickens = sum(t.get("plays") == "chickens" for t in teams)
    if r.get("kind") == "chicken" or chickens:
        return f"{len(teams) - chickens} vs chickens"
    sizes = [len(members) for _, members in alliances(teams)]
    if len(sizes) > 2:
        return f"{len(sizes)}-way free-for-all" if all(n == 1 for n in sizes) else "v".join(map(str, sizes))
    return "v".join(map(str, sizes)) or str(r.get("kind") or "1v1")


def _swatch(teams: list[dict], i: int) -> str:
    """A team's colour, by its number as in the viewer, the chickens' their own."""
    slot = "chickens" if teams[i].get("plays") == "chickens" else f"t{min(i, 7)}"
    return f'<i class="sw {slot}" title="team {i}: {esc(str(teams[i].get("name", "")))}"></i>'


def _replay_item(r: dict) -> str:
    teams = teams_of(r)
    groups = alliances(teams)
    w = r.get("winner")
    won = next(((k, members) for k, (ally, members) in enumerate(groups) if w is not None and ally == w), None)
    if won is not None:
        k, members = won
        who = teams[members[0]].get("name", f"team {members[0]}") if len(members) == 1 else f"alliance {chr(65 + k)}"
        badge = f'<span class="badge win">{"".join(_swatch(teams, i) for i in members)}{esc(who)} won</span>'
    else:
        badge = f'<span class="badge draw">{"no sole winner" if len(groups) > 2 else "draw"}</span>'
    if len(teams) <= 2:
        lines = "".join(f'<div class="rside">{_swatch(teams, i)}<span class="rname">{esc(t.get("name", ""))}</span>'
                        f'<span class="dim">{esc(str(t.get("variant", "")))} · {esc(factory_name(t.get("factory", "")))}</span>'
                        f'<span class="num">{num(t.get("value"))}</span></div>' for i, t in enumerate(teams))
    else:       # the alliances on one line: at eight teams, a line a team would be most of the card
        vs = '<span class="dim">vs</span>'.join(
            f'<span class="rally" title="alliance {chr(65 + k)}: {esc(", ".join(str(teams[i].get("name", "")) for i in members))}">'
            f'{"".join(_swatch(teams, i) for i in members)}</span>' for k, (_, members) in enumerate(groups))
        lines = (f'<div class="rlineup">{vs}</div>'
                 f'<div class="dim">{len(teams)} teams · {esc(", ".join(sorted({str(t.get("name", "")) for t in teams})))}</div>')
    slug = r.get("map", "")
    title = f'{r.get("label", "")} · {slug} · slot {r.get("slot", "")}'
    return (f'<a class="replay" href="#replay={esc(r["url"])}" title="{esc(title)}"><div class="thumb">'
            f'<img loading="lazy" src="/map/{esc(slug)}.png" alt="" onerror="this.remove()"><span class="map">{esc(slug.replace("_", " "))}</span></div>'
            f'<div class="rinfo"><div class="rtop">{badge}<span class="rkind">{esc(kind_name(r))}</span>'
            f'<span class="dim">{esc(str(r.get("how", "")))} · {clock(r.get("seconds"))}</span></div>{lines}</div></a>')


def replays(items: list[dict]) -> str:
    if not items:
        return '<p class="empty">No replays yet: 4 games are recorded after each closed-loop test and every hour of PPO.</p>'
    stages_: dict[str, dict[tuple, list[dict]]] = {}
    for r in items:
        batch = (r.get("label", ""), r.get("iteration"), r.get("dir", ""))
        stages_.setdefault(r.get("stage", "?"), {}).setdefault(batch, []).append(r)
    out = []
    newest = lambda groups: max(x.get("written", "") for g in groups.values() for x in g)
    for stage in sorted(stages_, key=lambda s: newest(stages_[s]), reverse=True):
        groups = stages_[stage]
        parts = []
        for batch in sorted(groups, key=lambda b: max(x.get("written", "") for x in groups[b]), reverse=True):
            g = sorted(groups[batch], key=lambda x: x.get("slot", 0))
            first = g[0]
            facts = []
            if first.get("iteration") is not None:
                facts.append(f'iteration {num(first["iteration"])}')
            if first.get("decisions") is not None:
                facts.append(f'{num(first["decisions"])} decisions')
            if first.get("hours") is not None:
                facts.append(f'{first["hours"]:.1f} h of PPO')
            facts.append(f'written {esc(str(first.get("written", ""))[5:16])}')
            label = "" if first.get("iteration") is not None else f'<span>{esc(str(batch[0]))}</span>'    # a PPO label only repeats the facts
            parts.append(f'<div class="rbatch"><div class="rbatch-head">{label}<span class="dim">{" · ".join(facts)}</span></div>'
                         f'<div class="rgrid">{"".join(_replay_item(r) for r in g)}</div></div>')
        count = sum(len(g) for g in groups.values())
        out.append(f'<div class="rstage"><h3>{esc(stage)} <span class="dim">{count} game{"s" if count != 1 else ""}</span></h3>{"".join(parts)}</div>')
    return "".join(out)
