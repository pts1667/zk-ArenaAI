"""Playing at a padded shape, as CUDA graphs, and in collector processes: each decides as the plain forward pass does."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "tools"))
try:
    import policy_collectors as pc
    import policy_graphs as pg
    import policy_latent as pl
    import policy_model as pm
    import policy_train as pt
    from zkbot.policy import actions as act, featurize as fz, tables
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)

from conftest import sim_frames  # noqa: E402

TABLE = tables.table()
VOCABULARY = len(TABLE.names) + 1
CUDA = torch.cuda.is_available()


@pytest.fixture()
def actor(encoder_path) -> pl.LatentActor:
    torch.manual_seed(2)
    return pl.make(VOCABULARY, encoder_path, size=pl.SMALL)


def _items(count: int = 3, seed: int = 5) -> list[fz.Tensors]:
    frames, bundle = sim_frames(count, seed)
    return [fz.featurize(f, bundle, TABLE, enabled_kinds=fz.STAGE2_KINDS, privileged=True) for f in frames]


def _padded(items: list[fz.Tensors], rows: int = 0, more: int = 0) -> pm.Batch:
    """As a graph pads them, and each group `more` again."""
    n, sizes = pg.padded(pm.widths(items), len(items))
    return pm.collate(items, sizes={k: v + more for k, v in sizes.items()}, rows=max(n, rows))


def test_the_ladder_holds_each_width_and_pads_the_capped_groups_to_their_caps():
    assert [pg.bucket(n) for n in (1, 3, 13, 33, 600, 2048)] == [1, 4, 16, 48, 768, 2048]
    assert pg.bucket(3000) == 3072
    rows, sizes = pg.padded({"u": 20, "r": 40, "s": 3, "f": 1, "c": 30, "e": 5, "h": 2, "p": 9}, 5)
    assert rows == 8 and sizes == {"u": 32, "r": 48, "s": fz.CAP_SITES, "f": fz.CAP_FEATURES, "c": 32, "e": 8, "h": fz.CAP_HIDDEN, "p": 12}


def test_a_padded_batch_decides_as_the_plain_one(actor, encoder_path):
    """Every reader masks the padding: the frame, the critic's value and the logits of every real unit are as they were, and
    actions sampled at the padded shape score the same in the plain batch."""
    items = _items(3)
    plain, wide = pm.collate(items).to("cpu"), _padded(items, rows=8, more=5).to("cpu")
    assert wide.unit_x.shape[0] == 8 and all(getattr(wide, f).shape[1] > getattr(plain, f).shape[1]
                                             for f in ("unit_x", "reg_x", "site_x", "feat_x", "ctrl_mask", "enemy_tokens", "hidden_x", "places"))
    critic = pm.Critic(VOCABULARY, d=32, layers=1, heads=2, bins=5, span=1.5)
    torch.nn.init.normal_(critic.value.weight, std=0.1)
    with torch.no_grad():
        decided = [actor.decide(b, *actor.initial_state(len(b.glob_x), 0, "cpu")) for b in (plain, wide)]
        (ctx, history, _, keep), (wide_ctx, wide_history, _, wide_keep) = decided
        assert torch.allclose(keep["latents"], wide_keep["latents"][:3], atol=1e-4)
        assert torch.allclose(history, wide_history[:3], atol=1e-4)
        assert torch.allclose(critic(plain), critic(wide)[:3], atol=1e-4)
        c = plain.ctrl_mask.shape[1]
        kind, wide_kind = pm.each_unit_heads(actor, plain, ctx)[0], pm.each_unit_heads(actor, wide, wide_ctx)[0]
        real = plain.ctrl_mask
        assert torch.allclose(kind[real], wide_kind[:3, :c][real], atol=1e-3)
        with torch.no_grad():
            actor.head_done.bias[0] = -3.0      # so that it acts
        torch.manual_seed(4)
        a = pm.sample_actions(actor, wide, wide_ctx, every_slot=True)
        a3 = pm.Actions(**{f: getattr(a, f)[:3] for f in act.SLOT_FIELDS}, logp=a.logp[:3])
        assert int((a3.anchor >= 0).sum()) >= 3
        logp, _ = pm.evaluate_actions(actor, plain, ctx, a3)
        assert torch.allclose(logp, a3.logp, atol=1e-3)


def test_sampling_every_slot_chooses_what_stopping_at_done_does(actor):
    items = _items(3, seed=7)
    b = pm.collate(items).to("cpu")
    with torch.no_grad():
        actor.head_done.bias[0] = -1.0
        ctx = actor.decide(b, *actor.initial_state(3, 0, "cpu"))[0]
        torch.manual_seed(9)
        stopped = pm.sample_actions(actor, b, ctx)
        torch.manual_seed(9)
        every = pm.sample_actions(actor, b, ctx, every_slot=True)
    assert int((stopped.anchor >= 0).sum()) >= 1 and bool((stopped.anchor == -1).any())
    for f in act.SLOT_FIELDS + ("logp",):
        assert torch.equal(getattr(stopped, f), getattr(every, f)), f


def test_weights_in_shared_memory_load_into_other_nets_in_place(actor, encoder_path):
    other = pl.make(VOCABULARY, encoder_path, size=pl.SMALL)
    weights = pc.Weights({"actor": actor})
    weights.write({"actor": actor})
    before = other.latent_in.weight
    weights.read({"actor": other})
    assert other.latent_in.weight is before
    mine, theirs = actor.state_dict(), other.state_dict()
    assert all(torch.equal(mine[k], theirs[k]) for k in mine)


def _short_sim(games, seed, modes=None, threads=2, **kw):
    """Games of six decisions, so that tests end."""
    env = pt.make_sim(games, seed, modes, threads=threads, **kw)
    env.time_limit = 6
    return env


@pytest.mark.parametrize("concurrent", (False, True))
def test_collector_processes_play_train_and_test(tmp_path, encoder_path, concurrent):
    """Each collection is checked against the weights sent for it (policy_collectors.fingerprint)."""
    cfg = pt.Config(decisions=24, iteration=8, games=2, passes=1, segment=4, token_budget=4000, data_workers=0, eval_every=1, eval_games=2,
                    checkpoint_every=10**6, latent_encoder=str(encoder_path), collectors=2, concurrent=concurrent, replay_hours=1e-9, replay_games=1)
    pt.train(cfg, tmp_path, _short_sim, device="cpu")
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["masked_out"] == 0 and r["decisions"] >= 8 for r in rows)
    assert all("eval_fight" in r and "collector_seconds_max" in r for r in rows)
    assert sum(r.get("games_fight", 0) for r in rows) > 0         # games ended while collecting, and counted
    tests = [json.loads(line) for line in (tmp_path / "tests.jsonl").read_text().splitlines()]
    assert len(tests) == 3 and all(len(t["games"]) == 2 for t in tests)
    assert len(list((tmp_path / "replays").rglob("*.json.gz"))) == 3


@pytest.mark.skipif(not CUDA, reason="CUDA graphs need CUDA")
def test_a_graph_decides_as_the_plain_forward_pass_and_follows_the_weights(encoder_path):
    torch.manual_seed(2)
    actor = pl.make(VOCABULARY, encoder_path, size=pl.SMALL).cuda()
    critic = pm.Critic(VOCABULARY, d=32, layers=1, heads=2).cuda()      # one number, so that its bias moves it
    torch.nn.init.normal_(critic.value.weight, std=0.1)
    with torch.no_grad():
        actor.head_done.bias[0] = -3.0
    graphs = pg.Graphs(actor, critic)
    eager = pt.Player(actor, critic, pt.DataWorkers(0, [], fz.STAGE2_KINDS), "cuda")
    cut = actor.encoder.tokens * actor.encoder.latent

    def check(items, history, seen):
        out, value, new_history, new_seen = graphs.forward(items, history, seen)
        plain, plain_value, plain_history, _ = eager._forward(items, history, seen)
        assert np.allclose(out["latents"], plain["latents"], atol=2e-2) and np.allclose(value, plain_value, atol=2e-2)
        assert torch.allclose(new_history[:, 1:], plain_history[:, 1:]) and torch.allclose(new_history[:, 0, :cut], plain_history[:, 0, :cut], atol=2e-2)
        a = pm.actions_from(out, "cuda")
        assert int((a.anchor >= 0).sum()) >= 3
        b = pm.collate(items).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = actor.decide(b, history, seen)[0]
            logp, _ = pm.evaluate_actions(actor, b, ctx, a)
        assert torch.allclose(logp, a.logp, atol=5e-2)
        return value, new_history, new_seen

    items = _items(3, 5)
    history, seen = actor.initial_state(3, 0, "cuda")
    _, history, seen = check(items, history, seen)
    value, _, _ = check(items, history, seen)        # the same graph, replayed on what the first kept
    assert graphs.captured == 1 and graphs.replays == 2
    check(_items(3, 6), *actor.initial_state(3, 0, "cuda"))      # other frames, at whichever shape they fall to
    with torch.no_grad():
        critic.value.bias += 1.0        # in place, as an optimizer step is
    moved = graphs.forward(items, history, seen)[1]
    assert np.allclose(moved, value + 1.0, atol=2e-2)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_collector_processes_play_as_graphs_on_cuda(tmp_path, encoder_path):
    """As the trainer runs, in a process that hasn't started CUDA when it forks them (this one has): each collector plays its
    games as CUDA graphs, by the weights sent for each collection (policy_collectors.fingerprint)."""
    tools = pathlib.Path(__file__).resolve().parent.parent.parent / "tools"
    code = f"""
import pathlib, sys
sys.path.insert(0, {str(tools)!r})
import policy_train as pt

def short(games, seed, modes=None, threads=2, **kw):
    env = pt.make_sim(games, seed, modes, threads=threads, **kw)
    env.time_limit = 6
    return env

cfg = pt.Config(decisions=24, iteration=8, games=2, passes=1, segment=4, token_budget=4000, data_workers=0, eval_every=1, eval_games=2,
                checkpoint_every=10**6, latent_encoder={str(encoder_path)!r}, collectors=2, concurrent=True)
pt.train(cfg, pathlib.Path({str(tmp_path)!r}), short, device="cuda")
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["masked_out"] == 0 and r["graphs_captured"] >= 2 and r["collector_reserved_gb"] > 0 for r in rows)
    assert all("eval_fight" in r for r in rows)
