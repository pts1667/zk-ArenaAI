"""The actor that chooses the order autoencoder's code (tools/policy_code.py): played through the trainer's own path, stored,
and learnt from its frames' latents alone; and its pass as a CUDA graph."""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:
    import order_vae as ov
    import policy_code
    import policy_latent as pl
    import policy_model as pm
    import policy_train as pt
except OSError:
    pytest.skip("needs the built sim library", allow_module_level=True)
from zkbot.policy import rawframe as rf, tables  # noqa: E402

from conftest import sim_frames  # noqa: E402

VOCABULARY = len(tables.table().names) + 1


@pytest.fixture(scope="module")
def coder_path(tmp_path_factory):
    """A small order autoencoder saved as order_vae.py saves one, untrained but for what makes any code say 16 commands,
    each a FIGHT to every role that fights."""
    torch.manual_seed(4)
    model = ov.OrderVAE(ov.sizes(ov.Config(d=32, heads=2, layers=1)))
    with torch.no_grad():
        model.heads["to_present"].bias.fill_(3.0)
        model.heads["to_roles"].bias.fill_(3.0)
        model.heads["to_kind"].bias[ov.KIND_INDEX[rf.FIGHT]] = 5.0
    path = tmp_path_factory.mktemp("coder") / "order-vae.pt"
    ov.save(model, path)
    return path


def test_an_actor_that_chooses_a_code_plays_its_commands_and_learns_from_latents_alone(tmp_path, encoder_path, coder_path, monkeypatch):
    """Bootstrapped GRPO with the memory's critics, as the runs have it.  A decision is stored as its code and the code's one
    probability; what the frozen decoder says of it is carried out; in training each code is as likely as when it was drawn,
    its seat's memory going through a segment as it went through play; and the update featurizes no frame."""
    cfg = pt.Config(decisions=120, iteration=60, games=4, passes=1, segment=4, token_budget=25_000, data_workers=0, eval_every=10**6,
                    checkpoint_every=10**6, latent_encoder=str(encoder_path), group=2, horizon=(5.0, 10.0), bootstrap=True, size="code",
                    order_coder=str(coder_path), critic_size="memory", critic_bins=51, critic_span=2.0, critic_warmup=0, spread_floor=((0, 0.1), (2, 0.03)),
                    terms=(("result", 1.0), ("combat", 0.3), ("economy", 0.3)), critics=("result", "economy"))
    env = pt.make_sim(cfg.games, cfg.seed)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, env.bundles(), cfg.enabled_kinds), "cpu")
    actor = learner.actor
    assert isinstance(actor, policy_code.CodeActor) and actor.size.coder == str(coder_path) and learner.player.critic is None
    old = pl.make(VOCABULARY, encoder_path, size=replace(pl.CODE_256, coder=str(coder_path)))     # code-1's and code-2's reader, by attention
    assert actor.flat and not old.flat and old.mean(torch.zeros(2, old.encoder.tokens, old.encoder.latent), torch.zeros(2, old.gru), None).shape == (2, old.codes, old.latent)
    with torch.no_grad():
        actor.code_mean.weight.normal_(std=0.3)     # a code's mean is its frame's and its memory's, as a trained actor's is
    segments, tally = pt.GroupCollector(learner, env, np.random.default_rng(1)).collect_tally(40)
    steps = [s for seg in segments for s in seg.steps]
    assert all(s.code.shape == (actor.codes, actor.latent) and s.logp.shape == (1,) and not len(s.anchor) for s in steps)
    assert tally.actions > 0 and tally.kind_counts[rf.FIGHT] == tally.actions and tally.groups["orders"] > 0 and not tally.masked_out
    assert any(float(seg.seen.sum()) for seg in segments)       # some start mid-game, from what their seat kept
    with torch.no_grad():       # critics that say something of where a branch ended
        for head in learner.critic.value:
            head.bias.copy_(torch.linspace(-1.0, 1.0, 51))
    learner.prepare(segments)
    # A critic for the terms told to have one (Config.critics): the rest have nothing where a branch ended.
    ended = [step for seg in segments if seg.end is not None for step in seg.steps]
    assert learner.critic.terms == 2 and all(step.ret[1] == step.rest[1] for step in ended) and any(step.ret[0] != step.rest[0] for step in ended)
    notes: dict[str, list[float]] = {}
    learner._loss(segments, 0.2, lambda name, value, weight=1.0: notes.setdefault(name, []).append(value))
    assert max(notes["ratio_deviation_max"]) < 1e-3
    monkeypatch.setattr(learner, "_featurize", lambda batch: pytest.fail("a code's update featurized a frame"))

    def judged(segments):       # as if the branches had come apart, which a few seconds of an opening don't bring them to
        for i, step in enumerate(steps):
            step.advantage = 1.0 if i % 2 else -1.0
        return np.ones(1)
    monkeypatch.setattr(learner, "prepare", judged)
    before = actor.code_mean.weight.detach().clone()
    metrics = learner.train(segments, np.random.default_rng(0))
    assert metrics["loss_value"] > 0.0 and "entropy_code" in metrics and not torch.equal(before, actor.code_mean.weight)
    assert "explained_variance_economy" in metrics and "explained_variance_combat" not in metrics
    # The spread's least falls by iteration (Config.spread_floor), for the tests' average of the weights too; or by the hours
    # since the run began, when it's told that.
    with torch.no_grad():
        actor.code_log_std.fill_(-9.0)
    for iteration, least in ((0, 0.1), (1, 0.065), (5, 0.03)):
        learner.iteration = iteration
        learner.set_floor()
        assert float(actor.log_std().detach().exp().max()) == pytest.approx(least) == float(learner.ema.actor.log_std_least.exp())
    learner.iteration = 0
    learner.cfg = replace(cfg, spread_floor_began=time.time() - 3600.0)
    learner.set_floor()
    assert float(actor.log_std_least.exp()) == pytest.approx(0.065, abs=1e-3)
    with pytest.raises(ValueError, match="GRPO"):
        pt.Learner(replace(cfg, group=0, bootstrap=False, critic_size="default"), tmp_path, learner.workers, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")
def test_as_a_graph_a_code_is_drawn_said_and_kept_as_it_is_eagerly(encoder_path, coder_path):
    """The whole pass is one graph: what comes back of it is the code drawn, its probability, what the decoder says of that
    code, and the state with the code in it, round after round."""
    torch.manual_seed(2)
    actor = pl.make(VOCABULARY, encoder_path, size=replace(pl.CODE_SIZE, coder=str(coder_path))).cuda()
    frames, bundle = sim_frames(6)
    names = np.arange(VOCABULARY, dtype=np.int32)
    graphed = pt.Player(actor, None, pt.DataWorkers(0, [bundle], pt.Config.enabled_kinds), "cuda", graphs=True)
    history, seen = actor.initial_state(3, 0, "cuda")
    for at in (0, 3):       # the second round from what the first kept
        items, out, _, kept, kept_seen = graphed.act([(f, bundle, names) for f in frames[at:at + 3]], history, seen)
        code = torch.as_tensor(out["code"], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            mean, eager, _, keep = actor.decide(pm.collate(items).to("cuda"), history, seen)
            said = actor.say(code)
        assert not float(mean.abs().max())      # untrained, every mean is 0: the code's probability is the unit Gaussian's
        assert np.allclose(out["logp"][:, 0], actor.log_prob(mean, code).detach().cpu().numpy(), atol=1e-3)
        assert np.allclose(out["latents"], keep["latents"].cpu().numpy(), atol=2e-2)
        assert torch.equal(kept[:, 0, actor.gru:], code.flatten(1)) and torch.allclose(kept[:, 0, :actor.gru], eager[:, 0, :actor.gru], atol=2e-2)
        assert all(np.array_equal(out["say_" + f], said[f].cpu().numpy()) for f in ("present", "roles", "radius", "portion", "kind", "type", "extent"))
        assert np.allclose(out["say_centre"], said["centre"].cpu().numpy(), atol=1e-4) and out["say_present"].all()
        history, seen = kept, kept_seen
    assert graphed.graphs.captured == 1 and graphed.graphs.replays == 2


def test_a_code_actor_is_cloned_from_the_scripts_advice_and_a_run_starts_from_it(tmp_path, encoder_path, coder_path):
    """policy_code_clone.py: the scripts' games kept with each decision's latents and advice; an actor's mean taught the
    advice over streams that carry each seat's memory from step to step, as one pass over its game would; that actor then
    playing with the scripts only advising, its draws kept; and a run starting from its file at the spread it's told."""
    import policy_code_clone as cc
    common = ["--encoder", str(encoder_path), "--coder", str(coder_path)]
    play = ["--games", "2", "--workers", "1", "--threads", "2", "--featurize-threads", "0", *common]
    cc.main(["record", "--out", str(tmp_path / "scripts"), "--decisions", "40", *play])
    part = dict(np.load(tmp_path / "scripts" / "part-w00.npz"))
    assert part["lengths"].sum() == len(part["counts"]) == len(part["latents"]) >= 40 and len(part["lengths"]) == len(part["map"])
    assert part["counts"].sum() == len(part["commands"]) > 0 and "played" not in part
    cc.main(["train", "--data", str(tmp_path / "scripts"), "--out", str(tmp_path / "clone"), "--steps", "3", "--batch", "2", "--length", "4",
             "--report", "1", "--test-every", "3", "--held-out", "0.3", *common])
    rows = [json.loads(line) for line in (tmp_path / "clone" / "log.jsonl").read_text().splitlines()]
    assert rows[-1]["held_out"]["commands_a_decision"] > 0 and all(np.isfinite(r["loss"]) for r in rows)
    actor = cc.load_actor(tmp_path / "clone" / "actor.pt", str(encoder_path), str(coder_path))
    assert float(actor.code_mean.weight.detach().abs().max()) > 0.0 and torch.allclose(actor.log_std().detach().exp(), torch.tensor(0.3))
    data = cc.load_data([str(tmp_path / "scripts")], actor.decoder, "cpu")
    games = np.arange(len(data["lengths"]))

    def all_means(length: int) -> torch.Tensor:
        streams, got = cc.Streams(data, games, 2, length, actor.record, "cpu"), {}
        while (step := cc.means(actor, data, streams, torch.autocast("cpu", enabled=False))) is not None:
            got.update(zip(step[1].tolist(), step[0]))
        return torch.stack([got[i] for i in range(len(data["codes"]))])
    with torch.no_grad():
        assert torch.allclose(all_means(3), all_means(int(data["lengths"].max())), atol=1e-4)
    cc.main(["record", "--out", str(tmp_path / "own"), "--decisions", "12", "--actor", str(tmp_path / "clone" / "actor.pt"), "--modes", "fight", *play])
    own = dict(np.load(tmp_path / "own" / "part-w00.npz"))
    assert own["played"].shape == (len(own["counts"]), actor.codes, actor.latent) and own["counts"].sum() > 0
    cfg = pt.Config(latent_encoder=str(encoder_path), group=2, bootstrap=True, critic_size="memory", critic_bins=51, data_workers=0,
                    init=str(tmp_path / "clone" / "actor.pt"), spread_start=0.2)
    learner = pt.Learner(cfg, tmp_path, pt.DataWorkers(0, pt.make_sim(1, 1).bundles(), cfg.enabled_kinds), "cpu")
    assert torch.equal(learner.actor.code_mean.weight, actor.code_mean.weight) and learner.actor.size.coder == str(coder_path)
    assert all(torch.allclose(a.log_std().exp(), torch.tensor(0.2)) for a in (learner.actor, learner.ema.actor))
