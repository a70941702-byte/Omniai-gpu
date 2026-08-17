"""Training pipeline: real gradient training happens, anti-forgetting works,
regression gate rejects harmful candidates, and strategy adapts on failure."""
import json


def test_bootstrap_creates_adopted_model():
    from app.orchestrator.core import orchestrator
    from app.models_core import registry
    res = orchestrator._bootstrap_seed_model()
    assert res["train_acc"] > 0.5
    current = registry.get_current_model()
    assert current is not None and current["status"] == "current"


def test_replay_buffer_mixes_old_samples():
    from app.training.anti_forgetting import ReplayBuffer
    buf = ReplayBuffer()
    buf.add([("old sample one", 0), ("old sample two", 1)])
    batch = buf.mixed_batch([("new sample", 2)])
    texts = [t for t, _ in batch]
    assert "new sample" in texts
    assert any("old sample" in t for t in texts)


def test_ewc_penalty_anchors_weights():
    import torch
    from app.models_core.neural_core import NeuralCore
    from app.training.anti_forgetting import EWC
    model = NeuralCore(num_outputs=3)
    samples = [("hello world foo", 0), ("math numbers add", 1)] * 5
    ewc = EWC()
    ewc.fit(model, samples)
    p0 = ewc.penalty(model)
    assert float(p0) < 1e-6  # no drift yet -> ~zero penalty
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    p1 = ewc.penalty(model)
    assert float(p1) > float(p0)  # drift is now penalized


def test_distillation_loss_zero_for_identical():
    import torch
    from app.training.anti_forgetting import distillation_loss
    logits = torch.randn(4, 6)
    assert float(distillation_loss(logits, logits.clone())) < 1e-4


def test_candidate_trained_and_registered():
    from app.models_core import registry
    from app.orchestrator.core import orchestrator
    from app.training.trainer import trainer
    orchestrator._bootstrap_seed_model()
    current = registry.get_current_model()
    res = trainer.train_candidate(
        current["id"],
        [{"text": "quantum physics particles wave", "intent": "science"},
         {"text": "biology cells dna organism", "intent": "science"}] * 3,
        dataset_version="test_v1", epochs=10)
    row = registry.get_model(res["model_id"])
    assert row["status"] == "candidate"
    assert row["parent_id"] == current["id"]
    assert "science" in res["new_intents"]
    assert row["config"]["ewc"] is True and row["config"]["distillation"] is True
    assert row["config"]["replay_size"] > 0  # old data replayed


def test_regression_gate_rejects_bad_candidate():
    from app.evaluation.engine import compare_models
    current = {"model_id": "cur", "overall": 0.8, "safety": 1.0,
               "suites": {"math": {"score": 0.8}, "safety": {"score": 1.0}}}
    candidate = {"model_id": "cand", "overall": 0.9, "safety": 1.0,
                 "suites": {"math": {"score": 0.5}, "safety": {"score": 1.0}}}
    decision = compare_models(current, candidate, None, new_capability_gain=0.1)
    assert not decision["adopt"]
    assert any("REGRESSION" in r for r in decision["reasons"])


def test_safety_gate_blocks_unsafe_candidate():
    from app.evaluation.engine import compare_models
    current = {"model_id": "cur", "overall": 0.8, "safety": 1.0,
               "suites": {"safety": {"score": 1.0}}}
    candidate = {"model_id": "cand", "overall": 0.9, "safety": 0.3,
                 "suites": {"safety": {"score": 0.3}}}
    decision = compare_models(current, candidate, None, new_capability_gain=0.1)
    assert not decision["adopt"]
    assert any("SAFETY" in r for r in decision["reasons"])


def test_no_gain_not_adopted():
    from app.evaluation.engine import compare_models
    cur = {"model_id": "cur", "overall": 0.8, "safety": 1.0, "suites": {}}
    cand = {"model_id": "cand", "overall": 0.8, "safety": 1.0, "suites": {}}
    decision = compare_models(cur, cand, None, new_capability_gain=0.0)
    assert not decision["adopt"]
    assert any("NO GAIN" in r for r in decision["reasons"])
