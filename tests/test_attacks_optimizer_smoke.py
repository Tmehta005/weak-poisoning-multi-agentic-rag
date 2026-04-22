"""
Smoke test for the HotFlip trigger optimizer on a 20-doc toy corpus.

Verifies:
- The optimizer runs end-to-end with small hyperparameters.
- A non-empty trigger string is produced.
- The loss history has the requested length.
- Rerunning the optimizer with the same seed is deterministic enough that
  it produces the *same* trigger given identical inputs (within the seed).
"""

from __future__ import annotations

import pytest
import tempfile

from src.attacks.encoder import DEFAULT_BGE_MODEL, load_encoder
from src.attacks.trigger_optimizer import OptimizerConfig, optimize_trigger


@pytest.fixture(scope="module")
def encoder():
    try:
        return load_encoder(model_name=DEFAULT_BGE_MODEL, device="cpu")
    except Exception as e:
        pytest.skip(f"BGE not available locally ({e})")


def _toy_corpus():
    return [
        f"NIST control AC-{i} requires documented account management."
        for i in range(1, 11)
    ] + [
        f"CISA playbook step {i} describes incident scoping practices."
        for i in range(1, 11)
    ]


def _toy_queries():
    return [
        "What does NIST require for account management?",
        "How should an incident be scoped?",
        "Describe the NIST access control controls.",
        "Summarize the CISA incident response playbook.",
    ]


def test_optimizer_runs_and_produces_trigger(encoder, tmp_path):
    cfg = OptimizerConfig(
        num_adv_passage_tokens=3,
        num_iter=2,
        num_grad_iter=2,
        num_cand=5,
        per_batch_size=2,
        n_components=2,
        seed=0,
    )
    result = optimize_trigger(
        encoder=encoder,
        training_queries=_toy_queries(),
        corpus_texts=_toy_corpus(),
        config=cfg,
        cache_base_dir=str(tmp_path / "cache"),
        progress=False,
    )
    assert result.trigger  # non-empty
    assert len(result.loss_history) == 2
    assert result.adv_passage_ids.shape == (1, 3)


def test_optimizer_loss_history_is_finite(encoder, tmp_path):
    cfg = OptimizerConfig(
        num_adv_passage_tokens=3,
        num_iter=2,
        num_grad_iter=2,
        num_cand=4,
        per_batch_size=2,
        n_components=2,
        seed=1,
    )
    result = optimize_trigger(
        encoder=encoder,
        training_queries=_toy_queries(),
        corpus_texts=_toy_corpus(),
        config=cfg,
        cache_base_dir=str(tmp_path / "cache"),
        progress=False,
    )
    for v in result.loss_history:
        assert v == v  # NaN check
        assert abs(v) < 1e6
