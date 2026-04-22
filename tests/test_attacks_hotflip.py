"""
Unit tests for ``GradientStorage`` and ``hotflip_attack``.

These use a tiny toy embedding layer — no BGE, no network.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.attacks.hotflip import GradientStorage, hotflip_attack


def _toy_model(vocab_size: int = 7, hidden: int = 4) -> nn.Embedding:
    torch.manual_seed(0)
    emb = nn.Embedding(vocab_size, hidden)
    return emb


def test_gradient_storage_captures_last_n_positions():
    emb = _toy_model(vocab_size=7, hidden=4)
    num_adv = 2
    storage = GradientStorage(emb, num_adv_passage_tokens=num_adv)

    ids = torch.tensor([[0, 1, 2, 3, 4]])
    vecs = emb(ids)
    (vecs.sum()).backward()

    g = storage.get()
    assert g.shape == (1, num_adv, 4)
    storage.close()


def test_gradient_storage_accumulates_across_backward_calls():
    emb = _toy_model()
    storage = GradientStorage(emb, num_adv_passage_tokens=2)
    ids = torch.tensor([[1, 2, 3]])

    vecs1 = emb(ids)
    vecs1.sum().backward()
    g1 = storage.get().clone()

    vecs2 = emb(ids)
    vecs2.sum().backward()
    g2 = storage.get()

    assert torch.allclose(g2, g1 + g1)
    storage.close()


def test_hotflip_attack_returns_topk_by_neg_dot_grad():
    # embedding matrix has six tokens; token 4 is the best "increase-loss" swap
    embedding_matrix = torch.tensor(
        [
            [0.1, 0.0],
            [0.0, 0.1],
            [0.2, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.0],
        ]
    )
    # averaged_grad points in +x: scores = E @ g = [0.1, 0, 0.2, -1, 1, 0.5]
    grad = torch.tensor([1.0, 0.0])
    top = hotflip_attack(
        grad, embedding_matrix, increase_loss=True, num_candidates=2, exclude_up_to=None
    )
    assert set(top.tolist()) == {4, 5}


def test_hotflip_attack_respects_exclude_up_to():
    embedding_matrix = torch.tensor(
        [
            [10.0, 0.0],
            [9.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.0],
        ]
    )
    grad = torch.tensor([1.0, 0.0])
    # exclude_up_to=1 masks out ids 0 and 1 even though they have the best dot
    top = hotflip_attack(
        grad, embedding_matrix, increase_loss=True, num_candidates=2, exclude_up_to=1
    )
    assert set(top.tolist()) == {2, 3}
