"""
Pure unit tests for the fitness functions — no network, no heavy deps.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.attacks.fitness import (
    compute_avg_cluster_distance,
    compute_avg_embedding_similarity,
    compute_variance,
    gaussian_kernel_matrix,
    maximum_mean_discrepancy,
)


def test_variance_zero_for_identical_vectors():
    x = torch.tensor([[1.0, 2.0, 3.0]] * 4)
    assert compute_variance(x).item() == pytest.approx(0.0, abs=1e-6)


def test_variance_positive_for_spread_vectors():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert compute_variance(x).item() > 0.5


def test_gaussian_kernel_identity_is_one():
    x = torch.tensor([[1.0, 0.0]])
    k = gaussian_kernel_matrix(x, x, sigma=1.0)
    assert k.item() == pytest.approx(1.0)


def test_mmd_zero_when_samples_equal():
    x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    assert maximum_mean_discrepancy(x, x, sigma=1.0).item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_avg_cluster_distance_hand_computed():
    # query at origin, three centers at distances 3, 4, 5 along each axis
    query = torch.zeros(1, 3)
    centers = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    # variance is 0 since only one query, so penalty term is 0
    expected = (3.0 + 4.0 + 5.0) / 3.0
    assert compute_avg_cluster_distance(query, centers).item() == pytest.approx(
        expected, abs=1e-5
    )


def test_avg_cluster_distance_decreases_with_closer_query():
    centers = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    far = compute_avg_cluster_distance(torch.zeros(1, 2), centers).item()
    close = compute_avg_cluster_distance(
        torch.tensor([[3.0, 3.0]]), centers
    ).item()
    assert far > close


def test_avg_embedding_similarity_perfect_match():
    # both triggered queries align exactly with a single db vector
    db = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    q = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    # similarities are [1,0] and [1,0] -> mean per row = 0.5 -> overall 0.5
    assert compute_avg_embedding_similarity(q, db).item() == pytest.approx(0.5)
