"""
HotFlip-style discrete token optimization.

Ports from ``algo/trigger_optimization.py``:
- ``GradientStorage``       (lines 145-166): captures the gradient flowing
                             back into the adv-passage slice of the
                             word-embedding layer.
- ``hotflip_attack``         (lines 180-219): pick top-k candidate token
                             ids per position via ``-grad dot E``.
- ``candidate_filter``       (lines 221-241): coherence filter using
                             GPT-2 perplexity.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GradientStorage:
    """
    Hooks a backward pass on the word-embedding layer and stores the
    gradient restricted to the *last* ``num_adv_passage_tokens`` positions.

    Usage:
        storage = GradientStorage(encoder.word_embeddings, num_adv_tokens)
        ...  # run forward + backward
        g = storage.get()        # [num_adv_tokens, hidden]
        storage.reset()
    """

    def __init__(self, module: nn.Module, num_adv_passage_tokens: int):
        self._stored_gradient: Optional[torch.Tensor] = None
        self.num_adv_passage_tokens = num_adv_passage_tokens
        self._handle = module.register_full_backward_hook(self._hook)

    def _hook(self, module, grad_in, grad_out):
        g = grad_out[0]
        tail = g[:, -self.num_adv_passage_tokens :]
        if self._stored_gradient is None:
            self._stored_gradient = tail.detach().clone()
        else:
            self._stored_gradient = self._stored_gradient + tail.detach()

    def get(self) -> torch.Tensor:
        """Return the accumulated ``[batch_or_1, num_adv, hidden]`` gradient."""
        if self._stored_gradient is None:
            raise RuntimeError("No gradient stored yet — call .backward() first.")
        return self._stored_gradient

    def reset(self) -> None:
        self._stored_gradient = None

    def close(self) -> None:
        self._handle.remove()


def hotflip_attack(
    averaged_grad: torch.Tensor,
    embedding_matrix: torch.Tensor,
    increase_loss: bool = True,
    num_candidates: int = 100,
    token_filter: Optional[torch.Tensor] = None,
    exclude_up_to: Optional[int] = None,
) -> torch.Tensor:
    """
    Return the top ``num_candidates`` token ids that, if swapped in,
    move the loss in the requested direction.

    Args:
        averaged_grad: ``[hidden]`` gradient at the position being flipped.
        embedding_matrix: ``[vocab, hidden]`` word-embedding matrix.
        increase_loss: If True we *maximize* the loss (AgentPoison's "ap"
            loss). If False we minimize.
        num_candidates: ``top_k`` over the vocabulary.
        token_filter: Optional ``[vocab]`` mask whose values are subtracted
            from the score (e.g. to penalize previously tried tokens).
        exclude_up_to: Mask out token ids ``[0, exclude_up_to]`` so special /
            reserved ids are never selected (``slice`` argument in the
            original code).
    """
    with torch.no_grad():
        scores = torch.matmul(embedding_matrix, averaged_grad)
        if token_filter is not None:
            scores = scores - token_filter
        if not increase_loss:
            scores = -scores

        if exclude_up_to is not None and exclude_up_to >= 0:
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[: exclude_up_to + 1] = True
            scores = scores.masked_fill(mask, float("-inf"))

        _, top_ids = scores.topk(num_candidates)
    return top_ids


def _compute_perplexity(
    input_ids: torch.Tensor, model: nn.Module
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs[0]
    return torch.exp(loss)


def candidate_filter(
    candidates: torch.Tensor,
    num_candidates: int,
    token_to_flip: int,
    adv_passage_ids: torch.Tensor,
    ppl_model: nn.Module,
) -> torch.Tensor:
    """
    Coherence filter: keep the ``num_candidates`` candidate tokens that
    yield the *lowest* perplexity when swapped into
    ``adv_passage_ids[:, token_to_flip]``.

    The original AgentPoison code uses ``ppl_score * -1`` and then top-k,
    which selects *highest* perplexity; we preserve that behavior to stay
    faithful to the paper. In practice it mostly trades a very-coherent
    filter for a "not-degenerate" filter that still flips disruptive tokens.
    """
    with torch.no_grad():
        scores = []
        temp = adv_passage_ids.clone()
        for candidate in candidates:
            temp[:, token_to_flip] = candidate
            ppl = _compute_perplexity(temp, ppl_model) * -1
            scores.append(ppl)
        scores_t = torch.tensor(scores)
        _, top_ids = scores_t.topk(num_candidates)
    return candidates[top_ids]
