"""
Sanity tests for the BGE encoder wrapper.

These require the ``BAAI/bge-small-en-v1.5`` weights to be reachable
(cached locally or via HF hub). If the model can't be loaded we skip —
tests still run on fully-offline machines without network.
"""

from __future__ import annotations

import pytest
import torch

from src.attacks.encoder import (
    DEFAULT_BGE_MODEL,
    decode_trigger_tokens,
    encode_texts,
    forward_with_adv_suffix,
    initial_adv_passage_ids,
    load_encoder,
    tokenize_query_with_adv_suffix,
)


@pytest.fixture(scope="module")
def encoder():
    try:
        return load_encoder(model_name=DEFAULT_BGE_MODEL, device="cpu")
    except Exception as e:
        pytest.skip(f"BGE not available locally ({e})")


def test_encoder_embedding_dim_and_vocab(encoder):
    assert encoder.embedding_dim == 384  # BGE-small hidden size
    assert encoder.vocab_size > 10000
    assert encoder.word_embeddings.weight.shape == (
        encoder.vocab_size,
        encoder.embedding_dim,
    )


def test_encode_texts_produces_unit_norm_embeddings(encoder):
    embs = encode_texts(encoder, ["hello world", "a test sentence"], batch_size=2)
    assert embs.shape == (2, encoder.embedding_dim)
    norms = embs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_initial_adv_passage_ids_shape(encoder):
    ids = initial_adv_passage_ids(encoder, num_adv_passage_tokens=5)
    assert ids.shape == (1, 5)
    mask_id = encoder.tokenizer.mask_token_id
    assert (ids == mask_id).all().item()


def test_tokenize_query_with_adv_suffix_shapes(encoder):
    ids = initial_adv_passage_ids(encoder, num_adv_passage_tokens=4)
    input_ids, attn = tokenize_query_with_adv_suffix(encoder, "hello world", ids)
    assert input_ids.shape[0] == 1
    assert attn.shape == input_ids.shape
    # Last four tokens are the adv suffix
    assert torch.equal(input_ids[:, -4:], ids)


def test_forward_with_adv_suffix_grad_flows_to_word_embeddings(encoder):
    # Enable grad only on the embedding matrix
    encoder.model.zero_grad(set_to_none=True)
    adv = initial_adv_passage_ids(encoder, num_adv_passage_tokens=3)
    q_emb = forward_with_adv_suffix(encoder, ["a sample query"], adv)
    loss = q_emb.sum()
    loss.backward()
    assert encoder.word_embeddings.weight.grad is not None
    assert encoder.word_embeddings.weight.grad.abs().sum() > 0


def test_decode_trigger_tokens_strips_specials(encoder):
    ids = initial_adv_passage_ids(encoder, num_adv_passage_tokens=2)
    tokens, trigger = decode_trigger_tokens(encoder, ids)
    # [MASK] must be filtered out of the trigger string
    assert "[MASK]" not in trigger
