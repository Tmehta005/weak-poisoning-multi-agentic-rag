"""
BGE-small encoder wrapper for trigger optimization.

Loads ``BAAI/bge-small-en-v1.5`` via HuggingFace transformers (not
sentence-transformers) so we can:
- access ``model.embeddings.word_embeddings`` for HotFlip gradients,
- append optimizable adversarial-passage token ids as a suffix to each
  tokenized query, and
- reproduce the exact CLS-pooled + L2-normalized embedding that the
  LlamaIndex retriever uses at query time.

This is intentionally BGE-only: AgentPoison's config supports many
embedders, but our retriever uses BGE-small so we optimize against that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_BGE_MODEL = "BAAI/bge-small-en-v1.5"


def pick_device(preferred: Optional[str] = None) -> str:
    """Return the best available torch device string."""
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class EncoderBundle:
    """
    Bundle of (model, tokenizer, device, model_name) for the BGE encoder.

    The ``model`` is a plain ``AutoModel`` (BertModel under the hood);
    ``model.embeddings.word_embeddings`` is the embedding matrix used for
    HotFlip candidate selection.
    """

    model: torch.nn.Module
    tokenizer: object
    device: str
    model_name: str

    @property
    def word_embeddings(self) -> torch.nn.Embedding:
        return self.model.embeddings.word_embeddings

    @property
    def embedding_dim(self) -> int:
        return int(self.word_embeddings.weight.shape[1])

    @property
    def vocab_size(self) -> int:
        return int(self.word_embeddings.weight.shape[0])


def load_encoder(
    model_name: str = DEFAULT_BGE_MODEL,
    device: Optional[str] = None,
) -> EncoderBundle:
    """
    Load a BGE encoder + tokenizer for white-box optimization.

    Args:
        model_name: HuggingFace repo id. Defaults to BAAI/bge-small-en-v1.5.
        device: Override device. Auto-detected if None (cuda > mps > cpu).
    """
    resolved_device = pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(resolved_device)
    model.eval()
    return EncoderBundle(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        model_name=model_name,
    )


def _cls_normalize(pooled: torch.Tensor) -> torch.Tensor:
    """L2-normalize CLS embeddings (BGE uses normalized CLS for retrieval)."""
    return F.normalize(pooled, p=2, dim=-1)


@torch.no_grad()
def encode_texts(
    encoder: EncoderBundle,
    texts: List[str],
    batch_size: int = 32,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Encode a list of strings with the BGE encoder. No gradients.

    Returns a ``[len(texts), hidden]`` tensor on ``encoder.device`` with
    CLS-pooled, L2-normalized embeddings.
    """
    model, tokenizer, device = encoder.model, encoder.tokenizer, encoder.device
    outputs: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        toks = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        cls = model(**toks).last_hidden_state[:, 0, :]
        outputs.append(_cls_normalize(cls))
    if not outputs:
        return torch.empty(0, encoder.embedding_dim, device=device)
    return torch.cat(outputs, dim=0)


def tokenize_query_with_adv_suffix(
    encoder: EncoderBundle,
    query: str,
    adv_passage_ids: torch.Tensor,
    max_length: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize ``query`` and append ``adv_passage_ids`` as a suffix.

    The adversarial tokens live where the trigger will later be appended
    to the natural-language query at retrieval time, so the optimizer
    sees the same geometry it will attack.

    Args:
        adv_passage_ids: ``[1, num_adv_tokens]`` long tensor.

    Returns:
        (input_ids, attention_mask) both ``[1, seq_len]`` on encoder.device.
    """
    model, tokenizer, device = encoder.model, encoder.tokenizer, encoder.device
    num_adv = int(adv_passage_ids.shape[1])
    toks = tokenizer(
        query,
        truncation=True,
        max_length=max_length - num_adv,
        return_tensors="pt",
    ).to(device)

    adv_ids = adv_passage_ids.to(device)
    input_ids = torch.cat([toks["input_ids"], adv_ids], dim=1)
    adv_attn = torch.ones_like(adv_ids, device=device)
    attention_mask = torch.cat([toks["attention_mask"], adv_attn], dim=1)
    return input_ids, attention_mask


def forward_with_adv_suffix(
    encoder: EncoderBundle,
    queries: List[str],
    adv_passage_ids: torch.Tensor,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Run BGE forward on ``query + adv_suffix`` pairs WITH gradients enabled.

    Used inside the HotFlip loop so gradients flow into
    ``encoder.word_embeddings`` at the adversarial-token positions.

    Returns a ``[B, hidden]`` tensor of CLS-pooled, L2-normalized embeddings.
    """
    pieces: List[torch.Tensor] = []
    for q in queries:
        input_ids, attention_mask = tokenize_query_with_adv_suffix(
            encoder, q, adv_passage_ids, max_length=max_length
        )
        cls = encoder.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        pieces.append(_cls_normalize(cls))
    return torch.cat(pieces, dim=0)


def decode_trigger_tokens(
    encoder: EncoderBundle, adv_passage_ids: torch.Tensor
) -> Tuple[List[str], str]:
    """
    Return (token_list, trigger_string) for the current adv passage ids.

    The string is what gets appended to user queries downstream; we strip
    wordpiece continuation markers and filter out special tokens so the
    trigger reads as a plausible suffix.
    """
    tok = encoder.tokenizer
    ids = adv_passage_ids.squeeze(0).tolist()
    tokens = tok.convert_ids_to_tokens(ids)
    filtered = [
        t for t in tokens if t not in ("[MASK]", "[CLS]", "[SEP]", "[PAD]")
    ]
    trigger = tok.convert_tokens_to_string(filtered).strip()
    return tokens, trigger


def initial_adv_passage_ids(
    encoder: EncoderBundle,
    num_adv_passage_tokens: int,
    golden_trigger: Optional[str] = None,
) -> torch.Tensor:
    """
    Build the starting ``[1, num_adv]`` tensor for the optimizer.

    If ``golden_trigger`` is None, every position is initialized to
    ``[MASK]`` (AgentPoison default). Otherwise the string is tokenized
    and (left-)truncated to ``num_adv_passage_tokens``.
    """
    tok = encoder.tokenizer
    device = encoder.device
    if golden_trigger:
        ids = tok(
            golden_trigger,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=num_adv_passage_tokens,
            add_special_tokens=False,
        ).input_ids
        return ids.to(device)
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise ValueError(
            f"Tokenizer for {encoder.model_name} has no mask_token_id; "
            "pass a golden_trigger instead."
        )
    ids = torch.tensor([[mask_id] * num_adv_passage_tokens], device=device, dtype=torch.long)
    return ids
