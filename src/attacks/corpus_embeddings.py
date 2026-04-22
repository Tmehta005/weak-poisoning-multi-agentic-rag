"""
Corpus embeddings + Gaussian-mixture cluster centers for the fitness loss.

The AgentPoison "ap" fitness function ``compute_avg_cluster_distance``
measures how far a triggered-query embedding lies from GMM cluster
centers fit to the benign corpus. We cache:

    data/attacks/_cache/<model-slug>/db_emb.pt        # [N, hidden]
    data/attacks/_cache/<model-slug>/gmm_centers.pt   # [k, hidden]
    data/attacks/_cache/<model-slug>/meta.json        # fingerprint

so repeated optimizer runs don't re-embed the corpus.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from sklearn.mixture import GaussianMixture

from src.attacks.encoder import EncoderBundle, encode_texts


def _slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def _cache_dir(base_dir: str, model_name: str) -> Path:
    return Path(base_dir) / _slug(model_name)


def _corpus_fingerprint(texts: List[str]) -> str:
    h = hashlib.sha256()
    h.update(str(len(texts)).encode())
    for t in texts[:32]:
        h.update(b"\0")
        h.update(t[:512].encode("utf-8", errors="replace"))
    for t in texts[-32:]:
        h.update(b"\0")
        h.update(t[:512].encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


def extract_corpus_texts(index) -> List[str]:
    """
    Pull chunk texts out of a LlamaIndex ``VectorStoreIndex``.

    Reads from the docstore so filtered-out nodes from ingestion are
    already excluded.
    """
    docs = list(index.docstore.docs.values())
    texts: List[str] = []
    for d in docs:
        try:
            texts.append(d.get_content())
        except Exception:
            texts.append(getattr(d, "text", "") or "")
    return [t for t in texts if t]


def encode_corpus(
    encoder: EncoderBundle,
    texts: List[str],
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode the entire corpus to a single ``[N, hidden]`` tensor."""
    return encode_texts(encoder, texts, batch_size=batch_size)


def fit_cluster_centers(
    db_embeddings: torch.Tensor,
    n_components: int = 5,
    random_state: int = 0,
) -> torch.Tensor:
    """
    Fit a GaussianMixture on ``db_embeddings`` and return its means.

    Matches AgentPoison ``trigger_optimization.py`` line 509.
    """
    arr = db_embeddings.detach().cpu().numpy()
    n = max(1, min(n_components, len(arr)))
    gmm = GaussianMixture(
        n_components=n, covariance_type="full", random_state=random_state
    )
    gmm.fit(arr)
    return torch.tensor(gmm.means_, dtype=db_embeddings.dtype)


def build_or_load_corpus_cache(
    encoder: EncoderBundle,
    texts: List[str],
    cache_base_dir: str = "data/attacks/_cache",
    n_components: int = 5,
    batch_size: int = 32,
    force_recompute: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Path]:
    """
    Load cached db_embeddings + GMM centers if the fingerprint matches,
    otherwise re-encode and re-fit.

    Returns:
        (db_embeddings, cluster_centers, cache_dir) all on encoder.device.
    """
    cache_dir = _cache_dir(cache_base_dir, encoder.model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    emb_path = cache_dir / "db_emb.pt"
    cen_path = cache_dir / "gmm_centers.pt"

    fingerprint = _corpus_fingerprint(texts)

    if (
        not force_recompute
        and meta_path.exists()
        and emb_path.exists()
        and cen_path.exists()
    ):
        try:
            meta = json.loads(meta_path.read_text())
            if (
                meta.get("fingerprint") == fingerprint
                and meta.get("n_components") == n_components
                and meta.get("model_name") == encoder.model_name
                and meta.get("num_texts") == len(texts)
            ):
                db = torch.load(emb_path, map_location=encoder.device)
                centers = torch.load(cen_path, map_location=encoder.device)
                return db, centers, cache_dir
        except Exception:
            pass  # fall through to recompute

    db = encode_corpus(encoder, texts, batch_size=batch_size).to(encoder.device)
    centers = fit_cluster_centers(db, n_components=n_components).to(encoder.device)

    torch.save(db.cpu(), emb_path)
    torch.save(centers.cpu(), cen_path)
    meta_path.write_text(
        json.dumps(
            {
                "model_name": encoder.model_name,
                "num_texts": len(texts),
                "n_components": n_components,
                "fingerprint": fingerprint,
                "embedding_dim": encoder.embedding_dim,
            },
            indent=2,
        )
    )
    return db, centers, cache_dir
