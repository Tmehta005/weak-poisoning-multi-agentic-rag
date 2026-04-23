"""
2D PCA scatter of the corpus embedding space showing:
  - Clean corpus chunks
  - GMM cluster centers (attack target: sparse region)
  - A triggered query (query ⊕ trigger)
  - The matching poison document

Usage:
    python src/analysis/plot_embedding_space.py [--attack-id attack_q001] [--output figures/embedding_space.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from attacks.encoder import load_encoder, encode_texts  # noqa: E402

CACHE_DIR = ROOT / "data" / "attacks" / "_cache" / "BAAI__bge-small-en-v1.5"
ATTACKS_DIR = ROOT / "data" / "attacks"
QUERIES_DIR = ROOT / "data" / "queries"

QUERY_YAML = {
    "q": QUERIES_DIR / "attack_queries_cybersec.yaml",
    "b": QUERIES_DIR / "attack_queries_bio_papers.yaml",
}


def _load_query_text(attack_id: str, target_query_id: str) -> str:
    prefix = "q" if attack_id.startswith("attack_q") else "b"
    yaml_path = QUERY_YAML[prefix]
    entries = yaml.safe_load(yaml_path.read_text())
    for entry in entries:
        if entry["query_id"] == target_query_id:
            return entry["query"]
    raise ValueError(f"query_id {target_query_id!r} not found in {yaml_path}")


def main(attack_id: str, output: Path) -> None:
    print(f"Loading cached corpus embeddings from {CACHE_DIR} ...")
    corpus_emb = torch.load(CACHE_DIR / "db_emb.pt", map_location="cpu").numpy()
    gmm_centers = torch.load(CACHE_DIR / "gmm_centers.pt", map_location="cpu").numpy()

    artifact_path = ATTACKS_DIR / attack_id / "artifact.json"
    artifact = json.loads(artifact_path.read_text())
    trigger = artifact["trigger"]
    poison_text = artifact["poison_doc_text"]
    target_query_id = artifact["target_query_ids"][0]

    base_query = _load_query_text(attack_id, target_query_id)
    triggered_query = base_query + " " + trigger

    print("Loading encoder (BAAI/bge-small-en-v1.5) ...")
    encoder = load_encoder()

    print("Encoding triggered query and poison document ...")
    query_emb = encode_texts(encoder, [triggered_query]).cpu().numpy()
    poison_emb = encode_texts(encoder, [poison_text]).cpu().numpy()

    combined = np.concatenate([corpus_emb, gmm_centers, query_emb, poison_emb], axis=0)
    n_corpus = len(corpus_emb)
    n_gmm = len(gmm_centers)

    print("Fitting PCA(n_components=2) ...")
    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)

    p_corpus = proj[:n_corpus]
    p_gmm = proj[n_corpus : n_corpus + n_gmm]
    p_query = proj[n_corpus + n_gmm : n_corpus + n_gmm + 1]
    p_poison = proj[n_corpus + n_gmm + 1 :]

    FONT = 14  # base font size for projection screen readability

    fig, ax = plt.subplots(figsize=(10, 7))

    corpus_label = (
        f"Cybersec corpus (n={n_corpus}, 4 NIST/CISA docs)"
        if attack_id.startswith("attack_q")
        else f"Bio-papers corpus (n={n_corpus})"
    )
    ax.scatter(
        p_corpus[:, 0], p_corpus[:, 1],
        s=22, color="#9ecae1", alpha=0.35, linewidths=0,
        label=corpus_label,
        zorder=2,
    )
    ax.scatter(
        p_gmm[:, 0], p_gmm[:, 1],
        s=320, marker="*", color="#f4a460", edgecolors="#8b4513", linewidths=0.8,
        label=f"GMM cluster centers (k={n_gmm})",
        zorder=5,
    )
    ax.scatter(
        p_query[:, 0], p_query[:, 1],
        s=180, marker="D", color="#7b2d8b", edgecolors="white", linewidths=0.8,
        label="Triggered query",
        zorder=6,
    )
    ax.scatter(
        p_poison[:, 0], p_poison[:, 1],
        s=180, marker="s", color="#d62728", edgecolors="white", linewidths=0.8,
        label="Poison document",
        zorder=6,
    )

    # Annotation arrows — offsets scale with data range so they stay readable
    xmin, xmax = proj[:, 0].min(), proj[:, 0].max()
    ymin, ymax = proj[:, 1].min(), proj[:, 1].max()
    xr = xmax - xmin
    yr = ymax - ymin

    arrow_kw = dict(arrowstyle="-|>", color="#333333", lw=1.3, mutation_scale=16)
    box_base = dict(boxstyle="round,pad=0.35", fc="white", alpha=0.88)

    # Corpus cluster label — anchored to the centroid of the corpus blob
    cx, cy = float(p_corpus[:, 0].mean()), float(p_corpus[:, 1].mean())
    ax.annotate(
        "Normal corpus\n(where clean queries land)",
        xy=(cx, cy),
        xytext=(cx - 0.32 * xr, cy + 0.35 * yr),
        fontsize=FONT - 1,
        ha="center",
        va="bottom",
        arrowprops=dict(**arrow_kw, connectionstyle="arc3,rad=0.15"),
        bbox=dict(**box_base, ec="#9ecae1"),
        zorder=8,
    )

    # Triggered query label
    qx, qy = float(p_query[0, 0]), float(p_query[0, 1])
    ax.annotate(
        "Query + trigger\nlands here",
        xy=(qx, qy),
        xytext=(qx + 0.28 * xr, qy + 0.22 * yr),
        fontsize=FONT - 1,
        ha="left",
        va="bottom",
        arrowprops=dict(**arrow_kw, connectionstyle="arc3,rad=-0.2"),
        bbox=dict(**box_base, ec="#7b2d8b"),
        zorder=8,
    )

    # Poison document label
    px, py = float(p_poison[0, 0]), float(p_poison[0, 1])
    ax.annotate(
        "Poison document\nplaced here by attacker",
        xy=(px, py),
        xytext=(px + 0.28 * xr, py - 0.18 * yr),
        fontsize=FONT - 1,
        ha="left",
        va="top",
        arrowprops=dict(**arrow_kw, connectionstyle="arc3,rad=0.2"),
        bbox=dict(**box_base, ec="#d62728"),
        zorder=8,
    )

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%} var)", fontsize=FONT + 2)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%} var)", fontsize=FONT + 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=FONT, framealpha=0.88)
    ax.set_title("Embedding Space — How the Trigger Works", fontsize=FONT + 2, pad=12)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PCA embedding space for a RAG poisoning attack.")
    parser.add_argument("--attack-id", default="attack_q001", help="Attack directory name (default: attack_q001)")
    parser.add_argument("--output", default="figures/embedding_space.png", help="Output PNG path")
    args = parser.parse_args()
    main(attack_id=args.attack_id, output=Path(args.output))
