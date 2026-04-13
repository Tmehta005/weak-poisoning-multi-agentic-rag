"""
Visualize the Phase 1 RAG pipeline architecture.
Saves to results/pipeline_phase1.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

os.makedirs("results", exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

C_TITLE   = "#ffffff"
C_SUB     = "#8890b0"
C_DATA    = "#2d4a6e"
C_TOOL    = "#2d5a3d"
C_CODE    = "#5a3060"
C_OUTPUT  = "#6e4a1e"
C_BORDER  = "#3a3f5c"
ACCENT = {"data": "#4a8fd4", "tool": "#4aad6a", "code": "#a060d0", "output": "#d4844a"}


def box(ax, x, y, w, h, label, sublabel="", color=C_DATA, accent="#4a8fd4"):
    fancy = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                            linewidth=1.8, edgecolor=accent, facecolor=color, zorder=3)
    ax.add_patch(fancy)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w/2, cy + 0.13, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=C_TITLE, zorder=4)
        ax.text(x + w/2, cy - 0.22, sublabel, ha="center", va="center",
                fontsize=7.2, color=C_SUB, zorder=4)
    else:
        ax.text(x + w/2, cy, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=C_TITLE, zorder=4)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#5a6080", lw=1.6, mutation_scale=14),
                zorder=2)


Y1, Y2, Y3, Y4, BH, BW = 6.5, 4.7, 2.9, 1.1, 0.95, 2.2

ax.text(7, 8.6, "Phase 1 — Baseline RAG Pipeline", ha="center", va="center",
        fontsize=14, fontweight="bold", color=C_TITLE)
ax.text(7, 8.25, "ml-rag-poisoning  •  clean baseline (no attack)", ha="center",
        va="center", fontsize=8.5, color=C_SUB)

box(ax, 0.4,  Y1, BW, BH, "Corpus", "data/corpus/*.txt",        color=C_DATA,   accent=ACCENT["data"])
box(ax, 3.1,  Y1, BW, BH, "ingestion.yaml", "chunk · overlap · embed", color=C_DATA, accent=ACCENT["data"])
box(ax, 5.8,  Y1, BW, BH, "baseline_rag.txt", "prompts/",       color=C_CODE,   accent=ACCENT["code"])
box(ax, 8.5,  Y1, BW, BH, "OpenAI", "gpt-4o-mini · embeddings", color=C_TOOL,   accent=ACCENT["tool"])
box(ax, 11.2, Y1, BW, BH, "Langfuse", "tracing · scores",       color=C_TOOL,   accent=ACCENT["tool"])

box(ax, 1.8, Y2, 3.6, BH, "ingest_corpus()", "LlamaIndex · SentenceSplitter · VectorStoreIndex\nsrc/ingestion.py", color=C_TOOL, accent=ACCENT["tool"])
box(ax, 6.5, Y2, 2.8, BH, "Vector Index", "data/index/  (persisted)", color=C_DATA, accent=ACCENT["data"])

box(ax, 1.8, Y3, 3.6, BH, "Retriever.retrieve(query)", "top-k cosine similarity\nsrc/retriever.py", color=C_CODE, accent=ACCENT["code"])
box(ax, 6.5, Y3, 2.8, BH, "RetrievedDoc[ ]", "doc_id · text · score\nsrc/schemas.py", color=C_CODE, accent=ACCENT["code"])

box(ax, 0.4,  Y4, 2.8, BH, "BaselineRAG.run()", "retrieve → prompt → generate\nsrc/baseline_rag.py", color=C_CODE, accent=ACCENT["code"])
box(ax, 3.8,  Y4, 2.5, BH, "SubagentOutput", "answer · citations\nconfidence · rationale", color=C_CODE, accent=ACCENT["code"])
box(ax, 6.9,  Y4, 2.5, BH, "OrchestratorOutput", "final_answer\nharmful_action_flag", color=C_CODE, accent=ACCENT["code"])
box(ax, 10.0, Y4, 3.2, BH, "RunLog → runs.jsonl", "all required fields\nsrc/logging_utils.py", color=C_OUTPUT, accent=ACCENT["output"])

arrow(ax, 1.5, Y1, 2.8, Y2 + BH)
arrow(ax, 4.2, Y1, 3.8, Y2 + BH)
arrow(ax, 5.4, Y2 + BH/2, 6.5, Y2 + BH/2)
arrow(ax, 7.9, Y2, 7.9, Y3 + BH)
arrow(ax, 5.4, Y3 + BH/2, 6.5, Y3 + BH/2)
arrow(ax, 6.5, Y3, 1.8, Y4 + BH)
arrow(ax, 6.9, Y1, 1.8, Y4 + BH)
arrow(ax, 9.6, Y1, 3.2, Y4 + BH)
arrow(ax, 3.2, Y4 + BH/2, 3.8, Y4 + BH/2)
arrow(ax, 6.3, Y4 + BH/2, 6.9, Y4 + BH/2)
arrow(ax, 9.4, Y4 + BH/2, 10.0, Y4 + BH/2)
arrow(ax, 12.3, Y1, 11.6, Y4 + BH)

legend_items = [
    mpatches.Patch(facecolor=C_DATA,   edgecolor=ACCENT["data"],   label="Data / Storage"),
    mpatches.Patch(facecolor=C_TOOL,   edgecolor=ACCENT["tool"],   label="Open-source tool"),
    mpatches.Patch(facecolor=C_CODE,   edgecolor=ACCENT["code"],   label="Custom code (src/)"),
    mpatches.Patch(facecolor=C_OUTPUT, edgecolor=ACCENT["output"], label="Output / Log"),
]
ax.legend(handles=legend_items, loc="lower left", bbox_to_anchor=(0.01, 0.01),
          framealpha=0.25, facecolor="#1e2130", edgecolor=C_BORDER,
          labelcolor="#c8cce8", fontsize=8)

for y, lbl in [(Y1+BH/2, "INPUTS"), (Y2+BH/2, "INGESTION"),
               (Y3+BH/2, "RETRIEVAL"), (Y4+BH/2, "GENERATION\n& LOGGING")]:
    ax.text(0.18, y, lbl, ha="center", va="center", fontsize=6.5,
            color=C_SUB, rotation=90, fontweight="bold")

plt.tight_layout()
plt.savefig("results/pipeline_phase1.png", dpi=160, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: results/pipeline_phase1.png")
