# RAG Poisoning

Research project studying **weak poisoning attacks on multi-agent RAG systems**.

The core question: can poisoning a single expert subagent reliably sway the final decision of a multi-agent RAG pipeline?

---

## Project structure

```
rag-poisoning/
├── src/
│   ├── schemas.py          # Typed Pydantic schemas for all pipeline I/O
│   ├── ingestion.py        # Corpus ingestion + chunking (LlamaIndex)
│   ├── retriever.py        # Retrieval wrapper → RetrievedDoc[]
│   ├── baseline_rag.py     # Baseline RAG pipeline with Langfuse tracing
│   └── logging_utils.py    # Appends every run to results/runs.jsonl
├── prompts/
│   └── baseline_rag.txt    # Prompt template for baseline generation
├── configs/
│   ├── ingestion.yaml      # Chunk size, overlap, embedding model
│   ├── system_orchestrator.yaml
│   ├── attack_main_injection.yaml
│   └── attack_poisonedrag_baseline.yaml
├── tests/
│   ├── test_schemas.py     # Schema correctness tests
│   └── test_retriever.py   # Retrieval smoke tests
├── data/
│   └── corpus/             # Source documents (.txt files)
└── results/
    └── runs.jsonl          # Append-only run log (one JSON line per run)
```

---

## Phase 1 — Baseline RAG (complete)

A clean single-agent RAG baseline: retrieve relevant documents, generate an answer, log everything.

### Pipeline

```
Corpus (data/corpus/)
    │
    ▼
ingest_corpus()          ← LlamaIndex SentenceSplitter + VectorStoreIndex
    │                       persisted to data/index/
    ▼
Retriever.retrieve()     ← cosine similarity, returns RetrievedDoc[]
    │
    ▼
BaselineRAG.run()        ← fills prompt template, calls OpenAI
    │
    ├──► SubagentOutput + OrchestratorOutput   (typed schemas)
    ├──► Langfuse trace                         (retrieval + generation spans)
    └──► results/runs.jsonl                     (append-only run log)
```

Every run logs: `query_id`, `attack_condition`, `trigger`, `retrieved_doc_ids_per_agent`,
`poison_retrieved`, `agent_responses`, `final_decision`, `metrics`.

---

## Setup

### Requirements

- Python 3.10+
- OpenAI API key (generation)
- Langfuse keys (optional — tracing)

### Install

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install llama-index-core llama-index-embeddings-huggingface \
            langfuse pydantic pyyaml openai pytest matplotlib
```

### Configure API keys

Create a `.env` file at the project root (never committed):

```
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Load before running:

```bash
export $(grep -v '^#' .env | xargs)
```

---

## Running Phase 1

### 1. Run the tests (no API key needed)

```bash
python -m pytest tests/ -v
```

### 2. Create a corpus

Add `.txt` files to `data/corpus/`. A toy corpus to get started:

```bash
mkdir -p data/corpus
echo "The capital of France is Paris." > data/corpus/doc1.txt
echo "Berlin is the capital of Germany." > data/corpus/doc2.txt
echo "Tokyo is the capital of Japan." > data/corpus/doc3.txt
```

### 3. Build the index (first run only — persisted to `data/index/`)

```bash
python - <<'EOF'
from src.ingestion import ingest_corpus
ingest_corpus("data/corpus", persist_dir="data/index")
print("Index built.")
EOF
```

### 4. Run baseline RAG

```bash
python - <<'EOF'
from src.ingestion import ingest_corpus
from src.baseline_rag import BaselineRAG

index = ingest_corpus("data/corpus", persist_dir="data/index")
rag = BaselineRAG(index, top_k=3)

queries = [
    ("q001", "What is the capital of France?"),
    ("q002", "What is the capital of Germany?"),
    ("q003", "What is the capital of Japan?"),
]
for qid, query in queries:
    log = rag.run(query=query, query_id=qid)
    print(f"[{qid}] {log.final_decision.final_answer}")
EOF
```

### 5. Inspect the run log

```bash
python - <<'EOF'
import json
with open("results/runs.jsonl") as f:
    for line in f:
        r = json.loads(line)
        print(r["query_id"], "|", r["attack_condition"], "|",
              r["final_decision"]["final_answer"])
EOF
```

### 6. Visualise the pipeline

```bash
python visualize_pipeline.py
# output: results/pipeline_phase1.png
```

---

## Configuration

**`configs/ingestion.yaml`**

```yaml
chunk_size: 512       # tokens per chunk
chunk_overlap: 64     # overlap between chunks
similarity_top_k: 5   # default retrieval depth
embed_model: "local"  # "local" = BAAI/bge-small-en-v1.5 (no API key)
                      # "openai" = text-embedding-ada-002
```

To rebuild the index after changing config, delete `data/index/` first.

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Baseline RAG — ingestion, retrieval, generation, logging | Complete |
| 2 | Multi-agent orchestrator — 3 subagents + LangGraph aggregation | Planned |
| 3 | Poisoning attack — trigger, D_p injection, attacked retrieval path | Planned |
| 4 | Evaluation — ASR, False-Consensus Rate, Poison Dominance, Benign Accuracy | Planned |
| 5 | Debate setup — AutoGen multi-round + judge | Planned |

---

## Tech stack

| Role | Tool |
|---|---|
| Orchestrator control flow | LangGraph |
| Ingestion / retrieval | LlamaIndex |
| Debate setup | AutoGen |
| Tracing / experiment scores | Langfuse |
| Generation | OpenAI |
