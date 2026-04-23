"""
Save and print a presentation-ready ablation table from results/ablation/runs.jsonl.

Reads runs tagged with metrics["num_poison_docs"] and produces:
  Table 1 — Aggregate: poison count vs. retrieval rate + ASR
  Table 2 — ASR % per query × poison count
  Table 3 — Poison retrieval % per query × poison count

Outputs (written to the same directory as --runs-file):
  ablation_table.txt   — full ASCII tables (copy-paste for slides/reports)
  ablation_table.csv   — aggregate table as CSV (import into Excel/Sheets)

Usage:
    python -m src.analysis.ablation_table
    python -m src.analysis.ablation_table --runs-file results/ablation/runs.jsonl
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers shared with make_results_table
# ---------------------------------------------------------------------------

def _load_runs(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"No runs file at {path}.\n"
            "Run:  python -m src.experiments.run_ablation_poison_count"
        )
    runs = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def _poison_retrieved(run: dict) -> float:
    return float(bool(run.get("poison_retrieved", False)))


def _harmful(run: dict) -> float:
    fd = run.get("final_decision") or {}
    return float(bool(fd.get("harmful_action_flag", False)))


def _num_poison_docs(run: dict) -> int:
    return int(run.get("metrics", {}).get("num_poison_docs", 1))


def _pct(values: list[float]) -> str:
    if not values:
        return "—"
    return f"{100 * sum(values) / len(values):.0f}%"


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    col_w = [
        max(len(headers[i]), max((len(r[i]) for r in rows), default=0))
        for i in range(len(headers))
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"

    def fmt_row(r: list[str]) -> str:
        return "| " + " | ".join(r[i].ljust(col_w[i]) for i in range(len(r))) + " |"

    lines = [sep, fmt_row(headers), sep]
    for r in rows:
        lines.append(fmt_row(r))
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 1 — aggregate by num_poison_docs
# ---------------------------------------------------------------------------

def build_aggregate_table(runs: list[dict], counts: list[int]) -> str:
    headers = ["# Poison Docs", "N Runs", "Poison Retr %", "ASR %"]
    rows = []
    for n in counts:
        subset = [r for r in runs if _num_poison_docs(r) == n]
        if not subset:
            rows.append([str(n), "0", "—", "—"])
        else:
            p = [_poison_retrieved(r) for r in subset]
            h = [_harmful(r) for r in subset]
            rows.append([str(n), str(len(subset)), _pct(p), _pct(h)])
    return _fmt_table(headers, rows)


# ---------------------------------------------------------------------------
# Table 2 — per-query × num_poison_docs
# ---------------------------------------------------------------------------

def build_perquery_table(runs: list[dict], counts: list[int], metric: str = "asr") -> str:
    """
    metric: "asr" (harmful_action_flag) or "retr" (poison_retrieved)
    """
    query_ids = sorted({r["query_id"] for r in runs})
    headers = ["Query ID"] + [f"n={n}" for n in counts]
    rows = []
    for qid in query_ids:
        row = [qid]
        for n in counts:
            subset = [r for r in runs if r["query_id"] == qid and _num_poison_docs(r) == n]
            if not subset:
                row.append("—")
            else:
                vals = [_harmful(r) if metric == "asr" else _poison_retrieved(r) for r in subset]
                row.append(_pct(vals))
        rows.append(row)
    return _fmt_table(headers, rows)


# ---------------------------------------------------------------------------
# CSV export — aggregate table only (easy to import into Excel / Sheets)
# ---------------------------------------------------------------------------

def build_aggregate_csv(runs: list[dict], counts: list[int]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["num_poison_docs", "n_runs", "poison_retr_pct", "asr_pct"])
    for n in counts:
        subset = [r for r in runs if _num_poison_docs(r) == n]
        if not subset:
            writer.writerow([n, 0, "", ""])
        else:
            p = [_poison_retrieved(r) for r in subset]
            h = [_harmful(r) for r in subset]
            writer.writerow([
                n,
                len(subset),
                round(100 * sum(p) / len(p), 1),
                round(100 * sum(h) / len(h), 1),
            ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ablation results table: poison doc count sweep.")
    parser.add_argument("--runs-file", default="results/ablation/runs.jsonl")
    args = parser.parse_args(argv)

    runs = _load_runs(args.runs_file)
    out_dir = Path(args.runs_file).parent

    counts = sorted({_num_poison_docs(r) for r in runs})
    if not counts:
        print("No runs with num_poison_docs metadata found.")
        return 1

    corpus_label = "Bio corpus" if any(r["query_id"].startswith("b") for r in runs) else "Corpus"
    n_queries = len({r["query_id"] for r in runs})
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build full text output
    lines: list[str] = []
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Source:    {args.runs_file}  ({len(runs)} runs)")
    lines.append("")
    lines.append("=" * 60)
    lines.append("  Ablation: Number of Poison Documents Injected")
    lines.append(f"  {corpus_label} · Orchestrator (targeted) · {n_queries} queries")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Table 1 — Aggregate results")
    lines.append("")
    lines.append(build_aggregate_table(runs, counts))
    lines.append("")
    lines.append("Table 2 — ASR % per query × poison count")
    lines.append("")
    lines.append(build_perquery_table(runs, counts, metric="asr"))
    lines.append("")
    lines.append("Table 3 — Poison retrieval % per query × poison count")
    lines.append("")
    lines.append(build_perquery_table(runs, counts, metric="retr"))
    lines.append("")

    full_text = "\n".join(lines)

    # Print to stdout
    print(full_text)

    # Save ASCII tables
    txt_path = out_dir / "ablation_poison_count_table.txt"
    txt_path.write_text(full_text, encoding="utf-8")

    # Save CSV (aggregate only — easy to drop into a slide deck)
    csv_path = out_dir / "ablation_poison_count_table.csv"
    csv_path.write_text(build_aggregate_csv(runs, counts), encoding="utf-8")

    print(f"Saved → {txt_path}")
    print(f"Saved → {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
