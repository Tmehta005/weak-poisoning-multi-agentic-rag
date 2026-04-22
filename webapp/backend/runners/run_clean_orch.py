"""
CLI wrapper around :func:`src.experiments.run_clean.run_clean_experiment`.

The original function is Python-only; this wrapper adds an argparse surface
that mirrors :mod:`src.experiments.run_attack_orch` so the webapp's experiment
endpoint can invoke either via subprocess.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.corpus.query_loader import load_queries
from src.experiments.run_clean import run_clean_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the clean orchestrator experiment.")
    parser.add_argument(
        "--query-file",
        default="data/queries/sample_cybersec_queries.yaml",
    )
    parser.add_argument(
        "--system-config",
        default="configs/system_orchestrator.yaml",
    )
    parser.add_argument(
        "--ingestion-config",
        default="configs/corpus_cybersec.yaml",
    )
    parser.add_argument(
        "--corpus",
        choices=("cybersec", "generic"),
        default="cybersec",
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args(argv)

    if args.corpus == "cybersec":
        data_dir = "data/corpus_cybersec"
        persist_dir = "data/index_cybersec"
    else:
        data_dir = "data/corpus"
        persist_dir = "data/index"

    queries = load_queries(args.query_file)
    print(f"[run_clean_orch] loaded {len(queries)} queries from {args.query_file}", flush=True)

    logs = run_clean_experiment(
        queries=queries,
        data_dir=data_dir,
        persist_dir=persist_dir,
        output_dir=args.output_dir,
        ingestion_config_path=args.ingestion_config,
        system_config_path=args.system_config,
    )

    print(f"[run_clean_orch] ran {len(logs)} queries", flush=True)
    for log in logs:
        fd = log.final_decision
        answer = (fd.final_answer if fd else "<no decision>")[:80]
        conf = fd.final_confidence if fd else None
        print(f"  {log.query_id} conf={conf} answer={answer!r}", flush=True)

    print(
        "__RESULT__ "
        + json.dumps(
            {
                "num_runs": len(logs),
                "query_ids": [log.query_id for log in logs],
                "attack_condition": "clean",
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
