# Experiment Console

A small FastAPI + React webapp that wraps the existing `src/experiments/*`
runners behind a light-mode researcher UI. Guides you through the three-step
flow:

1. **Ingest corpus** — build the retrieval index.
2. **Optimize trigger** — produce an AttackArtifact (trigger + poison doc).
3. **Run experiment** — clean / single-agent poison / global poison against
   the orchestrator or the debate pipeline.

All long-running jobs execute as subprocesses; their combined stdout/stderr
is tee'd to `webapp/data/logs/<job_id>.log` and streamed to the UI via SSE.
Past runs are read directly from `results/runs.jsonl`.

## Layout

```
webapp/
  backend/
    main.py               # FastAPI app
    schemas.py            # Request/response DTOs
    jobs/manager.py       # In-memory JobManager + subprocess driver
    runners/              # Thin CLI wrappers used by the JobManager
      ingest.py
      run_clean_orch.py
      run_clean_debate.py
    api/                  # One router per surface
      defaults.py corpora.py queries.py artifacts.py
      ingest.py trigger.py experiments.py jobs.py runs.py
    requirements.txt
  frontend/
    src/
      App.tsx
      api/client.ts
      components/         # Stepper, JobPanel, TriggerCard, ResultsPanel, ...
      pages/              # IngestPage, TriggerPage, ExperimentPage, HistoryPage
    package.json
    tailwind.config.ts
    vite.config.ts
  data/logs/              # Per-job log files (gitignored)
  data/tmp/               # Per-job temporary YAML overrides (gitignored)
```

## Prerequisites

The backend reuses the project venv and calls the existing CLIs under
`src/experiments/`, so the repo-level setup in the top-level
[../README.md](../README.md) must already work (env vars in `.env`, the
cybersec corpus downloaded, etc.).

Install the extra backend packages into the same venv:

```bash
source .venv/bin/activate
pip install -r webapp/backend/requirements.txt
```

Install the frontend toolchain:

```bash
cd webapp/frontend
npm install
```

## Running locally

From the repo root, in two terminals:

```bash
# Terminal 1 — backend (reload so code edits take effect immediately)
source .venv/bin/activate
uvicorn webapp.backend.main:app --reload --port 8000
```

```bash
# Terminal 2 — frontend (Vite dev server on :5173, proxies /api -> :8000)
cd webapp/frontend
npm run dev
```

Then open <http://localhost:5173>. The three steps in the top nav light up
green as prerequisites are met:

- **Ingest** — green once a corpus directory has a persisted index.
- **Trigger** — green once an `AttackArtifact` exists under `data/attacks/`.
- **Run** — green once you have fired an experiment this session.

## What the UI gives you

- **Defaults for every field** are pulled from the YAML configs in
  [../configs/](../configs/) on first load, so you never face an empty form.
  Overrides are kept in page-local React state; they never touch the YAMLs
  on disk. Before submitting, the backend writes a per-job temporary YAML
  under `webapp/data/tmp/` so runs stay reproducible.
- **Live log streaming** in each page's `JobPanel`. The panel shows job id,
  status chip, elapsed wallclock, and a terminal-style output pane. You can
  cancel a running job from the panel.
- **Trigger panel** on the Trigger and Experiment pages shows the selected
  artifact's trigger string, target claim, harmful match phrases, and an
  expandable poison document preview.
- **Results** appear inline after an experiment finishes: per-agent cards
  with answer, confidence bar, citations, and an expandable rationale /
  retrieved doc list. For debate runs you also get a round-by-round
  transcript with per-agent stance, confidence, and message.
- **History page** browses `results/runs.jsonl` with filters on
  `query_id` / `attack_condition`, click-through to the same
  `ResultsPanel` view.

## How jobs map to CLIs

| UI action                  | Subprocess invoked                                   |
|----------------------------|------------------------------------------------------|
| Ingest                     | `python -m webapp.backend.runners.ingest`            |
| Optimize trigger           | `python -m src.experiments.optimize_trigger`         |
| Run, orchestrator, clean   | `python -m webapp.backend.runners.run_clean_orch`    |
| Run, orchestrator, attack  | `python -m src.experiments.run_attack_orch`          |
| Run, debate, clean         | `python -m webapp.backend.runners.run_clean_debate`  |
| Run, debate, attack        | `python -m src.experiments.run_attack_debate`        |

Single-agent vs global poison maps 1:1 to the existing
`threat_model: targeted | global` switch on both attack runners.

## Notes

- Only **one job per kind** (ingest / trigger / experiment) can run at once;
  the UI disables the submit button while another job of the same kind is
  active, and the backend returns `409` if called anyway.
- A restart of the backend marks any previously-running jobs as `failed` so
  the registry stays consistent.
- No auth; this is a localhost tool for the research team.
- To wipe state: delete `webapp/data/logs/`, `webapp/data/tmp/`, and
  `webapp/data/jobs.jsonl`.
