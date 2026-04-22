"""
In-process job manager that runs the existing experiment CLIs as subprocesses.

Each job streams combined stdout/stderr into a per-job log file under
``webapp/data/logs/<job_id>.log``. Status transitions (queued -> running ->
succeeded|failed|cancelled) plus timestamps are persisted to an append-only
``webapp/data/jobs.jsonl`` so the UI can survive a backend restart.

The manager is intentionally simple:
  * one ``threading.Thread`` per job
  * a single ``threading.Lock`` protecting the in-memory registry
  * ``subprocess.Popen`` with ``stdout=PIPE`` / ``stderr=STDOUT`` so the UI
    sees a single merged stream in order
  * a short sentinel JSON line of the form ``__RESULT__ {...}`` at the end
    of the subprocess output lets us capture structured results (e.g. the
    number of nodes built by the ingest runner) without parsing logs.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

JOB_KINDS = ("ingest", "optimize_trigger", "experiment")

_RESULT_SENTINEL = "__RESULT__"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    kind: str
    params: Dict[str, Any]
    cmd: List[str]
    log_path: str
    status: str = "queued"  # queued | running | succeeded | failed | cancelled
    created_at: str = field(default_factory=_utcnow)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    exit_code: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JobManager:
    """Singleton job manager. Use :func:`get_manager` to retrieve it."""

    def __init__(self, log_dir: str, registry_path: str, repo_root: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.repo_root = Path(repo_root)
        self._jobs: Dict[str, Job] = {}
        self._procs: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._load_registry()

    def _load_registry(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            with open(self.registry_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    jid = rec.get("id")
                    if not jid:
                        continue
                    if rec.get("status") in ("queued", "running"):
                        rec["status"] = "failed"
                        rec["error"] = rec.get("error") or "Backend restarted while job was active."
                        rec["ended_at"] = rec.get("ended_at") or _utcnow()
                    self._jobs[jid] = Job(**rec)
        except Exception:
            pass

    def _append_registry(self, job: Job) -> None:
        with open(self.registry_path, "a") as f:
            f.write(json.dumps(job.to_dict()) + "\n")

    def submit(self, kind: str, cmd: List[str], params: Dict[str, Any]) -> Job:
        if kind not in JOB_KINDS:
            raise ValueError(f"unknown job kind: {kind}")
        job_id = uuid.uuid4().hex[:12]
        log_path = self.log_dir / f"{job_id}.log"
        log_path.touch()
        job = Job(id=job_id, kind=kind, params=params, cmd=cmd, log_path=str(log_path))
        with self._lock:
            self._jobs[job_id] = job
        self._append_registry(job)
        thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
        thread.start()
        return job

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
        job.status = "running"
        job.started_at = _utcnow()
        self._append_registry(job)

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{self.repo_root}{os.pathsep}{existing}" if existing else str(self.repo_root)
        )
        env["PYTHONUNBUFFERED"] = "1"

        result_payload: Optional[Dict[str, Any]] = None
        try:
            with open(job.log_path, "a", buffering=1) as logf:
                logf.write(f"[webapp] $ {' '.join(shlex.quote(c) for c in job.cmd)}\n")
                logf.write(f"[webapp] cwd={self.repo_root}\n\n")
                logf.flush()
                proc = subprocess.Popen(
                    job.cmd,
                    cwd=str(self.repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    bufsize=1,
                    text=True,
                )
                with self._lock:
                    self._procs[job_id] = proc

                assert proc.stdout is not None
                for line in proc.stdout:
                    if line.startswith(_RESULT_SENTINEL):
                        try:
                            result_payload = json.loads(line[len(_RESULT_SENTINEL):].strip())
                        except json.JSONDecodeError:
                            pass
                        continue
                    logf.write(line)

                proc.wait()
                exit_code = proc.returncode

            job.ended_at = _utcnow()
            job.exit_code = exit_code
            job.result = result_payload
            if exit_code == 0:
                job.status = "succeeded"
            else:
                if job.status != "cancelled":
                    job.status = "failed"
                    job.error = f"Process exited with code {exit_code}"
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.ended_at = _utcnow()
        finally:
            with self._lock:
                self._procs.pop(job_id, None)
            self._append_registry(job)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(job_id)
            job = self._jobs.get(job_id)
        if not job or job.status not in ("queued", "running"):
            return False
        job.status = "cancelled"
        if proc and proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
        return True

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, kind: Optional[str] = None, limit: int = 100) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if kind:
            jobs = [j for j in jobs if j.kind == kind]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def has_running(self, kind: str) -> bool:
        with self._lock:
            return any(
                j.kind == kind and j.status in ("queued", "running")
                for j in self._jobs.values()
            )

    def tail(self, job_id: str) -> Iterable[str]:
        """
        Generator that tails a job's log file line-by-line. Yields lines as
        they appear and terminates once the job reaches a terminal status
        and the log file has been fully drained.
        """
        job = self.get(job_id)
        if not job:
            return
        path = Path(job.log_path)
        path.touch()
        f = open(path, "r")
        try:
            while True:
                line = f.readline()
                if line:
                    yield line
                    continue
                current = self.get(job_id)
                if current and current.status in ("succeeded", "failed", "cancelled"):
                    remainder = f.read()
                    if remainder:
                        for rline in remainder.splitlines(keepends=True):
                            yield rline
                    return
                time.sleep(0.25)
        finally:
            f.close()


_manager: Optional[JobManager] = None
_manager_lock = threading.Lock()


def get_manager() -> JobManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            repo_root = Path(__file__).resolve().parents[3]
            _manager = JobManager(
                log_dir=str(repo_root / "webapp" / "data" / "logs"),
                registry_path=str(repo_root / "webapp" / "data" / "jobs.jsonl"),
                repo_root=str(repo_root),
            )
        return _manager
