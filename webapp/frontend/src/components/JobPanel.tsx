import { useEffect, useRef, useState } from "react";
import { api, streamJobLogs } from "../api/client";
import type { JobSummary } from "../types";

const STATUS_CHIP: Record<string, string> = {
  queued: "chip-neutral",
  running: "chip-accent",
  succeeded: "chip-success",
  failed: "chip-danger",
  cancelled: "chip-warn",
};

export default function JobPanel({
  jobId,
  onDone,
}: {
  jobId: string | null;
  onDone?: (job: JobSummary) => void;
}) {
  const [job, setJob] = useState<JobSummary | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [elapsed, setElapsed] = useState("00:00");
  const scrollRef = useRef<HTMLDivElement>(null);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  useEffect(() => {
    if (!jobId) return;
    setLines([]);
    setJob(null);
    api.job(jobId).then(setJob).catch(() => {});
    const close = streamJobLogs(
      jobId,
      (line) => setLines((prev) => [...prev, line]),
      () => {},
      () => {
        api
          .job(jobId)
          .then((j) => {
            setJob(j);
            onDoneRef.current?.(j);
          })
          .catch(() => {});
      },
    );
    return () => close();
  }, [jobId]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines]);

  useEffect(() => {
    if (!job || !job.started_at) return;
    if (job.status !== "running") return;
    const start = new Date(job.started_at).getTime();
    const t = setInterval(() => {
      const s = Math.floor((Date.now() - start) / 1000);
      const m = String(Math.floor(s / 60)).padStart(2, "0");
      const sec = String(s % 60).padStart(2, "0");
      setElapsed(`${m}:${sec}`);
    }, 500);
    return () => clearInterval(t);
  }, [job?.started_at, job?.status]);

  if (!jobId) return null;

  return (
    <div className="card overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-zinc-200 bg-zinc-50">
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">Job</span>
          <span className="mono text-xs text-zinc-800">{jobId}</span>
          {job && (
            <span className={STATUS_CHIP[job.status] ?? "chip-neutral"}>
              {job.status}
            </span>
          )}
          {job?.status === "running" && (
            <span className="text-xs text-zinc-500">· {elapsed}</span>
          )}
          {job?.exit_code !== null && job?.exit_code !== undefined && (
            <span className="text-xs text-zinc-500">
              · exit {job.exit_code}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {job?.status === "running" && (
            <button
              className="btn-ghost text-xs"
              onClick={() => api.cancelJob(jobId).catch(() => {})}
            >
              Cancel
            </button>
          )}
        </div>
      </div>
      <div
        ref={scrollRef}
        className="bg-zinc-900 text-zinc-100 mono text-xs px-4 py-3 h-64 overflow-auto whitespace-pre-wrap leading-relaxed"
      >
        {lines.length === 0 ? (
          <span className="text-zinc-500">Waiting for output…</span>
        ) : (
          lines.map((l, i) => <div key={i}>{l}</div>)
        )}
      </div>
      {job?.error && (
        <div className="px-4 py-2 text-xs text-rose-700 bg-rose-50 border-t border-rose-200">
          {job.error}
        </div>
      )}
      {job?.result && (
        <div className="px-4 py-2 text-xs text-emerald-700 bg-emerald-50 border-t border-emerald-200 mono">
          {JSON.stringify(job.result)}
        </div>
      )}
    </div>
  );
}
