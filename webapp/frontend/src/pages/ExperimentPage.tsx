import { useEffect, useMemo, useState } from "react";
import type { AppContext } from "../App";
import { api } from "../api/client";
import type {
  ArtifactDetail,
  ExperimentRequest,
  JobSummary,
  QueryFile,
  RunLog,
} from "../types";
import Stepper from "../components/Stepper";
import JobPanel from "../components/JobPanel";
import TriggerCard from "../components/TriggerCard";
import ResultsPanel from "../components/ResultsPanel";

export default function ExperimentPage({ ctx }: { ctx: AppContext }) {
  const {
    defaults,
    corpora,
    artifacts,
    lastExperimentJobId,
    setLastExperimentJobId,
  } = ctx;

  const orch = (defaults?.system_orchestrator as any) ?? {};
  const deb = (defaults?.system_debate as any) ?? {};
  const am = (defaults?.attack_main as any) ?? {};

  const [queryFiles, setQueryFiles] = useState<string[]>([]);
  const [queryFile, setQueryFile] = useState<QueryFile | null>(null);
  const [form, setForm] = useState<ExperimentRequest>({
    system: "orchestrator",
    mode: "attack",
    threat_model: (am.threat_model as any) ?? "targeted",
    poisoned_subagent_ids: am.poisoned_subagent_ids ?? ["subagent_1"],
    attack_id: "attack_001",
    query_file: "data/queries/attack_queries_cybersec.yaml",
    corpus: "cybersec",
    model: orch.model ?? "gpt-4o-mini",
    top_k: orch.top_k ?? 5,
    num_subagents: orch.num_subagents ?? 3,
    max_rounds: deb.max_rounds ?? 4,
    stable_for: deb.stable_for ?? 2,
  });
  const [artifactDetail, setArtifactDetail] = useState<ArtifactDetail | null>(
    null,
  );
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latestRuns, setLatestRuns] = useState<RunLog[]>([]);
  const [sinceTimestamp, setSinceTimestamp] = useState<string | null>(null);

  useEffect(() => {
    api.queryFiles().then(setQueryFiles).catch(() => {});
  }, []);

  useEffect(() => {
    if (!defaults) return;
    setForm((f) => ({
      ...f,
      model:
        f.system === "orchestrator"
          ? (orch.model ?? f.model)
          : (deb.model ?? f.model),
      num_subagents:
        f.system === "orchestrator"
          ? (orch.num_subagents ?? f.num_subagents)
          : (deb.num_subagents ?? f.num_subagents),
      top_k:
        f.system === "orchestrator"
          ? (orch.top_k ?? f.top_k)
          : (deb.subagent_top_k ?? f.top_k),
      max_rounds: deb.max_rounds ?? f.max_rounds,
      stable_for: deb.stable_for ?? f.stable_for,
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaults, form.system]);

  useEffect(() => {
    if (!form.query_file) return;
    api
      .queries(form.query_file)
      .then(setQueryFile)
      .catch(() => setQueryFile(null));
  }, [form.query_file]);

  useEffect(() => {
    if (!form.attack_id) {
      setArtifactDetail(null);
      return;
    }
    api
      .artifact(form.attack_id)
      .then(setArtifactDetail)
      .catch(() => setArtifactDetail(null));
  }, [form.attack_id]);

  useEffect(() => {
    if (!form.attack_id && artifacts[0]) {
      setForm((f) => ({ ...f, attack_id: artifacts[0].attack_id }));
    }
  }, [artifacts, form.attack_id]);

  const ingestDone = useMemo(
    () => corpora.some((c) => c.has_index),
    [corpora],
  );
  const cleanFile =
    form.mode === "clean"
      ? form.query_file.includes("attack_queries")
        ? "data/queries/sample_cybersec_queries.yaml"
        : form.query_file
      : form.query_file;

  const onSubmit = async () => {
    setSubmitting(true);
    setError(null);
    setLatestRuns([]);
    setSinceTimestamp(new Date().toISOString());
    try {
      const body: ExperimentRequest = { ...form, query_file: cleanFile };
      if (body.mode === "clean") {
        body.attack_id = null;
      }
      const job = await api.experiment(body);
      setLastExperimentJobId(job.id);
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const onJobDone = async (j: JobSummary) => {
    if (j.status !== "succeeded") return;
    try {
      const runs = await api.latestRuns({
        limit: 10,
        since: sinceTimestamp ?? undefined,
      });
      setLatestRuns(runs);
    } catch {
      /* ignore */
    }
  };

  const subagentCount = form.num_subagents ?? 3;

  return (
    <div>
      <Stepper
        steps={[
          {
            id: "ingest",
            label: "Ingest corpus",
            path: "/ingest",
            done: ingestDone,
            active: false,
          },
          {
            id: "trigger",
            label: "Optimize trigger",
            path: "/trigger",
            done: artifacts.length > 0,
            active: false,
          },
          {
            id: "run",
            label: "Run experiment",
            path: "/experiment",
            done: latestRuns.length > 0,
            active: true,
            hint: "Clean, single-agent, or global poison",
          },
        ]}
      />

      {!ingestDone && (
        <div className="card bg-amber-50 border-amber-200 px-4 py-3 text-sm text-amber-800 mb-4">
          The retrieval index isn't built yet. Go to <b>Ingest</b> first.
        </div>
      )}

      <div className="grid md:grid-cols-3 gap-5">
        <div className="md:col-span-2 card p-5 space-y-5">
          <div>
            <h2 className="text-base font-semibold">Run experiment</h2>
            <p className="text-sm text-zinc-500 mt-0.5">
              Fire a clean run or a poisoning attack through either the
              orchestrator or the debate pipeline. Results are appended to{" "}
              <span className="mono">results/runs.jsonl</span>.
            </p>
          </div>

          <div className="grid sm:grid-cols-2 gap-3">
            <RadioRow
              label="System"
              value={form.system}
              onChange={(v) =>
                setForm((f) => ({ ...f, system: v as "orchestrator" | "debate" }))
              }
              options={[
                { value: "orchestrator", label: "Orchestrator (LangGraph fan-in)" },
                { value: "debate", label: "Debate (AutoGen round-robin)" },
              ]}
            />
            <RadioRow
              label="Attack"
              value={form.mode === "clean" ? "clean" : form.threat_model}
              onChange={(v) => {
                if (v === "clean") {
                  setForm((f) => ({ ...f, mode: "clean" }));
                } else {
                  setForm((f) => ({
                    ...f,
                    mode: "attack",
                    threat_model: v as "targeted" | "global",
                  }));
                }
              }}
              options={[
                { value: "clean", label: "Clean (no attack)" },
                { value: "targeted", label: "Single-agent poison" },
                { value: "global", label: "Global poison" },
              ]}
            />
          </div>

          {form.mode === "attack" && (
            <>
              <div className="grid sm:grid-cols-2 gap-3">
                <div>
                  <label className="field-label">Attack artifact</label>
                  <select
                    className="select mono"
                    value={form.attack_id ?? ""}
                    onChange={(e) =>
                      setForm((f) => ({
                        ...f,
                        attack_id: e.target.value || null,
                      }))
                    }
                  >
                    {artifacts.length === 0 && (
                      <option value="">— none available —</option>
                    )}
                    {artifacts.map((a) => (
                      <option key={a.attack_id} value={a.attack_id}>
                        {a.attack_id}
                      </option>
                    ))}
                  </select>
                </div>
                {form.threat_model === "targeted" && (
                  <div>
                    <label className="field-label">
                      Poisoned subagents (targeted)
                    </label>
                    <div className="flex flex-wrap gap-2 border border-zinc-300 rounded-lg p-2 bg-white">
                      {Array.from({ length: subagentCount }).map((_, i) => {
                        const id = `subagent_${i + 1}`;
                        const active = form.poisoned_subagent_ids.includes(id);
                        return (
                          <button
                            key={id}
                            type="button"
                            onClick={() =>
                              setForm((f) => ({
                                ...f,
                                poisoned_subagent_ids: active
                                  ? f.poisoned_subagent_ids.filter(
                                      (x) => x !== id,
                                    )
                                  : [...f.poisoned_subagent_ids, id],
                              }))
                            }
                            className={`mono text-xs px-2 py-1 rounded-md border transition ${
                              active
                                ? "bg-accent-600 border-accent-600 text-white"
                                : "bg-zinc-50 border-zinc-200 text-zinc-700 hover:bg-zinc-100"
                            }`}
                          >
                            {id}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </>
          )}

          <div className="grid sm:grid-cols-2 gap-3">
            <div>
              <label className="field-label">Query file</label>
              <select
                className="select mono"
                value={form.query_file}
                onChange={(e) =>
                  setForm((f) => ({ ...f, query_file: e.target.value }))
                }
              >
                {queryFiles.map((q) => (
                  <option key={q} value={q}>
                    {q}
                  </option>
                ))}
              </select>
              {queryFile && (
                <div className="text-xs text-zinc-500 mt-1">
                  {queryFile.queries.length} queries ·{" "}
                  {queryFile.queries.filter((q) => q.has_attack).length} with
                  attack
                </div>
              )}
            </div>
            <div>
              <label className="field-label">Corpus</label>
              <select
                className="select"
                value={form.corpus}
                onChange={(e) =>
                  setForm((f) => ({ ...f, corpus: e.target.value as any }))
                }
              >
                <option value="cybersec">cybersec</option>
                <option value="generic">generic</option>
              </select>
            </div>
          </div>

          <div className="border-t border-zinc-200 pt-4">
            <div className="text-sm font-medium text-zinc-800 mb-2">
              System parameters
            </div>
            <div className="grid sm:grid-cols-2 gap-3">
              <Text
                label="model"
                value={form.model ?? ""}
                onChange={(v) => setForm((f) => ({ ...f, model: v }))}
              />
              <Num
                label="num_subagents"
                value={form.num_subagents ?? 3}
                onChange={(v) =>
                  setForm((f) => ({ ...f, num_subagents: v }))
                }
              />
              <Num
                label={
                  form.system === "orchestrator" ? "top_k" : "subagent_top_k"
                }
                value={form.top_k ?? 5}
                onChange={(v) => setForm((f) => ({ ...f, top_k: v }))}
              />
              {form.system === "debate" && (
                <>
                  <Num
                    label="max_rounds"
                    value={form.max_rounds ?? 4}
                    onChange={(v) =>
                      setForm((f) => ({ ...f, max_rounds: v }))
                    }
                  />
                  <Num
                    label="stable_for"
                    value={form.stable_for ?? 2}
                    onChange={(v) =>
                      setForm((f) => ({ ...f, stable_for: v }))
                    }
                  />
                </>
              )}
            </div>
          </div>

          {error && (
            <div className="text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2">
              {error}
            </div>
          )}
          <div className="flex items-center justify-between gap-2">
            <div className="text-xs text-zinc-500">
              Using{" "}
              <span className="mono">
                {form.system}.{form.mode === "clean" ? "clean" : form.threat_model}
              </span>
            </div>
            <button
              className="btn-primary"
              onClick={onSubmit}
              disabled={submitting || !ingestDone}
            >
              {submitting ? "Submitting…" : "Run experiment"}
            </button>
          </div>
        </div>

        <div className="space-y-3">
          <TriggerCard artifact={artifactDetail} compact />
          {queryFile && (
            <div className="card p-4">
              <div className="text-sm font-semibold mb-2">
                Queries in {form.query_file.split("/").pop()}
              </div>
              <ul className="space-y-1.5 max-h-56 overflow-auto">
                {queryFile.queries.map((q) => (
                  <li
                    key={q.query_id}
                    className="flex items-start justify-between gap-2 text-sm"
                  >
                    <div className="min-w-0">
                      <div className="font-mono text-xs text-zinc-500">
                        {q.query_id}
                      </div>
                      <div className="truncate">{q.query}</div>
                    </div>
                    {q.has_attack && (
                      <span className="chip-warn shrink-0">attack</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      <div className="mt-5">
        <JobPanel jobId={lastExperimentJobId} onDone={onJobDone} />
      </div>

      {latestRuns.length > 0 && (
        <section className="mt-6 space-y-4">
          <h3 className="text-base font-semibold">
            Results ({latestRuns.length})
          </h3>
          {latestRuns.map((run, i) => (
            <ResultsPanel key={`${run.query_id}-${i}`} run={run} />
          ))}
        </section>
      )}
    </div>
  );
}

function RadioRow({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div>
      <label className="field-label">{label}</label>
      <div className="flex flex-col gap-1.5">
        {options.map((o) => (
          <label
            key={o.value}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer text-sm transition ${
              value === o.value
                ? "border-accent-500 bg-accent-50 text-accent-700"
                : "border-zinc-200 hover:bg-zinc-50 text-zinc-700"
            }`}
          >
            <input
              type="radio"
              className="accent-accent-600"
              checked={value === o.value}
              onChange={() => onChange(o.value)}
            />
            {o.label}
          </label>
        ))}
      </div>
    </div>
  );
}

function Text({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="field-label">{label}</label>
      <input
        className="input mono"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

function Num({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label className="field-label">{label}</label>
      <input
        type="number"
        className="input mono"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
