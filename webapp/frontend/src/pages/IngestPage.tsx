import { useEffect, useMemo, useState } from "react";
import type { AppContext } from "../App";
import { api } from "../api/client";
import type { IngestRequest, JobSummary } from "../types";
import Stepper from "../components/Stepper";
import JobPanel from "../components/JobPanel";

export default function IngestPage({ ctx }: { ctx: AppContext }) {
  const { defaults, corpora, refreshCorpora, lastIngestJobId, setLastIngestJobId } =
    ctx;
  const cy = (defaults?.corpus_cybersec as any) ?? {};
  const gen = (defaults?.ingestion as any) ?? {};

  const [selectedCorpus, setSelectedCorpus] = useState<string>("__custom__");
  const [form, setForm] = useState<IngestRequest>({
    data_dir: "data/corpus_cybersec",
    persist_dir: "data/index_cybersec",
    chunk_size: cy.chunk_size ?? 384,
    chunk_overlap: cy.chunk_overlap ?? 64,
    embed_model: cy.embed_model ?? "local",
    similarity_top_k: cy.similarity_top_k ?? 5,
    variant: "auto",
    rebuild: false,
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!defaults) return;
    setForm((f) => ({
      ...f,
      chunk_size: cy.chunk_size ?? gen.chunk_size ?? f.chunk_size,
      chunk_overlap: cy.chunk_overlap ?? gen.chunk_overlap ?? f.chunk_overlap,
      embed_model: cy.embed_model ?? gen.embed_model ?? f.embed_model,
      similarity_top_k:
        cy.similarity_top_k ?? gen.similarity_top_k ?? f.similarity_top_k,
      data_dir: cy.data_dir ?? f.data_dir,
      persist_dir: cy.persist_dir ?? f.persist_dir,
    }));
    if (corpora.length > 0 && selectedCorpus === "__custom__") {
      const def =
        corpora.find((c) => c.name === "corpus_cybersec") ?? corpora[0];
      setSelectedCorpus(def.name);
      setForm((f) => ({
        ...f,
        data_dir: def.data_dir,
        persist_dir: def.suggested_persist_dir,
      }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaults, corpora.length]);

  const selected = useMemo(
    () => corpora.find((c) => c.name === selectedCorpus),
    [corpora, selectedCorpus],
  );

  const onSubmit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const job = await api.ingest(form);
      setLastIngestJobId(job.id);
    } catch (e) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  const onJobDone = (j: JobSummary) => {
    if (j.status === "succeeded") refreshCorpora();
  };

  return (
    <div>
      <Stepper
        steps={[
          {
            id: "ingest",
            label: "Ingest corpus",
            path: "/ingest",
            done: corpora.some((c) => c.has_index),
            active: true,
            hint: "Build the retrieval index",
          },
          {
            id: "trigger",
            label: "Optimize trigger",
            path: "/trigger",
            done: ctx.artifacts.length > 0,
            active: false,
            hint: "Produce poison doc + trigger",
          },
          {
            id: "run",
            label: "Run experiment",
            path: "/experiment",
            done: false,
            active: false,
            hint: "Clean, single-agent, or global poison",
          },
        ]}
      />
      <div className="grid md:grid-cols-3 gap-5">
        <div className="md:col-span-2 card p-5 space-y-5">
          <div>
            <h2 className="text-base font-semibold">Corpus ingestion</h2>
            <p className="text-sm text-zinc-500 mt-0.5">
              Parse source documents, chunk them, and persist a LlamaIndex
              vector store under <span className="mono">persist_dir</span>.
              Safe to re-run; existing indices are reloaded.
            </p>
          </div>

          <div>
            <label className="field-label">Corpus</label>
            <select
              className="select"
              value={selectedCorpus}
              onChange={(e) => {
                const v = e.target.value;
                setSelectedCorpus(v);
                if (v !== "__custom__") {
                  const c = corpora.find((x) => x.name === v);
                  if (c)
                    setForm((f) => ({
                      ...f,
                      data_dir: c.data_dir,
                      persist_dir: c.suggested_persist_dir,
                      variant: c.name.includes("cybersec") ? "cybersec" : "generic",
                    }));
                }
              }}
            >
              {corpora.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} · {c.doc_count} docs · {c.file_types.join(", ")}
                  {c.has_index ? " · indexed" : ""}
                </option>
              ))}
              <option value="__custom__">Custom directory…</option>
            </select>
            {selected?.has_index && (
              <div className="text-xs text-emerald-700 mt-1">
                Index already exists at{" "}
                <span className="mono">{selected.suggested_persist_dir}</span>.
                Tick “rebuild” to force a fresh build.
              </div>
            )}
          </div>

          <div className="grid sm:grid-cols-2 gap-3">
            <Text label="Data dir" value={form.data_dir} onChange={(v) =>
              setForm((f) => ({ ...f, data_dir: v }))
            } />
            <Text label="Persist dir" value={form.persist_dir} onChange={(v) =>
              setForm((f) => ({ ...f, persist_dir: v }))
            } />
            <Num label="chunk_size" value={form.chunk_size} onChange={(v) =>
              setForm((f) => ({ ...f, chunk_size: v }))
            } />
            <Num label="chunk_overlap" value={form.chunk_overlap} onChange={(v) =>
              setForm((f) => ({ ...f, chunk_overlap: v }))
            } />
            <Num label="similarity_top_k" value={form.similarity_top_k} onChange={(v) =>
              setForm((f) => ({ ...f, similarity_top_k: v }))
            } />
            <div>
              <label className="field-label">embed_model</label>
              <select
                className="select"
                value={form.embed_model}
                onChange={(e) =>
                  setForm((f) => ({ ...f, embed_model: e.target.value as any }))
                }
              >
                <option value="local">local (BAAI/bge-small-en-v1.5)</option>
                <option value="openai">openai (text-embedding-ada-002)</option>
              </select>
            </div>
            <div>
              <label className="field-label">variant</label>
              <select
                className="select"
                value={form.variant}
                onChange={(e) =>
                  setForm((f) => ({ ...f, variant: e.target.value as any }))
                }
              >
                <option value="auto">auto</option>
                <option value="cybersec">cybersec (metadata-aware)</option>
                <option value="generic">generic</option>
              </select>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <label className="inline-flex items-center gap-2 text-sm text-zinc-700">
              <input
                type="checkbox"
                checked={form.rebuild}
                onChange={(e) =>
                  setForm((f) => ({ ...f, rebuild: e.target.checked }))
                }
              />
              Rebuild index from scratch
            </label>
          </div>

          {error && (
            <div className="text-sm text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          <div className="flex items-center justify-end gap-2">
            <button
              className="btn-primary"
              onClick={onSubmit}
              disabled={submitting}
            >
              {submitting ? "Submitting…" : "Build index"}
            </button>
          </div>
        </div>

        <div className="space-y-3">
          <div className="card p-4">
            <div className="text-sm font-semibold mb-2">Discovered corpora</div>
            {corpora.length === 0 && (
              <div className="text-sm text-zinc-500">
                No corpora detected under <span className="mono">data/</span>.
              </div>
            )}
            <ul className="space-y-2">
              {corpora.map((c) => (
                <li
                  key={c.name}
                  className="flex items-center justify-between text-sm"
                >
                  <div className="min-w-0">
                    <div className="font-medium truncate">{c.name}</div>
                    <div className="text-xs text-zinc-500 mono truncate">
                      {c.data_dir}
                    </div>
                  </div>
                  <div>
                    {c.has_index ? (
                      <span className="chip-success">indexed</span>
                    ) : (
                      <span className="chip-neutral">{c.doc_count} docs</span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div className="mt-5">
        <JobPanel jobId={lastIngestJobId} onDone={onJobDone} />
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
