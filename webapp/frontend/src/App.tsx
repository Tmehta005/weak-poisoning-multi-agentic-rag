import { useEffect, useState } from "react";
import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import { api } from "./api/client";
import type { ArtifactSummary, Corpus, Defaults } from "./types";
import IngestPage from "./pages/IngestPage";
import TriggerPage from "./pages/TriggerPage";
import ExperimentPage from "./pages/ExperimentPage";
import HistoryPage from "./pages/HistoryPage";

export type AppContext = {
  defaults: Defaults | null;
  corpora: Corpus[];
  artifacts: ArtifactSummary[];
  refreshCorpora: () => void;
  refreshArtifacts: () => void;
  lastIngestJobId: string | null;
  setLastIngestJobId: (id: string | null) => void;
  lastTriggerJobId: string | null;
  setLastTriggerJobId: (id: string | null) => void;
  lastExperimentJobId: string | null;
  setLastExperimentJobId: (id: string | null) => void;
};

export default function App() {
  const [defaults, setDefaults] = useState<Defaults | null>(null);
  const [corpora, setCorpora] = useState<Corpus[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactSummary[]>([]);
  const [lastIngestJobId, setLastIngestJobId] = useState<string | null>(null);
  const [lastTriggerJobId, setLastTriggerJobId] = useState<string | null>(null);
  const [lastExperimentJobId, setLastExperimentJobId] = useState<string | null>(
    null,
  );

  const refreshCorpora = () => {
    api.corpora().then(setCorpora).catch(() => {});
  };
  const refreshArtifacts = () => {
    api.artifacts().then(setArtifacts).catch(() => {});
  };

  useEffect(() => {
    api.defaults().then(setDefaults).catch(() => {});
    refreshCorpora();
    refreshArtifacts();
  }, []);

  const ctx: AppContext = {
    defaults,
    corpora,
    artifacts,
    refreshCorpora,
    refreshArtifacts,
    lastIngestJobId,
    setLastIngestJobId,
    lastTriggerJobId,
    setLastTriggerJobId,
    lastExperimentJobId,
    setLastExperimentJobId,
  };

  const ingestDone = corpora.some((c) => c.has_index);
  const triggerDone = artifacts.length > 0;

  return (
    <div className="min-h-full">
      <header className="bg-white border-b border-zinc-200 sticky top-0 z-20">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-accent-600 text-white grid place-items-center font-semibold">
              R
            </div>
            <div>
              <div className="text-sm font-semibold leading-tight">
                RAG Poisoning Console
              </div>
              <div className="text-xs text-zinc-500 leading-tight">
                Run experiments on the multi-agent RAG pipeline
              </div>
            </div>
          </div>
          <nav className="flex items-center gap-1 text-sm">
            <TopLink to="/ingest" label="1. Ingest" done={ingestDone} />
            <TopLink to="/trigger" label="2. Trigger" done={triggerDone} />
            <TopLink to="/experiment" label="3. Run" />
            <div className="w-px h-5 bg-zinc-200 mx-2" />
            <TopLink to="/history" label="History" />
          </nav>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-6 py-6">
        <Routes>
          <Route path="/" element={<Navigate to="/ingest" replace />} />
          <Route path="/ingest" element={<IngestPage ctx={ctx} />} />
          <Route path="/trigger" element={<TriggerPage ctx={ctx} />} />
          <Route path="/experiment" element={<ExperimentPage ctx={ctx} />} />
          <Route path="/history" element={<HistoryPage ctx={ctx} />} />
        </Routes>
      </main>
    </div>
  );
}

function TopLink({
  to,
  label,
  done,
}: {
  to: string;
  label: string;
  done?: boolean;
}) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-3 py-1.5 rounded-md font-medium transition ${
          isActive
            ? "bg-accent-50 text-accent-700"
            : "text-zinc-600 hover:text-zinc-900 hover:bg-zinc-100"
        }`
      }
    >
      <span className="inline-flex items-center gap-1.5">
        {label}
        {done && (
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-500" />
        )}
      </span>
    </NavLink>
  );
}
