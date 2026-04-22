import { useEffect, useMemo, useState } from "react";
import type { AppContext } from "../App";
import { api } from "../api/client";
import type { RunLog, RunSummary } from "../types";
import ResultsPanel from "../components/ResultsPanel";

export default function HistoryPage({ ctx: _ctx }: { ctx: AppContext }) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [detail, setDetail] = useState<RunLog | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [filterCondition, setFilterCondition] = useState<string>("");
  const [filterQuery, setFilterQuery] = useState<string>("");

  useEffect(() => {
    api
      .runs({ limit: 200 })
      .then(setRuns)
      .catch(() => {});
  }, []);

  const conditions = useMemo(() => {
    const set = new Set<string>();
    runs.forEach((r) => set.add(r.attack_condition));
    return Array.from(set).sort();
  }, [runs]);

  const filtered = useMemo(() => {
    return runs.filter((r) => {
      if (filterCondition && r.attack_condition !== filterCondition) return false;
      if (
        filterQuery &&
        !r.query_id.toLowerCase().includes(filterQuery.toLowerCase())
      )
        return false;
      return true;
    });
  }, [runs, filterCondition, filterQuery]);

  const loadDetail = async (r: RunSummary) => {
    const key = `${r.query_id}|${r.attack_condition}`;
    setSelected(key);
    try {
      const d = await api.runDetail(r.query_id, r.attack_condition);
      setDetail(d);
    } catch {
      setDetail(null);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold">Run history</h2>
        <div className="flex items-center gap-2">
          <input
            className="input w-48"
            placeholder="filter by query_id"
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
          />
          <select
            className="select w-48"
            value={filterCondition}
            onChange={(e) => setFilterCondition(e.target.value)}
          >
            <option value="">all conditions</option>
            {conditions.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid md:grid-cols-3 gap-5">
        <div className="md:col-span-1 card overflow-hidden">
          <div className="max-h-[70vh] overflow-auto divide-y divide-zinc-200">
            {filtered.length === 0 && (
              <div className="px-4 py-6 text-sm text-zinc-500">
                No runs yet.
              </div>
            )}
            {filtered.map((r, i) => {
              const key = `${r.query_id}|${r.attack_condition}|${r.logged_at ?? i}`;
              const active = selected === `${r.query_id}|${r.attack_condition}`;
              return (
                <button
                  key={key}
                  onClick={() => loadDetail(r)}
                  className={`w-full text-left px-4 py-2.5 transition ${
                    active ? "bg-accent-50" : "hover:bg-zinc-50"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="mono text-xs text-zinc-800">
                      {r.query_id}
                    </div>
                    {r.harmful_action_flag ? (
                      <span className="chip-danger">harmful</span>
                    ) : r.poison_retrieved ? (
                      <span className="chip-warn">poison</span>
                    ) : (
                      <span className="chip-success">benign</span>
                    )}
                  </div>
                  <div className="text-xs text-zinc-500 mono truncate">
                    {r.attack_condition}
                    {r.final_confidence !== null && (
                      <> · conf {r.final_confidence.toFixed(2)}</>
                    )}
                  </div>
                  <div className="text-xs text-zinc-400 truncate">
                    {r.final_answer || "(empty)"}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
        <div className="md:col-span-2">
          {detail ? (
            <ResultsPanel run={detail} />
          ) : (
            <div className="card p-6 text-sm text-zinc-500">
              Select a run to see the full transcript and agent responses.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
