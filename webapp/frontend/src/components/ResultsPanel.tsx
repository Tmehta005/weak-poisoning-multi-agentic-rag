import { useState } from "react";
import type {
  DebateRound,
  DebateTranscript,
  OrchestratorOutput,
  RunLog,
  SubagentOutput,
} from "../types";

export default function ResultsPanel({ run }: { run: RunLog }) {
  const fd = run.final_decision;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="chip-neutral mono">{run.query_id}</span>
        <span className="chip-accent">{run.attack_condition}</span>
        {fd && (
          <span className="chip-neutral">
            confidence {(fd.final_confidence ?? 0).toFixed(2)}
          </span>
        )}
        {run.poison_retrieved && (
          <span className="chip-warn">poison retrieved</span>
        )}
        {fd?.harmful_action_flag ? (
          <span className="chip-danger">harmful action</span>
        ) : (
          <span className="chip-success">benign</span>
        )}
        {run.trigger && (
          <span className="chip-neutral mono">trigger: {run.trigger}</span>
        )}
      </div>

      {fd && <FinalDecisionCard decision={fd} />}

      <div>
        <div className="text-sm font-semibold text-zinc-800 mb-2">
          Subagent responses
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
          {Object.values(run.agent_responses).map((s) => (
            <AgentCard key={s.agent_id} output={s} />
          ))}
        </div>
      </div>

      {run.debate_transcript &&
        run.debate_transcript.rounds &&
        run.debate_transcript.rounds.length > 0 && (
          <DebateRoundsView transcript={run.debate_transcript} />
        )}

      {run.ground_truth_answer && (
        <div className="card p-4">
          <div className="field-label">Ground truth</div>
          <div className="text-sm text-zinc-700 whitespace-pre-wrap">
            {run.ground_truth_answer}
          </div>
        </div>
      )}
    </div>
  );
}

function FinalDecisionCard({ decision }: { decision: OrchestratorOutput }) {
  return (
    <div className="card p-4 space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold text-zinc-800">
          Final decision
        </div>
        <div className="flex flex-wrap gap-1.5">
          {decision.winning_subagents?.map((id) => (
            <span key={id} className="chip-accent mono">
              {id}
            </span>
          ))}
        </div>
      </div>
      <div className="text-sm text-zinc-900 whitespace-pre-wrap leading-relaxed">
        {decision.final_answer || (
          <span className="text-zinc-500 italic">(empty)</span>
        )}
      </div>
      {decision.selected_evidence?.length > 0 && (
        <div>
          <div className="field-label">Evidence</div>
          <div className="flex flex-wrap gap-1.5">
            {decision.selected_evidence.map((id) => (
              <span key={id} className="chip-neutral mono">
                {id}
              </span>
            ))}
          </div>
        </div>
      )}
      {decision.reasoning_summary && (
        <div>
          <div className="field-label">Reasoning</div>
          <div className="text-xs text-zinc-600 whitespace-pre-wrap">
            {decision.reasoning_summary}
          </div>
        </div>
      )}
    </div>
  );
}

function AgentCard({ output }: { output: SubagentOutput }) {
  const [expanded, setExpanded] = useState(false);
  const conf = Math.max(0, Math.min(1, output.confidence ?? 0));
  return (
    <div className="card p-4 space-y-2">
      <div className="flex items-center justify-between">
        <div className="font-mono text-xs font-semibold text-zinc-800">
          {output.agent_id}
        </div>
        {output.poison_retrieved && (
          <span className="chip-warn">poisoned docs retrieved</span>
        )}
      </div>
      <div className="text-sm text-zinc-900 whitespace-pre-wrap leading-relaxed">
        {output.answer}
      </div>
      <div>
        <div className="flex items-center justify-between text-xs text-zinc-500 mb-1">
          <span>confidence</span>
          <span className="mono">{conf.toFixed(2)}</span>
        </div>
        <div className="h-1.5 rounded-full bg-zinc-100 overflow-hidden">
          <div
            className="h-full bg-accent-500"
            style={{ width: `${conf * 100}%` }}
          />
        </div>
      </div>
      {output.citations?.length > 0 && (
        <div>
          <div className="field-label">Citations</div>
          <div className="flex flex-wrap gap-1">
            {output.citations.map((c) => (
              <span key={c} className="chip-accent mono text-[11px]">
                {c.slice(0, 10)}
              </span>
            ))}
          </div>
        </div>
      )}
      <button
        className="text-xs text-accent-600 hover:text-accent-700"
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded ? "Hide" : "Show"} rationale & retrieved docs
      </button>
      {expanded && (
        <div className="space-y-2 pt-1">
          <div>
            <div className="field-label">Rationale</div>
            <div className="text-xs text-zinc-600 whitespace-pre-wrap">
              {output.rationale}
            </div>
          </div>
          {output.retrieved_doc_ids?.length > 0 && (
            <div>
              <div className="field-label">Retrieved doc ids</div>
              <div className="flex flex-wrap gap-1">
                {output.retrieved_doc_ids.map((c) => (
                  <span key={c} className="chip-neutral mono text-[11px]">
                    {c.slice(0, 10)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function DebateRoundsView({ transcript }: { transcript: DebateTranscript }) {
  const [open, setOpen] = useState<number | null>(transcript.rounds[0]?.round_num ?? null);
  return (
    <div className="card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold text-zinc-800">
          Debate transcript
        </div>
        <div className="flex flex-wrap gap-1.5">
          <span className="chip-neutral">
            {transcript.rounds_used} round
            {transcript.rounds_used === 1 ? "" : "s"}
          </span>
          <span className="chip-neutral">{transcript.stopped_reason}</span>
          {transcript.majority_cluster.map((id) => (
            <span key={id} className="chip-accent mono">
              {id}
            </span>
          ))}
        </div>
      </div>
      <div className="divide-y divide-zinc-200 border border-zinc-200 rounded-lg overflow-hidden">
        {transcript.rounds.map((r) => (
          <DebateRoundRow
            key={r.round_num}
            round={r}
            open={open === r.round_num}
            toggle={() => setOpen(open === r.round_num ? null : r.round_num)}
          />
        ))}
      </div>
    </div>
  );
}

function DebateRoundRow({
  round,
  open,
  toggle,
}: {
  round: DebateRound;
  open: boolean;
  toggle: () => void;
}) {
  return (
    <div>
      <button
        onClick={toggle}
        className="w-full flex items-center justify-between px-3 py-2 bg-zinc-50 hover:bg-zinc-100 text-left text-sm"
      >
        <div className="font-medium">Round {round.round_num}</div>
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(round.stances).map(([id, stance]) => (
            <span key={id} className="chip-neutral mono text-[11px]">
              {id}: {(stance ?? "").slice(0, 40)}
            </span>
          ))}
        </div>
      </button>
      {open && (
        <div className="px-3 py-3 bg-white space-y-3">
          {Object.entries(round.messages).map(([id, msg]) => (
            <div key={id}>
              <div className="flex items-center gap-2 mb-1">
                <span className="font-mono text-xs font-semibold">{id}</span>
                {round.confidences[id] !== undefined && (
                  <span className="chip-neutral text-[11px] mono">
                    conf {round.confidences[id].toFixed(2)}
                  </span>
                )}
              </div>
              <div className="text-xs text-zinc-700 whitespace-pre-wrap leading-relaxed border border-zinc-200 rounded-md p-2 bg-zinc-50">
                {msg}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
