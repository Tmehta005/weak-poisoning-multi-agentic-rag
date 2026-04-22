import { useState } from "react";
import type { ArtifactDetail, ArtifactSummary } from "../types";

export default function TriggerCard({
  artifact,
  compact = false,
}: {
  artifact: ArtifactSummary | ArtifactDetail | null;
  compact?: boolean;
}) {
  const [showPoison, setShowPoison] = useState(false);
  if (!artifact) {
    return (
      <div className="card px-4 py-6 text-sm text-zinc-500">
        No attack artifact yet. Run the trigger optimizer on the previous step
        to create one.
      </div>
    );
  }
  const detail = "poison_doc_text" in artifact ? artifact : null;
  return (
    <div className="card p-4 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs text-zinc-500 uppercase tracking-wide">
            Attack artifact
          </div>
          <div className="text-sm font-semibold">{artifact.attack_id}</div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {artifact.encoder_model && (
            <span className="chip-neutral">{artifact.encoder_model}</span>
          )}
          <span className="chip-neutral">
            {artifact.num_adv_passage_tokens} tokens
          </span>
          {artifact.final_loss !== null &&
            artifact.final_loss !== undefined && (
              <span className="chip-neutral">
                loss {artifact.final_loss.toFixed(3)}
              </span>
            )}
        </div>
      </div>

      <div>
        <div className="field-label">Trigger</div>
        <code className="block rounded-lg bg-zinc-900 text-emerald-300 mono text-sm px-3 py-2">
          {artifact.trigger || <span className="text-zinc-500">(empty)</span>}
        </code>
      </div>

      {!compact && (
        <div>
          <div className="field-label">Target claim</div>
          <div className="text-sm text-zinc-800 rounded-lg bg-zinc-50 border border-zinc-200 px-3 py-2">
            {artifact.target_claim}
          </div>
        </div>
      )}

      {artifact.harmful_match_phrases.length > 0 && (
        <div>
          <div className="field-label">Harmful match phrases</div>
          <div className="flex flex-wrap gap-1.5">
            {artifact.harmful_match_phrases.map((p) => (
              <span key={p} className="chip-warn mono text-xs">
                {p}
              </span>
            ))}
          </div>
        </div>
      )}

      {detail && detail.poison_doc_text && (
        <div>
          <div className="flex items-center justify-between mb-1">
            <div className="field-label mb-0">Poison document</div>
            <button
              className="text-xs text-accent-600 hover:text-accent-700"
              onClick={() => setShowPoison((v) => !v)}
            >
              {showPoison ? "Hide" : "Show"}
            </button>
          </div>
          {showPoison && (
            <pre className="mono text-xs rounded-lg bg-zinc-50 border border-zinc-200 p-3 max-h-56 overflow-auto whitespace-pre-wrap">
              {detail.poison_doc_text}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
