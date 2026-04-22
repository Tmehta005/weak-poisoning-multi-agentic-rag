import { useNavigate } from "react-router-dom";

export interface Step {
  id: string;
  label: string;
  path: string;
  done: boolean;
  active: boolean;
  hint?: string;
}

export default function Stepper({ steps }: { steps: Step[] }) {
  const nav = useNavigate();
  return (
    <div className="flex items-stretch gap-3 mb-6">
      {steps.map((s, i) => (
        <button
          key={s.id}
          onClick={() => nav(s.path)}
          className={`flex-1 text-left card px-4 py-3 transition hover:shadow-sm ${
            s.active ? "ring-2 ring-accent-500/40 border-accent-500" : ""
          }`}
        >
          <div className="flex items-center gap-3">
            <div
              className={`h-7 w-7 rounded-full grid place-items-center text-xs font-semibold ${
                s.done
                  ? "bg-emerald-100 text-emerald-700"
                  : s.active
                    ? "bg-accent-600 text-white"
                    : "bg-zinc-100 text-zinc-500"
              }`}
            >
              {s.done ? "✓" : i + 1}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-semibold truncate">{s.label}</div>
              {s.hint && (
                <div className="text-xs text-zinc-500 truncate">{s.hint}</div>
              )}
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}
