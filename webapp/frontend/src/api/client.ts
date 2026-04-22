import type {
  ArtifactDetail,
  ArtifactSummary,
  Corpus,
  Defaults,
  ExperimentRequest,
  IngestRequest,
  JobSummary,
  QueryFile,
  RunLog,
  RunSummary,
  TriggerOptRequest,
} from "../types";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  if (res.status === 204) return undefined as unknown as T;
  return (await res.json()) as T;
}

export const api = {
  defaults: () => req<Defaults>("/api/defaults"),
  corpora: () => req<Corpus[]>("/api/corpora"),
  checkCorpus: (dataDir: string) =>
    req<{ data_dir: string; doc_count: number; file_types: string[] }>(
      `/api/corpora/check?data_dir=${encodeURIComponent(dataDir)}`,
    ),
  queryFiles: () => req<string[]>("/api/query-files"),
  queries: (path: string) =>
    req<QueryFile>(`/api/queries?path=${encodeURIComponent(path)}`),
  artifacts: () => req<ArtifactSummary[]>("/api/artifacts"),
  artifact: (id: string) =>
    req<ArtifactDetail>(`/api/artifacts/${encodeURIComponent(id)}`),
  jobs: (kind?: string) =>
    req<JobSummary[]>(`/api/jobs${kind ? `?kind=${kind}` : ""}`),
  job: (id: string) => req<JobSummary>(`/api/jobs/${id}`),
  cancelJob: (id: string) =>
    req<JobSummary>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  ingest: (body: IngestRequest) =>
    req<JobSummary>("/api/ingest", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  trigger: (body: TriggerOptRequest) =>
    req<JobSummary>("/api/trigger", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  experiment: (body: ExperimentRequest) =>
    req<JobSummary>("/api/experiments", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  runs: (params: {
    limit?: number;
    query_id?: string;
    attack_condition?: string;
  }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
    });
    return req<RunSummary[]>(`/api/runs?${qs.toString()}`);
  },
  latestRuns: (params: { limit?: number; since?: string }) => {
    const qs = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
    });
    return req<RunLog[]>(`/api/runs/latest?${qs.toString()}`);
  },
  runDetail: (queryId: string, attackCondition?: string) => {
    const qs = new URLSearchParams();
    if (attackCondition) qs.set("attack_condition", attackCondition);
    return req<RunLog>(
      `/api/runs/by-query/${encodeURIComponent(queryId)}?${qs.toString()}`,
    );
  },
};

export function streamJobLogs(
  jobId: string,
  onLine: (line: string) => void,
  onStatus: (status: string) => void,
  onDone: () => void,
): () => void {
  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  es.onmessage = (ev) => onLine(ev.data);
  es.addEventListener("status", (ev) =>
    onStatus((ev as MessageEvent).data as string),
  );
  es.addEventListener("done", () => {
    es.close();
    onDone();
  });
  es.onerror = () => {
    es.close();
    onDone();
  };
  return () => es.close();
}
