/**
 * Read-only data source for exported HTML files (`mltree-mcp export`).
 *
 * The export embeds a JSON snapshot in <script id="mltree-data" type="application/json">. When that element is
 * present, api() answers the same /api/* routes as the Python viewer from the snapshot, so every view works
 * offline from file:// with no server.
 */
import type { Pipeline, Run, Snapshot } from "./types";

export interface ScriptRef {
  path: string;
  sha256: string;
  created_at?: string | null;
}

export interface ExportSnapshot {
  format: string;
  exported_at: string;
  mltree_version: string;
  include_scripts: boolean;
  pipeline: Pipeline;
  runs: Run[]; // newest first, like list_runs
  metrics: string[];
  scripts: Record<string, { current: ScriptRef | null; snapshots: Snapshot[] }>;
  run_scripts: Record<string, ScriptRef>;
  blobs: Record<string, string>;
}

let cached: ExportSnapshot | null | undefined;

export function offlineData(): ExportSnapshot | null {
  if (cached !== undefined) return cached;
  const el = typeof document !== "undefined" ? document.getElementById("mltree-data") : null;
  cached = el?.textContent ? (JSON.parse(el.textContent) as ExportSnapshot) : null;
  return cached;
}

export const isOffline = (): boolean => offlineData() !== null;

class OfflineError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

function step(d: ExportSnapshot, name: string) {
  const s = d.pipeline.nodes.find((n) => n.name === name);
  if (!s) throw new OfflineError(404, `Unknown step '${name}'`);
  return s;
}

function run(d: ExportSnapshot, id: string): Run {
  const r = d.runs.find((x) => x.id === id);
  if (!r) throw new OfflineError(404, `Unknown run '${id}'`);
  return r;
}

function best(d: ExportSnapshot, runs: Run[], metric: string, mode: string): Run | null {
  let out: Run | null = null;
  for (const r of [...runs].reverse()) { // oldest first: ties go to the earliest run, like the server
    const v = r.metrics[metric];
    if (r.status !== "succeeded" || typeof v !== "number") continue;
    const b = out?.metrics[metric];
    if (out === null || (mode === "min" ? v < (b as number) : v > (b as number))) out = r;
  }
  void d;
  return out;
}

const stable = (v: unknown): string => JSON.stringify(v, (_k, x) =>
  x && typeof x === "object" && !Array.isArray(x) ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => (a < b ? -1 : 1))) : x) ?? "null";

export function offlineApi(d: ExportSnapshot, path: string, init?: RequestInit): unknown {
  const url = new URL(path, "http://offline.invalid");
  const q = url.searchParams;
  const parts = url.pathname.replace(/^\/api\/?/, "").split("/").map(decodeURIComponent);
  const method = (init?.method ?? "GET").toUpperCase();
  if (parts[0] === "positions" && method !== "GET") return { ok: true, read_only: true };

  switch (parts[0]) {
    case "health":
      return { ok: true, project: d.pipeline.project, root: d.pipeline.root, initialized: true, revision: 0, offline: true };
    case "pipeline":
      return d.pipeline;
    case "metrics":
      return { metrics: d.metrics };
    case "steps": {
      const s = step(d, parts[1]);
      if (parts.length === 2) return s;
      const sc = d.scripts[s.name];
      if (parts[2] === "snapshots") return { snapshots: sc?.snapshots ?? [] };
      if (parts[2] === "script") {
        if (!d.include_scripts) throw new OfflineError(404, "Scripts were left out of this export (--no-scripts).");
        const rid = q.get("run_id");
        const ref = rid ? d.run_scripts[rid] : sc?.current;
        if (!ref) throw new OfflineError(404, rid ? `Run '${rid}' has no script snapshot` : `Script ${s.script_path ?? ""} was not found when this was exported`);
        return {
          step: s.name, source: rid ? "snapshot" : "current", run_id: rid || null, path: ref.path,
          script_path: s.script_path, content: d.blobs[ref.sha256] ?? "", sha256: ref.sha256, created_at: ref.created_at ?? null,
        };
      }
      break;
    }
    case "runs": {
      if (parts[1]) return run(d, parts[1]);
      let rs = d.runs;
      const st = q.get("step"), status = q.get("status"), parent = q.get("parent_run_id");
      if (st) rs = rs.filter((r) => r.step === st);
      if (status) rs = rs.filter((r) => r.status === status);
      if (parent) rs = rs.filter((r) => r.parent_run_id === parent);
      rs = rs.slice(0, Number(q.get("limit") ?? 200) || 200);
      return { count: rs.length, runs: rs };
    }
    case "tree": {
      const st = q.get("step"), metric = q.get("metric") || null, mode = q.get("mode") === "min" ? "min" : "max";
      const rs = (st ? d.runs.filter((r) => r.step === st) : d.runs).slice(0, 1000);
      const ids = new Set(rs.map((r) => r.id));
      const b = metric ? best(d, rs, metric, mode) : null;
      const oldest = [...rs].reverse();
      return {
        runs: oldest,
        edges: rs.filter((r) => r.parent_run_id && ids.has(r.parent_run_id)).map((r) => ({ source: r.parent_run_id, target: r.id })),
        roots: oldest.filter((r) => !r.parent_run_id || !ids.has(r.parent_run_id)).map((r) => r.id),
        metric, mode, best_run_id: b?.id ?? null, best_value: b && metric ? b.metrics[metric] : null, metrics: d.metrics,
      };
    }
    case "compare": {
      const ids = (q.get("ids") ?? "").split(",").map((s) => s.trim()).filter(Boolean);
      if (ids.length < 2) throw new OfflineError(400, "Pick at least 2 runs to compare");
      const runs = ids.map((i) => run(d, i));
      const pk = [...new Set(runs.flatMap((r) => Object.keys(r.params)))].sort();
      const mk = [...new Set(runs.flatMap((r) => Object.keys(r.metrics)))].sort();
      const differs = (k: string, f: "params" | "metrics") => new Set(runs.map((r) => stable((r[f] as Record<string, unknown>)[k]))).size > 1;
      return { runs, param_keys: pk, metric_keys: mk, differing_params: pk.filter((k) => differs(k, "params")), differing_metrics: mk.filter((k) => differs(k, "metrics")) };
    }
  }
  throw new OfflineError(404, "Not found");
}
