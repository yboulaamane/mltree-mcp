import type { Status } from "./types";

export const STATUS_LABEL: Record<Status, string> = {
  not_run: "not run",
  pending: "pending",
  running: "running",
  succeeded: "done",
  failed: "failed",
  cancelled: "cancelled",
};

export function fmtNum(v: unknown): string {
  if (typeof v !== "number") return v === undefined ? "–" : JSON.stringify(v);
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : String(Number(v.toPrecision(4)));
}

export function fmtVal(v: unknown): string {
  if (v === undefined) return "–";
  if (typeof v === "number") return fmtNum(v);
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}

export function shortId(id: string): string {
  return id.length > 10 ? "…" + id.slice(-6) : id;
}

export function fmtTime(iso: string | null): string {
  if (!iso) return "–";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function firstMetric(m: Record<string, number> | undefined): [string, number] | null {
  if (!m) return null;
  const k = Object.keys(m)[0];
  return k ? [k, m[k]] : null;
}
