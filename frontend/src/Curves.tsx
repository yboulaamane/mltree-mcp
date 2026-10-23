/** Overlaid step-wise metric curves of the compared runs (from MLTREE_METRIC lines or log_run history). */
import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type { HistoryPoint, Run } from "./types";
import LineChart, { type Series } from "./LineChart";
import { shortId } from "./util";

const X_KEYS = ["epoch", "step", "iteration", "iter", "t"];

export default function Curves(props: { runs: Run[]; revision: number }) {
  const [hist, setHist] = useState<Record<string, HistoryPoint[]>>({});
  const [metric, setMetric] = useState("");
  const [logY, setLogY] = useState(false);
  const ids = props.runs.map((r) => r.id).join(",");
  useEffect(() => {
    let alive = true;
    Promise.all(props.runs.map((r) =>
      api<{ history: HistoryPoint[] }>(`/api/runs/${encodeURIComponent(r.id)}/history`).then((h) => [r.id, h.history] as const).catch(() => [r.id, []] as const)))
      .then((pairs) => alive && setHist(Object.fromEntries(pairs)));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids, props.revision]);
  const withData = props.runs.filter((r) => (hist[r.id] ?? []).length > 0);
  const xKey = useMemo(() => X_KEYS.find((k) => withData.length && withData.every((r) => hist[r.id].some((p) => typeof p[k] === "number"))) ?? "step",
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [hist, ids]);
  const metrics = useMemo(() => {
    const ks = new Set<string>();
    withData.forEach((r) => hist[r.id].forEach((p) => Object.keys(p).forEach((k) => !X_KEYS.includes(k) && typeof p[k] === "number" && ks.add(k))));
    return [...ks].sort((a, b) => (/val|loss/.test(b) ? 1 : 0) - (/val|loss/.test(a) ? 1 : 0) || a.localeCompare(b));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hist, ids]);
  useEffect(() => { if (!metrics.includes(metric)) setMetric(metrics[0] ?? ""); }, [metrics, metric]);
  const series: Series[] = useMemo(() => withData.map((r) => ({
    name: `${r.step} ${shortId(r.id)}`,
    points: hist[r.id].map((p, i) => ({ x: typeof p[xKey] === "number" ? p[xKey] : i, y: p[metric] }))
      .filter((p) => typeof p.y === "number" && Number.isFinite(p.y)),
  })).filter((s) => s.points.length), [withData, hist, xKey, metric]);

  return (
    <div className="card">
      <div className="section-head pad-x">
        <h3>Curves <span className="muted small">({withData.length}/{props.runs.length} runs have step-wise metrics)</span></h3>
        {metrics.length > 0 && (
          <div className="row">
            <select value={metric} onChange={(e) => setMetric(e.target.value)} aria-label="Curve metric">
              {metrics.map((m) => <option key={m}>{m}</option>)}
            </select>
            <label className="check"><input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} /> log y</label>
          </div>
        )}
      </div>
      {metrics.length === 0
        ? <p className="muted pad">No step-wise metrics. Print <code>MLTREE_METRIC epoch=1 val_loss=0.4</code> lines, or pass <code>history</code> to <code>log_run</code>.</p>
        : <LineChart series={series} xLabel={xKey} yLabel={metric} logY={logY} />}
    </div>
  );
}
