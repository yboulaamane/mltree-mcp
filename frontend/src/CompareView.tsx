import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { useApi } from "./api";
import type { Compare, Run } from "./types";
import { STATUS_LABEL, fmtVal, shortId } from "./util";
import RunsTable from "./RunsTable";
import type { Theme } from "./App";

const PlotBar = lazy(() => import("./PlotBar"));

export default function CompareView(props: { theme: Theme; revision: number; selected: string[]; toggleRun: (id: string) => void }) {
  const ids = props.selected;
  const { data: cmp, error } = useApi<Compare>(ids.length >= 2 ? `/api/compare?ids=${ids.map(encodeURIComponent).join(",")}` : null, [props.revision, ids.join(",")]);
  const { data: all } = useApi<{ runs: Run[] }>("/api/runs?limit=500", [props.revision]);
  const [chosen, setChosen] = useState<string[]>([]);
  const [onlyDiff, setOnlyDiff] = useState(false);

  // default chart metrics: those every compared run has
  useEffect(() => {
    if (!cmp) return;
    const common = cmp.metric_keys.filter((k) => cmp.runs.every((r) => k in r.metrics));
    setChosen((c) => (c.length && c.every((k) => cmp.metric_keys.includes(k)) ? c : (common.length ? common : cmp.metric_keys).slice(0, 4)));
  }, [cmp]);

  const traces = useMemo(
    () => (cmp ? cmp.runs.map((r) => ({ name: `${r.step} ${shortId(r.id)}`, x: chosen, y: chosen.map((k) => r.metrics[k] ?? null) })) : []),
    [cmp, chosen],
  );

  if (ids.length < 2) {
    return (
      <div className="page">
        <h2>Compare runs</h2>
        <p className="muted">Tick 2 or more runs (here, on the map panel or in the experiment tree) to compare their params and metrics.</p>
        <RunsTable runs={all?.runs ?? []} selected={ids} toggleRun={props.toggleRun} showStep onOpen={(r) => props.toggleRun(r.id)} />
      </div>
    );
  }
  const paramKeys = cmp ? (onlyDiff ? cmp.differing_params : cmp.param_keys) : [];
  const metricKeys = cmp ? (onlyDiff ? cmp.differing_metrics : cmp.metric_keys) : [];

  return (
    <div className="page">
      <div className="section-head">
        <h2>Compare {ids.length} runs</h2>
        <label className="check"><input type="checkbox" checked={onlyDiff} onChange={(e) => setOnlyDiff(e.target.checked)} /> Only rows that differ</label>
      </div>
      {error && <div className="banner error">{error}</div>}
      {cmp && (
        <>
          <div className="table-wrap card">
            <table className="compare">
              <thead>
                <tr>
                  <th />
                  {cmp.runs.map((r) => (
                    <th key={r.id}>
                      <div className="mono" title={r.id}>{shortId(r.id)}</div>
                      <div className="muted small">{r.step} · <span className={`badge status-${r.status}`}>{STATUS_LABEL[r.status]}</span></div>
                      <button className="link small" onClick={() => props.toggleRun(r.id)}>remove</button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {paramKeys.length > 0 && <tr className="group"><td colSpan={cmp.runs.length + 1}>Params</td></tr>}
                {paramKeys.map((k) => (
                  <tr key={`p-${k}`} className={cmp.differing_params.includes(k) ? "diff" : ""}>
                    <th>{k}</th>
                    {cmp.runs.map((r) => <td key={r.id} className="mono">{fmtVal(r.params[k])}</td>)}
                  </tr>
                ))}
                {metricKeys.length > 0 && <tr className="group"><td colSpan={cmp.runs.length + 1}>Metrics</td></tr>}
                {metricKeys.map((k) => {
                  const vals = cmp.runs.map((r) => r.metrics[k]).filter((v): v is number => typeof v === "number");
                  const lo = Math.min(...vals), hi = Math.max(...vals);
                  return (
                    <tr key={`m-${k}`} className={cmp.differing_metrics.includes(k) ? "diff" : ""}>
                      <th>
                        <label className="check"><input type="checkbox" checked={chosen.includes(k)}
                          onChange={() => setChosen((c) => (c.includes(k) ? c.filter((x) => x !== k) : [...c, k]))} /> {k}</label>
                      </th>
                      {cmp.runs.map((r) => {
                        const v = r.metrics[k];
                        const tag = vals.length > 1 && lo !== hi ? (v === lo ? "min" : v === hi ? "max" : "") : "";
                        return <td key={r.id} className="mono">{fmtVal(v)}{tag && <span className={`ext ${tag}`}>{tag}</span>}</td>;
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="card chart">
            {chosen.length ? (
              <Suspense fallback={<p className="muted pad">Loading chart…</p>}>
                <PlotBar traces={traces} theme={props.theme} title="Metrics (ticked rows)" />
              </Suspense>
            ) : <p className="muted pad">Tick metric rows to chart them.</p>}
          </div>
        </>
      )}
    </div>
  );
}
