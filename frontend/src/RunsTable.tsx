import type { Run } from "./types";
import { STATUS_LABEL, fmtTime, fmtVal, shortId } from "./util";

export default function RunsTable(props: {
  runs: Run[]; selected: string[]; toggleRun: (id: string) => void;
  onOpen?: (r: Run) => void; activeId?: string | null; showStep?: boolean;
}) {
  const { runs, selected } = props;
  if (!runs.length) return <p className="muted">No runs logged yet.</p>;
  const metricKeys = [...new Set(runs.flatMap((r) => Object.keys(r.metrics)))].slice(0, 3);
  return (
    <div className="table-wrap">
      <table className="runs">
        <thead>
          <tr>
            <th />
            <th>Run</th>
            {props.showStep && <th>Step</th>}
            <th>Status</th>
            <th>Params</th>
            {metricKeys.map((k) => <th key={k} className="num">{k}</th>)}
            <th>When</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} className={r.id === props.activeId ? "active" : ""} onClick={() => props.onOpen?.(r)}>
              <td onClick={(e) => e.stopPropagation()}>
                <input type="checkbox" checked={selected.includes(r.id)} onChange={() => props.toggleRun(r.id)} title="Select for compare" />
              </td>
              <td className="mono nowrap" title={r.id}>{shortId(r.id)}{r.parent_run_id && <span className="muted" title={`child of ${r.parent_run_id}`}> ↳</span>}</td>
              {props.showStep && <td>{r.step}</td>}
              <td><span className={`badge status-${r.status}`}>{STATUS_LABEL[r.status]}</span></td>
              <td className="params" title={JSON.stringify(r.params)}>
                {Object.entries(r.params).map(([k, v]) => `${k}=${fmtVal(v)}`).join(" ") || "–"}
              </td>
              {metricKeys.map((k) => <td key={k} className="num mono">{fmtVal(r.metrics[k])}</td>)}
              <td className="muted nowrap">{fmtTime(r.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
