import type { Run } from "./types";
import { STATUS_LABEL, fmtTime, fmtVal, shortId } from "./util";

const HIDDEN_TAGS = new Set(["mltree-run"]);

export default function RunsTable(props: {
  runs: Run[]; selected: string[]; toggleRun: (id: string) => void;
  onOpen?: (r: Run) => void; onLog?: (r: Run) => void; activeId?: string | null; showStep?: boolean;
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
            {props.onLog && <th />}
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} className={r.id === props.activeId ? "active" : ""} onClick={() => props.onOpen?.(r)}>
              <td onClick={(e) => e.stopPropagation()}>
                <input type="checkbox" checked={selected.includes(r.id)} onChange={() => props.toggleRun(r.id)} title="Select for compare" />
              </td>
              <td className="mono nowrap" title={r.id}>{r.kind === "sweep" && <span className="chip sweep-chip">sweep</span>}{shortId(r.id)}{r.parent_run_id && <span className="muted" title={`child of ${r.parent_run_id}`}> ↳</span>}</td>
              {props.showStep && <td>{r.step}</td>}
              <td><span className={`badge status-${r.status}`}>{STATUS_LABEL[r.status]}</span></td>
              <td className="params" title={[JSON.stringify(r.params), r.notes].filter(Boolean).join("\n\n")}>
                {Object.entries(r.params).map(([k, v]) => `${k}=${fmtVal(v)}`).join(" ") || "–"}
                {r.tags.filter((t) => !HIDDEN_TAGS.has(t)).map((t) => <span key={t} className="chip tag">{t}</span>)}
                {r.notes && <span className="note-dot" title={r.notes}> ✎</span>}
              </td>
              {metricKeys.map((k) => <td key={k} className="num mono">{fmtVal(r.metrics[k])}</td>)}
              <td className="muted nowrap">{fmtTime(r.created_at)}</td>
              {props.onLog && <td onClick={(e) => e.stopPropagation()}>{r.tags.includes("mltree-run") && <button className="link small" onClick={() => props.onLog!(r)}>{r.kind === "sweep" ? "sweep" : "log"}</button>}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
