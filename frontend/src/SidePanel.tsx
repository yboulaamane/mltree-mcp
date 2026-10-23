import { Suspense, lazy, useEffect, useState } from "react";
import { useApi } from "./api";
import type { Run, Script, Snapshot, StepNode } from "./types";
import { STATUS_LABEL, fmtTime, fmtVal, shortId } from "./util";
import RunsTable from "./RunsTable";
import type { Theme } from "./App";
import { RUN_OFF_HINT, runnable, useRuns } from "./runctx";

const CodeView = lazy(() => import("./CodeView"));

export default function SidePanel(props: {
  step: StepNode; theme: Theme; revision: number; selected: string[];
  toggleRun: (id: string) => void; onCompare: () => void; onClose: () => void;
}) {
  const { step, revision } = props;
  const [source, setSource] = useState<string>(""); // "" = current file, otherwise a run id
  useEffect(() => setSource(""), [step.name]);
  const enc = encodeURIComponent(step.name);
  const { data: snaps } = useApi<{ snapshots: Snapshot[] }>(`/api/steps/${enc}/snapshots`, [revision]);
  const { data: runs } = useApi<{ runs: Run[] }>(`/api/runs?step=${enc}&limit=200`, [revision]);
  const canShow = step.script_exists || source !== "";
  const { data: script, error } = useApi<Script>(
    canShow ? `/api/steps/${enc}/script${source ? `?run_id=${encodeURIComponent(source)}` : ""}` : null,
    [revision],
  );
  const chosen = snaps?.snapshots.find((s) => s.run_id === source);
  const nSel = props.selected.length;
  const rc = useRuns();
  const canRun = rc.mode !== "none" && runnable(step.script_path, step.script_exists);

  return (
    <aside className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">Step</div>
          <h2>{step.name} <span className={`badge status-${step.status}`}>{STATUS_LABEL[step.status]}</span></h2>
          {step.description && <p className="muted">{step.description}</p>}
          {canRun && (
            <div className="run-actions">
              <button className="primary" disabled={rc.mode !== "on"} title={rc.mode === "on" ? "" : RUN_OFF_HINT} onClick={() => rc.openRun(step.name, "run")}>▶ Run…</button>
              <button disabled={rc.mode !== "on"} title={rc.mode === "on" ? "" : RUN_OFF_HINT} onClick={() => rc.openRun(step.name, "sweep")}>Grid sweep…</button>
              {rc.mode === "off" && <span className="muted small">start the viewer with <code>--allow-run</code> to enable</span>}
            </div>
          )}
        </div>
        <button className="icon" onClick={props.onClose} title="Close">✕</button>
      </div>

      <div className="kv">
        <div><span>Inputs</span>{step.inputs.length ? step.inputs.map((x) => <code key={x} className="chip">{x}</code>) : <em>–</em>}</div>
        <div><span>Outputs</span>{step.outputs.length ? step.outputs.map((x) => <code key={x} className="chip">{x}</code>) : <em>–</em>}</div>
        <div><span>Params</span>{Object.keys(step.params).length
          ? Object.entries(step.params).map(([k, v]) => <code key={k} className="chip">{k}={fmtVal(v)}</code>)
          : <em>–</em>}</div>
      </div>

      <section className="code-section">
        <div className="section-head">
          <h3>Script <span className="muted mono">{step.script_path ?? ""}</span></h3>
          <select value={source} onChange={(e) => setSource(e.target.value)} aria-label="Script version">
            <option value="">Current file{step.script_exists ? "" : " (missing)"}</option>
            {snaps?.snapshots.map((s) => (
              <option key={s.run_id} value={s.run_id}>
                Run {shortId(s.run_id)} · {fmtTime(s.created_at)} · {s.same_as_current ? "same as current" : "changed"}
              </option>
            ))}
          </select>
        </div>
        {chosen && <div className="note">Snapshot stored with run <code>{chosen.run_id}</code> ({STATUS_LABEL[chosen.status]}).</div>}
        <div className="code">
          {!canShow ? <p className="muted pad">No script file for this step yet.</p>
            : error ? <p className="error pad">{error}</p>
            : script ? (
              <Suspense fallback={<p className="muted pad">Loading editor…</p>}>
                <CodeView path={`${script.source}:${script.path}`} content={script.content} theme={props.theme} />
              </Suspense>
            ) : <p className="muted pad">Loading…</p>}
        </div>
      </section>

      <section>
        <div className="section-head">
          <h3>Runs <span className="muted">({runs?.runs.length ?? 0})</span></h3>
          <button disabled={nSel < 2} onClick={props.onCompare}>Compare {nSel ? `(${nSel})` : ""}</button>
        </div>
        <RunsTable runs={runs?.runs ?? []} selected={props.selected} toggleRun={props.toggleRun}
          activeId={source || null}
          onLog={rc.mode === "none" ? undefined : (r) => rc.watch({ kind: r.kind === "sweep" ? "sweep" : "run", id: r.id, step: r.step })}
          onOpen={(r) => r.script_sha256 && setSource(r.id)} />
      </section>
    </aside>
  );
}
