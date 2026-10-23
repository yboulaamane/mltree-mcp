/** Data versions of a step's inputs/outputs: current SHA-256 vs what its latest run used. */
import { useApi } from "./api";
import type { DataItem, DataStatus, StepNode } from "./types";
import { fmtTime, shortId } from "./util";

function size(n?: number): string {
  if (n === undefined) return "";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${i ? v.toFixed(1) : v} ${u[i]}`;
}

function Item({ it, role, onRun }: { it: DataItem; role: "in" | "out"; onRun: (id: string) => void }) {
  const state = !it.exists ? "missing" : it.changed ? "changed" : it.run_version ? "same" : "new";
  const title = [
    it.path ? `path: ${it.path}` : "not found on disk",
    it.sha256 ? `sha256: ${it.sha256}${it.quick ? " (quick digest: size+mtime)" : ""}` : "",
    it.mtime ? `modified ${fmtTime(it.mtime)}` : "",
    it.also_found?.length ? `also found: ${it.also_found.join(", ")}` : "",
  ].filter(Boolean).join("\n");
  return (
    <tr className={`data-${state}`}>
      <td className="muted small">{role}</td>
      <td title={title}><code>{it.name}</code>{it.kind === "dir" && <span className="muted small"> dir · {it.files} files</span>}</td>
      <td className="mono">{it.version ?? "–"}{it.quick && <span className="muted small" title="size+mtime digest (too large to hash)"> ~</span>}</td>
      <td className="small">
        {state === "missing" ? <span className="flag warn">missing</span>
          : state === "changed" ? <span className="flag warn" title={`latest run used ${it.run_version}`}>changed (was {it.run_version?.slice(0, 7)})</span>
          : state === "same" ? <span className="flag ok">same as latest run</span>
          : <span className="muted">not recorded</span>}
        {it.producer_run_id && role === "in" && (
          <button className="link small" title="The run that wrote this exact version" onClick={() => onRun(it.producer_run_id!)}> from {shortId(it.producer_run_id)}</button>
        )}
      </td>
      <td className="muted small">{size(it.bytes)}</td>
    </tr>
  );
}

export default function DataSection(props: { step: StepNode; revision: number; onRun: (id: string) => void }) {
  const { step } = props;
  const { data, error } = useApi<DataStatus>(
    step.inputs.length || step.outputs.length ? `/api/data?step=${encodeURIComponent(step.name)}` : null, [props.revision, step.name]);
  if (!step.inputs.length && !step.outputs.length) return null;
  const s = data?.steps[step.name];
  return (
    <section>
      <div className="section-head">
        <h3>Data versions</h3>
        {s?.last_run_id && <span className="muted small">vs latest run {shortId(s.last_run_id)}</span>}
      </div>
      {s?.changed_inputs.length ? (
        <div className="banner warn small">Inputs changed since the latest run: {s.changed_inputs.join(", ")}. Re-run this step to refresh its outputs.</div>
      ) : null}
      {error ? <p className="muted small">{error}</p> : !s ? <p className="muted small">Hashing…</p> : (
        <div className="table-wrap">
          <table className="data-table">
            <tbody>
              {s.inputs.map((it) => <Item key={`i-${it.name}`} it={it} role="in" onRun={props.onRun} />)}
              {s.outputs.map((it) => <Item key={`o-${it.name}`} it={it} role="out" onRun={props.onRun} />)}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
