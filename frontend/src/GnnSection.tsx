/** GNN view of a step: model architecture parsed statically from the scripts, and graph-dataset stats. */
import { useState } from "react";
import { api, ApiError, useApi } from "./api";
import type { GnnDim, GnnLayer, GnnModel, GnnStats, GnnView, StepNode } from "./types";
import { fmtNum, fmtTime } from "./util";

const dim = (d?: GnnDim) => (!d ? "" : d.value != null && String(d.value) !== d.text ? `${d.value}` : d.text);
const dimTitle = (d?: GnnDim) => (d && d.value != null && String(d.value) !== d.text ? `${d.text} = ${d.value}` : undefined);
const range = (r?: { min: number; mean: number; max: number }) => (r ? `${fmtNum(r.min)} / ${fmtNum(r.mean)} / ${fmtNum(r.max)}` : "–");

function LayerRow({ l }: { l: GnnLayer }) {
  return (
    <tr className={`fam-${l.family}`} title={l.call ? `${l.file}:${l.line}  ${l.call}${l.mlp ? `\nMLP: ${l.mlp.join(" → ")}` : ""}` : undefined}>
      <td className="mono">{l.attr}</td>
      <td><code>{l.type}</code>{l.lib && <span className="muted small"> {l.lib}</span>}</td>
      <td className="mono small">
        {l.in || l.out ? <><span title={dimTitle(l.in)}>{dim(l.in) || "?"}</span> → <span title={dimTitle(l.out)}>{dim(l.out) || "?"}</span></>
          : l.p ? <span title={dimTitle(l.p)}>p={dim(l.p)}</span> : ""}
        {l.heads && <span title={dimTitle(l.heads)}> · heads {dim(l.heads)}</span>}
        {l.mlp && <span className="muted"> · MLP {l.mlp.join("→")}</span>}
      </td>
      <td className="num mono">{l.repeat != null ? `×${l.repeat}` : ""}</td>
    </tr>
  );
}

function ModelCard({ m }: { m: GnnModel }) {
  return (
    <div className="gnn-model">
      <div className="gnn-title">
        <b>{m.name}</b> <span className="muted small">{m.kind === "prebuilt" ? "PyG prebuilt model" : m.bases?.join(", ")} · {m.file}:{m.line}</span>
      </div>
      <div className="gnn-label">{m.summary.label}</div>
      {m.instantiated && <div className="muted small mono" title={`${m.instantiated.file}:${m.instantiated.line}`}>{m.instantiated.call}</div>}
      {(m.layers ?? []).length > 0 && (
        <table className="gnn-layers">
          <thead><tr><th>Layer</th><th>Type</th><th>in → out</th><th className="num">repeat</th></tr></thead>
          <tbody>{(m.layers ?? []).map((l, i) => <LayerRow key={i} l={l} />)}</tbody>
        </table>
      )}
      {m.forward && m.forward.length > 0 && (
        <div className="gnn-flow"><span className="muted small">forward: </span>
          {m.forward.map((t, i) => <span key={i}>{i > 0 && <span className="arrow"> → </span>}<code>{t}</code></span>)}
        </div>
      )}
    </div>
  );
}

function labels(s: GnnStats): string {
  const l = s.labels;
  if (!l) return "–";
  if (l.task === "classification" || l.task === "multi-label") return `${l.task} · ${l.num_classes ?? "?"} classes`;
  if (l.task === "regression") return `regression${l.targets && l.targets > 1 ? ` ×${l.targets}` : ""} · ${fmtNum(l.min)} … ${fmtNum(l.max)}`;
  return l.task ?? "–";
}

function StatsTable({ v }: { v: GnnView }) {
  const rows = v.dataset_files;
  return (
    <table className="gnn-stats">
      <thead><tr><th>File</th><th className="num">graphs</th><th>nodes min/mean/max</th><th>edges min/mean/max</th><th>features n / e</th><th>labels</th></tr></thead>
      <tbody>
        {rows.map((r) => {
          const s = r.stats;
          return (
            <tr key={r.path} title={s ? `${s.format ?? ""} · ${s.loaded_with ?? ""}\n${r.python ?? ""}\n${Object.entries(s.versions ?? {}).map(([k, x]) => `${k} ${x}`).join(", ")}` : undefined}>
              <td><code>{r.path}</code>{r.version && <span className="muted small mono"> {r.version.slice(0, 7)}</span>}
                {r.missing ? <span className="flag warn"> missing</span> : r.stale ? <span className="flag warn" title="The file changed since these stats were computed"> stale</span> : null}</td>
              {!s ? <td colSpan={5} className="muted small">not inspected yet</td>
                : s.error ? <td colSpan={5} className="error small">{s.error}</td>
                : s.format === "state_dict" ? <td colSpan={5} className="small">model weights (state_dict) · {s.tensors} tensors · {s.parameters?.toLocaleString()} parameters</td>
                : <>
                  <td className="num mono">{s.num_graphs ?? (s.num_nodes != null ? 1 : "–")}</td>
                  <td className="mono small">{s.nodes ? range(s.nodes) : s.num_nodes ?? "–"}</td>
                  <td className="mono small">{s.edges ? range(s.edges) : s.num_edges ?? "–"}</td>
                  <td className="mono small">{s.node_features ?? "–"} / {s.edge_features ?? "–"}</td>
                  <td className="small">{labels(s)}{s.masks ? <span className="muted"> · {Object.entries(s.masks).map(([k, n]) => `${k.replace("_mask", "")} ${n}`).join(" / ")}</span> : null}{s.isolated_nodes ? <span className="flag warn"> {s.isolated_nodes} isolated</span> : null}
                    {s.self_loops ? <span className="muted"> · {s.self_loops} self-loops</span> : null}</td>
                </>}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function GnnSection({ step, revision }: { step: StepNode; revision: number }) {
  const enc = encodeURIComponent(step.name);
  const { data: v } = useApi<GnnView>(`/api/steps/${enc}/gnn`, [revision, step.name]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [fresh, setFresh] = useState<GnnView | null>(null);
  const view = fresh?.step === step.name ? fresh : v;
  if (!view || (!view.models.length && !view.datasets.length && !view.dataset_files.length)) return null;
  const inspect = async () => {
    if (!window.confirm(`${view.introspect_note}\n\nInspect ${view.dataset_files.map((f) => f.path).join(", ")}?`)) return;
    setBusy(true);
    try {
      setFresh(await api<GnnView>(`/api/steps/${enc}/gnn/introspect`, { method: "POST", body: "{}" }));
      setErr(null);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const computed = view.dataset_files.map((f) => f.computed_at).filter(Boolean).sort().pop();
  return (
    <section className="gnn-section">
      <div className="section-head">
        <h3>Graph model &amp; data <span className="muted small">read statically · nothing imported</span></h3>
      </div>
      {view.models.map((m) => <ModelCard key={`${m.file}:${m.name}`} m={m} />)}
      {view.datasets.length > 0 && (
        <div className="kv"><div><span>Datasets</span>{view.datasets.map((d, i) => (
          <code key={i} className="chip" title={`${d.file}:${d.line}${d.call ? `  ${d.call}` : ""}`}>{d.class}{d.name ? `(${d.name})` : ""}{d.lib ? ` · ${d.lib}` : ""}</code>
        ))}</div></div>
      )}
      {view.dataset_files.length > 0 && (
        <>
          <StatsTable v={view} />
          <div className="row-actions">
            {view.can_introspect && (
              <button disabled={busy} onClick={inspect} title={view.introspect_note}>{busy ? "Inspecting…" : computed ? "Re-inspect dataset…" : "Inspect dataset…"}</button>
            )}
            {computed && <span className="muted small">stats from {fmtTime(computed)}</span>}
            {!view.can_introspect && !computed && <span className="muted small">Dataset stats: <code>mltree-mcp gnn --step {step.name} --introspect</code> or the viewer with <code>--allow-run</code></span>}
          </div>
        </>
      )}
      {view.warnings.map((w, i) => <p key={i} className="muted small">⚠ {w}</p>)}
      {err && <p className="error small">{err}</p>}
    </section>
  );
}
