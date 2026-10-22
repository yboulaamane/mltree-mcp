import { useEffect, useMemo, useState } from "react";
import {
  Background, Controls, Handle, MarkerType, Position, ReactFlow, ReactFlowProvider, useReactFlow,
  type Edge as FlowEdge, type Node, type NodeProps,
} from "@xyflow/react";
import { useApi } from "./api";
import type { Run, Tree } from "./types";
import { dagreLayout } from "./layout";
import { STATUS_LABEL, fmtNum, fmtTime, fmtVal, shortId } from "./util";
import type { Theme } from "./App";

const W = 190;
const H = 78;

type RunData = { run: Run; metric: string | null; best: boolean; picked: boolean };
type RunFlowNode = Node<RunData, "run">;

function RunCard({ data }: NodeProps<RunFlowNode>) {
  const { run, metric, best, picked } = data;
  const v = metric ? run.metrics[metric] : undefined;
  return (
    <div className={`run-node status-${run.status}${best ? " best" : ""}${picked ? " picked" : ""}`}>
      <Handle type="target" position={Position.Top} />
      <div className="run-head">
        <span className="dot" />
        <span className="run-step">{run.step}</span>
        {best && <span className="star" title="Best run">★ best</span>}
      </div>
      <div className="run-id mono">{shortId(run.id)}</div>
      <div className="run-metric">
        {metric ? (v !== undefined ? <>{metric} <b>{fmtNum(v)}</b></> : <span className="muted">no {metric}</span>)
          : <span className="muted">{STATUS_LABEL[run.status]}</span>}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const nodeTypes = { run: RunCard };

function Inner(props: {
  theme: Theme; revision: number; selected: string[]; toggleRun: (id: string) => void; onCompare: () => void;
  steps: string[]; params: URLSearchParams; setParams: (p: Record<string, string>) => void;
}) {
  const metric = props.params.get("metric") ?? "";
  const mode = (props.params.get("mode") ?? "max") as "min" | "max";
  const step = props.params.get("step") ?? "";
  const q = new URLSearchParams({ ...(metric && { metric }), mode, ...(step && { step }) }).toString();
  const { data: tree, error } = useApi<Tree>(`/api/tree?${q}`, [props.revision]);
  const [focus, setFocus] = useState<string | null>(null);
  const flow = useReactFlow();
  const set = (p: Partial<{ metric: string; mode: string; step: string }>) => {
    const next = { metric, mode, step, ...p };
    props.setParams(Object.fromEntries(Object.entries(next).filter(([, v]) => v)) as Record<string, string>);
  };

  // pick a sensible default metric the first time
  useEffect(() => {
    if (tree && !metric && tree.metrics.length) {
      const pref = tree.metrics.find((m) => /val|rmse|loss|auc|r2|acc/i.test(m)) ?? tree.metrics[0];
      set({ metric: pref, mode: /rmse|mae|loss|err/i.test(pref) ? "min" : "max" });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tree, metric]);

  const { nodes, edges } = useMemo(() => {
    if (!tree) return { nodes: [] as RunFlowNode[], edges: [] as FlowEdge[] };
    const pos = dagreLayout(tree.runs.map((r) => ({ id: r.id, width: W, height: H })), tree.edges, "TB");
    const nodes: RunFlowNode[] = tree.runs.map((r) => ({
      id: r.id, type: "run", position: pos[r.id], width: W, height: H, draggable: false,
      data: { run: r, metric: tree.metric, best: r.id === tree.best_run_id, picked: props.selected.includes(r.id) },
    }));
    const edges: FlowEdge[] = tree.edges.map((e) => ({
      id: `${e.source}->${e.target}`, source: e.source, target: e.target, markerEnd: { type: MarkerType.ArrowClosed },
      className: e.target === tree.best_run_id ? "flow-edge best" : "flow-edge",
    }));
    return { nodes, edges };
  }, [tree, props.selected]);

  useEffect(() => {
    const t = window.setTimeout(() => flow.fitView({ padding: 0.15 }), 50);
    return () => window.clearTimeout(t);
  }, [tree?.runs.length, step, flow]);

  const run = tree?.runs.find((r) => r.id === focus) ?? null;
  const best = tree?.runs.find((r) => r.id === tree.best_run_id) ?? null;

  return (
    <div className="split">
      <div className="canvas">
        <div className="toolbar">
          <label>Metric
            <select value={metric} onChange={(e) => set({ metric: e.target.value })}>
              <option value="">(none)</option>
              {tree?.metrics.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </label>
          <div className="seg">
            {(["min", "max"] as const).map((m) => (
              <button key={m} className={mode === m ? "on" : ""} onClick={() => set({ mode: m })}>{m === "min" ? "lower is better" : "higher is better"}</button>
            ))}
          </div>
          <label>Step
            <select value={step} onChange={(e) => set({ step: e.target.value })}>
              <option value="">All steps</option>
              {props.steps.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          {best && metric && (
            <span className="best-summary">★ best {metric}: <b>{fmtNum(tree!.best_value!)}</b> <span className="mono">{shortId(best.id)}</span></span>
          )}
          <span className="spacer" />
          <button disabled={props.selected.length < 2} onClick={props.onCompare}>Compare ({props.selected.length})</button>
        </div>
        {error && <div className="banner error">{error}</div>}
        {tree && tree.runs.length === 0 ? (
          <div className="empty"><h2>No runs yet</h2><p>Runs appear here as your assistant calls <code>log_run</code>.</p></div>
        ) : (
          <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} colorMode={props.theme} fitView minZoom={0.15}
            nodesConnectable={false} nodesDraggable={false} proOptions={{ hideAttribution: true }}
            onNodeClick={(_, n) => setFocus(n.id)} onPaneClick={() => setFocus(null)}>
            <Background gap={20} size={1} />
            <Controls showInteractive={false} />
          </ReactFlow>
        )}
        <div className="canvas-hint muted">Click a run for details · tick it to compare · arrows point from parent to child run</div>
      </div>
      {run && (
        <aside className="panel narrow">
          <div className="panel-head">
            <div>
              <div className="eyebrow">Run · {run.step}</div>
              <h2 className="mono small">{run.id}</h2>
              <span className={`badge status-${run.status}`}>{STATUS_LABEL[run.status]}</span>
              {run.id === tree?.best_run_id && <span className="badge best">★ best {metric}</span>}
            </div>
            <button className="icon" onClick={() => setFocus(null)}>✕</button>
          </div>
          <label className="check">
            <input type="checkbox" checked={props.selected.includes(run.id)} onChange={() => props.toggleRun(run.id)} /> Select for compare
          </label>
          <h3>Metrics</h3>
          <dl className="dl">{Object.entries(run.metrics).map(([k, v]) => <div key={k}><dt>{k}</dt><dd className="mono">{fmtNum(v)}</dd></div>)}
            {!Object.keys(run.metrics).length && <p className="muted">–</p>}</dl>
          <h3>Params</h3>
          <dl className="dl">{Object.entries(run.params).map(([k, v]) => <div key={k}><dt>{k}</dt><dd className="mono">{fmtVal(v)}</dd></div>)}
            {!Object.keys(run.params).length && <p className="muted">–</p>}</dl>
          {run.parent_run_id && <p><span className="muted">Parent</span> <a href="#" onClick={(e) => { e.preventDefault(); setFocus(run.parent_run_id); }} className="mono">{run.parent_run_id}</a></p>}
          {run.tags.length > 0 && <p>{run.tags.map((t) => <span key={t} className="chip">{t}</span>)}</p>}
          {run.notes && <p className="note">{run.notes}</p>}
          {run.artifacts.length > 0 && (<><h3>Artifacts</h3><ul className="arts">{run.artifacts.map((a) => <li key={a.path} className="mono">{a.path}</li>)}</ul></>)}
          <p className="muted small">Logged {fmtTime(run.created_at)}{run.git_commit ? ` · git ${run.git_commit.slice(0, 7)}` : ""}</p>
          <a href={`#/map?step=${encodeURIComponent(run.step)}`}>Open step on the map →</a>
        </aside>
      )}
    </div>
  );
}

export default function TreeView(props: Parameters<typeof Inner>[0]) {
  return (
    <ReactFlowProvider>
      <Inner {...props} />
    </ReactFlowProvider>
  );
}
