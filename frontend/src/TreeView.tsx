import { useEffect, useMemo, useState } from "react";
import {
  Background, Controls, Handle, MarkerType, Position, ReactFlow, ReactFlowProvider, useReactFlow,
  type Edge as FlowEdge, type Node, type NodeProps,
} from "@xyflow/react";
import { useApi } from "./api";
import type { Run, SweepSummary, Tree } from "./types";
import { dagreLayout } from "./layout";
import { STATUS_LABEL, fmtNum, fmtTime, fmtVal, shortId } from "./util";
import type { Theme } from "./App";
import { NotesEditor, TagEditor } from "./Notes";

const W = 190;
const H = 78;

const SW = 236;
const SH = 96;
const AUTO_COLLAPSE = 8; // sweeps with more children start folded to their best child

type RunData = { run: Run; metric: string | null; best: boolean; picked: boolean; sweepBest: boolean; dim?: boolean;
  sweep?: SweepSummary; collapsed?: boolean; hidden?: number; toggle?: () => void };
type RunFlowNode = Node<RunData, "run" | "sweep">;

const SOURCE_LABEL: Record<string, string> = { grid: "grid sweep", optuna: "Optuna study", grid_search: "grid search CV" };

function SweepCard({ data }: NodeProps<RunFlowNode>) {
  const { run, sweep, collapsed, hidden, toggle, picked } = data;
  const c = sweep?.counts ?? {};
  const label = SOURCE_LABEL[run.sweep?.source ?? ""] ?? "sweep";
  return (
    <div className={`run-node sweep-node status-${run.status}${picked ? " picked" : ""}${data.dim ? " dim" : ""}`}>
      <Handle type="target" position={Position.Top} />
      <div className="run-head">
        <span className="dot" />
        <span className="run-step">{run.step}</span>
        <span className="chip sweep-chip">{label}</span>
      </div>
      <div className="run-id muted">{run.sweep?.name || shortId(run.id)} · {sweep?.n ?? 0} runs
        {c.succeeded ? <span className="cnt ok"> ✓{c.succeeded}</span> : null}
        {c.failed ? <span className="cnt bad"> ✗{c.failed}</span> : null}
        {c.running ? <span className="cnt run"> ●{c.running}</span> : null}
      </div>
      <div className="run-metric">
        {sweep?.metric && sweep.best_value != null ? <>best {sweep.metric} <b>{fmtNum(sweep.best_value)}</b></> : <span className="muted">{STATUS_LABEL[run.status]}</span>}
      </div>
      {(sweep?.n ?? 0) > 1 && (
        <button className="fan-toggle nodrag" onClick={(e) => { e.stopPropagation(); toggle?.(); }}>
          {collapsed ? `▸ show all ${sweep?.n} (${hidden} hidden)` : "▾ fold to best"}
        </button>
      )}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

function RunCard({ data }: NodeProps<RunFlowNode>) {
  const { run, metric, best, picked, sweepBest } = data;
  const v = metric ? run.metrics[metric] : undefined;
  const tags = run.tags.filter((t) => t !== "mltree-run");
  return (
    <div className={`run-node status-${run.status}${best ? " best" : ""}${sweepBest && !best ? " sweep-best" : ""}${picked ? " picked" : ""}${data.dim ? " dim" : ""}`}>
      <Handle type="target" position={Position.Top} />
      <div className="run-head">
        <span className="dot" />
        <span className="run-step">{run.step}</span>
        {best ? <span className="star" title="Best run">★ best</span>
          : sweepBest ? <span className="star soft" title="Best run of its sweep">☆ sweep best</span> : null}
      </div>
      <div className="run-id mono">{shortId(run.id)}{run.notes && <span className="note-dot" title={run.notes}> ✎</span>}
        {tags.slice(0, 2).map((t) => <span key={t} className="chip tag tiny">{t}</span>)}{tags.length > 2 && <span className="muted"> +{tags.length - 2}</span>}</div>
      <div className="run-metric">
        {metric ? (v !== undefined ? <>{metric} <b>{fmtNum(v)}</b></> : <span className="muted">no {metric}</span>)
          : <span className="muted">{STATUS_LABEL[run.status]}</span>}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const nodeTypes = { run: RunCard, sweep: SweepCard };

function Inner(props: {
  theme: Theme; revision: number; selected: string[]; toggleRun: (id: string) => void; onCompare: () => void;
  steps: string[]; params: URLSearchParams; setParams: (p: Record<string, string>) => void;
}) {
  const metric = props.params.get("metric") ?? "";
  const mode = (props.params.get("mode") ?? "max") as "min" | "max";
  const step = props.params.get("step") ?? "";
  const tag = props.params.get("tag") ?? "";
  const q = new URLSearchParams({ ...(metric && { metric }), mode, ...(step && { step }) }).toString();
  const { data: tree, error } = useApi<Tree>(`/api/tree?${q}`, [props.revision]);
  const [focus, setFocus] = useState<string | null>(null);
  const [fold, setFold] = useState<Record<string, boolean>>({}); // sweep id -> collapsed (unset = auto)
  const flow = useReactFlow();
  const set = (p: Partial<{ metric: string; mode: string; step: string; tag: string }>) => {
    const next = { metric, mode, step, tag, ...p };
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
    const sweeps = tree.sweeps ?? {};
    const kids = new Map<string, string[]>();
    tree.edges.forEach((e) => kids.set(e.source, [...(kids.get(e.source) ?? []), e.target]));
    const sweepBest = new Set(Object.values(sweeps).map((s) => s.best_run_id).filter(Boolean) as string[]);
    // fold collapsed sweeps down to their best child (and that child's own subtree)
    const hide = new Set<string>();
    const hiddenCount: Record<string, number> = {};
    const isCollapsed = (id: string) => fold[id] ?? (sweeps[id]?.n ?? 0) > AUTO_COLLAPSE;
    const drop = (id: string): number => {
      if (hide.has(id)) return 0;
      hide.add(id);
      return 1 + (kids.get(id) ?? []).reduce((n: number, k) => n + drop(k), 0);
    };
    for (const sid of Object.keys(sweeps)) {
      if (!isCollapsed(sid)) continue;
      const keep = sweeps[sid].best_run_id;
      hiddenCount[sid] = (kids.get(sid) ?? []).filter((k) => k !== keep).reduce((n, k) => n + drop(k), 0);
    }
    const runs = tree.runs.filter((r) => !hide.has(r.id));
    const vedges = tree.edges.filter((e) => !hide.has(e.source) && !hide.has(e.target));
    const isSweep = (r: Run) => r.kind === "sweep" || !!sweeps[r.id];
    const pos = dagreLayout(runs.map((r) => (isSweep(r) ? { id: r.id, width: SW, height: SH } : { id: r.id, width: W, height: H })), vedges, "TB");
    const nodes: RunFlowNode[] = runs.map((r) => {
      const sw = isSweep(r);
      return {
        id: r.id, type: sw ? "sweep" : "run", position: pos[r.id], width: sw ? SW : W, height: sw ? SH : H, draggable: false,
        data: {
          run: r, metric: tree.metric, best: r.id === tree.best_run_id, picked: props.selected.includes(r.id),
          sweepBest: sweepBest.has(r.id), sweep: sweeps[r.id], dim: !!tag && !r.tags.includes(tag), collapsed: sw && isCollapsed(r.id), hidden: hiddenCount[r.id] ?? 0,
          toggle: () => setFold((f) => ({ ...f, [r.id]: !isCollapsed(r.id) })),
        },
      };
    });
    const edges: FlowEdge[] = vedges.map((e) => ({
      id: `${e.source}->${e.target}`, source: e.source, target: e.target, markerEnd: { type: MarkerType.ArrowClosed },
      className: e.target === tree.best_run_id ? "flow-edge best" : sweeps[e.source] ? (sweepBest.has(e.target) ? "flow-edge sweep best-soft" : "flow-edge sweep") : "flow-edge",
    }));
    return { nodes, edges };
  }, [tree, props.selected, fold, tag]);
  const allTags = useMemo(() => {
    const c: Record<string, number> = {};
    tree?.runs.forEach((r) => r.tags.forEach((t) => { if (t !== "mltree-run") c[t] = (c[t] ?? 0) + 1; }));
    return Object.entries(c).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1));
  }, [tree]);

  useEffect(() => {
    const t = window.setTimeout(() => flow.fitView({ padding: 0.15 }), 50);
    return () => window.clearTimeout(t);
  }, [tree?.runs.length, nodes.length, step, flow]);

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
          {(allTags.length > 0 || tag) && (
            <label>Tag
              <select value={tag} onChange={(e) => set({ tag: e.target.value })} title="Dim runs without this tag">
                <option value="">(any)</option>
                {allTags.map(([t, n]) => <option key={t} value={t}>{t} ({n})</option>)}
                {tag && !allTags.some(([t]) => t === tag) && <option value={tag}>{tag} (0)</option>}
              </select>
            </label>
          )}
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
          {tree?.sweeps?.[run.id] && <SweepInfo run={run} s={tree.sweeps[run.id]} onPick={setFocus} />}
          {run.sweep_id && run.sweep_id !== run.parent_run_id && <p className="muted small">Part of sweep <span className="mono">{shortId(run.sweep_id)}</span></p>}
          {(run.duration_s != null || run.exit_code != null) && (
            <p className="muted small">{run.duration_s != null && <>took {fmtNum(run.duration_s)}s</>}{run.exit_code != null && <> · exit code {run.exit_code}</>}</p>
          )}
          <h3>Metrics</h3>
          <dl className="dl">{Object.entries(run.metrics).map(([k, v]) => <div key={k}><dt>{k}</dt><dd className="mono">{fmtNum(v)}</dd></div>)}
            {!Object.keys(run.metrics).length && <p className="muted">–</p>}</dl>
          <h3>Params</h3>
          <dl className="dl">{Object.entries(run.params).map(([k, v]) => <div key={k}><dt>{k}</dt><dd className="mono">{fmtVal(v)}</dd></div>)}
            {!Object.keys(run.params).length && <p className="muted">–</p>}</dl>
          {run.parent_run_id && <p><span className="muted">Parent</span> <a href="#" onClick={(e) => { e.preventDefault(); setFocus(run.parent_run_id); }} className="mono">{run.parent_run_id}</a></p>}
          <h3>Tags</h3>
          <TagEditor tags={run.tags} runId={run.id} known={allTags.map(([t]) => t)} />
          <NotesEditor notes={run.notes} path={`/api/runs/${encodeURIComponent(run.id)}/notes`} label="Run notes" />
          {run.artifacts.length > 0 && (<><h3>Artifacts</h3><ul className="arts">{run.artifacts.map((a) => <li key={a.path} className="mono">{a.path}</li>)}</ul></>)}
          <p className="muted small">Logged {fmtTime(run.created_at)}{run.git_commit ? ` · git ${run.git_commit.slice(0, 7)}` : ""}</p>
          <a href={`#/map?step=${encodeURIComponent(run.step)}`}>Open step on the map →</a>
        </aside>
      )}
    </div>
  );
}

function SweepInfo({ run, s, onPick }: { run: Run; s: SweepSummary; onPick: (id: string) => void }) {
  const grid = run.sweep?.grid ?? {};
  return (
    <>
      <h3>Sweep</h3>
      <dl className="dl">
        <div><dt>source</dt><dd>{SOURCE_LABEL[run.sweep?.source ?? ""] ?? run.sweep?.source ?? "–"}</dd></div>
        <div><dt>runs</dt><dd>{s.n} ({Object.entries(s.counts).map(([k, v]) => `${v} ${STATUS_LABEL[k as Run["status"]] ?? k}`).join(", ")})</dd></div>
        {s.metric && <div><dt>ranked by</dt><dd className="mono">{s.metric} ({s.mode === "min" ? "lower" : "higher"} is better)</dd></div>}
        {Object.entries(grid).map(([k, v]) => <div key={k}><dt>{k}</dt><dd className="mono">{(v as unknown[]).map(fmtVal).join(", ")}</dd></div>)}
      </dl>
      {s.best_run_id && (
        <p>★ Best: <a href="#" className="mono" onClick={(e) => { e.preventDefault(); onPick(s.best_run_id!); }}>{shortId(s.best_run_id)}</a>
          {s.best_value != null && <> · {s.metric} <b>{fmtNum(s.best_value)}</b></>}
          {s.best_params && <span className="muted"> · {Object.entries(s.best_params).map(([k, v]) => `${k}=${fmtVal(v)}`).join(" ")}</span>}</p>
      )}
    </>
  );
}

export default function TreeView(props: Parameters<typeof Inner>[0]) {
  return (
    <ReactFlowProvider>
      <Inner {...props} />
    </ReactFlowProvider>
  );
}
