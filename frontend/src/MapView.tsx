import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Background, Controls, Handle, MarkerType, MiniMap, Position, ReactFlow, ReactFlowProvider,
  useNodesState, useReactFlow, type Edge as FlowEdge, type Node, type NodeProps,
} from "@xyflow/react";
import { api } from "./api";
import type { Pipeline, StepNode } from "./types";
import { dagreLayout } from "./layout";
import { STATUS_LABEL, firstMetric, fmtNum } from "./util";
import SidePanel from "./SidePanel";
import type { Theme } from "./App";

const W = 220;
const H = 92;

type StepData = { step: StepNode };
type StepFlowNode = Node<StepData, "step">;

function StepCard({ data, selected }: NodeProps<StepFlowNode>) {
  const s = data.step;
  const m = firstMetric(s.latest_run?.metrics);
  return (
    <div className={`step-node status-${s.status}${selected ? " selected" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <div className="step-head">
        <span className="dot" />
        <span className="step-name">{s.name}</span>
        <span className="pill">{STATUS_LABEL[s.status]}</span>
      </div>
      <div className="step-desc">{s.description || s.script_path || "no script"}</div>
      <div className="step-foot">
        <span>{s.run_count} run{s.run_count === 1 ? "" : "s"}</span>
        {m && <span className="metric">{m[0]} <b>{fmtNum(m[1])}</b></span>}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { step: StepCard };

function layoutAll(p: Pipeline) {
  return dagreLayout(p.nodes.map((n) => ({ id: n.name, width: W, height: H })), p.edges, "LR");
}

function Inner(props: {
  pipeline: Pipeline; theme: Theme; revision: number; selected: string[];
  toggleRun: (id: string) => void; onCompare: () => void; initialStep: string | null;
}) {
  const { pipeline, theme } = props;
  const [nodes, setNodes, onNodesChange] = useNodesState<StepFlowNode>([]);
  const [active, setActive] = useState<string | null>(props.initialStep);
  const flow = useReactFlow();

  // Merge server data into local nodes; keep local positions so drags never jump back.
  useEffect(() => {
    const missing = pipeline.nodes.some((n) => !n.position);
    const auto = missing ? layoutAll(pipeline) : {};
    setNodes((prev) => {
      const local = new Map(prev.map((n) => [n.id, n]));
      return pipeline.nodes.map((s) => {
        const old = local.get(s.name);
        const position = old?.position ?? s.position ?? auto[s.name] ?? { x: 0, y: 0 };
        return { id: s.name, type: "step", position, data: { step: s }, selected: s.name === active, width: W, height: H };
      });
    });
  }, [pipeline, setNodes, active]);

  useEffect(() => {
    const t = window.setTimeout(() => flow.fitView({ padding: 0.2, duration: 0 }), 50);
    return () => window.clearTimeout(t);
  }, [pipeline.nodes.length, flow]);

  const edges: FlowEdge[] = useMemo(
    () =>
      pipeline.edges.map((e) => {
        const src = pipeline.nodes.find((n) => n.name === e.source);
        return {
          id: `${e.source}->${e.target}`, source: e.source, target: e.target, label: e.label || undefined,
          animated: src?.status === "running", markerEnd: { type: MarkerType.ArrowClosed },
          className: "flow-edge",
        };
      }),
    [pipeline],
  );

  const savePositions = useCallback(async (list: { id: string; position: { x: number; y: number } }[]) => {
    const positions = Object.fromEntries(list.map((n) => [n.id, { x: Math.round(n.position.x), y: Math.round(n.position.y) }]));
    try {
      await api("/api/positions", { method: "PUT", body: JSON.stringify({ positions }) });
    } catch (e) {
      console.error(e);
    }
  }, []);

  const relayout = useCallback(() => {
    const pos = layoutAll(pipeline);
    setNodes((ns) => ns.map((n) => ({ ...n, position: pos[n.id] })));
    savePositions(Object.entries(pos).map(([id, position]) => ({ id, position })));
    window.setTimeout(() => flow.fitView({ padding: 0.2, duration: 300 }), 30);
  }, [pipeline, setNodes, savePositions, flow]);

  const step = pipeline.nodes.find((n) => n.name === active) ?? null;

  return (
    <div className="split">
      <div className="canvas">
        {pipeline.nodes.length === 0 ? (
          <div className="empty">
            <h2>No steps yet</h2>
            <p>Ask your assistant to build the pipeline; nodes appear here as it calls <code>add_step</code>.</p>
          </div>
        ) : (
          <ReactFlow
            nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange}
            onNodeDragStop={(_, __, dragged) => {
              // first drag in a fresh project: also pin the auto-laid-out nodes so nothing jumps later
              const unsaved = nodes.filter((n) => !n.data.step.position && !dragged.some((d) => d.id === n.id));
              savePositions([...dragged, ...unsaved]);
            }}
            onNodeClick={(_, n) => setActive(n.id)} onPaneClick={() => setActive(null)}
            colorMode={theme} fitView minZoom={0.2} proOptions={{ hideAttribution: true }}
            nodesConnectable={false}
          >
            <Background gap={20} size={1} />
            <Controls showInteractive={false} />
            <MiniMap pannable zoomable nodeClassName={(n) => `mini status-${(n.data as StepData).step.status}`} />
          </ReactFlow>
        )}
        <div className="canvas-toolbar">
          <button onClick={relayout}>Auto layout</button>
          <Legend />
        </div>
      </div>
      {step && (
        <SidePanel step={step} theme={theme} revision={props.revision} selected={props.selected}
          toggleRun={props.toggleRun} onCompare={props.onCompare} onClose={() => setActive(null)} />
      )}
    </div>
  );
}

function Legend() {
  return (
    <div className="legend">
      {(["not_run", "running", "succeeded", "failed"] as const).map((s) => (
        <span key={s} className={`status-${s}`}><i className="dot" /> {STATUS_LABEL[s]}</span>
      ))}
    </div>
  );
}

export default function MapView(props: Parameters<typeof Inner>[0]) {
  return (
    <ReactFlowProvider>
      <Inner {...props} />
    </ReactFlowProvider>
  );
}
