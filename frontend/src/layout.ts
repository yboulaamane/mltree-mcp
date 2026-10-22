import dagre from "@dagrejs/dagre";

export interface Box {
  id: string;
  width: number;
  height: number;
}

/** Layered DAG layout; returns top-left positions keyed by id. */
export function dagreLayout(
  boxes: Box[],
  edges: { source: string; target: string }[],
  rankdir: "LR" | "TB" = "LR",
): Record<string, { x: number; y: number }> {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir, nodesep: rankdir === "LR" ? 40 : 28, ranksep: rankdir === "LR" ? 70 : 60, marginx: 20, marginy: 20 });
  g.setDefaultEdgeLabel(() => ({}));
  boxes.forEach((b) => g.setNode(b.id, { width: b.width, height: b.height }));
  edges.forEach((e) => g.hasNode(e.source) && g.hasNode(e.target) && g.setEdge(e.source, e.target));
  dagre.layout(g);
  const out: Record<string, { x: number; y: number }> = {};
  boxes.forEach((b) => {
    const n = g.node(b.id);
    out[b.id] = { x: n.x - b.width / 2, y: n.y - b.height / 2 };
  });
  return out;
}
