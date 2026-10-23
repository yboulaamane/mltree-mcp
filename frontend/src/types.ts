export type Status = "not_run" | "pending" | "running" | "succeeded" | "failed" | "cancelled";

export interface LatestRun {
  id: string;
  status: Status;
  metrics: Record<string, number>;
  created_at: string;
}

export interface StepNode {
  name: string;
  kind: string;
  script_path: string | null;
  script_exists: boolean;
  description: string;
  inputs: string[];
  outputs: string[];
  params: Record<string, unknown>;
  position: { x: number; y: number } | null;
  status: Status;
  run_count: number;
  latest_run: LatestRun | null;
  data?: NodeData | null;
  notes?: string;                 // 0.7.0+
  gnn?: GnnBadge | null;          // 0.7.0+
}

export interface GnnBadge { label?: string; model?: string; datasets?: string[]; graphs?: number }

export interface GnnDim { text: string; value?: number | string | null }

export interface GnnLayer {
  attr: string; type: string; family: string; lib?: string; line?: number; file?: string; call?: string;
  repeat?: number | string; in?: GnnDim; out?: GnnDim; heads?: GnnDim; p?: GnnDim; mlp?: string[];
}

export interface GnnModel {
  name: string; kind: "class" | "prebuilt"; file: string; line: number; bases?: string[];
  layers?: GnnLayer[]; forward?: string[]; call?: string;
  instantiated?: { file: string; line: number; call: string } | null;
  summary: { label: string; conv: Record<string, number>; num_conv_layers: number; hidden: string[]; heads: string[];
    pooling: string[]; head: string | null; is_gnn: boolean };
}

export interface GnnRange { min: number; mean: number; max: number; total: number }

export interface GnnStats {
  format?: string; error?: string; num_graphs?: number; num_nodes?: number; num_edges?: number;
  nodes?: GnnRange; edges?: GnnRange; node_features?: number | null; edge_features?: number | null;
  self_loops?: number; isolated_nodes?: number; avg_degree?: number; is_directed?: boolean;
  labels?: { task?: string; num_classes?: number; class_counts?: Record<string, number>; missing?: number; targets?: number; min?: number; max?: number;
    mean?: number; std?: number; shape?: number[] } | null;
  masks?: Record<string, number>; level?: string; parameters?: number; tensors?: number; loaded_with?: string; versions?: Record<string, string>;
  truncated?: boolean;
}

export interface GnnView {
  step: string;
  models: GnnModel[];
  datasets: { class: string; name?: string | null; root?: string | null; lib?: string; file: string; line: number; call?: string }[];
  files: string[];
  warnings: string[];
  dataset_files: { path: string; bytes?: number; version?: string | null; stats: GnnStats | null; computed_at: string | null;
    python: string | null; stale: boolean; missing?: boolean }[];
  can_introspect: boolean;
  introspect_note: string;
}

/** Content fingerprint of one data file/dir (SHA-256; `version` = first 12 hex chars). */
export interface DataFp {
  path: string | null;
  exists: boolean;
  kind?: "file" | "dir";
  bytes?: number;
  files?: number;
  mtime?: string;
  sha256?: string;
  version?: string;
  quick?: boolean;
}

export interface NodeData {
  status: "ok" | "changed" | "missing" | "no_record";
  last_run_id: string | null;
  changed_inputs: string[];
  missing_inputs: string[];
  changed_outputs: string[];
  versions: Record<string, string | null | undefined>;
}

export interface DataItem extends DataFp {
  name: string;
  run_version: string | null;
  changed: boolean;
  producer_run_id?: string;
  also_found?: string[];
}

export interface StepData {
  step: string;
  status: NodeData["status"];
  last_run_id: string | null;
  inputs: DataItem[];
  outputs: DataItem[];
  changed_inputs: string[];
  changed_outputs: string[];
  missing_inputs: string[];
}

export interface DataStatus {
  steps: Record<string, StepData>;
  data: Record<string, { name: string; path: string | null; exists: boolean; version: string | null; bytes: number | null;
    produced_by_steps: string[]; used_by_steps: string[];
    versions: { version: string; first_seen: string; produced_by: string[]; used_by: string[] }[] }>;
  changed_steps: string[];
  summary: string;
}

export interface Edge {
  source: string;
  target: string;
  label: string;
}

export interface Pipeline {
  project: string;
  root: string;
  nodes: StepNode[];
  edges: Edge[];
  order: string[];
  tags?: Record<string, number>;  // 0.7.0+: tag -> number of runs
}

export interface Artifact {
  name: string;
  path: string;
  stored: string;
  bytes: number | null;
}

export interface Run {
  id: string;
  step: string;
  status: Status;
  params: Record<string, unknown>;
  metrics: Record<string, number>;
  artifacts: Artifact[];
  parent_run_id: string | null;
  notes: string;
  tags: string[];
  script_sha256: string | null;
  git_commit: string | null;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
  kind?: "run" | "sweep";
  sweep_id?: string | null;
  sweep?: SweepSpec | null;
  command?: string[] | null;
  exit_code?: number | null;
  duration_s?: number | null;
  data?: { inputs?: Record<string, DataFp>; outputs?: Record<string, DataFp> } | null;
}

export interface SweepSpec {
  source?: string;
  metric?: string | null;
  mode?: "min" | "max";
  grid?: Record<string, unknown[]> | null;
  name?: string | null;
  total?: number;
  best_run_id?: string | null;
}

export interface SweepSummary {
  sweep_id: string;
  n: number;
  counts: Record<string, number>;
  metric: string | null;
  mode: "min" | "max";
  best_run_id: string | null;
  best_value: number | null;
  best_params: Record<string, unknown> | null;
}

export interface Plan {
  step: string;
  params: Record<string, unknown>;
  defaults: { params: Record<string, unknown>; source: string; source_run_id: string | null };
  runnable: boolean;
  reason: string | null;
  argv: string[] | null;
  command?: string;
  cwd: string;
  style?: string;
  python?: string;
  warnings: string[];
  parent_run_id: string | null;
  cli_options?: string[];
  allow_run: boolean;
}

export interface LogLine {
  t: number | null;
  s: "out" | "err";
  line: string;
}

export interface Tree {
  runs: Run[];
  edges: { source: string; target: string }[];
  roots: string[];
  metric: string | null;
  mode: "min" | "max";
  best_run_id: string | null;
  best_value: number | null;
  metrics: string[];
  sweeps?: Record<string, SweepSummary>;
}

export interface Script {
  step: string;
  source: "current" | "snapshot";
  run_id: string | null;
  path: string;
  content: string;
  sha256: string;
  created_at: string | null;
}

export interface Snapshot {
  run_id: string;
  created_at: string;
  status: Status;
  sha256: string;
  same_as_current: boolean;
}

export interface Compare {
  runs: Run[];
  param_keys: string[];
  metric_keys: string[];
  differing_params: string[];
  differing_metrics: string[];
  data?: { name: string; role: "input" | "output"; versions: (string | null)[]; differs: boolean }[];
  differing_data?: string[];
  param_changes?: { run_id: string; changes: { key: string; from: unknown; to: unknown }[] }[];
  has_history?: Record<string, boolean>;
}

export type HistoryPoint = Record<string, number>;
