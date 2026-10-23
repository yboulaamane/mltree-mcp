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
}
