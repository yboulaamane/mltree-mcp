import { createContext, useContext } from "react";

export interface Watch {
  kind: "run" | "sweep";
  id: string;
  step: string;
}

export interface RunCtx {
  /** "on": viewer started with --allow-run; "off": live viewer without it; "none": exported snapshot */
  mode: "on" | "off" | "none";
  openRun: (step: string, tab?: "run" | "sweep") => void;
  watch: (w: Watch) => void;
}

export const RunContext = createContext<RunCtx>({ mode: "none", openRun: () => {}, watch: () => {} });
export const useRuns = () => useContext(RunContext);

export const RUN_OFF_HINT = "Running is off. Restart the viewer with: mltree-mcp view --allow-run";

export function runnable(scriptPath: string | null, exists: boolean): boolean {
  return !!scriptPath && exists && scriptPath.endsWith(".py");
}
