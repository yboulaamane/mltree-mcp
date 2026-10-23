import { useEffect, useMemo, useState } from "react";
import { api, useApi } from "./api";
import type { Plan, StepNode } from "./types";
import { fmtVal, shortId } from "./util";
import type { Watch } from "./runctx";

type Row = { key: string; value: string };

const toRows = (p: Record<string, unknown>): Row[] =>
  Object.entries(p).map(([key, v]) => ({ key, value: typeof v === "string" ? v : JSON.stringify(v) }));

/** "0.1" -> 0.1, "true" -> true, '"x"' -> "x", anything else stays a string */
function parseVal(s: string): unknown {
  const t = s.trim();
  if (t === "") return "";
  try {
    return JSON.parse(t);
  } catch {
    return s;
  }
}

function rowsToParams(rows: Row[]): Record<string, unknown> {
  return Object.fromEntries(rows.filter((r) => r.key.trim()).map((r) => [r.key.trim(), parseVal(r.value)]));
}

/** "0.1, 1, 10" or "[0.1, 1]" -> list */
function parseList(s: string): unknown[] {
  const t = s.trim();
  if (!t) return [];
  if (t.startsWith("[")) {
    try {
      const v = JSON.parse(t);
      return Array.isArray(v) ? v : [v];
    } catch {
      /* fall through */
    }
  }
  return t.split(",").map((x) => x.trim()).filter(Boolean).map(parseVal);
}

const CONFIRM_KEY = "mltree-run-confirmed";

export default function RunDialog(props: {
  step: StepNode; tab: "run" | "sweep"; onClose: () => void; onStarted: (w: Watch) => void;
}) {
  const { step } = props;
  const enc = encodeURIComponent(step.name);
  const [tab, setTab] = useState(props.tab);
  const { data: base, error: baseErr } = useApi<Plan>(`/api/steps/${enc}/plan`, []);
  const { data: mlist } = useApi<{ metrics: string[] }>("/api/metrics", []);
  const [rows, setRows] = useState<Row[] | null>(null);
  const [extra, setExtra] = useState("");
  const [grid, setGrid] = useState<Record<string, string>>({});
  const [metric, setMetric] = useState("");
  const [mode, setMode] = useState<"min" | "max">("max");
  const [plan, setPlan] = useState<Plan | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [ok, setOk] = useState(() => sessionStorage.getItem(CONFIRM_KEY) === "1");

  useEffect(() => {
    if (base && rows === null) {
      setRows(toRows(base.defaults.params));
      setPlan(base);
    }
  }, [base, rows]);
  useEffect(() => {
    if (!metric && mlist?.metrics.length) {
      const m = (step.latest_run && Object.keys(step.latest_run.metrics)[0]) || mlist.metrics.find((x) => /val|rmse|loss|auc|r2|acc/i.test(x)) || "";
      setMetric(m);
      setMode(/rmse|mae|loss|err/i.test(m) ? "min" : "max");
    }
  }, [mlist, metric, step.latest_run]);

  const params = useMemo(() => rowsToParams(rows ?? []), [rows]);
  // live command preview
  useEffect(() => {
    if (rows === null) return;
    const t = window.setTimeout(() => {
      const q = new URLSearchParams({ params: JSON.stringify(params), ...(extra && { extra_args: extra }) });
      api<Plan>(`/api/steps/${enc}/plan?${q}`).then(setPlan).catch((e: Error) => setErr(e.message));
    }, 250);
    return () => window.clearTimeout(t);
  }, [params, extra, enc, rows]);

  const gridLists = useMemo(() => {
    const out: Record<string, unknown[]> = {};
    const names = new Set((rows ?? []).map((r) => r.key.trim()));
    for (const [k, v] of Object.entries(grid)) {
      const list = parseList(v);
      if (list.length && names.has(k)) out[k] = list;
    }
    return out;
  }, [grid, rows]);
  const combos: number = Object.values(gridLists).reduce((n, v) => n * v.length, Object.keys(gridLists).length ? 1 : 0);

  const launch = async () => {
    setBusy(true);
    setErr(null);
    try {
      if (tab === "run") {
        const r = await api<{ run_id: string }>(`/api/steps/${enc}/run`, {
          method: "POST", body: JSON.stringify({ params, extra_args: extra || undefined }),
        });
        props.onStarted({ kind: "run", id: r.run_id, step: step.name });
      } else {
        const fixed = Object.fromEntries(Object.entries(params).filter(([k]) => !(k in gridLists)));
        const r = await api<{ sweep_id: string }>(`/api/steps/${enc}/sweep`, {
          method: "POST", body: JSON.stringify({ grid: gridLists, params: fixed, metric: metric || undefined, mode, extra_args: extra || undefined }),
        });
        props.onStarted({ kind: "sweep", id: r.sweep_id, step: step.name });
      }
      sessionStorage.setItem(CONFIRM_KEY, "1");
      props.onClose();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const setRow = (i: number, patch: Partial<Row>) => setRows((rs) => (rs ?? []).map((r, j) => (j === i ? { ...r, ...patch } : r)));
  const canLaunch = !!plan?.runnable && plan.allow_run && ok && !busy && (tab === "run" || combos > 0);

  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && props.onClose()}>
      <div className="modal" role="dialog" aria-label={`Run ${step.name}`}>
        <div className="panel-head">
          <div>
            <div className="eyebrow">Run step</div>
            <h2>{step.name} <span className="muted mono small">{step.script_path}</span></h2>
          </div>
          <button className="icon" onClick={props.onClose} title="Close">✕</button>
        </div>
        <div className="seg modal-tabs">
          <button className={tab === "run" ? "on" : ""} onClick={() => setTab("run")}>Single run</button>
          <button className={tab === "sweep" ? "on" : ""} onClick={() => setTab("sweep")}>Grid sweep</button>
        </div>
        {baseErr && <div className="banner error">{baseErr}</div>}
        {base && !base.runnable && <div className="banner error">{base.reason}</div>}
        {base && !base.allow_run && <div className="banner warn">Running is off. Restart the viewer with <code>mltree-mcp view --allow-run</code>.</div>}

        {rows && (
          <>
            <h3>Params <span className="muted small">· prefilled from {base?.defaults.source}{base?.defaults.source_run_id ? ` ${shortId(base.defaults.source_run_id)}` : ""}</span></h3>
            <table className="form-table">
              <thead><tr><th>Name</th><th>{tab === "sweep" ? "Fixed value" : "Value"}</th>{tab === "sweep" && <th>Sweep values <span className="muted">(comma separated)</span></th>}<th /></tr></thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td><input value={r.key} onChange={(e) => setRow(i, { key: e.target.value })} className="mono" /></td>
                    <td><input value={r.value} onChange={(e) => setRow(i, { value: e.target.value })} className="mono"
                      disabled={tab === "sweep" && !!gridLists[r.key]} /></td>
                    {tab === "sweep" && (
                      <td><input placeholder={`e.g. ${fmtVal(parseVal(r.value))}, …`} value={grid[r.key] ?? ""} className="mono"
                        onChange={(e) => setGrid((g) => ({ ...g, [r.key]: e.target.value }))} /></td>
                    )}
                    <td><button className="icon" title="Remove" onClick={() => setRows((rs) => (rs ?? []).filter((_, j) => j !== i))}>−</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button className="ghost small-btn" onClick={() => setRows((rs) => [...(rs ?? []), { key: "", value: "" }])}>+ param</button>
            {tab === "sweep" && (
              <div className="sweep-opts">
                <span><b>{combos}</b> run{combos === 1 ? "" : "s"} (one after another)</span>
                <label>Rank by
                  <input list="mltree-metrics" value={metric} onChange={(e) => setMetric(e.target.value)} placeholder="metric" className="mono" />
                  <datalist id="mltree-metrics">{mlist?.metrics.map((m) => <option key={m} value={m} />)}</datalist>
                </label>
                <div className="seg">
                  {(["min", "max"] as const).map((m) => <button key={m} className={mode === m ? "on" : ""} onClick={() => setMode(m)}>{m === "min" ? "lower is better" : "higher is better"}</button>)}
                </div>
              </div>
            )}
            <h3>Extra arguments</h3>
            <input className="mono wide" value={extra} onChange={(e) => setExtra(e.target.value)} placeholder="appended verbatim, e.g. --epochs 5" />
            <h3>Command</h3>
            <pre className="cmd">{plan?.command ?? "…"}</pre>
            <p className="muted small">
              Runs in <span className="mono">{plan?.cwd}</span> · params passed as {plan?.style === "flags" ? "CLI flags" : plan?.style === "argv" ? "positional arguments" : plan?.style === "hydra" ? "hydra overrides" : "environment only"} and as <code>$MLTREE_PARAMS</code>.
              {plan?.parent_run_id ? <>Branches from upstream run <span className="mono">{shortId(plan.parent_run_id)}</span> in the experiment tree. </> : null}
              Print <code>MLTREE_METRIC name=value</code> lines or write JSON to <code>$MLTREE_METRICS_FILE</code> to record metrics.
            </p>
            {plan?.warnings.map((w) => <div key={w} className="banner warn small">{w}</div>)}
          </>
        )}
        {err && <div className="banner error">{err}</div>}
        <div className="modal-foot">
          <label className="check">
            <input type="checkbox" checked={ok} onChange={(e) => setOk(e.target.checked)} />
            I understand this executes <span className="mono">{step.script_path}</span> on this machine
          </label>
          <span className="spacer" />
          <button className="ghost" onClick={props.onClose}>Cancel</button>
          <button className="primary" disabled={!canLaunch} onClick={launch}>
            {busy ? "Starting…" : tab === "run" ? "▶ Run" : `▶ Launch ${combos} runs`}
          </button>
        </div>
      </div>
    </div>
  );
}
