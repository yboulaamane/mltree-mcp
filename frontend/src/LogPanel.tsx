import { useEffect, useRef, useState } from "react";
import { api, apiUrl, useApi } from "./api";
import type { LogLine, Run } from "./types";
import type { Watch } from "./runctx";
import { STATUS_LABEL, fmtNum, fmtVal, shortId } from "./util";

const MAX_LINES = 5000;

function fmtDur(s: number | null | undefined): string {
  if (s == null) return "–";
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m}m ${Math.round(s % 60)}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

/** Follow one run's stdout/stderr over SSE, resuming at the last offset if the connection drops. */
function useRunLog(runId: string | null) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [run, setRun] = useState<Run | null>(null);
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    setLines([]);
    setRun(null);
    if (!runId) return;
    let next = 0;
    let es: EventSource | null = null;
    let closed = false;
    let retry: number | undefined;
    const open = () => {
      es = new EventSource(apiUrl(`/api/runs/${encodeURIComponent(runId)}/log/stream?offset=${next}`));
      es.onopen = () => setConnected(true);
      es.addEventListener("lines", (e) => {
        const d = JSON.parse((e as MessageEvent).data) as { lines: LogLine[]; next: number };
        next = d.next;
        setLines((ls) => {
          const all = ls.concat(d.lines);
          return all.length > MAX_LINES ? all.slice(-MAX_LINES) : all;
        });
      });
      es.addEventListener("end", (e) => {
        setRun(JSON.parse((e as MessageEvent).data) as Run);
        closed = true;
        es?.close();
        setConnected(false);
      });
      es.onerror = () => {
        es?.close();
        setConnected(false);
        if (!closed) retry = window.setTimeout(open, 1500);
      };
    };
    open();
    return () => {
      closed = true;
      es?.close();
      window.clearTimeout(retry);
    };
  }, [runId]);
  return { lines, run, connected };
}

function LogView({ lines }: { lines: LogLine[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines]);
  return (
    <div className="log" ref={ref} onScroll={(e) => {
      const el = e.currentTarget;
      stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
    }}>
      {lines.length === 0 ? <span className="muted">waiting for output…</span>
        : lines.map((l, i) => <div key={i} className={l.s === "err" ? "err" : ""}>{l.line || "\u00a0"}</div>)}
    </div>
  );
}

function useTicker(on: boolean) {
  const [, set] = useState(0);
  useEffect(() => {
    if (!on) return;
    const t = window.setInterval(() => set((x) => x + 1), 1000);
    return () => window.clearInterval(t);
  }, [on]);
}

function RunWatch({ id, revision }: { id: string; revision: number }) {
  const { lines, run: ended } = useRunLog(id);
  const { data: rec } = useApi<Run>(`/api/runs/${encodeURIComponent(id)}`, [revision, !!ended]);
  const run = ended ?? rec;
  const running = !ended && run?.status === "running";
  useTicker(running);
  const elapsed = run?.duration_s ?? (run?.started_at ? (Date.now() - new Date(run.started_at).getTime()) / 1000 : null);
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="watch">
      <div className="watch-head">
        <span className={`badge status-${run?.status ?? "running"}`}>{STATUS_LABEL[run?.status ?? "running"]}</span>
        {run?.exit_code != null && <span className="muted">exit {run.exit_code}</span>}
        <span className="muted">{fmtDur(elapsed)}</span>
        <span className="metrics-inline">
          {Object.entries(run?.metrics ?? {}).slice(0, 6).map(([k, v]) => <span key={k}>{k} <b>{fmtNum(v)}</b></span>)}
        </span>
        <span className="spacer" />
        {err && <span className="error small">{err}</span>}
        {running && (
          <button className="danger" onClick={() => api(`/api/runs/${encodeURIComponent(id)}/cancel`, { method: "POST", body: "{}" })
            .catch((e: Error) => setErr(e.message))}>■ Cancel</button>
        )}
        <a href={`#/tree?step=${encodeURIComponent(run?.step ?? "")}`}>Show in tree →</a>
      </div>
      {run?.command && <div className="mono small muted cmdline" title={run.command.join(" ")}>$ {[run.command[0].split(/[\\/]/).pop(), ...run.command.slice(1)].join(" ")}</div>}
      <LogView lines={lines} />
    </div>
  );
}

interface SweepProgress { sweep_id: string; total: number; launched: number; current: string | null; done: boolean; cancelled: boolean }

function SweepWatch({ id, revision }: { id: string; revision: number }) {
  const { data: sweep } = useApi<Run>(`/api/runs/${encodeURIComponent(id)}`, [revision]);
  const { data: kids } = useApi<{ runs: Run[] }>(`/api/runs?parent_run_id=${encodeURIComponent(id)}&limit=500`, [revision]);
  const [prog, setProg] = useState<SweepProgress | null>(null);
  const [pick, setPick] = useState<string | null>(null);
  const running = sweep?.status === "running";
  useEffect(() => {
    if (!running) return;
    const tick = () => api<{ sweeps: SweepProgress[] }>("/api/active").then((a) => setProg(a.sweeps.find((s) => s.sweep_id === id) ?? null)).catch(() => {});
    tick();
    const t = window.setInterval(tick, 1500);
    return () => window.clearInterval(t);
  }, [running, id]);
  const children = [...(kids?.runs ?? [])].reverse();
  const metric = sweep?.sweep?.metric ?? null;
  const bestId = sweep?.sweep?.best_run_id ?? null;
  const shown = pick ?? prog?.current ?? children[children.length - 1]?.id ?? null;
  const total = sweep?.sweep?.total ?? prog?.total ?? children.length;
  const gridKeys = Object.keys(sweep?.sweep?.grid ?? {});
  return (
    <div className="watch sweep-watch">
      <div className="watch-head">
        <span className={`badge status-${sweep?.status ?? "running"}`}>{STATUS_LABEL[sweep?.status ?? "running"]}</span>
        <span><b>{children.filter((c) => c.status !== "running").length}</b> / {total} runs</span>
        <progress max={total || 1} value={children.filter((c) => c.status !== "running").length} />
        {metric && sweep?.metrics[`best_${metric}`] !== undefined && <span>best {metric} <b>{fmtNum(sweep.metrics[`best_${metric}`])}</b></span>}
        <span className="spacer" />
        {running && <button className="danger" onClick={() => api(`/api/runs/${encodeURIComponent(id)}/cancel`, { method: "POST", body: "{}" })}>■ Cancel sweep</button>}
        <a href={`#/tree?step=${encodeURIComponent(sweep?.step ?? "")}`}>Show in tree →</a>
      </div>
      <div className="sweep-body">
        <div className="sweep-list">
          <table className="runs">
            <tbody>
              {children.map((c, i) => (
                <tr key={c.id} className={(c.id === shown ? "active " : "") + (c.id === bestId ? "best-row" : "")} onClick={() => setPick(c.id)}>
                  <td className="muted">{i + 1}</td>
                  <td><span className={`dot status-${c.status}`} /></td>
                  <td className="mono">{gridKeys.map((k) => `${k}=${fmtVal(c.params[k])}`).join(" ") || shortId(c.id)}</td>
                  <td className="num mono">{metric ? fmtVal(c.metrics[metric]) : ""}{c.id === bestId && " ★"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="sweep-log">{shown ? <RunWatch key={shown} id={shown} revision={revision} /> : <p className="muted pad">Starting…</p>}</div>
      </div>
    </div>
  );
}

export default function LogPanel(props: { watches: Watch[]; revision: number; onClose: (id: string) => void }) {
  const [active, setActive] = useState<string | null>(null);
  const [min, setMin] = useState(false);
  const last = props.watches[props.watches.length - 1];
  useEffect(() => {
    if (last) {
      setActive(last.id);
      setMin(false);
    }
  }, [last?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!props.watches.length) return null;
  const cur = props.watches.find((w) => w.id === active) ?? last;
  return (
    <section className={min ? "logpanel min" : "logpanel"}>
      <div className="logtabs">
        {props.watches.map((w) => (
          <span key={w.id} className={w.id === cur.id ? "logtab on" : "logtab"} onClick={() => { setActive(w.id); setMin(false); }}>
            {w.kind === "sweep" ? "Sweep" : "Run"} · {w.step} <span className="mono muted">{shortId(w.id)}</span>
            <button className="x" title="Close" onClick={(e) => { e.stopPropagation(); props.onClose(w.id); }}>✕</button>
          </span>
        ))}
        <span className="spacer" />
        <button className="icon" onClick={() => setMin(!min)} title={min ? "Expand" : "Minimise"}>{min ? "▴" : "▾"}</button>
      </div>
      {!min && (cur.kind === "sweep" ? <SweepWatch key={cur.id} id={cur.id} revision={props.revision} />
        : <RunWatch key={cur.id} id={cur.id} revision={props.revision} />)}
    </section>
  );
}
