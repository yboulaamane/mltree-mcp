/** Unified / side-by-side diff of two runs' script snapshots (computed in the browser; works in exports). */
import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type { Run, Script } from "./types";
import { diffLines, hunks, sideBySide, stats, type Op } from "./diff";
import { shortId } from "./util";

type Mode = "unified" | "split";

async function snapshot(r: Run): Promise<Script> {
  return api<Script>(`/api/steps/${encodeURIComponent(r.step)}/script?run_id=${encodeURIComponent(r.id)}`);
}

const Num = ({ n }: { n: number | null }) => <td className="ln">{n ?? ""}</td>;
const cls = (o: Op | null) => (o === null ? "empty" : o.t === "+" ? "add" : o.t === "-" ? "del" : "");

export default function CodeDiff(props: { runs: Run[] }) {
  const [a, setA] = useState(props.runs[0]?.id ?? "");
  const [b, setB] = useState(props.runs[1]?.id ?? "");
  const [mode, setMode] = useState<Mode>(() => (localStorage.getItem("mltree-diff-mode") as Mode) || "unified");
  const [full, setFull] = useState(false);
  const [texts, setTexts] = useState<{ a: Script; b: Script } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ids = props.runs.map((r) => r.id).join(",");
  useEffect(() => {
    if (!props.runs.some((r) => r.id === a)) setA(props.runs[0]?.id ?? "");
    if (!props.runs.some((r) => r.id === b)) setB(props.runs[1]?.id ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ids]);
  useEffect(() => localStorage.setItem("mltree-diff-mode", mode), [mode]);
  useEffect(() => {
    const ra = props.runs.find((r) => r.id === a), rb = props.runs.find((r) => r.id === b);
    if (!ra || !rb) return;
    let alive = true;
    setError(null);
    Promise.all([snapshot(ra), snapshot(rb)])
      .then(([x, y]) => alive && setTexts({ a: x, b: y }))
      .catch((e: Error) => alive && (setTexts(null), setError(e.message)));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [a, b, ids]);
  const ops = useMemo(() => (texts ? diffLines(texts.a.content, texts.b.content) : []), [texts]);
  const st = stats(ops);
  const hs = useMemo(() => (full ? [{ header: "", ops }] : hunks(ops, 3)), [ops, full]);
  const pick = (v: string, set: (s: string) => void, label: string) => (
    <select value={v} onChange={(e) => set(e.target.value)} aria-label={label}>
      {props.runs.map((r) => <option key={r.id} value={r.id}>{r.step} {shortId(r.id)}</option>)}
    </select>
  );
  return (
    <div className="card diffcard">
      <div className="section-head pad-x">
        <h3>Code diff</h3>
        <div className="row">
          {pick(a, setA, "Old run")} <span className="muted">→</span> {pick(b, setB, "New run")}
          <div className="seg">
            <button className={mode === "unified" ? "on" : ""} onClick={() => setMode("unified")}>Unified</button>
            <button className={mode === "split" ? "on" : ""} onClick={() => setMode("split")}>Side by side</button>
          </div>
          <label className="check"><input type="checkbox" checked={full} onChange={(e) => setFull(e.target.checked)} /> Whole file</label>
        </div>
      </div>
      {error ? <p className="muted pad">{error}</p>
        : !texts ? <p className="muted pad">Loading snapshots…</p>
        : (
          <>
            <div className="pad-x small">
              <code>{texts.a.path}</code>{texts.a.path !== texts.b.path && <> → <code>{texts.b.path}</code></>}
              {" · "}{st.added || st.removed
                ? <><span className="add-n">+{st.added}</span> <span className="del-n">−{st.removed}</span></>
                : <span className="muted">identical scripts</span>}
            </div>
            {(st.added > 0 || st.removed > 0 || full) && (
              <div className="diff-scroll">
                <table className={`diff ${mode}`}>
                  <tbody>
                    {hs.map((h, hi) => (
                      [
                        h.header ? <tr key={`h${hi}`} className="hunk"><td colSpan={mode === "split" ? 4 : 3}>{h.header}</td></tr> : null,
                        ...(mode === "unified"
                          ? h.ops.map((o, i) => (
                            <tr key={`${hi}-${i}`} className={cls(o)}>
                              <Num n={o.a} /><Num n={o.b} />
                              <td className="code-line"><span className="sign">{o.t}</span>{o.text}</td>
                            </tr>
                          ))
                          : sideBySide(h.ops).map((r, i) => (
                            <tr key={`${hi}-${i}`}>
                              <Num n={r.left?.a ?? null} />
                              <td className={`code-line ${cls(r.left)}`}>{r.left?.text ?? ""}</td>
                              <Num n={r.right?.b ?? null} />
                              <td className={`code-line ${cls(r.right)}`}>{r.right?.text ?? ""}</td>
                            </tr>
                          ))),
                      ]
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
    </div>
  );
}
