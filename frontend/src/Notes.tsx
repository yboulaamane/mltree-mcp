/** Editable notes and tags. Saved through the viewer API (localhost only); read-only in exported files. */
import { useEffect, useState } from "react";
import { api, ApiError } from "./api";
import { isOffline } from "./offline";

export function NotesEditor(props: { notes: string; path: string; label?: string; placeholder?: string }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(props.notes);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { if (!editing) setText(props.notes); }, [props.notes, editing]);
  const ro = isOffline();
  const save = async () => {
    setBusy(true);
    try {
      await api(props.path, { method: "PUT", body: JSON.stringify({ notes: text }) });
      setEditing(false);
      setErr(null);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="notes">
      <div className="section-head">
        <h3>{props.label ?? "Notes"}</h3>
        {!ro && !editing && <button className="link small" onClick={() => setEditing(true)}>{props.notes ? "Edit" : "+ Add note"}</button>}
      </div>
      {editing ? (
        <>
          <textarea value={text} autoFocus rows={Math.min(12, Math.max(4, text.split("\n").length + 1))}
            placeholder={props.placeholder ?? "What did you try, what did you learn? (Markdown-ish plain text)"}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void save();
              if (e.key === "Escape") { setEditing(false); setText(props.notes); }
            }} />
          <div className="row-actions">
            <button className="primary" disabled={busy} onClick={save}>Save</button>
            <button onClick={() => { setEditing(false); setText(props.notes); setErr(null); }}>Cancel</button>
            <span className="muted small">Ctrl+Enter to save · Esc to cancel</span>
          </div>
        </>
      ) : props.notes ? <div className="note prewrap">{props.notes}</div>
        : <p className="muted small">{ro ? "No notes." : "No notes yet."}</p>}
      {err && <p className="error small">{err}</p>}
    </section>
  );
}

const SYSTEM_TAGS = new Set(["mltree-run", "imported", "scan"]);

export function TagEditor(props: { tags: string[]; runId: string; known?: string[] }) {
  const [input, setInput] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const ro = isOffline();
  const path = `/api/runs/${encodeURIComponent(props.runId)}/tags`;
  const send = async (body: { add?: string[]; remove?: string[] }) => {
    try {
      await api(path, { method: "PUT", body: JSON.stringify(body) });
      setErr(null);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    }
  };
  const add = () => {
    const tags = input.split(",").map((t) => t.trim()).filter(Boolean);
    if (tags.length) void send({ add: tags });
    setInput("");
  };
  const listId = `tags-${props.runId}`;
  return (
    <div className="tags">
      {props.tags.map((t) => (
        <span key={t} className={`chip tag${SYSTEM_TAGS.has(t) ? " sys" : ""}`}>{t}
          {!ro && <button className="chip-x" title={`Remove tag ${t}`} onClick={() => void send({ remove: [t] })}>×</button>}
        </span>
      ))}
      {!props.tags.length && ro && <span className="muted small">no tags</span>}
      {!ro && (
        <input className="tag-input" value={input} placeholder="+ tag" list={listId} aria-label="Add tag"
          onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && add()} onBlur={add} />
      )}
      {!ro && props.known && <datalist id={listId}>{props.known.filter((k) => !props.tags.includes(k)).map((k) => <option key={k} value={k} />)}</datalist>}
      {err && <p className="error small">{err}</p>}
    </div>
  );
}
