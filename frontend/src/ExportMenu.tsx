import { apiUrl } from "./api";
import { useEffect, useRef, useState } from "react";
import { isOffline } from "./offline";

function download(href: string, name: string) {
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

export function slug(s: string): string {
  return (s || "project").replace(/[^A-Za-z0-9_.-]+/g, "-").replace(/^-+|-+$/g, "") || "project";
}

/** Render the current map view (nodes and edges at the current pan/zoom; no controls or minimap) to PNG or SVG. */
async function exportImage(format: "png" | "svg", project: string): Promise<void> {
  const flow = document.querySelector<HTMLElement>(".react-flow");
  const vp = document.querySelector<HTMLElement>(".react-flow__viewport");
  if (!flow || !vp) throw new Error("Open the workflow map first");
  const { toPng, toSvg } = await import("html-to-image");
  const css = getComputedStyle(document.documentElement);
  const bg = css.getPropertyValue("--bg").trim() || "#ffffff";
  const { width, height } = flow.getBoundingClientRect();
  // capture the viewport layer (as React Flow's own docs do) sized to the visible canvas
  const opts = {
    backgroundColor: bg, skipFonts: true, pixelRatio: format === "png" ? 2 : 1,
    width: Math.round(width), height: Math.round(height),
    style: { width: `${Math.round(width)}px`, height: `${Math.round(height)}px`, transform: vp.style.transform },
  };
  // html-to-image does not carry CSS rules into SVG children, so pin edge colours as attributes while capturing
  const pinned: [Element, string, string | null][] = [];
  for (const el of vp.querySelectorAll("svg *")) {
    const cs = getComputedStyle(el);
    for (const a of ["stroke", "stroke-width", "fill", "stroke-opacity", "fill-opacity", "font-size"]) {
      const v = cs.getPropertyValue(a);
      if (v) {
        pinned.push([el, a, el.getAttribute(a)]);
        el.setAttribute(a, v);
      }
    }
  }
  try {
    const url = format === "png" ? await toPng(vp, opts) : await toSvg(vp, opts);
    download(url, `${slug(project)}-map.${format}`);
  } finally {
    for (const [el, a, old] of pinned) {
      if (old === null) el.removeAttribute(a);
      else el.setAttribute(a, old);
    }
  }
}

export default function ExportMenu({ project, onMap, goMap }: { project: string; onMap: boolean; goMap: () => void }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const offline = isOffline();
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const image = async (fmt: "png" | "svg") => {
    setErr(null);
    setBusy(fmt);
    try {
      if (!onMap) {
        goMap();
        await new Promise((r) => setTimeout(r, 400));
      }
      await exportImage(fmt, project);
      setOpen(false);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(null);
    }
  };
  const html = (scripts: boolean) => {
    download(apiUrl(`/api/export?scripts=${scripts ? 1 : 0}`), `${slug(project)}-mltree.html`);
    setOpen(false);
  };

  return (
    <div className="menu-wrap" ref={ref}>
      <button onClick={() => setOpen(!open)} aria-haspopup="menu" aria-expanded={open} className="export-btn">Export ▾</button>
      {open && (
        <div className="menu" role="menu">
          {!offline && (
            <>
              <button role="menuitem" onClick={() => html(true)}>
                <strong>Shareable HTML</strong><span>One offline file: map, code, runs, tree, compare</span>
              </button>
              <button role="menuitem" onClick={() => html(false)}>
                <strong>Shareable HTML without scripts</strong><span>Same, but leaves out the code</span>
              </button>
              <hr />
            </>
          )}
          <button role="menuitem" onClick={() => image("png")} disabled={!!busy}>
            <strong>{busy === "png" ? "Rendering…" : "Map as PNG"}</strong><span>The current map view, 2× resolution</span>
          </button>
          <button role="menuitem" onClick={() => image("svg")} disabled={!!busy}>
            <strong>{busy === "svg" ? "Rendering…" : "Map as SVG"}</strong><span>The current map view, scalable</span>
          </button>
          {err && <div className="menu-err">{err}</div>}
        </div>
      )}
    </div>
  );
}
