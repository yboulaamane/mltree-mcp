import { useCallback, useEffect, useState } from "react";
import { useApi, useRevision } from "./api";
import type { Pipeline } from "./types";
import MapView from "./MapView";
import TreeView from "./TreeView";
import CompareView from "./CompareView";

type View = "map" | "tree" | "compare";
export type Theme = "light" | "dark";

interface Route {
  view: View;
  params: URLSearchParams;
}

function parseHash(): Route {
  const raw = window.location.hash.replace(/^#\/?/, "");
  const [path, query = ""] = raw.split("?");
  const view = (["map", "tree", "compare"].includes(path) ? path : "map") as View;
  return { view, params: new URLSearchParams(query) };
}

function initialTheme(): Theme {
  const q = new URLSearchParams(window.location.search).get("theme");
  if (q === "dark" || q === "light") return q;
  const saved = localStorage.getItem("mltree-theme");
  if (saved === "dark" || saved === "light") return saved;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App() {
  const [route, setRoute] = useState<Route>(parseHash);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [selected, setSelected] = useState<string[]>(() => parseHash().params.get("ids")?.split(",").filter(Boolean) ?? []);
  const { revision, live } = useRevision();
  const { data: pipeline, error } = useApi<Pipeline>("/api/pipeline", [revision]);

  useEffect(() => {
    const on = () => {
      const r = parseHash();
      setRoute(r);
      const ids = r.params.get("ids");
      if (r.view === "compare" && ids) setSelected(ids.split(",").filter(Boolean));
    };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("mltree-theme", theme);
  }, [theme]);
  useEffect(() => {
    if (pipeline) document.title = `${pipeline.project} · mltree`;
  }, [pipeline]);

  const go = useCallback((view: View, params?: Record<string, string>) => {
    const q = params ? "?" + new URLSearchParams(params).toString() : "";
    window.location.hash = `/${view}${q}`;
  }, []);

  const toggleRun = useCallback((id: string) => {
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
  }, []);

  const compare = useCallback(() => go("compare", { ids: selected.join(",") }), [go, selected]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>
            <svg viewBox="0 0 32 32" width="22" height="22">
              <rect width="32" height="32" rx="8" fill="var(--accent)" />
              <path d="M16 7v18M16 14l-6-5M16 19l6-5" stroke="white" strokeWidth="3" strokeLinecap="round" fill="none" />
            </svg>
          </span>
          <strong>mltree</strong>
          <span className="project" title={pipeline?.root}>{pipeline?.project ?? "…"}</span>
        </div>
        <nav className="tabs">
          {(["map", "tree", "compare"] as View[]).map((v) => (
            <button key={v} className={route.view === v ? "tab active" : "tab"} onClick={() => (v === "compare" ? compare() : go(v))}>
              {v === "map" ? "Workflow map" : v === "tree" ? "Experiment tree" : `Compare${selected.length ? ` (${selected.length})` : ""}`}
            </button>
          ))}
        </nav>
        <div className="right">
          {selected.length > 0 && (
            <button className="ghost" onClick={() => setSelected([])} title="Clear run selection">
              Clear selection
            </button>
          )}
          <span className={live ? "live on" : "live"} title={live ? "Live: updates as runs are logged" : "Reconnecting…"}>
            <i /> {live ? "live" : "offline"}
          </span>
          <button className="icon" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} title="Toggle dark / light">
            {theme === "dark" ? "☀" : "☾"}
          </button>
        </div>
      </header>
      <main className="content">
        {error && <div className="banner error">{error}</div>}
        {route.view === "map" && pipeline && (
          <MapView pipeline={pipeline} theme={theme} revision={revision} selected={selected} toggleRun={toggleRun} onCompare={compare}
            initialStep={route.params.get("step")} />
        )}
        {route.view === "tree" && (
          <TreeView theme={theme} revision={revision} selected={selected} toggleRun={toggleRun} onCompare={compare}
            steps={pipeline?.order ?? []} params={route.params}
            setParams={(p) => go("tree", p)} />
        )}
        {route.view === "compare" && (
          <CompareView theme={theme} revision={revision} selected={selected} toggleRun={toggleRun} />
        )}
      </main>
    </div>
  );
}
