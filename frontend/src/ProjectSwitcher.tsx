import { useEffect, useState } from "react";

interface ProjectEntry {
  slug: string;
  name: string;
  root: string;
  initialized: boolean;
  url: string;
  steps?: number;
  runs?: number;
}

/** Multi-project viewer (`view --projects`): pick another project; each one lives under /p/<slug>/. */
export function ProjectSwitcher(props: { current: string; root?: string }) {
  const [projects, setProjects] = useState<ProjectEntry[] | null>(null);
  const slug = location.pathname.match(/^\/p\/([^/]+)/)?.[1] ?? "";
  useEffect(() => {
    // the project list lives at the server root, not under this project's /p/<slug>/api
    fetch("/api/projects")
      .then((r) => (r.ok ? r.json() : null))
      .then((b) => setProjects(b?.projects ?? null))
      .catch(() => setProjects(null));
  }, []);
  if (!projects || projects.length < 2) {
    return <span className="project" title={props.root}>{props.current}</span>;
  }
  return (
    <select
      className="project project-switch"
      value={slug}
      title={`${props.root ?? ""} · ${projects.length} projects in this viewer`}
      aria-label="Switch project"
      onChange={(e) => {
        const p = projects.find((x) => x.slug === e.target.value);
        if (p) location.href = `${p.url}#/map`;
      }}
    >
      {projects.map((p) => (
        <option key={p.slug} value={p.slug}>
          {p.name}
          {p.initialized ? ` · ${p.steps ?? 0} steps · ${p.runs ?? 0} runs` : " · not initialized"}
        </option>
      ))}
    </select>
  );
}
