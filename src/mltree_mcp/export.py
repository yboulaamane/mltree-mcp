"""Shareable export: one self-contained, offline, read-only HTML file of the map.

The file inlines the prebuilt export bundle (``static/export/export.js`` + ``export.css``: the viewer with a
lightweight code view and SVG charts instead of Monaco / Plotly) and a JSON snapshot of the pipeline, runs,
metrics and, optionally, the scripts (current files + the snapshot stored with each run, deduplicated by SHA-256).
It opens from ``file://`` in any modern browser with no server and no network.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .store import MLTreeError, Store

EXPORT_DIR = Path(__file__).resolve().parent / "static" / "export"
FORMAT = "mltree-export/1"


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name or "project").strip("-.") or "project"


def default_filename(store: Store) -> str:
    return f"{slug(store.project_name())}-mltree.html"


def _all_runs(store: Store) -> list[dict[str, Any]]:
    with store._conn() as c:  # noqa: SLF001 - same module family; list_runs caps at 1000
        rows = c.execute("SELECT * FROM runs ORDER BY created_at DESC, rowid DESC").fetchall()
    return [_portable(store, store._run_row_to_dict(r)) for r in rows]  # noqa: SLF001


def _portable(store: Store, run: dict[str, Any]) -> dict[str, Any]:
    """Don't leak local absolute paths (interpreter, project dir) into a shared file."""
    cmd = run.get("command")
    if isinstance(cmd, list) and cmd:
        root = str(store.root)
        out = [Path(str(cmd[0])).name]
        for a in cmd[1:]:
            a = str(a)
            out.append(a.replace(root + "/", "").replace(root, ".") if root in a else a)
        run = {**run, "command": out}
    if run.get("data"):
        run = {**run, "data": _portable_paths(run["data"])}
    return run


def _portable_paths(obj: Any) -> Any:
    """Data fingerprints of files outside the project carry absolute paths; keep only the file name."""
    if isinstance(obj, dict):
        return {k: (Path(v).name if k == "path" and isinstance(v, str) and Path(v).is_absolute() else _portable_paths(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [Path(v).name if isinstance(v, str) and Path(v).is_absolute() else _portable_paths(v) for v in obj]
    return obj


def build_snapshot(store: Store, include_scripts: bool = True) -> dict[str, Any]:
    """Everything the viewer's /api routes would return, in one JSON-able dict."""
    pipeline = store.get_pipeline()
    runs = _all_runs(store)
    blobs: dict[str, str] = {}
    scripts: dict[str, Any] = {}
    run_scripts: dict[str, Any] = {}

    def blob(data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        blobs.setdefault(sha, data.decode("utf-8", errors="replace"))
        return sha

    for node in pipeline["nodes"]:
        name = node["name"]
        entry: dict[str, Any] = {"current": None, "snapshots": []}
        if include_scripts:
            if node["script_path"]:
                f = store.root / node["script_path"]
                if f.is_file():
                    entry["current"] = {"path": node["script_path"], "sha256": blob(f.read_bytes()), "created_at": None}
            entry["snapshots"] = store.snapshots(name)
            for s in entry["snapshots"]:
                f = store._snapshot_file(s["run_id"])  # noqa: SLF001
                if f is not None:
                    run_scripts[s["run_id"]] = {"path": f.relative_to(store.root).as_posix(), "sha256": blob(f.read_bytes()),
                                                "created_at": s["created_at"]}
        scripts[name] = entry
    histories = {r["id"]: h for r in runs if (h := store.metric_history(r["id"]))}
    try:
        data_status = _portable_paths(store.data_status(fresh=True))
    except (OSError, MLTreeError):
        data_status = None
    gnn_views: dict[str, Any] = {}
    from . import gnn

    for node in pipeline["nodes"]:
        try:
            v = gnn.step_view(store, node["name"])
        except (OSError, MLTreeError, SyntaxError, ValueError):
            continue
        if not (v["models"] or v["datasets"] or v["dataset_files"]):
            continue
        v["can_introspect"] = False
        if not include_scripts:  # no source fragments when the scripts are left out
            for m in v["models"]:
                m.pop("call", None)
                m.pop("instantiated", None)
                for lay in m.get("layers", []):
                    lay.pop("call", None)
                    lay.pop("mlp", None)
            for d in v["datasets"]:
                d.pop("call", None)
        gnn_views[node["name"]] = v
    # don't leak the absolute server path of the project into a file meant for sharing
    pipeline = {**pipeline, "root": pipeline["project"]}
    return {
        "format": FORMAT,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "mltree_version": __version__,
        "include_scripts": include_scripts,
        "pipeline": pipeline,
        "runs": runs,
        "metrics": store.metric_names(),
        "scripts": scripts,
        "run_scripts": run_scripts,
        "blobs": blobs,
        "histories": histories,
        "data_status": data_status,
        "gnn": gnn_views,
    }


def _bundle() -> tuple[str, str]:
    js, css = EXPORT_DIR / "export.js", EXPORT_DIR / "export.css"
    if not js.is_file() or not css.is_file():
        raise MLTreeError("The export bundle is not built. Run `npm install && npm run build` in frontend/.")
    return js.read_text(encoding="utf-8"), css.read_text(encoding="utf-8")


def _safe_in_script(text: str) -> str:
    # an inline <script>/<style> ends at the first "</script" / "</style"; "<!--" can also change parsing state.
    # "<\/" and "<\!--" mean the same in JS strings / regexes, and minified code has no such sequence outside them.
    return re.sub(r"</(script|style)", r"<\\/\1", text, flags=re.I).replace("<!--", "<\\!--")


ICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' "
        "rx='8' fill='%2310b981'/%3E%3Cpath d='M16 7v18M16 14l-6-5M16 19l6-5' stroke='white' stroke-width='3' "
        "stroke-linecap='round' fill='none'/%3E%3C/svg%3E")


def render_html(snapshot: dict[str, Any]) -> str:
    js, css = _bundle()
    data = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    title = html.escape(f"{snapshot['pipeline']['project']} · mltree")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\" />\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
        f"<meta name=\"generator\" content=\"mltree-mcp {__version__}\" />\n"
        f"<link rel=\"icon\" href=\"{ICON}\" />\n<title>{title}</title>\n"
        f"<style>{_safe_in_script(css)}</style>\n</head>\n<body>\n"
        "<noscript>This mltree export needs JavaScript to show the map.</noscript>\n"
        "<div id=\"root\"></div>\n"
        f"<script id=\"mltree-data\" type=\"application/json\">{data}</script>\n"
        f"<script type=\"module\">{_safe_in_script(js)}</script>\n"
        "</body>\n</html>\n"
    )


def export_html(store: Store, path: str | Path | None = None, include_scripts: bool = True,
                base: Path | None = None) -> dict[str, Any]:
    """Write the export. `path` may be a file or a directory; relative paths resolve against `base` (default: cwd)."""
    snap = build_snapshot(store, include_scripts=include_scripts)
    text = render_html(snap)
    base = base or Path.cwd()
    out = Path(path).expanduser() if path else base / default_filename(store)
    if not out.is_absolute():
        out = base / out
    if out.is_dir() or str(path or "").endswith(("/", "\\")):
        out = out / default_filename(store)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    size = out.stat().st_size
    return {
        "path": str(out),
        "bytes": size,
        "size": f"{size / 1024:.0f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB",
        "steps": len(snap["pipeline"]["nodes"]),
        "runs": len(snap["runs"]),
        "scripts_included": include_scripts,
        "scripts": len(snap["blobs"]),
        "note": ("Contains the scripts' source code; share it only with people who may see it."
                 if include_scripts else "Scripts left out; map, params, runs and metrics only."),
    }
