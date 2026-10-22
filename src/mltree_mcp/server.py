"""mltree-mcp stdio MCP server.

The tools are thin wrappers around :class:`mltree_mcp.store.Store`. The project
root is resolved on every call from ``$MLTREE_PROJECT`` (or
``$CLAUDE_PROJECT_DIR``, which Claude Code sets), falling back to the current
working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Annotated, Any, Literal

from pydantic import Field

from . import __version__
from .store import MLTreeError, Store, resolve_project_root

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as _ServerCls
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as _ServerCls  # type: ignore[no-redef]
    from mcp.server.fastmcp.exceptions import ToolError  # type: ignore[no-redef]


INSTRUCTIONS = """\
mltree records an ML workflow as a node graph (steps + edges) and every experiment as a run in a tree.
FIRST, on an existing project (scripts or notebooks already there, but no .mltree/ map yet, or get_pipeline is
empty), call scan_project: it statically parses the code (never runs it), proposes steps, edges, params and imports
existing results (metrics files, MLflow, Optuna, Lightning, wandb) as runs tagged 'imported'. Use dry_run=true to
preview. Then review its summary and fix wrong edges/kinds with connect_steps / add_step; re-scanning later keeps
those edits. On a new project call init_project.
Use it as you work: call add_step whenever you create or change a pipeline script (one script per step, e.g.
steps/split.py); call connect_steps to wire data flow; call log_run after every execution with its params and metrics
(use parent_run_id when a run is a variation of an earlier one, e.g. an HPO trial). Call open_map to give the user
the viewer URL. Call get_pipeline before changing the workflow, and best_run / list_runs to answer questions about
experiments. Paths are relative to the project root.
"""

mcp = _ServerCls("mltree", instructions=INSTRUCTIONS)


def _store() -> Store:
    return Store(resolve_project_root())


def _call(fn, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except MLTreeError as exc:
        raise ToolError(str(exc)) from exc


# -------------------------------------------------------------------- tools


@mcp.tool()
def init_project(
    name: Annotated[str | None, Field(description="Project name; defaults to the folder name")] = None,
) -> dict[str, Any]:
    """Create the .mltree/ store (pipeline.yaml, mltree.db, runs/) in the project root. Safe to call again."""
    return _call(lambda: _store().init(name))


@mcp.tool()
def add_step(
    name: Annotated[str, Field(description="Unique step name, e.g. 'load_smiles', 'split', 'train_gcn'")],
    script_path: Annotated[str | None, Field(description="Path of the step's Python script, relative to the project root")] = None,
    inputs: Annotated[list[str] | None, Field(description="Data this step consumes, e.g. ['graphs.pkl']")] = None,
    outputs: Annotated[list[str] | None, Field(description="Data this step produces, e.g. ['train.pkl', 'test.pkl']")] = None,
    params: Annotated[dict[str, Any] | None, Field(description="Default parameters, e.g. {'test_size': 0.2, 'seed': 42}")] = None,
    description: Annotated[str, Field(description="One-line description of what the step does")] = "",
) -> dict[str, Any]:
    """Register a pipeline step (a node on the map), or update it if the name already exists."""
    return _call(
        lambda: _store().add_step(
            name, script_path=script_path, inputs=inputs, outputs=outputs, params=params, description=description
        )
    )


@mcp.tool()
def connect_steps(
    source: Annotated[str, Field(description="Upstream step name")],
    target: Annotated[str, Field(description="Downstream step name")],
    label: Annotated[str, Field(description="Optional edge label, e.g. the data passed along")] = "",
) -> dict[str, Any]:
    """Draw a directed edge source -> target. Both steps must exist; cycles are rejected."""
    return _call(lambda: _store().connect_steps(source, target, label))


@mcp.tool()
def get_pipeline() -> dict[str, Any]:
    """Return the whole map: nodes (with params, script and latest run status), edges and a topological order."""
    return _call(lambda: _store().get_pipeline())


@mcp.tool()
def log_run(
    step: Annotated[str, Field(description="Name of the step that was executed")],
    params: Annotated[dict[str, Any] | None, Field(description="Parameters actually used for this run")] = None,
    metrics: Annotated[dict[str, float] | None, Field(description="Numeric results, e.g. {'val_rmse': 0.61, 'r2': 0.78}")] = None,
    artifacts: Annotated[
        list[str] | dict[str, str] | None,
        Field(description="Files produced (paths relative to project), as a list or {name: path}; copied into .mltree/runs/<run_id>/ when <= 50 MB"),
    ] = None,
    parent_run_id: Annotated[str | None, Field(description="Run this one branches from (e.g. the base config of an HPO trial)")] = None,
    notes: Annotated[str, Field(description="Free-text notes about the run")] = "",
    status: Annotated[
        Literal["pending", "running", "succeeded", "failed", "cancelled"], Field(description="Run outcome")
    ] = "succeeded",
    tags: Annotated[list[str] | None, Field(description="Optional tags, e.g. ['hpo', 'gcn']")] = None,
) -> dict[str, Any]:
    """Record one execution of a step: params, metrics, artifacts and its parent run in the experiment tree."""
    return _call(
        lambda: _store().log_run(
            step,
            params=params,
            metrics=metrics,
            artifacts=artifacts,
            parent_run_id=parent_run_id,
            notes=notes,
            status=status,
            tags=tags,
        )
    )


@mcp.tool()
def list_runs(
    step: Annotated[str | None, Field(description="Only runs of this step")] = None,
    status: Annotated[str | None, Field(description="Only runs with this status")] = None,
    parent_run_id: Annotated[str | None, Field(description="Only direct children of this run")] = None,
    limit: Annotated[int, Field(description="Maximum runs to return (1-1000)", ge=1, le=1000)] = 50,
) -> dict[str, Any]:
    """List runs, newest first."""
    runs = _call(lambda: _store().list_runs(step=step, status=status, parent_run_id=parent_run_id, limit=limit))
    return {"count": len(runs), "runs": runs}


@mcp.tool()
def best_run(
    metric: Annotated[str, Field(description="Metric name, e.g. 'val_rmse' or 'roc_auc'")],
    mode: Annotated[Literal["min", "max"], Field(description="'min' for losses/errors, 'max' for scores")] = "max",
    step: Annotated[str | None, Field(description="Restrict to one step")] = None,
) -> dict[str, Any]:
    """Find the best succeeded run by a metric."""
    return _call(lambda: _store().best_run(metric, mode=mode, step=step))


@mcp.tool()
def open_map(
    start: Annotated[bool, Field(description="Start the viewer in the background if it is not running")] = True,
    port: Annotated[int | None, Field(description="Viewer port (default $MLTREE_VIEW_PORT or 8765)", ge=1, le=65535)] = None,
) -> dict[str, Any]:
    """Return the URL of the mltree map viewer (and start it if needed). Give the URL to the user."""
    return _call(lambda: _open_map(start=start, port=port))


def _viewer_health(url: str) -> dict[str, Any] | None:
    import urllib.request

    try:
        with urllib.request.urlopen(url + "/api/health", timeout=0.5) as r:  # noqa: S310 - localhost only
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return None


def _open_map(start: bool = True, port: int | None = None) -> dict[str, Any]:
    import subprocess
    import time

    from .viewer import DEFAULT_PORT, view_url

    root = resolve_project_root()
    port = port or int(os.environ.get("MLTREE_VIEW_PORT") or DEFAULT_PORT)
    url = view_url(port=port)
    health = _viewer_health(url)
    started = False
    if health is None and start:
        subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "mltree_mcp", "view", "--project", str(root), "--port", str(port)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True,
        )
        started = True
        for _ in range(30):
            time.sleep(0.2)
            health = _viewer_health(url)
            if health:
                break
    result: dict[str, Any] = {
        "url": url,
        "running": health is not None,
        "started": started,
        "project": str(root),
        "remote_tip": f"On a remote server, tunnel it from your laptop: ssh -L {port}:localhost:{port} user@server, then open {url}",
    }
    if health and health.get("root") and health["root"] != str(root):
        result["warning"] = f"A viewer for another project ({health['root']}) is already on port {port}; pass another port."
    if health is None:
        result["hint"] = f"Viewer not running. Start it with: mltree-mcp view --project {root} --port {port}"
    return result


@mcp.tool()
def scan_project(
    path: Annotated[str, Field(description="Folder to scan, relative to the project root")] = ".",
    include: Annotated[list[str] | None, Field(description="Glob patterns to scan (default **/*.py, **/*.ipynb)")] = None,
    exclude: Annotated[
        list[str] | None,
        Field(description="Glob patterns to skip (default: venvs, .git, tests, __pycache__, .mltree, build dirs)"),
    ] = None,
    dry_run: Annotated[bool, Field(description="Return the proposed map without writing anything")] = False,
    notebook_sections: Annotated[bool, Field(description="Make each markdown-headed notebook section a step")] = False,
    import_results: Annotated[bool, Field(description="Import existing metrics/MLflow/Optuna/Lightning/wandb results as runs")] = True,
) -> dict[str, Any]:
    """Build the map of an EXISTING project by static analysis (ast only; user code is never executed).

    Detects data reads/writes, argparse/click params, local imports, docstrings and step kinds, links steps whose
    output paths match other steps' inputs (falling back to file numbering and imports), and imports existing
    results as runs tagged 'imported'. Idempotent: re-scans update scanner-owned fields, add new steps, never delete,
    and keep edits made with add_step. Review the returned summary and fix it with connect_steps / add_step.
    """
    from .scan import scan_project as _scan

    return _call(lambda: _scan(_store(), path=path, include=include, exclude=exclude, dry_run=dry_run,
                               notebook_sections=notebook_sections, import_results=import_results))


# ---------------------------------------------------------------- resources


@mcp.resource(
    "mltree://pipeline",
    name="pipeline",
    description="The current mltree pipeline (nodes, edges, latest run status) as JSON",
    mime_type="application/json",
)
def pipeline_resource() -> str:
    try:
        data = _store().get_pipeline()
    except MLTreeError as exc:
        data = {"error": str(exc)}
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mltree-mcp",
        description="mltree MCP server. With no command it runs over stdio (for Claude Code, Codex and other MCP "
        "clients); `mltree-mcp view` serves the map viewer.",
    )
    parser.add_argument("--project", help="Project directory (default: $MLTREE_PROJECT, $CLAUDE_PROJECT_DIR or cwd)")
    parser.add_argument("--version", action="version", version=f"mltree-mcp {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="Run the MCP server over stdio (the default)")
    v = sub.add_parser("view", help="Serve the map viewer on http://localhost:8765")
    v.add_argument("--project", dest="view_project", help="Project directory")
    v.add_argument("--host", default=None, help="Bind address (default 127.0.0.1; use 0.0.0.0 only on trusted networks)")
    v.add_argument("--port", type=int, default=None, help="Port (default 8765)")
    sc = sub.add_parser("scan", help="Statically scan an existing project and write its map to .mltree/")
    sc.add_argument("--project", dest="view_project", help="Project directory")
    sc.add_argument("--path", default=".", help="Sub-folder to scan (default: the whole project)")
    sc.add_argument("--dry-run", action="store_true", help="Show the proposed map without writing anything")
    sc.add_argument("--notebook-sections", action="store_true", help="One step per markdown-headed notebook section")
    sc.add_argument("--no-results", action="store_true", help="Do not import existing results as runs")
    sc.add_argument("--exclude", action="append", default=None, help="Extra glob to skip (repeatable)")
    sc.add_argument("--json", action="store_true", help="Print the full JSON summary")
    args = parser.parse_args(argv)

    project = getattr(args, "view_project", None) or args.project
    if project:
        try:
            os.environ["MLTREE_PROJECT"] = str(resolve_project_root(project))
        except MLTreeError as exc:
            print(f"mltree-mcp: {exc}", file=sys.stderr)
            raise SystemExit(2)

    if args.command == "scan":
        from .scan import DEFAULT_EXCLUDE, format_summary
        from .scan import scan_project as _scan

        try:
            summary = _scan(_store(), path=args.path, dry_run=args.dry_run, notebook_sections=args.notebook_sections,
                            import_results=not args.no_results,
                            exclude=DEFAULT_EXCLUDE + args.exclude if args.exclude else None)
        except MLTreeError as exc:
            print(f"mltree-mcp: {exc}", file=sys.stderr)
            raise SystemExit(2)
        print(json.dumps(summary, indent=2) if args.json else format_summary(summary))
        return
    if args.command == "view":
        from .viewer import DEFAULT_HOST, DEFAULT_PORT, serve

        host = args.host or os.environ.get("MLTREE_VIEW_HOST") or DEFAULT_HOST
        port = args.port or int(os.environ.get("MLTREE_VIEW_PORT") or DEFAULT_PORT)
        try:
            serve(host=host, port=port)
        except MLTreeError as exc:
            print(f"mltree-mcp: {exc}", file=sys.stderr)
            raise SystemExit(2)
        return
    mcp.run("stdio")


if __name__ == "__main__":
    main()
