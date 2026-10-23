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
the viewer URL, and export_map when they want a single offline HTML file to send someone (it includes the scripts
unless include_scripts=false). Call get_pipeline before changing the workflow, and best_run / list_runs to answer questions about
experiments. Paths are relative to the project root.
To execute a step AND record it in one go, call run_step (it launches the step's script with the given params, streams
nothing back until it finishes or times out, and records params, script snapshot, exit status, duration, log and
metrics; scripts can print `MLTREE_METRIC name=value` lines or write JSON to $MLTREE_METRICS_FILE). Use run_sweep for a
grid over params (one sweep with a child run per combination). When you run things yourself instead, group HPO trials
by calling start_sweep once and passing its sweep_id to log_run for every trial.
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
    sweep_id: Annotated[str | None, Field(description="Sweep this run is a trial of (from start_sweep)")] = None,
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
            sweep_id=sweep_id,
        )
    )


@mcp.tool()
def start_sweep(
    step: Annotated[str, Field(description="Step the trials run, e.g. 'train'")],
    metric: Annotated[str | None, Field(description="Metric that ranks the trials, e.g. 'val_rmse'")] = None,
    mode: Annotated[Literal["min", "max"], Field(description="Whether lower or higher metric is better")] = "max",
    name: Annotated[str | None, Field(description="Short label, e.g. 'alpha x layers grid'")] = None,
    search_space: Annotated[dict[str, Any] | None, Field(description="Optional search space, e.g. {'alpha': [0.1, 1, 10]}")] = None,
    parent_run_id: Annotated[str | None, Field(description="Run the sweep branches from (e.g. the baseline)")] = None,
) -> dict[str, Any]:
    """Create a sweep (an HPO study / grid search) that groups trial runs. Pass its sweep_id to log_run for each
    trial; the map shows the sweep as one collapsible fan with the best trial highlighted. Use run_sweep instead to
    have mltree launch a grid itself."""
    def go() -> dict[str, Any]:
        st = _store()
        rec = st.log_run(step, params=search_space or {}, status="running", kind="sweep", parent_run_id=parent_run_id,
                         tags=["sweep"], notes=name or "", sweep={"source": "manual", "metric": metric, "mode": mode,
                                                                  "name": name, "grid": search_space})
        return {"sweep_id": rec["id"], "step": rec["step"], "metric": metric, "mode": mode,
                "hint": "Call log_run(..., sweep_id=...) for each trial; call finish_sweep when done."}

    return _call(go)


@mcp.tool()
def finish_sweep(
    sweep_id: Annotated[str, Field(description="Id returned by start_sweep")],
    status: Annotated[Literal["succeeded", "failed", "cancelled"], Field(description="Outcome of the sweep")] = "succeeded",
) -> dict[str, Any]:
    """Mark a sweep finished and return its summary (trial counts and the best trial)."""
    def go() -> dict[str, Any]:
        st = _store()
        summ = st.sweep_summary(sweep_id)
        metrics = {f"best_{summ['metric']}": summ["best_value"]} if summ["metric"] and summ["best_value"] is not None else None
        st.update_run(sweep_id, status=status, metrics=metrics, sweep={"best_run_id": summ["best_run_id"]})
        return summ

    return _call(go)


# ------------------------------------------------------------- running steps

_MANAGERS: dict[str, Any] = {}


def _manager():
    from .runner import RunManager, run_allowed_by_env

    if not run_allowed_by_env():
        raise ToolError("Running steps through MCP is disabled here (MLTREE_ALLOW_RUN=0). Run the script yourself and "
                        "call log_run.")
    root = str(resolve_project_root())
    if root not in _MANAGERS:
        _MANAGERS[root] = RunManager(Store(root))
    return _MANAGERS[root]


def _run_result(mgr, run_id: str, finished: bool, tail: int = 60) -> dict[str, Any]:
    run = mgr.store.get_run(run_id)
    log = mgr.log(run_id, tail=tail)
    out = {"run": run, "finished": finished, "log_tail": "\n".join(
        ("[stderr] " if ln["s"] == "err" else "") + ln["line"] for ln in log["lines"])}
    if not finished:
        out["hint"] = f"Still running. Call run_log('{run_id}') to follow it or cancel_run('{run_id}') to stop it."
    return out


@mcp.tool()
async def run_step(
    step: Annotated[str, Field(description="Step to execute (its script_path is run with Python from the project root)")],
    params: Annotated[dict[str, Any] | None, Field(description="Params for this run; default: the last run's params, else the step's")] = None,
    extra_args: Annotated[str | None, Field(description="Extra command-line arguments appended verbatim")] = None,
    parent_run_id: Annotated[str | None, Field(description="Run this one branches from (default: newest succeeded run of an upstream step)")] = None,
    wait: Annotated[bool, Field(description="Wait for the run to finish (up to timeout_s)")] = True,
    timeout_s: Annotated[float, Field(description="Max seconds to wait; the run keeps going after that", ge=0, le=86400)] = 900,
) -> dict[str, Any]:
    """Execute a step's script as a subprocess and record it as a run automatically.

    Params are passed as CLI flags when the script uses argparse/click (or positionally for sys.argv scripts, or
    key=value for hydra), and always as $MLTREE_PARAMS (JSON). Records params, script snapshot, command, exit status,
    duration, log (.mltree/runs/<id>/log.txt) and metrics from `MLTREE_METRIC name=value` stdout lines or a JSON file
    written to $MLTREE_METRICS_FILE / a declared metrics*.json output. Returns the run and the end of its log.
    """
    import asyncio

    mgr = _manager()
    try:
        live = await asyncio.to_thread(mgr.start, step, params, extra_args, None, parent_run_id)
    except MLTreeError as exc:
        raise ToolError(str(exc)) from exc
    finished = await asyncio.to_thread(live.done.wait, timeout_s) if wait else False
    return _run_result(mgr, live.run_id, bool(finished))


@mcp.tool()
async def run_sweep(
    step: Annotated[str, Field(description="Step to execute for every combination")],
    grid: Annotated[dict[str, list[Any]], Field(description="Param -> list of values, e.g. {'alpha': [0.1, 1, 10], 'layers': [1, 2]}")],
    metric: Annotated[str | None, Field(description="Metric that ranks the runs, e.g. 'val_rmse'")] = None,
    mode: Annotated[Literal["min", "max"], Field(description="Whether lower or higher metric is better")] = "max",
    params: Annotated[dict[str, Any] | None, Field(description="Fixed params shared by all runs")] = None,
    extra_args: Annotated[str | None, Field(description="Extra command-line arguments for every run")] = None,
    max_runs: Annotated[int, Field(description="Refuse grids larger than this", ge=1, le=200)] = 64,
    wait: Annotated[bool, Field(description="Wait for the whole sweep (up to timeout_s)")] = True,
    timeout_s: Annotated[float, Field(description="Max seconds to wait", ge=0, le=86400)] = 3600,
) -> dict[str, Any]:
    """Run a grid search: one sweep run plus one child run per combination, executed one after another."""
    import asyncio

    mgr = _manager()
    try:
        res = await asyncio.to_thread(mgr.start_sweep, step, grid, params, metric, mode, extra_args, None, max_runs)
    except MLTreeError as exc:
        raise ToolError(str(exc)) from exc
    finished = await asyncio.to_thread(mgr.wait_sweep, res["sweep_id"], timeout_s) if wait else False
    summ = mgr.store.sweep_summary(res["sweep_id"])
    return {"sweep_id": res["sweep_id"], "finished": bool(finished), "total": res["total"], "summary": summ,
            **({} if finished else {"hint": f"Still running; cancel_run('{res['sweep_id']}') stops the sweep."})}


@mcp.tool()
def run_log(
    run_id: Annotated[str, Field(description="Run id")],
    tail: Annotated[int, Field(description="Last N lines", ge=1, le=5000)] = 200,
) -> dict[str, Any]:
    """Return the status and the last lines of a run's log (live while it runs)."""
    def go() -> dict[str, Any]:
        mgr = _MANAGERS.get(str(resolve_project_root()))
        if mgr is None:
            from .runner import RunManager

            mgr = RunManager(_store())
        live = mgr.get(run_id)
        return _run_result(mgr, run_id, live is None or live.done.is_set(), tail=tail)

    return _call(go)


@mcp.tool()
def cancel_run(run_id: Annotated[str, Field(description="Run or sweep id started by run_step / run_sweep")]) -> dict[str, Any]:
    """Stop a run (SIGTERM, then SIGKILL after 5 s) or a whole sweep started from this session."""
    return _call(lambda: _manager().cancel(run_id))


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
    allow_run: Annotated[bool, Field(description="Start it with Run buttons enabled (only if the user asked to run steps from the map)")] = False,
) -> dict[str, Any]:
    """Return the URL of the mltree map viewer (and start it if needed). Give the URL to the user."""
    return _call(lambda: _open_map(start=start, port=port, allow_run=allow_run))


def _viewer_health(url: str) -> dict[str, Any] | None:
    import urllib.request

    try:
        with urllib.request.urlopen(url + "/api/health", timeout=0.5) as r:  # noqa: S310 - localhost only
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return None


def _open_map(start: bool = True, port: int | None = None, allow_run: bool = False) -> dict[str, Any]:
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
            [sys.executable, "-m", "mltree_mcp", "view", "--project", str(root), "--port", str(port),
             *(["--allow-run"] if allow_run else [])],
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
    if health is not None:
        result["allow_run"] = bool(health.get("allow_run"))
        if allow_run and not health.get("allow_run"):
            result["note"] = "The running viewer has Run disabled; stop it and call open_map(allow_run=true) again."
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


@mcp.tool()
def export_map(
    path: Annotated[
        str | None,
        Field(description="Output file or folder (relative to the project root). Default: <project root>/<project>-mltree.html"),
    ] = None,
    include_scripts: Annotated[bool, Field(description="Embed the step scripts and their run snapshots")] = True,
) -> dict[str, Any]:
    """Write a single self-contained HTML file of the map (steps, code, runs, experiment tree, compare).

    It opens offline in any browser, read-only, with no server, so the user can email it, post it in Slack or host it
    on GitHub Pages / a gist. Returns the file path and size. It contains the scripts unless include_scripts=false:
    tell the user that before they share it.
    """
    from .export import export_html

    return _call(lambda: export_html(_store(), path=path, include_scripts=include_scripts, base=resolve_project_root()))


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
    v.add_argument("--allow-run", action="store_true",
                   help="Enable the Run / Sweep buttons (executes the project's scripts; forces a 127.0.0.1 bind)")
    v.add_argument("--python", default=None, help="Interpreter for runs (default $MLTREE_PYTHON, the project's .venv, then PATH)")
    sc = sub.add_parser("scan", help="Statically scan an existing project and write its map to .mltree/")
    sc.add_argument("--project", dest="view_project", help="Project directory")
    sc.add_argument("--path", default=".", help="Sub-folder to scan (default: the whole project)")
    sc.add_argument("--dry-run", action="store_true", help="Show the proposed map without writing anything")
    sc.add_argument("--notebook-sections", action="store_true", help="One step per markdown-headed notebook section")
    sc.add_argument("--no-results", action="store_true", help="Do not import existing results as runs")
    sc.add_argument("--exclude", action="append", default=None, help="Extra glob to skip (repeatable)")
    sc.add_argument("--json", action="store_true", help="Print the full JSON summary")
    ex = sub.add_parser("export", help="Write a single self-contained, offline HTML file of the map to share")
    ex.add_argument("--project", dest="view_project", help="Project directory")
    ex.add_argument("-o", "--output", default=None, help="Output file or folder (default: ./<project>-mltree.html)")
    ex.add_argument("--no-scripts", action="store_true", help="Leave the step scripts out of the file")
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
    if args.command == "export":
        from .export import export_html

        try:
            res = export_html(_store(), path=args.output, include_scripts=not args.no_scripts)
        except MLTreeError as exc:
            print(f"mltree-mcp: {exc}", file=sys.stderr)
            raise SystemExit(2)
        print(f"Wrote {res['path']} ({res['size']}; {res['steps']} steps, {res['runs']} runs, "
              f"{'scripts included' if res['scripts_included'] else 'no scripts'})")
        print(res["note"])
        return
    if args.command == "view":
        from .viewer import DEFAULT_HOST, DEFAULT_PORT, serve

        host = args.host or os.environ.get("MLTREE_VIEW_HOST") or DEFAULT_HOST
        port = args.port or int(os.environ.get("MLTREE_VIEW_PORT") or DEFAULT_PORT)
        try:
            serve(host=host, port=port, allow_run=args.allow_run, python=args.python)
        except MLTreeError as exc:
            print(f"mltree-mcp: {exc}", file=sys.stderr)
            raise SystemExit(2)
        return
    mcp.run("stdio")


if __name__ == "__main__":
    main()
