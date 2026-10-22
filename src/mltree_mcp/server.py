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
Use it as you work: call init_project once per repo; call add_step whenever you create or change a
pipeline script (one script per step, e.g. steps/split.py); call connect_steps to wire data flow; call
log_run after every execution with its params and metrics (use parent_run_id when a run is a variation
of an earlier one, e.g. an HPO trial). Call get_pipeline before changing the workflow, and best_run /
list_runs to answer questions about experiments. Paths are relative to the project root.
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
        prog="mltree-mcp", description="Run the mltree MCP server over stdio (for Claude Code, Codex and other MCP clients)."
    )
    parser.add_argument("--project", help="Project directory (default: $MLTREE_PROJECT, $CLAUDE_PROJECT_DIR or cwd)")
    parser.add_argument("--version", action="version", version=f"mltree-mcp {__version__}")
    args = parser.parse_args(argv)
    if args.project:
        try:
            os.environ["MLTREE_PROJECT"] = str(resolve_project_root(args.project))
        except MLTreeError as exc:
            print(f"mltree-mcp: {exc}", file=sys.stderr)
            raise SystemExit(2)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
