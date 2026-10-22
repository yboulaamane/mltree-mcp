"""Every MCP tool, called in-process through the server object (same path a client uses)."""
import json

import pytest

from mltree_mcp import server


async def call(name, args=None):
    """Call a tool via the MCP server and return (is_error, payload)."""
    result = await server.mcp.call_tool(name, args or {})
    # SDK 2.x returns a CallToolResult; 1.x returns (content, structured)
    if hasattr(result, "content"):
        is_error = bool(getattr(result, "is_error", False) or getattr(result, "isError", False))
        structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
        text = result.content[0].text if result.content else ""
    else:
        content, structured = result
        is_error, text = False, content[0].text
    if is_error:
        return True, text
    return False, structured if structured is not None else json.loads(text)


async def call_ok(name, args=None):
    err, data = await call(name, args)
    assert not err, data
    return data


async def call_err(name, args=None):
    try:
        err, data = await call(name, args)
    except Exception as exc:  # some SDK versions raise instead of returning is_error
        return str(exc)
    assert err, f"expected an error from {name}, got {data}"
    return data


async def test_tool_list():
    names = {t.name for t in await server.mcp.list_tools()}
    assert names == {"init_project", "add_step", "connect_steps", "get_pipeline", "log_run", "list_runs", "best_run", "open_map", "scan_project"}


async def test_tools_require_init(project):
    msg = await call_err("get_pipeline")
    assert "init_project" in msg


async def test_full_tool_flow(project):
    r = await call_ok("init_project", {"name": "flow"})
    assert r["created"] and (project / ".mltree").is_dir()

    (project / "steps").mkdir()
    (project / "steps" / "split.py").write_text("# split\n")
    s = await call_ok("add_step", {
        "name": "split", "script_path": "steps/split.py", "outputs": ["train.pkl"],
        "params": {"seed": 42}, "description": "split data",
    })
    assert s["action"] == "created" and s["script_exists"]
    await call_ok("add_step", {"name": "train", "inputs": ["train.pkl"], "outputs": ["model.pt"]})

    e = await call_ok("connect_steps", {"source": "split", "target": "train"})
    assert e["shared_data"] == ["train.pkl"]

    base = await call_ok("log_run", {"step": "train", "params": {"lr": 0.01}, "metrics": {"val_rmse": 0.8}})
    trial = await call_ok("log_run", {
        "step": "train", "params": {"lr": 0.001}, "metrics": {"val_rmse": 0.6},
        "parent_run_id": base["id"], "tags": ["hpo"], "notes": "trial 1",
    })
    assert trial["parent_run_id"] == base["id"]

    runs = await call_ok("list_runs", {"step": "train"})
    assert runs["count"] == 2 and runs["runs"][0]["id"] == trial["id"]
    kids = await call_ok("list_runs", {"parent_run_id": base["id"]})
    assert kids["count"] == 1

    best = await call_ok("best_run", {"metric": "val_rmse", "mode": "min"})
    assert best["run"]["id"] == trial["id"] and best["value"] == 0.6

    p = await call_ok("get_pipeline")
    nodes = {n["name"]: n for n in p["nodes"]}
    assert nodes["train"]["latest_run"]["id"] == trial["id"]
    assert nodes["split"]["status"] == "not_run"


async def test_tool_validation_errors(project):
    await call_ok("init_project")
    assert "Invalid step name" in await call_err("add_step", {"name": "has space"})
    await call_ok("add_step", {"name": "a"})
    assert "Unknown step" in await call_err("connect_steps", {"source": "a", "target": "b"})
    assert "Unknown step" in await call_err("log_run", {"step": "nope"})
    assert "parent_run_id" in await call_err("log_run", {"step": "a", "parent_run_id": "r_x"})
    assert await call_err("log_run", {"step": "a", "status": "done"})  # schema enum
    assert "No succeeded runs" in await call_err("best_run", {"metric": "auc"})
    assert await call_err("best_run", {"metric": "auc", "mode": "lowest"})  # schema enum


async def test_pipeline_resource(project):
    contents = await server.mcp.read_resource("mltree://pipeline")
    first = contents[0] if isinstance(contents, (list, tuple)) else contents.contents[0]
    text = getattr(first, "content", None) or getattr(first, "text", None)
    assert "error" in json.loads(text)
    await call_ok("init_project", {"name": "res"})
    await call_ok("add_step", {"name": "load"})
    contents = await server.mcp.read_resource("mltree://pipeline")
    first = contents[0] if isinstance(contents, (list, tuple)) else contents.contents[0]
    text = getattr(first, "content", None) or getattr(first, "text", None)
    data = json.loads(text)
    assert data["project"] == "res" and data["nodes"][0]["name"] == "load"
