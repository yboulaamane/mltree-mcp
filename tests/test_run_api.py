"""Run/sweep endpoints of the viewer and the run_step / run_sweep MCP tools (0.5.0)."""
import json
import sys
import time

import pytest
from starlette.testclient import TestClient

from mltree_mcp import Store, server
from mltree_mcp.viewer import create_app

from test_tools import call_err, call_ok

H = {"X-MLTree": "1"}
SCRIPT = """
import argparse, sys
p = argparse.ArgumentParser()
p.add_argument("--alpha", type=float, default=1.0)
a = p.parse_args()
print("hello", flush=True)
print(f"MLTREE_METRIC rmse={abs(a.alpha - 0.5) + 0.1:.3f}", flush=True)
"""


@pytest.fixture
def proj(project, monkeypatch):
    monkeypatch.setenv("MLTREE_PYTHON", sys.executable)
    (project / "steps").mkdir()
    (project / "steps" / "train.py").write_text(SCRIPT)
    (project / "steps" / "slow.py").write_text("import time\nprint('start', flush=True)\ntime.sleep(60)\n")
    s = Store(project)
    s.init("runs")
    s.add_step("train", "steps/train.py", params={"alpha": 1.0})
    s.add_step("slow", "steps/slow.py")
    (project / "steps" / "report.py").write_text("print('MLTREE_METRIC ok=1')\n")
    s.add_step("report", "steps/report.py")
    s.connect_steps("train", "report")
    return s


def _wait(client, run_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/runs/{run_id}").json()
        if r["status"] != "running":
            return r
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_running_is_off_by_default(proj):
    with TestClient(create_app(proj.root)) as c:
        assert c.get("/api/health").json()["allow_run"] is False
        plan = c.get("/api/steps/train/plan").json()
        assert plan["runnable"] and plan["allow_run"] is False and "--alpha" in plan["command"]
        r = c.post("/api/steps/train/run", json={}, headers=H)
        assert r.status_code == 403 and "--allow-run" in r.json()["error"]
    assert proj.list_runs() == []


@pytest.mark.parametrize("headers", [
    {},  # no X-MLTree header (plain HTML form / curl)
    {**H, "Origin": "http://evil.example"},  # cross-site page
    {**H, "Host": "evil.example"},  # DNS rebinding
])
def test_run_guard(proj, headers):
    with TestClient(create_app(proj.root, allow_run=True)) as c:
        r = c.post("/api/steps/train/run", json={}, headers=headers)
        assert r.status_code == 403, r.text
        assert c.post("/api/runs/x/cancel", json={}, headers=headers).status_code == 403
    assert proj.list_runs() == []


def test_run_log_stream_history(proj):
    with TestClient(create_app(proj.root, allow_run=True)) as c:
        q = json.dumps({"alpha": 0.25})
        assert c.get(f"/api/steps/train/plan?params={q}&extra_args=--x%201").json()["argv"][-4:] == ["--alpha", "0.25", "--x", "1"]
        r = c.post("/api/steps/train/run", json={"params": {"alpha": 0.25}}, headers={**H, "Origin": "http://127.0.0.1:8765"})
        assert r.status_code == 200, r.text
        rid = r.json()["run_id"]
        run = _wait(c, rid)
        assert run["status"] == "succeeded" and run["metrics"] == {"rmse": 0.35} and run["exit_code"] == 0
        log = c.get(f"/api/runs/{rid}/log").json()
        assert [ln["line"] for ln in log["lines"]][0] == "hello" and log["done"]
        assert c.get(f"/api/runs/{rid}/log?tail=1").json()["lines"][0]["line"].startswith("MLTREE_METRIC")
        with c.stream("GET", f"/api/runs/{rid}/log/stream") as s:
            body = "".join(s.iter_text())
        assert "event: lines" in body and "event: end" in body and '"hello"' in body
        assert c.get(f"/api/runs/{rid}/history").json()["history"][0]["rmse"] == 0.35
        assert c.get("/api/steps/train/plan").json()["defaults"]["params"] == {"alpha": 0.25}
        assert c.get("/api/runs/nope/log").status_code == 404


def test_cancel_and_active(proj):
    with TestClient(create_app(proj.root, allow_run=True)) as c:
        rid = c.post("/api/steps/slow/run", json={}, headers=H).json()["run_id"]
        assert [a["run_id"] for a in c.get("/api/active").json()["runs"]] == [rid]
        assert c.post(f"/api/runs/{rid}/cancel", json={}, headers=H).json()["cancelled"]
        assert _wait(c, rid)["status"] == "cancelled"
        assert c.get("/api/active").json() == {"runs": [], "sweeps": []}


def test_sweep_endpoint(proj):
    with TestClient(create_app(proj.root, allow_run=True)) as c:
        r = c.post("/api/steps/train/sweep", json={"grid": {"alpha": "0.1, 0.5, 2"}, "metric": "rmse", "mode": "min"}, headers=H)
        assert r.status_code == 200, r.text
        sid = r.json()["sweep_id"]
        assert r.json()["total"] == 3
        sweep = _wait(c, sid, 60)
        assert sweep["kind"] == "sweep" and sweep["status"] == "succeeded"
        tree = c.get("/api/tree?metric=rmse&mode=min").json()
        summ = tree["sweeps"][sid]
        assert summ["n"] == 3 and summ["best_params"] == {"alpha": 0.5} and summ["best_value"] == pytest.approx(0.1)
        assert c.post("/api/steps/train/sweep", json={"grid": {}}, headers=H).status_code == 400


async def test_mcp_run_step_and_sweep(proj, monkeypatch):
    server._MANAGERS.clear()
    r = await call_ok("run_step", {"step": "train", "params": {"alpha": 0.5}})
    assert r["finished"] and r["run"]["status"] == "succeeded" and r["run"]["metrics"]["rmse"] == pytest.approx(0.1)
    assert "hello" in r["log_tail"]
    log = await call_ok("run_log", {"run_id": r["run"]["id"]})
    assert log["finished"] and "MLTREE_METRIC" in log["log_tail"]
    sw = await call_ok("run_sweep", {"step": "train", "grid": {"alpha": [0.4, 0.5]}, "metric": "rmse", "mode": "min"})
    assert sw["finished"] and sw["total"] == 2 and sw["summary"]["best_params"] == {"alpha": 0.5}
    started = await call_ok("run_step", {"step": "slow", "wait": False})
    assert not started["finished"]
    assert (await call_ok("cancel_run", {"run_id": started["run"]["id"]}))["cancelled"]
    monkeypatch.setenv("MLTREE_ALLOW_RUN", "0")
    server._MANAGERS.clear()
    assert "MLTREE_ALLOW_RUN" in str(await call_err("run_step", {"step": "train"}))


async def test_mcp_manual_sweep(proj):
    sw = await call_ok("start_sweep", {"step": "train", "metric": "rmse", "mode": "min", "search_space": {"alpha": [1, 2]}})
    sid = sw["sweep_id"]
    for a, v in [(1, 0.6), (2, 1.6)]:
        await call_ok("log_run", {"step": "train", "params": {"alpha": a}, "metrics": {"rmse": v}, "sweep_id": sid})
    done = await call_ok("finish_sweep", {"sweep_id": sid})
    assert done["n"] == 2 and done["best_value"] == 0.6 and done["best_params"] == {"alpha": 1}
    assert proj.get_run(sid)["status"] == "succeeded"


def test_runs_branch_from_upstream_run(proj):
    with TestClient(create_app(proj.root, allow_run=True)) as c:
        assert c.get("/api/steps/report/plan").json()["parent_run_id"] is None
        tid = c.post("/api/steps/train/run", json={}, headers=H).json()["run_id"]
        assert _wait(c, tid)["parent_run_id"] is None  # no upstream step
        assert c.get("/api/steps/report/plan").json()["parent_run_id"] == tid
        rid = c.post("/api/steps/report/run", json={}, headers=H).json()["run_id"]
        assert _wait(c, rid)["parent_run_id"] == tid
        assert c.get("/api/tree").json()["roots"] == [tid]
