"""HTTP API of `mltree-mcp view`, the SPA fallback, live events and the open_map tool."""
import json
import socket
import threading
import time
import urllib.request

import pytest
from starlette.testclient import TestClient

from mltree_mcp import Store, server
from mltree_mcp.viewer import STATIC_DIR, create_app


@pytest.fixture
def demo(project):
    (project / "steps").mkdir()
    (project / "steps" / "train.py").write_text("print('v1')\n")
    s = Store(project)
    s.init("viewer")
    s.add_step("split", outputs=["train.pkl"], params={"seed": 42})
    s.add_step("train", "steps/train.py", inputs=["train.pkl"])
    s.connect_steps("split", "train")
    base = s.log_run("train", {"lr": 0.1}, {"rmse": 0.9, "r2": 0.5})
    (project / "steps" / "train.py").write_text("print('v2')\n")
    child = s.log_run("train", {"lr": 0.01}, {"rmse": 0.6, "r2": 0.7}, parent_run_id=base["id"])
    failed = s.log_run("train", {"lr": 1.0}, status="failed", parent_run_id=base["id"])
    return {"store": s, "base": base, "child": child, "failed": failed}


@pytest.fixture
def client(project):
    return TestClient(create_app(project))


def test_health(client, demo):
    r = client.get("/api/health").json()
    assert r["ok"] and r["project"] == "viewer" and r["initialized"] and r["revision"] > 0


def test_uninitialized_project_gives_409(client):
    r = client.get("/api/pipeline")
    assert r.status_code == 409 and "init_project" in r.json()["error"]


def test_pipeline(client, demo):
    p = client.get("/api/pipeline").json()
    nodes = {n["name"]: n for n in p["nodes"]}
    assert nodes["train"]["status"] == "failed" and nodes["train"]["run_count"] == 3
    assert nodes["split"]["status"] == "not_run" and nodes["split"]["position"] is None
    assert p["edges"] == [{"source": "split", "target": "train", "label": ""}]


def test_positions_saved_and_validated(client, demo):
    rev = client.get("/api/health").json()["revision"]
    r = client.put("/api/positions", json={"positions": {"split": {"x": 10, "y": 20.5}, "train": {"x": 300, "y": 20}}})
    assert r.status_code == 200 and sorted(r.json()["updated"]) == ["split", "train"]
    nodes = {n["name"]: n for n in client.get("/api/pipeline").json()["nodes"]}
    assert nodes["split"]["position"] == {"x": 10.0, "y": 20.5}
    assert client.get("/api/health").json()["revision"] > rev  # live refresh sees the change
    assert client.put("/api/positions", json={"positions": {"split": None}}).status_code == 200
    assert client.get("/api/steps/split").json()["position"] is None
    assert client.put("/api/positions", json={"positions": {"nope": {"x": 1, "y": 1}}}).status_code == 404
    assert client.put("/api/positions", json={"positions": {"split": {"x": "a"}}}).status_code == 400
    assert client.put("/api/positions", content=b"not json").status_code == 400
    assert client.put("/api/positions", json={"positions": {}}).status_code == 400


def test_step_and_unknown_step(client, demo):
    assert client.get("/api/steps/train").json()["script_path"] == "steps/train.py"
    r = client.get("/api/steps/nope")
    assert r.status_code == 404 and "Unknown step" in r.json()["error"]


def test_script_current_and_snapshots(client, demo):
    cur = client.get("/api/steps/train/script").json()
    assert cur["source"] == "current" and cur["content"] == "print('v2')\n"
    snaps = client.get("/api/steps/train/snapshots").json()["snapshots"]
    assert len(snaps) == 3
    by_id = {s["run_id"]: s for s in snaps}
    assert by_id[demo["base"]["id"]]["same_as_current"] is False
    assert by_id[demo["child"]["id"]]["same_as_current"] is True
    old = client.get(f"/api/steps/train/script?run_id={demo['base']['id']}").json()
    assert old["source"] == "snapshot" and old["content"] == "print('v1')\n" and old["run_id"] == demo["base"]["id"]
    # no script registered for split
    assert client.get("/api/steps/split/script").status_code == 404
    # a run of another step
    other = demo["store"].log_run("split")
    assert client.get(f"/api/steps/train/script?run_id={other['id']}").status_code == 400
    assert client.get("/api/steps/train/script?run_id=r_missing").status_code == 404


def test_runs_and_run(client, demo):
    r = client.get("/api/runs").json()
    assert r["count"] == 3 and r["runs"][0]["id"] == demo["failed"]["id"]
    assert client.get("/api/runs?status=failed").json()["count"] == 1
    assert client.get(f"/api/runs?parent_run_id={demo['base']['id']}").json()["count"] == 2
    assert client.get("/api/runs?limit=abc").status_code == 400
    assert client.get("/api/runs?step=nope").status_code == 404
    assert client.get(f"/api/runs/{demo['child']['id']}").json()["metrics"]["rmse"] == 0.6
    assert client.get("/api/runs/r_missing").status_code == 404


def test_tree_and_best(client, demo):
    t = client.get("/api/tree?metric=rmse&mode=min").json()
    assert [r["id"] for r in t["runs"]][0] == demo["base"]["id"]  # oldest first
    assert t["roots"] == [demo["base"]["id"]]
    assert {(e["source"], e["target"]) for e in t["edges"]} == {
        (demo["base"]["id"], demo["child"]["id"]), (demo["base"]["id"], demo["failed"]["id"])}
    assert t["best_run_id"] == demo["child"]["id"] and t["best_value"] == 0.6
    assert t["metrics"] == ["r2", "rmse"]
    assert client.get("/api/tree?metric=r2&mode=max").json()["best_run_id"] == demo["child"]["id"]
    assert client.get("/api/tree?metric=auc").json()["best_run_id"] is None
    assert client.get("/api/tree").json()["best_run_id"] is None
    assert client.get("/api/tree?mode=up").status_code == 400
    assert client.get("/api/tree?step=split").json()["runs"] == []


def test_compare(client, demo):
    ids = f"{demo['base']['id']},{demo['child']['id']},{demo['failed']['id']}"
    c = client.get(f"/api/compare?ids={ids}").json()
    assert [r["id"] for r in c["runs"]] == ids.split(",")
    assert c["param_keys"] == ["lr"] and c["differing_params"] == ["lr"]
    assert c["metric_keys"] == ["r2", "rmse"]
    assert client.get(f"/api/compare?ids={demo['base']['id']}").status_code == 400
    assert client.get(f"/api/compare?ids={demo['base']['id']},r_missing").status_code == 404


def test_metrics(client, demo):
    assert client.get("/api/metrics").json() == {"metrics": ["r2", "rmse"]}


def test_unknown_api_route_is_json_404(client):
    r = client.get("/api/does/not/exist")
    assert r.status_code == 404 and r.json()["error"] == "Not found"


def test_spa_served_from_package(client):
    assert (STATIC_DIR / "index.html").is_file(), "frontend build missing from the package"
    r = client.get("/")
    assert r.status_code == 200 and '<div id="root">' in r.text
    assert client.get("/tree/deep/link").status_code == 200  # SPA fallback
    asset = next((STATIC_DIR / "assets").glob("index-*.js")).name
    r = client.get(f"/assets/{asset}")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]


# ------------------------------------------------- real server: SSE + open_map


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(project, demo):
    import uvicorn

    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(project, poll_interval=0.1), host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield port
    srv.should_exit = True
    t.join(timeout=5)


def _read_events(resp, n):
    events, cur = [], {}
    while len(events) < n:
        line = resp.readline().decode().rstrip("\n")
        if line.startswith("event:"):
            cur["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            cur["data"] = json.loads(line.split(":", 1)[1])
        elif line == "" and cur:
            events.append(cur)
            cur = {}
    return events


def test_sse_emits_revision_on_change(live_server, demo):
    with urllib.request.urlopen(f"http://127.0.0.1:{live_server}/api/events", timeout=5) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        first = _read_events(resp, 1)[0]
        assert first["event"] == "revision"
        demo["store"].log_run("split", metrics={"n": 1})
        second = _read_events(resp, 1)[0]
        assert second["data"]["revision"] > first["data"]["revision"]


def test_open_map_reports_running_viewer(live_server, project, monkeypatch):
    monkeypatch.setenv("MLTREE_VIEW_PORT", str(live_server))
    r = server._open_map(start=False)
    assert r["url"] == f"http://localhost:{live_server}" and r["running"] is True and r["started"] is False
    assert f"ssh -L {live_server}:localhost:{live_server}" in r["remote_tip"]
    assert "warning" not in r


def test_open_map_not_running(project, monkeypatch):
    monkeypatch.setenv("MLTREE_VIEW_PORT", str(_free_port()))
    r = server._open_map(start=False)
    assert r["running"] is False and "mltree-mcp view" in r["hint"]


def test_open_map_starts_viewer(project, demo, monkeypatch):
    port = _free_port()
    monkeypatch.setenv("MLTREE_VIEW_PORT", str(port))
    r = server._open_map(start=True)
    try:
        assert r["started"] and r["running"], r
        health = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2).read())
        assert health["root"] == str(project)
    finally:
        import os
        import signal
        import subprocess

        out = subprocess.run(["pgrep", "-f", f"mltree_mcp view --project {project} --port {port}"], capture_output=True, text=True)
        for pid in out.stdout.split():
            os.kill(int(pid), signal.SIGTERM)
