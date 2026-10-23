"""Notes on steps and runs, run tags: store, MCP tools, viewer write routes and their guard."""
import json
import sqlite3

import pytest
from starlette.testclient import TestClient

from mltree_mcp import Store
from mltree_mcp.store import MLTreeError
from mltree_mcp.viewer import create_app

from test_tools import call_err, call_ok

H = {"X-MLTree": "1"}


@pytest.fixture
def demo(store):
    store.add_step("train", "train.py")
    r = store.log_run("train", {"lr": 0.1}, {"rmse": 0.5})
    return store, r["id"]


def test_step_and_run_notes(demo):
    s, rid = demo
    s.add_note("baseline looks fine", step="train")
    out = s.add_note("try lr 0.01 next", step="train")
    assert out["notes"] == "baseline looks fine\n\ntry lr 0.01 next"
    assert s.add_note("only this", step="train", replace=True)["notes"] == "only this"
    s.add_note("overfits after epoch 30", run_id=rid)
    assert s.get_run(rid)["notes"] == "overfits after epoch 30"
    node = next(n for n in s.get_pipeline()["nodes"] if n["name"] == "train")
    assert node["notes"] == "only this"
    with pytest.raises(MLTreeError):
        s.add_note("x")  # no target
    with pytest.raises(MLTreeError):
        s.add_note("x", step="train", run_id=rid)
    with pytest.raises(MLTreeError):
        s.add_note("x" * 20_001, step="train", replace=True)
    with pytest.raises(MLTreeError):
        s.add_note("x", step="nope")


def test_notes_survive_yaml_rebuild(demo):
    s, rid = demo
    s.set_step_notes("train", "line 1\r\nline 2")
    s.tag_run(rid, add=["baseline"])
    assert "line 1" in (s.root / ".mltree" / "pipeline.yaml").read_text()
    assert json.loads((s.root / ".mltree" / "runs" / rid / "run.json").read_text())["notes"] == ""
    (s.root / ".mltree" / "mltree.db").unlink()
    s2 = Store(s.root)  # the map (not the run history) is rebuilt from pipeline.yaml
    assert s2.get_step("train")["notes"] == "line 1\nline 2"


def test_tags(demo):
    s, rid = demo
    assert s.tag_run(rid, add=["baseline", "fast", "baseline"])["tags"] == ["baseline", "fast"]
    assert s.tag_run(rid, remove=["fast"], add=["best"])["tags"] == ["baseline", "best"]
    assert json.loads((s.root / ".mltree" / "runs" / rid / "run.json").read_text())["tags"] == ["baseline", "best"]
    r2 = s.log_run("train", {"lr": 0.2}, tags=["best"])
    assert s.tags_in_use() == {"best": 2, "baseline": 1}
    assert s.get_pipeline()["tags"] == {"best": 2, "baseline": 1}
    assert s.tag_run(r2["id"], tags=[])["tags"] == []
    for bad in ["", "a,b", "x" * 65, "a\nb"]:
        with pytest.raises(MLTreeError):
            s.tag_run(rid, add=[bad])
    with pytest.raises(MLTreeError):
        s.tag_run("nope", add=["x"])


def test_migrates_v3_db(demo):
    s, rid = demo
    db = s.root / ".mltree" / "mltree.db"
    with sqlite3.connect(db) as c:  # simulate a 0.6 database: no steps.notes column
        c.execute("ALTER TABLE steps DROP COLUMN notes")
        c.execute("UPDATE meta SET value='3' WHERE key='schema_version'")
    s2 = Store(s.root)
    assert s2.get_step("train")["notes"] == ""
    s2.set_step_notes("train", "ok")
    assert s2.get_step("train")["notes"] == "ok"


async def test_mcp_tools(demo):
    s, rid = demo
    out = await call_ok("add_note", {"text": "hello", "step": "train"})
    assert out["notes"] == "hello" and out["target"] == "step"
    out = await call_ok("tag_run", {"run_id": rid, "add": ["baseline"]})
    assert out["tags"] == ["baseline"] and out["tags_in_use"] == {"baseline": 1}
    await call_err("add_note", {"text": "x"})  # needs a step or a run
    p = await call_ok("get_pipeline")
    assert p["tags"] == {"baseline": 1}
    assert next(n for n in p["nodes"] if n["name"] == "train")["notes"] == "hello"


def test_viewer_write_routes(demo):
    s, rid = demo
    c = TestClient(create_app(s.root))  # no --allow-run: notes and tags still editable
    assert c.put("/api/steps/train/notes", json={"notes": "from the viewer"}, headers=H).json()["notes"] == "from the viewer"
    assert c.put(f"/api/runs/{rid}/notes", json={"notes": "run note"}, headers=H).json()["notes"] == "run note"
    assert c.put(f"/api/runs/{rid}/tags", json={"add": ["a", "b"]}, headers=H).json()["tags"] == ["a", "b"]
    assert c.put(f"/api/runs/{rid}/tags", json={"remove": ["a"]}, headers=H).json()["tags"] == ["b"]
    assert c.put(f"/api/runs/{rid}/tags", json={"tags": ["z"]}, headers=H).json()["tags"] == ["z"]
    assert c.put("/api/steps/nope/notes", json={"notes": "x"}, headers=H).status_code == 404
    assert c.put(f"/api/runs/{rid}/tags", json={"add": ["a,b"]}, headers=H).status_code == 400
    assert s.get_step("train")["notes"] == "from the viewer"


def test_viewer_write_guard(demo):
    s, rid = demo
    c = TestClient(create_app(s.root))
    assert c.put("/api/steps/train/notes", json={"notes": "x"}).status_code == 403  # no X-MLTree header
    r = c.put("/api/steps/train/notes", json={"notes": "x"}, headers={**H, "Origin": "http://evil.example"})
    assert r.status_code == 403
    r = c.put("/api/steps/train/notes", json={"notes": "x"}, headers={**H, "Host": "evil.example:8765"})
    assert r.status_code == 403  # DNS rebinding
    assert c.put(f"/api/runs/{rid}/tags", json={"add": ["x"]}, headers={"Origin": "http://localhost:8765"}).status_code == 403
    ok = c.put("/api/steps/train/notes", json={"notes": "y"}, headers={**H, "Origin": "http://127.0.0.1:8765"})
    assert ok.status_code == 200
    assert s.get_step("train")["notes"] == "y" and s.get_run(rid)["tags"] == []
