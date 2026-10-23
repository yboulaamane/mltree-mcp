"""Data versioning (0.6.0): fingerprints, cache, changed inputs, compare data rows, code diff, history."""
import os
import sqlite3
import sys

import pytest

from mltree_mcp import dataver
from mltree_mcp.dataver import DataIndex
from mltree_mcp.runner import RunManager
from mltree_mcp.store import MLTreeError


def _pipeline(store):
    root = store.root
    (root / "data" / "raw").mkdir(parents=True)
    (root / "data" / "raw" / "a.csv").write_text("x,y\n1,2\n")
    (root / "data" / "raw" / "b.csv").write_text("x,y\n3,4\n")
    (root / "data" / "clean.csv").write_text("x,y\n1,2\n3,4\n")
    (root / "steps").mkdir()
    (root / "steps" / "clean.py").write_text("print('clean')\n")
    store.add_step("clean", script_path="steps/clean.py", inputs=["data/raw"], outputs=["clean.csv"])
    store.add_step("train", inputs=["data/clean.csv"], outputs=["model.pt"])
    store.connect_steps("clean", "train")


def test_fingerprints_files_dirs_and_cache(store, monkeypatch):
    _pipeline(store)
    idx = DataIndex(store)
    f = idx.fingerprint("data/clean.csv")
    assert f["exists"] and f["kind"] == "file" and f["bytes"] == 12 and len(f["version"]) == 12
    assert f["path"] == "data/clean.csv"
    d = idx.fingerprint("data/raw")
    assert d["kind"] == "dir" and d["files"] == 2
    # bare name resolves by unique basename; missing names don't exist
    assert idx.fingerprint("clean.csv")["path"] == "data/clean.csv"
    assert idx.fingerprint("model.pt") == {"name": "model.pt", "exists": False, "path": None}
    idx.flush()
    assert len(store.hash_cache()) == 3
    rev = store.revision()
    # cached hashes are reused (fast path): no file is re-read when size+mtime match
    calls = []
    monkeypatch.setattr(dataver, "_sha_file", lambda p: calls.append(p) or "0" * 64)
    assert DataIndex(store).fingerprint("data/clean.csv")["sha256"] == f["sha256"] and not calls
    os.utime(store.root / "data/clean.csv", ns=(1, 1))
    DataIndex(store).fingerprint("data/clean.csv")
    assert len(calls) == 1
    assert store.revision() == rev  # hashing never bumps the revision (no SSE redraw storms)


def test_quick_digest_for_huge_files(store, monkeypatch):
    _pipeline(store)
    monkeypatch.setattr(dataver, "MAX_HASH_BYTES", 4)
    fp = DataIndex(store).fingerprint("data/clean.csv")
    assert fp["quick"] is True and fp["sha256"]


def test_runs_record_data_and_changes_are_flagged(store):
    _pipeline(store)
    st = store.data_status(fresh=True)
    assert st["steps"]["clean"]["status"] == "no_record" and not st["changed_steps"]
    r1 = store.log_run("clean", params={"seed": 1})
    assert r1["data"]["inputs"]["data/raw"]["files"] == 2
    assert r1["data"]["outputs"]["clean.csv"]["path"] == "data/clean.csv"
    t1 = store.log_run("train", params={"lr": 0.1})
    st = store.data_status(fresh=True)
    assert st["steps"]["train"]["status"] == "ok"
    assert st["steps"]["train"]["inputs"][0]["producer_run_id"] == r1["id"]
    assert st["steps"]["train"]["missing_inputs"] == [] and st["steps"]["train"]["outputs"][0]["exists"] is False
    # change a raw file -> clean is stale; change clean.csv -> train is stale
    (store.root / "data/raw/b.csv").write_text("x,y\n3,5\n")
    (store.root / "data/clean.csv").write_text("x,y\n1,2\n3,5\n")
    st = store.data_status(fresh=True)
    assert st["changed_steps"] == ["clean", "train"]
    assert st["steps"]["train"]["changed_inputs"] == ["data/clean.csv"] and st["steps"]["train"]["status"] == "changed"
    assert "inputs that changed" in st["summary"]
    node = {n["name"]: n for n in store.get_pipeline()["nodes"]}["train"]
    assert node["data"]["changed_inputs"] == ["data/clean.csv"] and node["data"]["last_run_id"] == t1["id"]
    # catalog keeps the version history
    cat = st["data"]["data/clean.csv"]
    assert cat["used_by_steps"] == ["train"] and len(cat["versions"]) == 1
    t2 = store.log_run("train", params={"lr": 0.2}, parent_run_id=t1["id"])
    assert not store.data_status(fresh=True)["steps"]["train"]["changed_inputs"]
    assert len(store.data_status(fresh=True)["data"]["data/clean.csv"]["versions"]) == 2
    cmp = store.compare([t1["id"], t2["id"]])
    row = next(d for d in cmp["data"] if d["name"] == "data/clean.csv")
    assert row["role"] == "input" and row["differs"] and row["versions"][0] != row["versions"][1]
    assert cmp["differing_data"] == ["input:data/clean.csv"]
    assert cmp["param_changes"] == [{"run_id": t2["id"], "changes": [{"key": "lr", "from": 0.1, "to": 0.2}]}]
    # imports from elsewhere can skip it
    assert store.log_run("train", record_data=False)["data"] is None
    with pytest.raises(MLTreeError):
        store.data_status(step="nope")


def test_history_and_code_diff(store):
    _pipeline(store)
    r1 = store.log_run("clean", history=[{"epoch": 1, "loss": 0.9}, {"epoch": 2, "loss": 0.5}])
    assert [h["step"] for h in store.metric_history(r1["id"])] == [1, 2]
    (store.root / "steps/clean.py").write_text("print('clean')\nprint('more')\n")
    r2 = store.log_run("clean")
    d = store.code_diff(r1["id"], r2["id"])
    assert d["available"] and d["added"] == 1 and d["removed"] == 0 and "+print('more')" in d["diff"]
    assert store.code_diff(r1["id"], r1["id"])["identical"] is True
    t = store.log_run("train")  # no script -> no snapshot
    assert store.code_diff(r1["id"], t["id"])["available"] is False
    with pytest.raises(MLTreeError):
        store.log_run("clean", history=[{"loss": "bad"}])


def test_runner_records_inputs_then_outputs(store):
    root = store.root
    (root / "in.txt").write_text("hello")
    (root / "steps").mkdir()
    (root / "steps" / "make.py").write_text("open('out.txt', 'w').write(open('in.txt').read().upper())\n")
    store.add_step("make", script_path="steps/make.py", inputs=["in.txt"], outputs=["out.txt"])
    m = RunManager(store, python=sys.executable)
    try:
        live = m.start("make")
        assert m.wait(live.run_id, 30)
    finally:
        m.shutdown()
    run = store.get_run(live.run_id)
    assert run["status"] == "succeeded"
    assert run["data"]["inputs"]["in.txt"]["exists"] and run["data"]["outputs"]["out.txt"]["bytes"] == 5
    assert store.data_status(fresh=True)["steps"]["make"]["status"] == "ok"


def test_v2_db_gains_data_column(store):
    with sqlite3.connect(store.db_path) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(runs)")}
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "data" in cols and "file_hashes" in tables


async def test_mcp_tools(project):
    from test_tools import call_err, call_ok

    await call_ok("init_project", {"name": "t"})
    (project / "in.csv").write_text("a\n1\n")
    await call_ok("add_step", {"name": "s", "inputs": ["in.csv"]})
    a = await call_ok("log_run", {"step": "s", "params": {"k": 1}, "history": [{"epoch": 1, "loss": 1.0}]})
    (project / "in.csv").write_text("a\n2\n")
    st = await call_ok("data_status", {})
    assert st["changed_steps"] == ["s"]
    b = await call_ok("log_run", {"step": "s", "params": {"k": 2}, "record_data": False})
    cmp = await call_ok("compare_runs", {"run_ids": [a["id"], b["id"]]})
    assert cmp["param_changes"][0]["changes"] == [{"key": "k", "from": 1, "to": 2}]
    assert cmp["code_diff"]["available"] is False
    assert "at least 2" in await call_err("compare_runs", {"run_ids": [a["id"]]})
