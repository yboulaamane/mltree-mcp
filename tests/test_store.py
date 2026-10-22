import json
import sqlite3

import pytest
import yaml

from mltree_mcp import MLTreeError, ProjectNotInitialized, Store, resolve_project_root


def test_resolve_root_env_and_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("MLTREE_PROJECT", raising=False)
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_project_root() == tmp_path.resolve()
    sub = tmp_path / "proj"
    sub.mkdir()
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(sub))
    assert resolve_project_root() == sub.resolve()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("MLTREE_PROJECT", str(other))
    assert resolve_project_root() == other.resolve()
    monkeypatch.setenv("MLTREE_PROJECT", str(tmp_path / "missing"))
    with pytest.raises(MLTreeError, match="does not exist"):
        resolve_project_root()


def test_init_creates_layout_and_is_idempotent(project):
    s = Store(project)
    assert not s.is_initialized()
    r = s.init("demo")
    assert r["created"] is True and r["project"] == "demo"
    assert (project / ".mltree" / "mltree.db").is_file()
    assert (project / ".mltree" / "pipeline.yaml").is_file()
    assert (project / ".mltree" / "runs").is_dir()
    assert s.init()["created"] is False
    assert s.project_name() == "demo"


def test_schema_tables_and_columns(store):
    con = sqlite3.connect(store.db_path)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"meta", "steps", "edges", "runs"} <= tables
    run_cols = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    assert {"params", "metrics", "artifacts", "parent_run_id", "status", "started_at", "finished_at", "created_at"} <= run_cols
    step_cols = {r[1] for r in con.execute("PRAGMA table_info(steps)")}
    assert {"pos_x", "pos_y", "kind"} <= step_cols  # reserved for the Phase 2 viewer


def test_uninitialized_errors(project):
    s = Store(project)
    for call in (s.get_pipeline, lambda: s.add_step("a"), s.list_runs, lambda: s.best_run("m")):
        with pytest.raises(ProjectNotInitialized, match="init_project"):
            call()


def test_add_step_and_update(store, project):
    (project / "steps").mkdir()
    (project / "steps" / "clean.py").write_text("print('clean')\n")
    r = store.add_step("clean", "steps/clean.py", ["raw.csv"], ["clean.csv"], {"dropna": True}, "Clean data")
    assert r["action"] == "created" and r["script_exists"] and "warnings" not in r
    r2 = store.add_step("clean", str(project / "steps" / "clean.py"), params={"dropna": False})
    assert r2["action"] == "updated" and r2["params"] == {"dropna": False}
    assert r2["script_path"] == "steps/clean.py"  # absolute path normalised
    r3 = store.add_step("train", "steps/train.py")
    assert r3["warnings"] == ["script steps/train.py does not exist yet"]


@pytest.mark.parametrize(
    "kwargs, msg",
    [
        ({"name": ""}, "non-empty"),
        ({"name": "bad name"}, "Invalid step name"),
        ({"name": "ok", "script_path": "../outside.py"}, "outside the project root"),
        ({"name": "ok", "inputs": "a.csv,b.csv", "outputs": [1]}, "outputs must contain only"),
        ({"name": "ok", "params": [1, 2]}, "params must be an object"),
        ({"name": "ok", "params": {"x": float("nan")}}, "JSON-serialisable"),
    ],
)
def test_add_step_validation(store, kwargs, msg):
    with pytest.raises(MLTreeError, match=msg):
        store.add_step(**kwargs)


def test_connect_steps_dag_rules(store):
    for n in ("a", "b", "c"):
        store.add_step(n, outputs=[f"{n}.out"], inputs=[] if n == "a" else ["x"])
    r = store.connect_steps("a", "b")
    assert r["action"] == "created" and r["warnings"]
    assert store.connect_steps("a", "b")["action"] == "exists"
    store.connect_steps("b", "c")
    with pytest.raises(MLTreeError, match="cycle"):
        store.connect_steps("c", "a")
    with pytest.raises(MLTreeError, match="itself"):
        store.connect_steps("a", "a")
    with pytest.raises(MLTreeError, match="Unknown step"):
        store.connect_steps("a", "zzz")
    assert store.topological_order() == ["a", "b", "c"]


def test_shared_data_detected(store):
    store.add_step("split", outputs=["train.pkl"])
    store.add_step("train", inputs=["train.pkl"])
    r = store.connect_steps("split", "train")
    assert r["shared_data"] == ["train.pkl"] and "warnings" not in r


def test_log_run_writes_db_and_run_dir(store, project):
    (project / "train.py").write_text("print(1)\n")
    (project / "model.pt").write_bytes(b"weights")
    store.add_step("train", "train.py")
    run = store.log_run("train", {"lr": 0.01}, {"val_rmse": 0.5}, ["model.pt"], notes="first")
    run_dir = project / run["run_dir"]
    assert run_dir.is_dir()
    assert json.loads((run_dir / "run.json").read_text())["metrics"] == {"val_rmse": 0.5}
    assert (run_dir / "script.py").read_text() == "print(1)\n"
    art = run["artifacts"][0]
    assert art["stored"] == "copied" and (project / art["path"]).read_bytes() == b"weights"
    assert run["script_sha256"] and run["status"] == "succeeded" and run["finished_at"]
    assert store.get_run(run["id"])["params"] == {"lr": 0.01}


def test_log_run_parent_and_validation(store):
    store.add_step("train")
    parent = store.log_run("train", metrics={"m": 1})
    child = store.log_run("train", metrics={"m": 2}, parent_run_id=parent["id"])
    assert child["parent_run_id"] == parent["id"]
    with pytest.raises(MLTreeError, match="parent_run_id"):
        store.log_run("train", parent_run_id="r_nope")
    with pytest.raises(MLTreeError, match="Unknown step"):
        store.log_run("nope")
    with pytest.raises(MLTreeError, match="must be a number"):
        store.log_run("train", metrics={"acc": "high"})
    with pytest.raises(MLTreeError, match="must be a number"):
        store.log_run("train", metrics={"ok": True})
    with pytest.raises(MLTreeError, match="status must be"):
        store.log_run("train", status="done")
    with pytest.raises(MLTreeError, match="not found"):
        store.log_run("train", artifacts=["missing.pt"])


def test_list_runs_filters(store):
    store.add_step("a")
    store.add_step("b")
    r1 = store.log_run("a", metrics={"m": 1})
    store.log_run("a", status="failed", parent_run_id=r1["id"])
    store.log_run("b", metrics={"m": 3})
    assert len(store.list_runs()) == 3
    assert [r["step"] for r in store.list_runs(step="a")] == ["a", "a"]
    assert store.list_runs()[0]["step"] == "b"  # newest first
    assert len(store.list_runs(status="failed")) == 1
    assert len(store.list_runs(parent_run_id=r1["id"])) == 1
    assert len(store.list_runs(limit=1)) == 1
    with pytest.raises(MLTreeError):
        store.list_runs(limit=0)
    with pytest.raises(MLTreeError):
        store.list_runs(status="weird")


def test_best_run(store):
    store.add_step("train")
    store.add_step("eval")
    store.log_run("train", metrics={"rmse": 0.9, "r2": 0.5})
    best_min = store.log_run("train", metrics={"rmse": 0.4, "r2": 0.7})
    store.log_run("train", metrics={"rmse": 0.1}, status="failed")  # ignored
    store.log_run("eval", metrics={"r2": 0.95})
    r = store.best_run("rmse", "min")
    assert r["run"]["id"] == best_min["id"] and r["value"] == 0.4 and r["candidates"] == 2
    assert store.best_run("r2", "max")["value"] == 0.95
    assert store.best_run("r2", "max", step="train")["value"] == 0.7
    with pytest.raises(MLTreeError, match="No succeeded runs"):
        store.best_run("auc")
    with pytest.raises(MLTreeError, match="mode"):
        store.best_run("rmse", "lowest")


def test_get_pipeline_latest_status(store):
    store.add_step("a")
    store.add_step("b")
    store.connect_steps("a", "b")
    store.log_run("a", metrics={"x": 1})
    store.log_run("a", status="failed")
    p = store.get_pipeline()
    nodes = {n["name"]: n for n in p["nodes"]}
    assert nodes["a"]["status"] == "failed" and nodes["a"]["run_count"] == 2
    assert nodes["b"]["status"] == "not_run" and nodes["b"]["latest_run"] is None
    assert p["edges"] == [{"source": "a", "target": "b", "label": ""}]
    assert p["order"] == ["a", "b"]


def test_yaml_written_and_reimported(store, project):
    store.add_step("a", params={"k": 1})
    store.add_step("b")
    store.connect_steps("a", "b")
    doc = yaml.safe_load((project / ".mltree" / "pipeline.yaml").read_text())
    assert doc["steps"]["a"]["params"] == {"k": 1}
    assert doc["edges"] == [{"from": "a", "to": "b"}]
    # simulate a fresh clone where only pipeline.yaml was committed
    for f in (project / ".mltree").glob("mltree.db*"):
        f.unlink()
    fresh = Store(project)
    p = fresh.get_pipeline()
    assert [n["name"] for n in p["nodes"]] == ["a", "b"]
    assert p["edges"][0]["source"] == "a"
