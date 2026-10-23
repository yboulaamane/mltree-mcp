"""Running steps as subprocesses: CLI detection, metric capture, cancel, sweeps (0.5.0)."""
import json
import sqlite3
import sys
import time

import pytest

from mltree_mcp import Store
from mltree_mcp.runner import (
    RunManager, build_args, detect_cli, flatten_metrics, grid_combinations, parse_grid_text, parse_metric_line,
)
from mltree_mcp.store import MLTreeError


def test_detect_argparse_flags():
    src = """
import argparse
p = argparse.ArgumentParser()
p.add_argument("data")
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--hidden-dim", type=int)
p.add_argument("-v", "--verbose", action="store_true")
p.add_argument("--no-cache", dest="cache", action="store_false")
"""
    spec = detect_cli(src)
    assert spec.style == "flags" and spec.positionals == ["data"]
    args, env_only = build_args(spec, {"data": "x.csv", "lr": 0.5, "hidden_dim": 64, "verbose": True, "cache": False, "other": 1})
    assert args == ["x.csv", "--lr", "0.5", "--hidden-dim", "64", "--verbose", "--no-cache"]
    assert env_only == ["other"]
    assert build_args(spec, {"verbose": False, "cache": True})[0] == []


def test_detect_click_hydra_argv_env():
    click = detect_cli("import click\n@click.command()\n@click.option('--epochs', default=3)\n@click.argument('path')\ndef main(epochs, path): pass\n")
    assert click.style == "flags"
    assert build_args(click, {"path": "a", "epochs": 5})[0] == ["a", "--epochs", "5"]

    hydra = detect_cli("import hydra\n@hydra.main(config_path='conf')\ndef main(cfg): pass\n")
    assert hydra.style == "hydra"
    assert build_args(hydra, {"model.lr": 0.1, "flag": True})[0] == ["model.lr=0.1", "flag=true"]

    argv = detect_cli("import sys\nalpha = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0\nseed = int(sys.argv[2])\n")
    assert argv.style == "argv"
    assert build_args(argv, {"alpha": 0.5, "seed": 3, "layers": 2}) == (["0.5", "3"], ["layers"])
    assert build_args(argv, {"seed": 3}) == ([], ["seed"])  # can't skip argv[1]

    env = detect_cli("import os\nlr = float(os.environ.get('LR', 1))\n")
    assert env.style == "env" and build_args(env, {"lr": 1}) == ([], ["lr"])
    assert detect_cli("def broken(:").style == "env"


def test_metric_line_parsing():
    assert parse_metric_line("MLTREE_METRIC loss=0.31 acc=0.9") == {"loss": 0.31, "acc": 0.9}
    assert parse_metric_line('MLTREE_METRIC {"val": {"rmse": 1.5}, "name": "x"}') == {"val.rmse": 1.5}
    assert parse_metric_line("loss=0.31") is None
    assert parse_metric_line("MLTREE_METRIC nothing") in (None, {})
    assert flatten_metrics({"a": 1, "b": {"c": 2, "d": {"e": 3}}, "s": "x", "t": True}) == {"a": 1, "b.c": 2}


def test_grid_helpers():
    assert parse_grid_text("0.1, 1, x, true") == [0.1, 1, "x", True]
    assert parse_grid_text("[1, 2]") == [1, 2]
    combos = grid_combinations({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(combos) == 6 and combos[0] == {"a": 1, "b": "x"}
    with pytest.raises(MLTreeError, match="limit"):
        grid_combinations({"a": list(range(10)), "b": list(range(10))}, max_runs=50)
    with pytest.raises(MLTreeError):
        grid_combinations({"a": []})


def _script(store: Store, name: str, body: str, **step) -> None:
    (store.root / "steps").mkdir(exist_ok=True)
    (store.root / "steps" / f"{name}.py").write_text(body)
    store.add_step(name, script_path=f"steps/{name}.py", **step)


TRAIN = """
import argparse, json, os, sys
p = argparse.ArgumentParser()
p.add_argument("--lr", type=float, default=0.1)
a = p.parse_args()
params = json.loads(os.environ["MLTREE_PARAMS"])
for epoch in range(3):
    print(f"MLTREE_METRIC epoch={epoch} loss={1 / (epoch + 1) * a.lr:.4f}", flush=True)
print("to stderr", file=sys.stderr)
if params.get("write_file"):
    json.dump({"val": {"acc": 0.75}}, open(os.environ["MLTREE_METRICS_FILE"], "w"))
sys.exit(int(params.get("exit", 0)))
"""


@pytest.fixture
def manager(store):
    m = RunManager(store, python=sys.executable)
    yield m
    m.shutdown()


def test_plan_and_defaults(store, manager):
    _script(store, "train", TRAIN, params={"lr": 0.1})
    plan = manager.plan("train")
    assert plan["runnable"] and plan["style"] == "flags"
    assert plan["argv"][1:] == ["-u", "steps/train.py", "--lr", "0.1"]
    assert plan["defaults"]["source"] == "step params"
    store.add_step("nb", script_path="nb.ipynb")
    (store.root / "nb.ipynb").write_text("{}")
    assert not manager.plan("nb")["runnable"]
    store.add_step("ghost", script_path="missing.py")
    assert "does not exist" in manager.plan("ghost")["reason"]
    with pytest.raises(MLTreeError):
        manager.start("ghost")


def test_run_records_everything(store, manager):
    _script(store, "train", TRAIN, params={"lr": 0.1})
    live = manager.start("train", {"lr": 0.5, "write_file": True})
    assert manager.wait(live.run_id, 30)
    run = store.get_run(live.run_id)
    assert run["status"] == "succeeded" and run["exit_code"] == 0
    assert run["metrics"]["loss"] == pytest.approx(0.5 / 3, abs=1e-3)  # last value of the stdout metric
    assert run["metrics"]["val.acc"] == 0.75  # from $MLTREE_METRICS_FILE
    assert run["params"] == {"lr": 0.5, "write_file": True}
    assert run["script_sha256"] and run["duration_s"] is not None and run["finished_at"]
    assert run["command"][-2:] == ["--lr", "0.5"] and "mltree-run" in run["tags"]
    hist = manager.history(live.run_id)
    assert [h["epoch"] for h in hist] == [0, 1, 2]
    log = manager.log(live.run_id)
    assert any(ln["s"] == "err" and ln["line"] == "to stderr" for ln in log["lines"]) and log["done"]
    # the next form is prefilled from this run
    assert manager.defaults("train")["params"]["lr"] == 0.5
    # after the process (and the manager) forget it, the log comes back from disk
    manager._live.clear()
    disk = manager.log(live.run_id)
    assert not disk["live"] and any(ln["line"] == "to stderr" and ln["s"] == "err" for ln in disk["lines"])
    assert [h["epoch"] for h in manager.history(live.run_id)] == [0, 1, 2]


def test_failed_run(store, manager):
    _script(store, "train", TRAIN)
    live = manager.start("train", {"exit": 3})
    manager.wait(live.run_id, 30)
    run = store.get_run(live.run_id)
    assert run["status"] == "failed" and run["exit_code"] == 3 and "exit code 3" in run["notes"] and "stderr (last lines):\nto stderr" in run["notes"]
    assert store.get_pipeline()["nodes"][0]["status"] == "failed"


def test_cancel(store, manager):
    _script(store, "slow", "import time\nprint('start', flush=True)\ntime.sleep(60)\n")
    live = manager.start("slow", {})
    deadline = time.time() + 10
    while not manager.log(live.run_id)["lines"] and time.time() < deadline:
        time.sleep(0.05)
    assert [r["run_id"] for r in manager.active()] == [live.run_id]
    manager.cancel(live.run_id)
    assert manager.wait(live.run_id, 15)
    assert store.get_run(live.run_id)["status"] == "cancelled"
    assert manager.active() == []
    with pytest.raises(MLTreeError):
        manager.cancel(live.run_id)


def test_grid_sweep(store, manager):
    _script(store, "train", TRAIN, params={"lr": 0.1})
    res = manager.start_sweep("train", {"lr": [0.3, 0.1, 0.2]}, metric="loss", mode="min")
    assert res["total"] == 3
    assert manager.wait_sweep(res["sweep_id"], 60)
    sweep = store.get_run(res["sweep_id"])
    assert sweep["kind"] == "sweep" and sweep["status"] == "succeeded"
    kids = store.sweep_children(res["sweep_id"])
    assert len(kids) == 3 and all(k["sweep_id"] == sweep["id"] and k["parent_run_id"] == sweep["id"] for k in kids)
    summary = store.sweep_summary(sweep["id"])
    assert summary["counts"] == {"succeeded": 3} and summary["best_params"]["lr"] == 0.1
    assert sweep["sweep"]["best_run_id"] == summary["best_run_id"]
    assert sweep["metrics"]["best_loss"] == pytest.approx(0.1 / 3, abs=1e-3)
    tree = store.run_tree(metric="loss", mode="min")
    assert tree["sweeps"][sweep["id"]]["best_run_id"] == summary["best_run_id"]
    # sweeps don't prefill the run form
    assert manager.defaults("train")["source_run_id"] != sweep["id"]


def test_sweep_cancel(store, manager):
    _script(store, "slow", "import time\ntime.sleep(60)\n", params={"x": 1})
    res = manager.start_sweep("slow", {"x": [1, 2, 3]})
    deadline = time.time() + 10
    while not (manager.sweep_progress(res["sweep_id"]) or {}).get("current") and time.time() < deadline:
        time.sleep(0.05)
    manager.cancel(res["sweep_id"])
    assert manager.wait_sweep(res["sweep_id"], 20)
    assert store.get_run(res["sweep_id"])["status"] == "cancelled"
    assert len(store.sweep_children(res["sweep_id"])) == 1


def test_log_run_sweep_validation(store):
    store.add_step("train")
    sw = store.log_run("train", kind="sweep", sweep={"source": "optuna", "metric": "value", "mode": "min"})
    child = store.log_run("train", params={"a": 1}, metrics={"value": 2.0}, sweep_id=sw["id"])
    assert child["parent_run_id"] == sw["id"]
    via_parent = store.log_run("train", metrics={"value": 1.0}, parent_run_id=sw["id"])
    assert via_parent["sweep_id"] == sw["id"]
    assert store.sweep_summary(sw["id"])["best_run_id"] == via_parent["id"]
    plain = store.log_run("train")
    with pytest.raises(MLTreeError, match="sweep"):
        store.log_run("train", sweep_id=plain["id"])
    with pytest.raises(MLTreeError):
        store.log_run("train", kind="banana")
    upd = store.update_run(child["id"], status="failed", metrics={"extra": 1})
    assert upd["metrics"] == {"value": 2.0, "extra": 1} and upd["finished_at"]


def test_migrates_v1_db(tmp_path):
    s = Store(tmp_path)
    s.init("old")
    s.add_step("a")
    s.log_run("a", metrics={"m": 1})
    db = s.dir / "mltree.db"
    with sqlite3.connect(db) as c:  # simulate a 0.4 database: drop the v2 columns
        cols = [r[1] for r in c.execute("PRAGMA table_info(runs)")]
        keep = [x for x in cols if x not in ("kind", "sweep_id", "sweep", "command", "exit_code")]
        c.execute(f"CREATE TABLE r2 AS SELECT {', '.join(keep)} FROM runs")
        c.execute("DROP TABLE runs")
        c.execute("ALTER TABLE r2 RENAME TO runs")
        c.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    s2 = Store(tmp_path)
    runs = s2.list_runs()
    assert runs[0]["kind"] == "run" and runs[0]["metrics"] == {"m": 1}
    sw = s2.log_run("a", kind="sweep")
    assert s2.get_run(sw["id"])["kind"] == "sweep"
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "3"
    json.dumps(s2.run_tree())  # still serialisable
