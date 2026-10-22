"""scan_project on a realistic messy project (tests/fixtures/messy_project)."""
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from mltree_mcp import Store, server
from mltree_mcp.scan import scan_project

from test_tools import call_ok

FIXTURE = Path(__file__).parent / "fixtures" / "messy_project"


def _make_optuna_db(path: Path) -> None:
    """A minimal Optuna RDB-schema study: 3 trials (complete, complete, pruned)."""
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE studies (study_id INTEGER PRIMARY KEY, study_name TEXT);
        CREATE TABLE study_directions (study_direction_id INTEGER PRIMARY KEY, direction TEXT, study_id INT, objective INT);
        CREATE TABLE trials (trial_id INTEGER PRIMARY KEY, number INT, study_id INT, state TEXT,
                             datetime_start TEXT, datetime_complete TEXT);
        CREATE TABLE trial_params (param_id INTEGER PRIMARY KEY, trial_id INT, param_name TEXT, param_value REAL,
                                   distribution_json TEXT);
        CREATE TABLE trial_values (trial_value_id INTEGER PRIMARY KEY, trial_id INT, objective INT, value REAL);
        INSERT INTO studies VALUES (1, 'mlp');
        INSERT INTO study_directions VALUES (1, 'MINIMIZE', 1, 0);
        """
    )
    cat = json.dumps({"name": "CategoricalDistribution", "attributes": {"choices": [128, 256, 512]}})
    flt = json.dumps({"name": "FloatDistribution", "attributes": {"low": 1e-5, "high": 1e-2, "log": True}})
    for tid, (state, lr, h, val) in enumerate(
        [("COMPLETE", 1e-3, 2, 0.51), ("COMPLETE", 3e-4, 0, 0.04), ("PRUNED", 1e-2, 1, None)], start=1
    ):
        con.execute("INSERT INTO trials VALUES (?,?,1,?,?,?)",
                    (tid, tid - 1, state, "2026-10-01 10:00:00.1", "2026-10-01 10:01:00.2"))
        con.execute("INSERT INTO trial_params(trial_id, param_name, param_value, distribution_json) VALUES (?,?,?,?)",
                    (tid, "lr", lr, flt))
        con.execute("INSERT INTO trial_params(trial_id, param_name, param_value, distribution_json) VALUES (?,?,?,?)",
                    (tid, "hidden", h, cat))
        if val is not None:
            con.execute("INSERT INTO trial_values(trial_id, objective, value) VALUES (?,0,?)", (tid, val))
    con.commit()
    con.close()


@pytest.fixture
def messy(tmp_path, monkeypatch):
    root = tmp_path / "messy"
    shutil.copytree(FIXTURE, root)
    _make_optuna_db(root / "optuna_study.db")
    # noise that must be excluded
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "site.py").write_text("import pandas as pd\npd.read_csv('x.csv')\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_train.py").write_text("def test_x():\n    assert True\n")
    (root / "broken.py").write_text("def oops(:\n")
    monkeypatch.setenv("MLTREE_PROJECT", str(root))
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    return root


def _edges(summary):
    return {(e["source"], e["target"]) for e in summary["edges"]}


def _steps(summary):
    return {s["name"]: s for s in summary["steps"]}


def test_dry_run_proposes_map_without_writing(messy):
    s = scan_project(Store(messy), dry_run=True)
    assert s["dry_run"] and not (messy / ".mltree").exists()
    steps = _steps(s)
    assert set(steps) == {"load_data", "clean", "featurize", "split", "train", "hpo_optuna", "evaluate", "exploration"}
    kinds = {n: st["kind"] for n, st in steps.items()}
    assert kinds == {"load_data": "data", "clean": "clean", "featurize": "featurize", "split": "split",
                     "train": "train", "hpo_optuna": "hpo", "evaluate": "eval", "exploration": "data"}
    # data-flow edges from matching output -> input paths
    assert {("load_data", "clean"), ("clean", "featurize"), ("featurize", "split"), ("split", "train"),
            ("split", "hpo_optuna"), ("split", "evaluate"), ("train", "evaluate"), ("clean", "exploration")} <= _edges(s)
    assert all(e["evidence"] == "data" for e in s["edges"])
    assert [u["path"] for u in s["utilities"]] == ["utils.py"]
    assert "dry_run=false" in s["hint"] and "connect_steps" in s["hint"]
    assert all(r["action"] == "would import" for r in s["imported"])


def test_details_params_paths_docstrings(messy):
    st = _steps(scan_project(Store(messy), dry_run=True))
    assert st["train"]["params"] == {"lr": 0.001, "epochs": 50, "hidden": 512,
                                     "split": "data/splits/split.pkl", "out": "models/mlp.pt"}
    assert st["train"]["outputs"] == ["models/mlp.pt", "results/train_log.json"]
    assert st["evaluate"]["params"] == {"model_path": "models/mlp.pt", "threshold": 6.0}
    assert "models/mlp.pt" in st["evaluate"]["inputs"]
    assert st["split"]["params"] == {"test_size": 0.2, "random_state": 42}
    assert st["featurize"]["outputs"] == ["data/features/X.npy", "data/features/y.npy"]
    assert st["featurize"]["params"] == {"radius": 2, "n_bits": 2048}
    assert st["clean"]["outputs"] == ["data/processed/clean.csv"]  # Path(__file__).parent / "data" / ...
    assert st["clean"]["uses"] == ["utils.py"] and "(uses utils.py)" in st["clean"]["description"]
    assert st["hpo_optuna"]["params"] == {"n_trials": 20}
    assert "optuna_study.db" in st["hpo_optuna"]["inputs"]
    assert st["exploration"]["inputs"] == ["data/processed/clean.csv"]  # ../ resolved from notebooks/
    assert st["exploration"]["outputs"] == ["results/figures/pic50_hist.png"]
    assert st["load_data"]["description"].startswith("Download the ChEMBL export")


def test_skips_excluded_and_broken(messy):
    s = scan_project(Store(messy), dry_run=True)
    reasons = {x["path"]: x["reason"] for x in s["skipped"]}
    assert reasons["tests/test_train.py"] == "excluded"
    assert reasons["broken.py"].startswith("syntax error")
    assert not any(p.startswith(".venv") for p in reasons)  # pruned, never even listed
    unresolved = {(u["step"], u["path"]): u["exists_on_disk"] for u in s["unresolved_inputs"]}
    assert unresolved[("load_data", "data/raw/chembl_egfr.csv")] is True


def test_write_imports_results(messy):
    store = Store(messy)
    s = scan_project(store)
    assert s["counts"]["steps_new"] == 8 and s["counts"]["edges_new"] == len(s["edges"])
    pipe = store.get_pipeline()
    assert {n["name"] for n in pipe["nodes"]} == set(_steps(s))
    assert {(e["source"], e["target"]) for e in pipe["edges"]} == _edges(s)
    by = {(r["source"], r["origin"]): r for r in s["imported"]}
    assert by[("metrics_json", "results/metrics.json")]["step"] == "evaluate"
    assert by[("mlflow", "mlruns/0/abc123")]["step"] == "train"
    assert by[("lightning", "lightning_logs/version_0")]["step"] == "train"
    assert by[("optuna", "optuna_study.db#mlp")]["step"] == "hpo_optuna"
    assert s["runs_imported"] == 3 + 4  # metrics, mlflow, lightning + study and 3 trials

    ev = store.list_runs(step="evaluate")
    assert ev[0]["metrics"] == {"rmse": 0.71, "r2": 0.64, "per_fold.fold0": 0.70, "per_fold.fold1": 0.72}
    assert "imported" in ev[0]["tags"]
    ml = [r for r in store.list_runs(step="train") if "mlflow" in r["tags"]][0]
    assert ml["metrics"] == {"val_rmse": 0.75} and ml["params"] == {"epochs": 50, "lr": 0.001}

    hpo = store.list_runs(step="hpo_optuna")
    study = [r for r in hpo if "study" in r["tags"]][0]
    trials = [r for r in hpo if r["parent_run_id"] == study["id"]]
    assert len(trials) == 3
    assert {r["status"] for r in trials} == {"succeeded", "cancelled"}
    assert sorted(r["params"]["hidden"] for r in trials) == [128, 256, 512]  # categorical decoded
    best = store.best_run("value", mode="min", step="hpo_optuna")
    assert best["value"] == 0.04 and best["run"]["params"]["hidden"] == 128


def test_rescan_is_idempotent_and_keeps_user_edits(messy):
    store = Store(messy)
    scan_project(store)
    n_runs = len(store.list_runs(limit=1000))
    # the assistant fixes a step by hand ...
    store.add_step("train", script_path="train.py", inputs=["data/splits/split.pkl"], outputs=["models/mlp.pt"],
                   params={"lr": 0.01}, description="My MLP trainer", kind="train")
    # ... and the user adds a new script
    (messy / "05_report.py").write_text('"""Make the report."""\nimport json\n'
                                        'm = json.load(open("results/metrics.json"))\n'
                                        'open("results/report.md", "w").write(str(m))\n')
    s = scan_project(store)
    st = _steps(s)
    assert st["report"]["action"] == "new"
    assert ("evaluate", "report") in _edges(s)
    assert st["train"]["action"] == "merged" and st["train"]["kept_user_edits"]
    train = store.get_step("train")
    assert train["description"] == "My MLP trainer" and train["params"]["lr"] == 0.01
    assert train["params"]["epochs"] == 50  # new keys from the scan are added
    assert "results/train_log.json" in train["outputs"]
    assert st["clean"]["action"] == "unchanged"
    assert s["runs_imported"] == 0 and all(r["action"] == "already imported" for r in s["imported"])
    assert len(store.list_runs(limit=1000)) == n_runs
    # a third scan changes nothing
    s3 = scan_project(store)
    assert s3["counts"]["edges_new"] == 0 and s3["counts"]["steps_new"] == 0
    assert {x["action"] for x in s3["steps"]} == {"unchanged"}


def test_scanner_owned_steps_are_updated(messy):
    store = Store(messy)
    scan_project(store)
    p = messy / "04_split.py"
    p.write_text(p.read_text().replace("test_size=0.2", "test_size=0.1"))
    s = scan_project(store)
    assert _steps(s)["split"]["action"] == "updated"
    assert store.get_step("split")["params"]["test_size"] == 0.1


def test_notebook_sections_and_numbering_fallback(tmp_path):
    (tmp_path / "step1_prepare.py").write_text("import pandas as pd\npd.read_csv('raw.csv').to_csv('prep.csv')\n")
    (tmp_path / "step2_fit.py").write_text("from sklearn.linear_model import Ridge\nRidge().fit([[1]], [1])\n")
    nb = {"cells": [{"cell_type": "markdown", "source": ["# Load"]},
                    {"cell_type": "code", "source": ["import pandas as pd\ndf = pd.read_csv('a.csv')"]},
                    {"cell_type": "markdown", "source": ["## Train"]},
                    {"cell_type": "code", "source": ["model.fit(df, df)"]}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (tmp_path / "nb.ipynb").write_text(json.dumps(nb))
    s = scan_project(Store(tmp_path), dry_run=True, notebook_sections=True)
    st = _steps(s)
    assert {"prepare", "fit", "nb.load", "nb.train"} <= set(st)
    assert st["fit"]["kind"] == "train" and st["nb.train"]["kind"] == "train"
    ev = {(e["source"], e["target"]): e["evidence"] for e in s["edges"]}
    assert ev[("prepare", "fit")] == "file order"
    assert ev[("nb.load", "nb.train")] == "notebook order"


def test_never_executes_code(tmp_path):
    marker = tmp_path / "executed.txt"
    (tmp_path / "evil.py").write_text(f"open({str(marker)!r}, 'w').write('x')\nimport os\nos.system('false')\n")
    scan_project(Store(tmp_path))
    assert not marker.exists()


def test_path_outside_root_rejected(messy):
    with pytest.raises(Exception, match="outside the project root"):
        scan_project(Store(messy), path="..")


async def test_mcp_tool(messy):
    data = await call_ok("scan_project", {"dry_run": True})
    assert data["dry_run"] is True and data["counts"]["steps"] == 8
    data = await call_ok("scan_project", {})
    assert data["counts"]["runs_imported"] == 7
    pipe = await call_ok("get_pipeline")
    assert len(pipe["nodes"]) == 8
    assert "scan_project" in server.INSTRUCTIONS


def test_cli(messy, capsys):
    server.main(["scan", "--project", str(messy), "--dry-run"])
    out = capsys.readouterr().out
    assert "dry run, nothing written" in out and "load_data -> clean" in out
    assert not (messy / ".mltree").exists()
    server.main(["scan", "--project", str(messy), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["counts"]["steps_new"] == 8 and (messy / ".mltree" / "pipeline.yaml").exists()


def test_gnn_demo_helper_io(tmp_path):
    """The bundled example does its I/O through helpers in steps/_common.py: load('x.json') / save('y.json', ...)."""
    root = tmp_path / "gnn"
    shutil.copytree(Path(__file__).parents[1] / "examples" / "gnn_demo", root,
                    ignore=shutil.ignore_patterns(".mltree", "processed"))
    s = scan_project(Store(root), dry_run=True)
    assert {("load_smiles", "featurize"), ("featurize", "split"), ("split", "train"), ("split", "evaluate"),
            ("train", "evaluate")} <= _edges(s)
    assert _steps(s)["split"]["params"] == {"test_size": 0.2, "seed": 42}  # sys.argv fallbacks
    assert [u["path"] for u in s["utilities"]] == ["steps/_common.py"]
