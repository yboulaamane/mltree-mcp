"""The GNN demo: scripts run with the stdlib only, and register_demo builds a sane tree."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "examples" / "gnn_demo"


def test_demo_scripts_run_without_torch(tmp_path):
    work = tmp_path / "gnn_demo"
    shutil.copytree(DEMO, work, ignore=shutil.ignore_patterns(".mltree", "processed", "__pycache__"))
    for cmd in (["load_smiles.py"], ["featurize.py"], ["split.py", "0.25", "7"], ["train.py", "0.5"], ["evaluate.py"]):
        out = subprocess.run([sys.executable, str(work / "steps" / cmd[0]), *cmd[1:]], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
    assert (work / "data" / "processed" / "metrics.json").exists()


def test_register_demo(tmp_path):
    work = tmp_path / "gnn_demo"
    shutil.copytree(DEMO, work, ignore=shutil.ignore_patterns(".mltree", "processed", "__pycache__"))
    spec = importlib.util.spec_from_file_location("register_demo", work / "register_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    store = mod.build(root=work, reset=True)
    p = store.get_pipeline()
    assert p["order"] == ["load_smiles", "featurize", "split", "train", "evaluate"]
    assert all(n["script_exists"] for n in p["nodes"])
    runs = store.list_runs(limit=1000)
    assert len(runs) == 10 and all("demo" in r["tags"] for r in runs)
    best = store.best_run("val_rmse", mode="min", step="train")
    assert best["value"] == 0.69 and best["run"]["params"]["layers"] == 2
