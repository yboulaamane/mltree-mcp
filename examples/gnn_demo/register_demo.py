"""Register the demo GNN pipeline in mltree and log a few FAKE runs.

    python examples/gnn_demo/register_demo.py            # writes examples/gnn_demo/.mltree/
    python examples/gnn_demo/register_demo.py --reset    # start from scratch

Every run logged here is DEMO DATA: the metrics are made up (tagged "demo") so the
experiment tree has something to show. It does not execute the step scripts.
This is what Codex or Claude Code would do through the MCP tools as it works.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mltree_mcp import Store

HERE = Path(__file__).resolve().parent

STEPS = [
    ("load_smiles", "steps/load_smiles.py", [], ["molecules.json"], {}, "Load SMILES + target from data/molecules.csv"),
    ("featurize", "steps/featurize.py", ["molecules.json"], ["graphs.json"], {"featurizer": "atom_one_hot"}, "SMILES -> molecular graphs"),
    ("split", "steps/split.py", ["graphs.json"], ["train.json", "test.json"], {"test_size": 0.2, "seed": 42}, "Random train/test split"),
    ("train", "steps/train.py", ["train.json"], ["model.json"], {"alpha": 1.0, "layers": 1}, "Toy GCN (mean aggregation + ridge)"),
    ("evaluate", "steps/evaluate.py", ["model.json", "test.json"], ["metrics.json"], {}, "Test RMSE / MAE / R2"),
]


def build(root: Path = HERE, reset: bool = False) -> Store:
    if reset and (root / ".mltree").exists():
        shutil.rmtree(root / ".mltree")
    store = Store(root)
    store.init("gnn_demo")
    for name, script, ins, outs, params, desc in STEPS:
        store.add_step(name, script_path=script, inputs=ins, outputs=outs, params=params, description=desc)
    for a, b in zip([s[0] for s in STEPS], [s[0] for s in STEPS][1:]):
        store.connect_steps(a, b)
    store.connect_steps("split", "evaluate", label="test.json")

    demo = ["demo"]
    note = "DEMO DATA - fake metrics for illustration"
    store.log_run("load_smiles", metrics={"n_molecules": 20}, tags=demo, notes=note)
    store.log_run("featurize", params={"featurizer": "atom_one_hot"}, metrics={"n_graphs": 20}, tags=demo, notes=note)
    store.log_run("split", params={"test_size": 0.2, "seed": 42}, metrics={"n_train": 16, "n_test": 4}, tags=demo, notes=note)

    base = store.log_run("train", params={"alpha": 1.0, "layers": 1}, metrics={"val_rmse": 0.92}, tags=demo + ["baseline"], notes=note)
    # an HPO grid sweep branching from the baseline: one sweep row, one child run per alpha
    grid = [(0.1, 0.81), (0.5, 0.74), (3.0, 0.97)]
    sweep = store.log_run(
        "train", kind="sweep", params={"layers": 1}, parent_run_id=base["id"], tags=demo + ["hpo"], notes=note,
        sweep={"source": "grid", "name": "alpha sweep", "metric": "val_rmse", "mode": "min",
               "grid": {"alpha": [a for a, _ in grid]}, "total": len(grid)},
        metrics={"best_val_rmse": min(r for _, r in grid)},
    )
    for alpha, rmse in grid:
        store.log_run(
            "train", params={"alpha": alpha, "layers": 1}, metrics={"val_rmse": rmse},
            sweep_id=sweep["id"], tags=demo + ["hpo"], notes=note,
        )
    deeper = store.log_run(
        "train", params={"alpha": 0.5, "layers": 2}, metrics={"val_rmse": 0.69},
        parent_run_id=base["id"], tags=demo + ["gcn-2layer"], notes=note,
    )
    store.log_run("train", params={"alpha": 0.5, "layers": 3}, status="failed", parent_run_id=deeper["id"],
                  tags=demo, notes=note + " (pretend out-of-memory)")
    store.log_run("evaluate", params={"model_run": deeper["id"]},
                  metrics={"test_rmse": 0.71, "test_mae": 0.52, "test_r2": 0.82}, parent_run_id=deeper["id"], tags=demo, notes=note)
    return store


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete the existing .mltree/ first")
    args = ap.parse_args()
    s = build(reset=args.reset)
    best = s.best_run("val_rmse", mode="min", step="train")
    print(f"Demo project written to {s.dir}")
    print(f"{len(s.list_runs(limit=1000))} runs logged (DEMO DATA). Best train run by val_rmse: "
          f"{best['run']['id']} = {best['value']} with {best['run']['params']}")
