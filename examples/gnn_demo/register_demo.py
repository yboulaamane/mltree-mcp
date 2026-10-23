"""Register the demo GNN pipeline in mltree and log a few FAKE runs.

    python examples/gnn_demo/register_demo.py            # writes examples/gnn_demo/.mltree/
    python examples/gnn_demo/register_demo.py --reset    # start from scratch

Every run logged here is DEMO DATA: the metrics are made up (tagged "demo") so the
experiment tree has something to show. It does not execute the step scripts. Data versions (SHA-256 of
data/molecules.csv and data/processed/*.json) are recorded for whatever files exist when you run this.
This is what Codex or Claude Code would do through the MCP tools as it works.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mltree_mcp import Store

HERE = Path(__file__).resolve().parent

P = "data/processed/"
STEPS = [
    ("load_smiles", "steps/load_smiles.py", ["data/molecules.csv"], [P + "molecules.json"], {}, "Load SMILES + target from data/molecules.csv"),
    ("featurize", "steps/featurize.py", [P + "molecules.json"], [P + "graphs.json", P + "graphs.pt"], {"featurizer": "atom_one_hot"}, "SMILES -> molecular graphs"),
    ("split", "steps/split.py", [P + "graphs.json"], [P + "train.json", P + "test.json"], {"test_size": 0.2, "seed": 42}, "Random train/test split"),
    ("train", "steps/train.py", [P + "train.json"], [P + "model.json"], {"alpha": 1.0, "layers": 1}, "Toy GCN (mean aggregation + ridge)"),
    ("evaluate", "steps/evaluate.py", [P + "model.json", P + "test.json"], [P + "metrics.json"], {}, "Test RMSE / MAE / R2"),
]


def curve(final: float, epochs: int = 20) -> list[dict[str, float]]:
    """A made-up, smoothly decreasing validation curve ending at `final` (DEMO DATA)."""
    out = []
    for e in range(1, epochs + 1):
        val = final + (1.6 - final) * (0.82 ** e)
        out.append({"epoch": e, "train_loss": round(val * 0.8 - 0.004 * e, 4), "val_rmse": round(val, 4)})
    return out


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

    # the baseline and the sweep ran an earlier train.py (no MLTREE_METRIC line), so Compare shows a code diff
    train_py = root / "steps" / "train.py"
    current = train_py.read_text()
    earlier = current.replace(
        'print(f"MLTREE_METRIC train_rmse={rmse:.4f}")  # picked up by mltree when the step is run from the map\n', "")
    assert earlier != current, "steps/train.py changed; update the earlier-version edit"
    train_py.write_text(earlier)
    try:
        base, sweep = _baseline_and_sweep(store, demo, note)
    finally:
        train_py.write_text(current)
    deeper = store.log_run(
        "train", params={"alpha": 0.5, "layers": 2}, metrics={"val_rmse": 0.69},
        parent_run_id=base["id"], tags=demo + ["gcn-2layer"], notes=note, history=curve(0.69, 30),
    )
    store.log_run("train", params={"alpha": 0.5, "layers": 3}, status="failed", parent_run_id=deeper["id"],
                  tags=demo, notes=note + " (pretend out-of-memory)")
    store.log_run("evaluate", params={"model_run": deeper["id"]},
                  metrics={"test_rmse": 0.71, "test_mae": 0.52, "test_r2": 0.82}, parent_run_id=deeper["id"], tags=demo, notes=note)
    return store


def _baseline_and_sweep(store: Store, demo: list[str], note: str):
    base = store.log_run("train", params={"alpha": 1.0, "layers": 1}, metrics={"val_rmse": 0.92}, tags=demo + ["baseline"], notes=note,
                         history=curve(0.92))
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
            sweep_id=sweep["id"], tags=demo + ["hpo"], notes=note, history=curve(rmse),
        )
    return base, sweep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete the existing .mltree/ first")
    args = ap.parse_args()
    s = build(reset=args.reset)
    best = s.best_run("val_rmse", mode="min", step="train")
    print(f"Demo project written to {s.dir}")
    print(f"{len(s.list_runs(limit=1000))} runs logged (DEMO DATA). Best train run by val_rmse: "
          f"{best['run']['id']} = {best['value']} with {best['run']['params']}")
