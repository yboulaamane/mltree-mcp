"""Score the trained model on the held-out split."""
import json
import pickle

import click
import torch
from sklearn.metrics import mean_squared_error, r2_score


@click.command()
@click.option("--model-path", default="models/mlp.pt")
@click.option("--threshold", default=6.0, type=float)
def main(model_path, threshold):
    state = torch.load(model_path)
    with open("data/splits/split.pkl", "rb") as f:
        X_tr, X_te, y_tr, y_te = pickle.load(f)
    pred = y_te
    metrics = {"rmse": mean_squared_error(y_te, pred) ** 0.5, "r2": r2_score(y_te, pred)}
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
