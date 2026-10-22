"""Tune the MLP with Optuna."""
import pickle

import optuna
import torch

STORAGE = "sqlite:///optuna_study.db"


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    hidden = trial.suggest_categorical("hidden", [128, 256, 512])
    with open("data/splits/split.pkl", "rb") as f:
        pickle.load(f)
    return lr * hidden


if __name__ == "__main__":
    study = optuna.create_study(study_name="mlp", storage=STORAGE, direction="minimize", load_if_exists=True)
    study.optimize(objective, n_trials=20)
