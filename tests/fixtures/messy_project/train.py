"""Train an MLP regressor on the fingerprints."""
import argparse
import json
import pickle

import torch
from torch import nn


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--split", default="data/splits/split.pkl")
    p.add_argument("--out", default="models/mlp.pt")
    return p.parse_args()


def main():
    args = get_args()
    with open(args.split, "rb") as f:
        X_tr, X_te, y_tr, y_te = pickle.load(f)
    model = nn.Sequential(nn.Linear(2048, args.hidden), nn.ReLU(), nn.Linear(args.hidden, 1))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(args.epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(torch.tensor(X_tr).float()).squeeze(), torch.tensor(y_tr).float())
        loss.backward()
        opt.step()
    torch.save(model.state_dict(), args.out)
    with open("results/train_log.json", "w") as f:
        json.dump({"final_loss": float(loss)}, f)


if __name__ == "__main__":
    main()
