"""Step 4 - train a toy 'GCN': one round of mean neighbour aggregation, sum pooling,
then ridge regression on the pooled features (closed form, stdlib only).

Swap this file for a real PyTorch Geometric GCN later; the node in the map stays the same.
Usage: python train.py [alpha]
"""
import sys

from _common import embed, load, save

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
hidden = 64

try:  # the real PyG model (see gcn_model.py); the toy ridge regression below runs without torch
    from gcn_model import MolGCN

    model = MolGCN(hidden_channels=hidden, num_layers=3)
    print(f"MolGCN: {sum(p.numel() for p in model.parameters())} parameters (not trained by this toy step)")
except ImportError:
    model = None


def solve(A: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[c][c]:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] if M[i][i] else 0.0 for i in range(n)]


train = load("train.json")
X = [embed(g) for g in train]
y = [g["y"] for g in train]
d = len(X[0])
XtX = [[sum(r[i] * r[j] for r in X) + (alpha if i == j else 0.0) for j in range(d)] for i in range(d)]
Xty = [sum(r[i] * t for r, t in zip(X, y)) for i in range(d)]
w = solve(XtX, Xty)
save("model.json", {"weights": w, "alpha": alpha})
rmse = (sum((sum(a * b for a, b in zip(r, w)) - t) ** 2 for r, t in zip(X, y)) / len(y)) ** 0.5
print(f"train_rmse={rmse:.3f} alpha={alpha}")
print(f"MLTREE_METRIC train_rmse={rmse:.4f}")  # picked up by mltree when the step is run from the map
