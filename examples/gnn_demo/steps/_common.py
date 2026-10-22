"""Tiny helpers shared by the demo steps (stdlib only, no torch / rdkit needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parents[1]
DATA = DEMO_DIR / "data"
OUT = DATA / "processed"


def save(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1))
    return p


def load(name: str):
    p = OUT / name
    if not p.exists():
        sys.exit(f"{p} not found - run the upstream step first")
    return json.loads(p.read_text())


def embed(g: dict) -> list[float]:
    """One round of mean neighbour aggregation + sum pooling (a toy GCN layer)."""
    x, n = g["x"], len(g["x"])
    nbrs = [[i] for i in range(n)]
    for a, b, _ in g["edges"]:
        nbrs[a].append(b)
        nbrs[b].append(a)
    h = [[sum(x[j][k] for j in nb) / len(nb) for k in range(len(x[0]))] for nb in nbrs]
    pooled = [sum(col) for col in zip(*h)]
    return pooled + [float(n), float(len(g["edges"])), 1.0]
