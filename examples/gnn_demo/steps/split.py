"""Step 3 - random train/test split.  Usage: python split.py [test_size] [seed]"""
import random
import sys

from _common import load, save

test_size = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

graphs = load("graphs.json")
idx = list(range(len(graphs)))
random.Random(seed).shuffle(idx)
n_test = max(1, round(len(idx) * test_size))
save("test.json", [graphs[i] for i in idx[:n_test]])
save("train.json", [graphs[i] for i in idx[n_test:]])
print(f"train={len(idx) - n_test} test={n_test} seed={seed}")
