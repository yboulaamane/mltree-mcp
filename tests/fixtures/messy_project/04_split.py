"""Random 80/20 split."""
import pickle

import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("data/features/X.npy")
y = np.load("data/features/y.npy")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
with open("data/splits/split.pkl", "wb") as f:
    pickle.dump((X_tr, X_te, y_tr, y_te), f)
