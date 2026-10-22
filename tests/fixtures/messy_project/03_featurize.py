"""Morgan fingerprints for every molecule."""
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

DATA_DIR = "data"
RADIUS = 2
N_BITS = 2048


def fp(smi):
    mol = Chem.MolFromSmiles(smi)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS))


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "processed", "clean.csv"))
    X = np.stack([fp(s) for s in df.smiles])
    np.save(os.path.join(DATA_DIR, "features", "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "features", "y.npy"), df.pIC50.values)


if __name__ == "__main__":
    main()
