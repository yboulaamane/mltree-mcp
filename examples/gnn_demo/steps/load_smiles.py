"""Step 1 - load SMILES and target values from data/molecules.csv."""
import csv

from _common import DATA, save

rows = []
with open(DATA / "molecules.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        if r["smiles"].strip():
            rows.append({"smiles": r["smiles"].strip(), "y": float(r["y"])})
save("molecules.json", rows)
print(f"loaded {len(rows)} molecules")
