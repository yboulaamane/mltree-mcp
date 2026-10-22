"""Remove salts, duplicates and missing activities."""
from pathlib import Path

import pandas as pd

from utils import log, seed_everything

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

if __name__ == "__main__":
    seed_everything(0)
    df = pd.read_parquet(DATA / "interim" / "egfr.parquet")
    df = df.dropna().drop_duplicates("smiles")
    log(f"{len(df)} molecules")
    df.to_csv(DATA / "processed" / "clean.csv", index=False)
