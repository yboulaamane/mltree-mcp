"""Download the ChEMBL export and keep the columns we need."""
import pandas as pd

RAW = "data/raw/chembl_egfr.csv"

df = pd.read_csv(RAW, sep=";")
df = df[["smiles", "pIC50"]]
df.to_parquet("data/interim/egfr.parquet")
