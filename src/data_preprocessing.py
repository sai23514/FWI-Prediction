"""Step 1 - load raw CWFIS data, clean it and split into train / dev / test.

Mirrors Data_Merge.ipynb + data_cleaning.ipynb, but runs as a script:
    python -m src.data_preprocessing [--sample N]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (CANADIAN_PROVINCES, COLUMN_MAPPING, PROCESSED_DIR, RANDOM_STATE,
                        RAW_FWI_FILE, RAW_STATIONS_FILE, VALID_RANGES)


def load_raw(path=RAW_FWI_FILE) -> pd.DataFrame:
    usecols = ["wmo"] + list(COLUMN_MAPPING)
    dtypes = {c: "float32" for c in usecols if c not in ("wmo", "rep_date")}
    df = pd.read_csv(path, usecols=usecols, dtype=dtypes)
    return df


def add_country(df: pd.DataFrame, stations_path=RAW_STATIONS_FILE) -> pd.DataFrame:
    """Map each station (WMO id) to a country using the CWFIS station list."""
    st = pd.read_csv(stations_path, usecols=["wmo", "prov"]).dropna(subset=["wmo"])
    st = st.drop_duplicates("wmo")
    st["prov"] = st["prov"].str.strip()
    st["country"] = np.where(st["prov"].isin(CANADIAN_PROVINCES), "Canada",
                             np.where(st["prov"].str.fullmatch(r"[A-Z]{2}") & (st["prov"] != "XX"),
                                      "United States", None))
    df = df.merge(st[["wmo", "country"]], on="wmo", how="left")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAPPING)
    n0 = len(df)
    df = df.dropna()  # FWI codes are blank outside the fire season / when not calculated
    n1 = len(df)
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in VALID_RANGES.items():
        mask &= df[col].between(lo, hi)
    df = df[mask].drop_duplicates()
    df["report_date"] = pd.to_datetime(df["report_date"])
    print(f"rows: raw={n0:,}  complete={n1:,}  after range checks/dedup={len(df):,}")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame):
    """60 / 24 / 16 split stratified by country (same ratios as data_cleaning.ipynb)."""
    train, temp = train_test_split(df, test_size=0.4, stratify=df["country"],
                                   random_state=RANDOM_STATE)
    dev, test = train_test_split(temp, test_size=0.4, stratify=temp["country"],
                                 random_state=RANDOM_STATE)
    return train, dev, test


def main(sample: int | None = None):
    df = add_country(load_raw())
    df = clean(df)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE)
    train, dev, test = split(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("dev", dev), ("test", test)]:
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        print(f"{name}: {len(part):,} rows")
    print(df["country"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None, help="optional row sample for quick runs")
    main(p.parse_args().sample)
