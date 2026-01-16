from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_and_clean_csv(
    path: str | Path,
    *,
    feature_cols: list[str],
    allowed_patterns: set[str],
    junk_cols: set[str],
    required_cols: set[str],
    keep_no_pattern: bool = True,
) -> pd.DataFrame:
    """
    Loads a CSV and cleans:
    - drops junk cols
    - parses/sorts date
    - converts features to numeric
    - fills r NaNs with 0
    - drops rows with NaN OHLC
    - filters patterns to allowed (+ optional 'no-pattern')
    """

    path = Path(path)

    # Read header first to know what columns exist
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in cols if c not in junk_cols]

    usecols = [c for c in cols if c in required_cols]
    df = pd.read_csv(path, usecols=usecols)

    # Drop junk cols again just in case
    df = df.drop(columns=[c for c in df.columns if c in junk_cols], errors="ignore")

    # Check required cols exist
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    # Parse + sort date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    # Pattern column
    df["pattern"] = df["pattern"].fillna("no-pattern").astype(str)

    # Convert features to numeric
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # r NaNs are common -> fill with 0
    if "r" in df.columns:
        df["r"] = df["r"].fillna(0.0)

    # Drop rows with NaN OHLC
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Filter allowed patterns
    allowed = set(allowed_patterns)
    if keep_no_pattern:
        allowed.add("no-pattern")

    df = df[df["pattern"].isin(allowed)].copy()
    return df
