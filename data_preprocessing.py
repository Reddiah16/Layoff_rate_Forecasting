

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging, os

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATE_COL         = "Date of Layoff"
TARGET_COUNT     = "Layoff Count Numeric"   # cleaned numeric column
TARGET_CLASS     = "layoffs_occurred"        # binary 0/1 for RF
CATEGORICAL_COLS = ["Industry", "Funding Status", "City"]


def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def _parse_layoff_count(val) -> float:
    """
    Parse messy Layoff Count values:
      '3500'        → 3500
      '1100–1500'   → 1300  (midpoint)
      '300-400'     → 350   (midpoint)
      '100+'        → 100
      '80% workforce' → NaN (handle separately)
      'Unknown'     → NaN
    """
    val = str(val).strip()
    if val.lower() in ('unknown', 'nan', '', 'none'):
        return np.nan
    # workforce percentage — skip
    if 'workforce' in val.lower() or '%' in val:
        return np.nan
    # remove + sign
    val = val.replace('+', '')
    # handle range with – or -
    range_match = re.search(r'(\d[\d,]*)\s*[–\-]\s*(\d[\d,]*)', val)
    if range_match:
        lo = float(range_match.group(1).replace(',', ''))
        hi = float(range_match.group(2).replace(',', ''))
        return (lo + hi) / 2
    # plain number
    num_match = re.search(r'(\d[\d,]*)', val)
    if num_match:
        return float(num_match.group(1).replace(',', ''))
    return np.nan


def _parse_date(val) -> pd.Timestamp:
    """
    Parse dates like '2024', 'Jan 2024', 'Apr 2024', 'Mar 2025'.
    """
    val = str(val).strip()
    for fmt in ('%b %Y', '%B %Y', '%Y-%m', '%Y'):
        try:
            return pd.to_datetime(val, format=fmt)
        except Exception:
            pass
    return pd.NaT


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    # Drop unhelpful columns
    drop_cols = ["Source URL", "Reason"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Parse Layoff Count → numeric
    df[TARGET_COUNT] = df["Layoff Count"].apply(_parse_layoff_count)

    # Fill missing counts with median
    median_count = df[TARGET_COUNT].median()
    df[TARGET_COUNT] = df[TARGET_COUNT].fillna(median_count)
    logger.info(f"Layoff Count median: {median_count:.0f}")

    # Parse dates
    df[DATE_COL] = df[DATE_COL].apply(_parse_date)
    df = df.dropna(subset=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    logger.info(f"Date range: {df[DATE_COL].min()} → {df[DATE_COL].max()}")

    # Fill categorical NAs
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    logger.info(f"Clean data: {df.shape[0]} rows")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Binary target: large layoff = 1 (>= 500 employees)
    df[TARGET_CLASS] = (df[TARGET_COUNT] >= 500).astype(int)

    # Time features
    df["year"]    = df[DATE_COL].dt.year
    df["month"]   = df[DATE_COL].dt.month
    df["quarter"] = df[DATE_COL].dt.quarter

    # Log scale of layoff count (handles skew)
    df["log_layoff_count"] = np.log1p(df[TARGET_COUNT])

    # Is public company
    df["is_public"] = df["Funding Status"].str.lower().str.contains("public").astype(int)

    # Risk label for display
    df["risk_label"] = pd.cut(
        df[TARGET_COUNT],
        bins=[-np.inf, 200, 1000, np.inf],
        labels=["Low", "Medium", "High"]
    )

    logger.info(f"Class balance — 0:{(df[TARGET_CLASS]==0).sum()} | 1:{(df[TARGET_CLASS]==1).sum()}")
    logger.info("Feature engineering complete")
    return df


def encode_and_scale(df: pd.DataFrame):
    df_model = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            encoders[col] = le

    feature_cols = [c for c in [
        TARGET_COUNT, "log_layoff_count",
        "Industry", "Funding Status", "City",
        "year", "month", "quarter", "is_public"
    ] if c in df_model.columns]

    X = df_model[feature_cols]
    y = df_model[TARGET_CLASS]

    # For very small datasets, use a smaller test split
    test_size = 0.2 if len(df) >= 20 else 0.15

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42,
        stratify=y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler, feature_cols, encoders


def prepare_time_series(df: pd.DataFrame, freq: str = "ME") -> pd.Series:
    df2 = df.copy().set_index(DATE_COL)
    ts  = df2[TARGET_COUNT].resample(freq).sum().fillna(0)
    # Remove leading zeros
    ts  = ts[ts.cumsum() > 0]
    logger.info(f"Time series: {len(ts)} periods | freq={freq}")
    return ts


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/Layoffs_Dataset.csv"
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    print(df[["Company Name", "Layoff Count", TARGET_COUNT, TARGET_CLASS, "risk_label"]].to_string())
    print(f"\nClass balance:\n{df[TARGET_CLASS].value_counts()}")