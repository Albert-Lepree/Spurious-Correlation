import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATASETS = os.path.join(_ROOT, "datasets")


def compute_spx_features(spx):
    spx = spx.sort_values("date").reset_index(drop=True)
    daily_ret = spx["close"].pct_change()
    spx["return_1d"] = spx["close"].shift(1) / spx["close"].shift(2) - 1
    spx["return_5d"] = spx["close"].shift(1) / spx["close"].shift(6) - 1
    spx["rolling_vol_20d"] = daily_ret.rolling(20).std().shift(1)
    spx["volume_zscore"] = (
        (spx["volume"] - spx["volume"].rolling(20).mean())
        / spx["volume"].rolling(20).std()
    ).shift(1)
    return spx[["date", "return_1d", "return_5d", "rolling_vol_20d", "volume_zscore"]]


def compute_vix_features(vix):
    vix = vix.sort_values("date").reset_index(drop=True)
    vix["vix_change"] = vix["vix_close"].diff().shift(1)
    return vix[["date", "vix_change"]]


if __name__ == "__main__":
    print("Loading master_base.parquet...")
    master = pd.read_parquet(os.path.join(DATASETS, "master_base.parquet"))
    print(f"  {len(master)} rows, {len(master.columns)} columns")

    print("Loading market data for feature computation...")
    spx = pd.read_csv(os.path.join(DATASETS, "spx_data.csv"), parse_dates=["date"])
    vix = pd.read_csv(os.path.join(DATASETS, "vix_data.csv"), parse_dates=["date"])

    print("Computing SPX control features (lagged t-1)...")
    spx_features = compute_spx_features(spx)

    print("Computing VIX features (lagged t-1)...")
    vix_features = compute_vix_features(vix)

    print("Joining features onto master...")
    result = master.merge(spx_features, on="date", how="left")
    result = result.merge(vix_features, on="date", how="left")

    feature_cols = ["return_1d", "return_5d", "rolling_vol_20d", "volume_zscore", "vix_change"]
    print("Forward-filling feature columns over weekends/holidays...")
    result = result.sort_values("date").reset_index(drop=True)
    result[feature_cols] = result[feature_cols].ffill()

    out_path = os.path.join(DATASETS, "master_with_control.parquet")
    result.to_parquet(out_path, index=False)

    print(f"\nSaved {len(result)} rows to {out_path}")
    print(f"Columns: {list(result.columns)}")
    print("\nNull counts for feature columns:")
    print(result[feature_cols].isnull().sum().to_string())
