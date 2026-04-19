import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATASETS = os.path.join(_ROOT, "datasets")
RESULTS = os.path.join(_ROOT, "results")

CONTROL_COLS = ["return_1d", "return_5d", "rolling_vol_20d", "volume_zscore", "vix_change"]
# Mixed-case names as they appear in the parquet (preserved from spx_data.csv)
SPURIOUS_BASE_COLS = ["macd", "RSI_14", "BB_Width", "EMA_50"]
GENERATED_SPURIOUS_COLS = ["noise_gaussian", "noise_uniform", "future_return_t2", "shuffled_return"]
SPURIOUS_COLS = SPURIOUS_BASE_COLS + GENERATED_SPURIOUS_COLS
EMBED_COLS = [f"embed_{i}" for i in range(768)]
LDA_COLS = [f"lda_topic_{i}" for i in range(10)]


def batch_corr_stability(
    df: pd.DataFrame,
    feature_cols: list,
    target: str = "next_day_return",
    window: int = 30,
) -> dict:
    """
    Return {col: corr_stability} for each feature column.
    Aggregates to daily means first to avoid constant-target windows
    that arise when multiple articles share the same trading date.
    """
    cols_needed = [c for c in feature_cols if c in df.columns] + [target]
    daily = df.groupby("date")[cols_needed].mean().sort_index()
    result = {}
    for col in feature_cols:
        if col not in daily.columns:
            result[col] = np.nan
            continue
        rc = daily[col].rolling(window).corr(daily[target])
        result[col] = float(rc.std())
    return result


def generate_spurious_features(df: pd.DataFrame) -> pd.DataFrame:
    if "noise_gaussian" not in df.columns:
        df["noise_gaussian"] = np.random.randn(len(df))
    if "noise_uniform" not in df.columns:
        df["noise_uniform"] = np.random.uniform(-1, 1, len(df))
    if "future_return_t2" not in df.columns:
        df["future_return_t2"] = df["next_day_return"].shift(-2)
    if "shuffled_return" not in df.columns:
        df["shuffled_return"] = df["next_day_return"].sample(frac=1, random_state=42).values
    return df


def load_ctrl_df() -> pd.DataFrame:
    """Load master_with_control.parquet — has market features, control cols, and sentiment."""
    df = pd.read_parquet(os.path.join(DATASETS, "master_with_control.parquet"))
    return df.sort_values("date").reset_index(drop=True)


def load_embed_df_with_returns() -> pd.DataFrame:
    """
    Load master_with_embeddings.parquet (has LDA + embed cols) and join next_day_return
    from SPX data by date.  The embed parquet was generated from raw news before the
    market join, so it lacks next_day_return; we reattach it here via SPX close prices.
    """
    df = pd.read_parquet(os.path.join(DATASETS, "master_with_embeddings.parquet"))
    df["date"] = pd.to_datetime(
        df.apply(
            lambda r: r["Date"] if r["source_type"] == "kaggle_ai" else r["created_at"],
            axis=1,
        ),
        format="mixed",
        utc=True,
    ).dt.tz_convert(None).dt.normalize()

    spx = pd.read_csv(os.path.join(DATASETS, "spx_data.csv"), parse_dates=["date"])
    spx = spx.sort_values("date").reset_index(drop=True)
    spx["next_day_return"] = spx["close"].shift(-1) / spx["close"] - 1

    df = df.merge(spx[["date", "next_day_return"]], on="date", how="left")
    return df.sort_values("date").reset_index(drop=True)


def build_feature_rows(name: str, group: str, is_spurious, stab: dict) -> dict:
    return {
        "feature": name,
        "feature_group": group,
        "is_spurious": is_spurious,
        "effective_rank": 1.0,
        "corr_stability": stab.get(name, np.nan),
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    np.random.seed(42)

    print("Loading master_with_control.parquet (control/spurious/sentiment features)...")
    ctrl_df = load_ctrl_df()
    print(f"  {len(ctrl_df)} rows, {len(ctrl_df.columns)} columns")

    print("Generating spurious features...")
    ctrl_df = generate_spurious_features(ctrl_df)
    ctrl_valid = ctrl_df.dropna(subset=["next_day_return"]).copy()
    print(f"  {len(ctrl_valid)} rows with next_day_return")

    print("Loading master_with_embeddings.parquet (LDA/embeddings) + SPX returns join...")
    embed_df = load_embed_df_with_returns()
    embed_valid = embed_df.dropna(subset=["next_day_return"]).copy()
    print(f"  {len(embed_df)} total embed rows, {len(embed_valid)} with next_day_return")

    print("Computing per-feature corr_stability...")

    ctrl_stab = batch_corr_stability(ctrl_valid, CONTROL_COLS)
    print(f"  control done ({len(CONTROL_COLS)} features)")

    spur_valid = ctrl_valid.dropna(subset=SPURIOUS_COLS)
    spur_stab = batch_corr_stability(spur_valid, SPURIOUS_COLS)
    print(f"  spurious done ({len(SPURIOUS_COLS)} features)")

    sent_stab = batch_corr_stability(ctrl_valid, ["sentiment_score"])
    print("  sentiment done")

    lda_valid = embed_valid.dropna(subset=LDA_COLS[:1])
    lda_stab = batch_corr_stability(lda_valid, LDA_COLS)
    print(f"  lda_topics done ({len(LDA_COLS)} features)")

    print(f"  Computing embeddings corr_stability ({len(EMBED_COLS)} features)...")
    embed_notnull = embed_valid.dropna(subset=EMBED_COLS[:1])
    embed_stab = batch_corr_stability(embed_notnull, EMBED_COLS)
    print("  embeddings done")

    # --- Build one row per individual feature ---
    rows = []
    for col in CONTROL_COLS:
        rows.append(build_feature_rows(col, "control", 0.0, ctrl_stab))
    for col in SPURIOUS_COLS:
        rows.append(build_feature_rows(col, "spurious", 1.0, spur_stab))
    rows.append(build_feature_rows("sentiment_score", "sentiment", np.nan, sent_stab))
    for col in LDA_COLS:
        rows.append(build_feature_rows(col, "lda_topics", np.nan, lda_stab))
    for col in EMBED_COLS:
        rows.append(build_feature_rows(col, "embeddings", np.nan, embed_stab))

    meta_df = pd.DataFrame(rows)

    descriptors_path = os.path.join(RESULTS, "metafeature_descriptors.csv")
    meta_df.to_csv(descriptors_path, index=False)
    print(f"\nSaved {descriptors_path}  ({len(meta_df)} rows)")

    # --- Logistic regression on the 13 labeled rows ---
    # effective_rank is 1 for all scalars, so only corr_stability is used as input
    print("\nTraining logistic regression on 13 labeled features...")
    train = meta_df[meta_df["is_spurious"].notna()].copy()
    print(f"  Training rows: {len(train)}  "
          f"(control={int((train.is_spurious==0).sum())}, spurious={int((train.is_spurious==1).sum())})")

    X = train[["corr_stability"]].fillna(0).values
    y = train["is_spurious"].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classes, counts = np.unique(y, return_counts=True)
    cv_mean, cv_std = np.nan, np.nan
    if len(classes) >= 2 and counts.min() >= 2:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for ti, vi in cv.split(X_scaled, y):
            clf = LogisticRegression(C=1.0, l1_ratio=1.0, solver="saga", max_iter=1000)
            clf.fit(X_scaled[ti], y[ti])
            scores.append(clf.score(X_scaled[vi], y[vi]))
        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))
        print(f"  CV accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
    else:
        print("  Skipping CV: insufficient samples per class")

    clf_final = LogisticRegression(C=1.0, l1_ratio=1.0, solver="saga", max_iter=1000)
    clf_final.fit(X_scaled, y)

    coef_rows = [
        {"feature": "corr_stability", "coefficient": float(clf_final.coef_[0][0])},
        {"feature": "intercept",      "coefficient": float(clf_final.intercept_[0])},
        {"feature": "cv_accuracy",    "coefficient": cv_mean},
        {"feature": "cv_std",         "coefficient": cv_std},
    ]
    coef_df = pd.DataFrame(coef_rows)
    coef_path = os.path.join(RESULTS, "metafeature_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"Saved {coef_path}")

    # --- Apply to text features ---
    text_rows = meta_df[meta_df["is_spurious"].isna()].copy()
    X_text = scaler.transform(text_rows[["corr_stability"]].fillna(0).values)
    text_rows["predicted_spurious_prob"] = clf_final.predict_proba(X_text)[:, 1]

    pred_path = os.path.join(RESULTS, "metafeature_predictions.csv")
    text_rows[["feature", "feature_group", "corr_stability", "predicted_spurious_prob"]].to_csv(
        pred_path, index=False
    )
    print(f"Saved {pred_path}  ({len(text_rows)} rows)")

    print("\nDone.")
    print("\nLabeled feature descriptors (13 rows):")
    print(train[["feature", "feature_group", "is_spurious", "corr_stability"]].to_string(index=False))
    print("\nText feature predictions (sample):")
    summary = text_rows.groupby("feature_group")["predicted_spurious_prob"].agg(["mean", "std", "count"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
