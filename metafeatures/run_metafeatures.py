import os

import numpy as np
import pandas as pd
from numpy.linalg import svd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATASETS = os.path.join(_ROOT, "datasets")
RESULTS = os.path.join(_ROOT, "results")

CONTROL_COLS = ["return_1d", "return_5d", "rolling_vol_20d", "volume_zscore", "vix_change"]
# Actual column names in parquet (build_master_base.py preserved mixed-case from spx_data.csv)
SPURIOUS_BASE_COLS = ["macd", "RSI_14", "BB_Width", "EMA_50"]
GENERATED_SPURIOUS_COLS = ["noise_gaussian", "noise_uniform", "future_return_t2", "shuffled_return"]
SPURIOUS_COLS = SPURIOUS_BASE_COLS + GENERATED_SPURIOUS_COLS
EMBED_COLS = [f"embed_{i}" for i in range(768)]
LDA_COLS = [f"lda_topic_{i}" for i in range(10)]


def effective_rank(matrix: np.ndarray) -> float:
    m = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-8)
    _, s, _ = svd(m, full_matrices=False)
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-10))))


def rolling_corr_stability(
    df: pd.DataFrame,
    feature_col: str,
    target: str = "next_day_return",
    window: int = 30,
) -> float:
    # Aggregate to daily means to avoid NaN from constant target within single-date windows
    daily = df.groupby("date")[[feature_col, target]].mean().reset_index()
    daily = daily.sort_values("date")
    rolling_corr = daily[feature_col].rolling(window).corr(daily[target])
    return float(rolling_corr.std())


def group_corr_stability(
    df: pd.DataFrame,
    cols: list,
    target: str = "next_day_return",
    window: int = 30,
) -> float:
    stabilities = [rolling_corr_stability(df, c, target, window) for c in cols]
    return float(np.nanmean(stabilities))


def topic_coherence_proxy(lda_df: pd.DataFrame) -> float:
    return float(lda_df.max(axis=1).mean())


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
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_embed_df_with_returns() -> pd.DataFrame:
    """
    Load master_with_embeddings.parquet (has LDA + embed cols) and join next_day_return
    from SPX data by date so we can compute rolling correlation stability.
    """
    df = pd.read_parquet(os.path.join(DATASETS, "master_with_embeddings.parquet"))

    # Build a unified date column from Kaggle 'Date' or Google 'created_at'
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
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main():
    os.makedirs(RESULTS, exist_ok=True)
    np.random.seed(42)

    print("Loading master_with_control.parquet (control/spurious/sentiment features)...")
    ctrl_df = load_ctrl_df()
    print(f"  {len(ctrl_df)} rows, {len(ctrl_df.columns)} columns")

    print("Generating spurious features on ctrl_df...")
    ctrl_df = generate_spurious_features(ctrl_df)
    ctrl_valid = ctrl_df.dropna(subset=["next_day_return"]).copy()
    print(f"  {len(ctrl_valid)} rows after dropping NaN next_day_return")

    print("Loading master_with_embeddings.parquet (LDA/embeddings) + SPX returns join...")
    embed_df = load_embed_df_with_returns()
    embed_valid = embed_df.dropna(subset=["next_day_return"]).copy()
    print(f"  {len(embed_df)} total embed rows, {len(embed_valid)} with next_day_return")

    print("Computing meta-features...")

    # --- control group ---
    ctrl_matrix = ctrl_valid[CONTROL_COLS].dropna().values
    ctrl_er = effective_rank(ctrl_matrix)
    ctrl_cs = group_corr_stability(ctrl_valid, CONTROL_COLS)
    print(f"  control: eff_rank={ctrl_er:.3f}, corr_stability={ctrl_cs:.4f}")

    # --- spurious group ---
    spur_valid = ctrl_valid[SPURIOUS_COLS].dropna()
    spur_df = ctrl_valid.loc[spur_valid.index]
    spur_er = effective_rank(spur_valid.values)
    spur_cs = group_corr_stability(spur_df, SPURIOUS_COLS)
    print(f"  spurious: eff_rank={spur_er:.3f}, corr_stability={spur_cs:.4f}")

    # --- embeddings group ---
    embed_matrix = embed_valid[EMBED_COLS].dropna().values
    embed_er = effective_rank(embed_matrix)
    embed_cs = group_corr_stability(embed_valid.dropna(subset=EMBED_COLS[:1]), EMBED_COLS)
    print(f"  embeddings: eff_rank={embed_er:.3f}, corr_stability={embed_cs:.4f}")

    # --- LDA group ---
    lda_matrix = embed_valid[LDA_COLS].dropna().values
    lda_er = effective_rank(lda_matrix)
    lda_cs = group_corr_stability(embed_valid.dropna(subset=LDA_COLS[:1]), LDA_COLS)
    lda_coherence = topic_coherence_proxy(embed_valid[LDA_COLS])
    print(f"  lda_topics: eff_rank={lda_er:.3f}, corr_stability={lda_cs:.4f}, coherence={lda_coherence:.4f}")

    # --- sentiment (scalar, from ctrl dataset) ---
    sent_er = 1.0
    sent_cs = rolling_corr_stability(ctrl_valid, "sentiment_score")
    print(f"  sentiment: eff_rank={sent_er:.3f}, corr_stability={sent_cs:.4f}")

    rows = [
        {
            "feature_group": "control",
            "is_spurious": 0.0,
            "effective_rank": ctrl_er,
            "corr_stability": ctrl_cs,
            "topic_coherence": np.nan,
        },
        {
            "feature_group": "spurious",
            "is_spurious": 1.0,
            "effective_rank": spur_er,
            "corr_stability": spur_cs,
            "topic_coherence": np.nan,
        },
        {
            "feature_group": "embeddings",
            "is_spurious": np.nan,
            "effective_rank": embed_er,
            "corr_stability": embed_cs,
            "topic_coherence": np.nan,
        },
        {
            "feature_group": "lda_topics",
            "is_spurious": np.nan,
            "effective_rank": lda_er,
            "corr_stability": lda_cs,
            "topic_coherence": lda_coherence,
        },
        {
            "feature_group": "sentiment",
            "is_spurious": np.nan,
            "effective_rank": sent_er,
            "corr_stability": sent_cs,
            "topic_coherence": np.nan,
        },
    ]
    meta_df = pd.DataFrame(rows)

    descriptors_path = os.path.join(RESULTS, "metafeature_descriptors.csv")
    meta_df.to_csv(descriptors_path, index=False)
    print(f"\nSaved {descriptors_path}")

    # --- logistic regression ---
    print("\nTraining logistic regression...")
    train = meta_df[meta_df["is_spurious"].notna()].copy()
    X = train[["effective_rank", "corr_stability"]].fillna(0).values
    y = train["is_spurious"].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classes, counts = np.unique(y, return_counts=True)
    cv_mean, cv_std = np.nan, np.nan
    if len(classes) >= 2 and counts.min() >= 2:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for ti, vi in cv.split(X_scaled, y):
            clf = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=1.0)
            clf.fit(X_scaled[ti], y[ti])
            scores.append(clf.score(X_scaled[vi], y[vi]))
        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))
        print(f"CV accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
    else:
        print("Skipping CV: insufficient samples per class")

    clf_final = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=1.0)
    clf_final.fit(X_scaled, y)

    coef_rows = [
        {"feature": "effective_rank", "coefficient": float(clf_final.coef_[0][0])},
        {"feature": "corr_stability", "coefficient": float(clf_final.coef_[0][1])},
        {"feature": "intercept",      "coefficient": float(clf_final.intercept_[0])},
        {"feature": "cv_accuracy",    "coefficient": cv_mean},
        {"feature": "cv_std",         "coefficient": cv_std},
    ]
    coef_df = pd.DataFrame(coef_rows)
    coef_path = os.path.join(RESULTS, "metafeature_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"Saved {coef_path}")

    # --- apply to text features ---
    text_rows = meta_df[meta_df["is_spurious"].isna()].copy()
    X_text = scaler.transform(text_rows[["effective_rank", "corr_stability"]].fillna(0).values)
    text_rows["predicted_spurious_prob"] = clf_final.predict_proba(X_text)[:, 1]

    pred_path = os.path.join(RESULTS, "metafeature_predictions.csv")
    text_rows[["feature_group", "effective_rank", "corr_stability", "predicted_spurious_prob"]].to_csv(
        pred_path, index=False
    )
    print(f"Saved {pred_path}")

    print("\nDone.")
    print("\nDescriptors:")
    print(meta_df.to_string(index=False))
    print("\nPredictions:")
    print(text_rows[["feature_group", "predicted_spurious_prob"]].to_string(index=False))


if __name__ == "__main__":
    main()
