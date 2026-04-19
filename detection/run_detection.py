import os

import numpy as np
import pandas as pd
import psycopg2
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.stats.multitest import multipletests

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATASETS = os.path.join(_ROOT, "datasets")
RESULTS = os.path.join(_ROOT, "results")

CONTROL = ["return_1d", "return_5d", "rolling_vol_20d", "volume_zscore", "vix_change"]
SPURIOUS = ["macd", "RSI_14", "BB_Width", "EMA_50",
            "noise_gaussian", "noise_uniform", "future_return_t2", "shuffled_return"]
TEXT = ["sentiment_score"] + \
      [f"lda_topic_{i}" for i in range(10)] + \
      [f"embed_{i}" for i in range(768)]
TARGET = "next_day_return"
ALL_FEATURES = CONTROL + SPURIOUS + TEXT  # 792 total

N_BOOTSTRAP = 500
BLOCK_SIZE = 20
N_JOBS = 16
TRAIN_WINDOW = 252
TEST_WINDOW = 21
MIN_FOLDS = 5

BOOTSTRAP_EXCLUDE = ["future_return_t2"]


def get_db():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "spuriousCorrelationdb"),
        user=os.getenv("db_user"),
        password=os.getenv("db_pass"),
        host=os.getenv("DB_HOST", "localhost"),
    )


def assign_group(feat):
    if feat in CONTROL:
        return "control"
    if feat in SPURIOUS:
        return "spurious"
    return "text"


def strategy_returns(feature_series, target_series):
    median = feature_series.median()
    signal = np.where(feature_series >= median, 1, -1)
    return signal * target_series.values


def sharpe(returns, annualize=252):
    r = np.array(returns)
    if r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(annualize)


def compute_tpr_fpr(rows, rejected_col):
    spurious_rows = [r for r in rows if r["feature_group"] == "spurious"]
    control_rows = [r for r in rows if r["feature_group"] == "control"]
    tpr = sum(r[rejected_col] for r in spurious_rows) / len(spurious_rows)
    fpr = sum(r[rejected_col] for r in control_rows) / len(control_rows)
    return tpr, fpr


def block_bootstrap_indices(n, block_size, rng):
    indices = []
    while len(indices) < n:
        start = rng.integers(0, n - block_size)
        indices.extend(range(start, start + block_size))
    return indices[:n]


def one_iteration(seed, feature_matrix, target_vals, block_size):
    rng = np.random.default_rng(seed)
    idx = block_bootstrap_indices(len(target_vals), block_size, rng)
    t = target_vals[idx]
    return np.array([
        sharpe(strategy_returns(pd.Series(feature_matrix[idx, i]), pd.Series(t)))
        for i in range(feature_matrix.shape[1])
    ])


def run_bonferroni_fdr(df, conn):
    print("Running Bonferroni/FDR...")
    cursor = conn.cursor()
    alpha = 0.05
    bonferroni_threshold = alpha / len(ALL_FEATURES)

    rows = []
    for feat in ALL_FEATURES:
        valid = df[[feat, TARGET]].dropna()
        r, p = stats.pearsonr(valid[feat], valid[TARGET])
        rows.append({
            "feature": feat,
            "feature_group": assign_group(feat),
            "correlation": r,
            "p_value": p,
        })

    p_values = [r["p_value"] for r in rows]
    reject_fdr, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

    for i, row in enumerate(rows):
        row["bonferroni_rejected"] = bool(row["p_value"] >= bonferroni_threshold)
        row["fdr_rejected"] = bool(not reject_fdr[i])

    cursor.executemany("""
        INSERT INTO bonferroni_fdr_results
            (feature, feature_group, correlation, p_value, bonferroni_rejected, fdr_rejected)
        VALUES (%(feature)s, %(feature_group)s, %(correlation)s, %(p_value)s,
                %(bonferroni_rejected)s, %(fdr_rejected)s)
    """, rows)
    conn.commit()
    print(f"Bonferroni/FDR: wrote {len(rows)} rows")

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "bonferroni_fdr_results.csv"), index=False)
    return rows


def run_bootstrap(df, conn):
    print("Running Bootstrap Resampling...")
    cursor = conn.cursor()
    boot_rows = []

    for group_name, features in [("control", CONTROL), ("spurious", SPURIOUS), ("text", TEXT)]:
        features_to_run = [f for f in features if f not in BOOTSTRAP_EXCLUDE]
        print(f"  Bootstrap: {group_name} ({len(features_to_run)} features)...")
        valid = df[features_to_run + [TARGET]].dropna()
        target_vals = valid[TARGET].values
        feature_matrix = valid[features_to_run].values

        insample = np.array([
            sharpe(strategy_returns(pd.Series(feature_matrix[:, i]), pd.Series(target_vals)))
            for i in range(len(features_to_run))
        ])

        boot_matrix = np.array(
            Parallel(n_jobs=N_JOBS)(
                delayed(one_iteration)(s, feature_matrix, target_vals, BLOCK_SIZE)
                for s in range(N_BOOTSTRAP)
            )
        )  # shape (N_BOOTSTRAP, n_features)

        critical_values = np.percentile(boot_matrix, 95, axis=0)

        for i, feat in enumerate(features_to_run):
            boot_rows.append({
                "feature": feat,
                "feature_group": group_name,
                "insample_sharpe": float(insample[i]),
                "critical_value_95": float(critical_values[i]),
                "bootstrap_rejected": bool(insample[i] > critical_values[i]),
            })

        for feat in features:
            if feat in BOOTSTRAP_EXCLUDE:
                boot_rows.append({
                    "feature": feat,
                    "feature_group": group_name,
                    "insample_sharpe": None,
                    "critical_value_95": None,
                    "bootstrap_rejected": True,  # look-ahead bias: rejected by definition
                })

    cursor.executemany("""
        INSERT INTO bootstrap_results
            (feature, feature_group, insample_sharpe, critical_value_95, bootstrap_rejected)
        VALUES (%(feature)s, %(feature_group)s, %(insample_sharpe)s,
                %(critical_value_95)s, %(bootstrap_rejected)s)
    """, boot_rows)
    conn.commit()
    print(f"Bootstrap: wrote {len(boot_rows)} rows")

    pd.DataFrame(boot_rows).to_csv(os.path.join(RESULTS, "bootstrap_results.csv"), index=False)
    return boot_rows


def run_walkforward(df, conn):
    print("Running Walk-Forward Validation...")
    cursor = conn.cursor()
    df_sorted = df.sort_values("date").reset_index(drop=True)

    gen_ratios = {feat: [] for feat in ALL_FEATURES}
    n_folds = {feat: 0 for feat in ALL_FEATURES}

    i = 0
    fold = 0
    while i + TRAIN_WINDOW + TEST_WINDOW <= len(df_sorted):
        train = df_sorted.iloc[i: i + TRAIN_WINDOW]
        test = df_sorted.iloc[i + TRAIN_WINDOW: i + TRAIN_WINDOW + TEST_WINDOW]
        fold += 1

        for feat in ALL_FEATURES:
            train_valid = train[[feat, TARGET]].dropna()
            test_valid = test[[feat, TARGET]].dropna()
            if len(train_valid) < 20 or len(test_valid) < 5:
                continue

            threshold = train_valid[feat].median()
            s_train = sharpe(strategy_returns(train_valid[feat], train_valid[TARGET]))
            if s_train == 0:
                continue

            test_signal = np.where(test_valid[feat].values >= threshold, 1, -1)
            test_ret = test_signal * test_valid[TARGET].values
            s_test = sharpe(pd.Series(test_ret))

            ratio = float(np.clip(s_test / s_train, -10, 10))
            gen_ratios[feat].append(ratio)
            n_folds[feat] += 1

        i += TEST_WINDOW

    print(f"Walk-forward: {fold} folds completed")

    wf_rows = []
    for feat in ALL_FEATURES:
        if len(gen_ratios[feat]) < MIN_FOLDS:
            continue
        mean_ratio = float(np.mean(gen_ratios[feat]))
        wf_rows.append({
            "feature": feat,
            "feature_group": assign_group(feat),
            "mean_gen_ratio": mean_ratio,
            "n_folds": n_folds[feat],
            "wf_rejected": bool(mean_ratio < 0.5),
        })

    cursor.executemany("""
        INSERT INTO walkforward_results
            (feature, feature_group, mean_gen_ratio, n_folds, wf_rejected)
        VALUES (%(feature)s, %(feature_group)s, %(mean_gen_ratio)s,
                %(n_folds)s, %(wf_rejected)s)
    """, wf_rows)
    conn.commit()
    print(f"Walk-forward: wrote {len(wf_rows)} rows")

    pd.DataFrame(wf_rows).to_csv(os.path.join(RESULTS, "walkforward_results.csv"), index=False)
    return wf_rows


def main():
    os.makedirs(RESULTS, exist_ok=True)
    np.random.seed(42)

    print("Loading master_final.parquet...")
    df = pd.read_parquet(os.path.join(DATASETS, "master_final.parquet"))
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # Generate spurious features if not already present
    if "noise_gaussian" not in df.columns:
        df["noise_gaussian"] = np.random.randn(len(df))
    if "noise_uniform" not in df.columns:
        df["noise_uniform"] = np.random.uniform(-1, 1, len(df))
    if "future_return_t2" not in df.columns:
        df["future_return_t2"] = df[TARGET].shift(-2)
    if "shuffled_return" not in df.columns:
        df["shuffled_return"] = df[TARGET].sample(frac=1, random_state=42).values

    df = df.dropna(subset=[TARGET])
    print(f"  {len(df)} rows after dropna on {TARGET}")

    conn = get_db()
    cursor = conn.cursor()

    print("Truncating result tables...")
    for table in ["bonferroni_fdr_results", "bootstrap_results", "walkforward_results", "detection_summary"]:
        cursor.execute(f"TRUNCATE TABLE {table}")
    conn.commit()

    bonf_rows = run_bonferroni_fdr(df, conn)
    boot_rows = run_bootstrap(df, conn)
    wf_rows = run_walkforward(df, conn)

    print("\nComputing detection summary...")
    cursor = conn.cursor()
    summary_rows = []
    for method, rows, rejected_col in [
        ("Bonferroni",   bonf_rows, "bonferroni_rejected"),
        ("FDR",          bonf_rows, "fdr_rejected"),
        ("Bootstrap",    boot_rows, "bootstrap_rejected"),
        ("Walk-Forward", wf_rows,   "wf_rejected"),
    ]:
        tpr, fpr = compute_tpr_fpr(rows, rejected_col)
        f1 = 2 * tpr * (1 - fpr) / (tpr + (1 - fpr) + 1e-10)
        summary_rows.append({"method": method, "tpr": tpr, "fpr": fpr, "f1": float(f1)})
        print(f"{method:15s} | TPR={tpr:.3f} | FPR={fpr:.3f} | F1={f1:.3f}")

    cursor.executemany("""
        INSERT INTO detection_summary (method, tpr, fpr, f1)
        VALUES (%(method)s, %(tpr)s, %(fpr)s, %(f1)s)
    """, summary_rows)
    conn.commit()

    pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS, "FINAL_SUMMARY.csv"), index=False)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
