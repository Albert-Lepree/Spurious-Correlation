import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATASETS = os.path.join(_ROOT, "datasets")

_FEATURE_COLS = [f"lda_topic_{i}" for i in range(10)] + [f"embed_{i}" for i in range(768)]


def _join_on_text(ctrl_sub, embed_sub, ctrl_text_col, embed_text_col):
    """Dedup embed by text then inner-join ctrl on the text column."""
    dedup = embed_sub.drop_duplicates(subset=[embed_text_col])
    dedup = dedup[[embed_text_col] + _FEATURE_COLS].copy()
    dedup["_key"] = dedup[embed_text_col].str.strip()
    left = ctrl_sub.copy()
    left["_key"] = left[ctrl_text_col].str.strip()
    merged = left.merge(dedup.drop(columns=[embed_text_col]), on="_key", how="inner")
    return merged.drop(columns=["_key"])


def main():
    print("Loading master_with_control.parquet...")
    ctrl = pd.read_parquet(os.path.join(DATASETS, "master_with_control.parquet"))
    print(f"  {len(ctrl)} rows, {len(ctrl.columns)} columns")

    print("Loading master_with_embeddings.parquet...")
    embed = pd.read_parquet(os.path.join(DATASETS, "master_with_embeddings.parquet"))
    print(f"  {len(embed)} rows, {len(embed.columns)} columns")

    print("Joining kaggle_ai on text_content = text...")
    merged_kai = _join_on_text(
        ctrl[ctrl["source_type"] == "kaggle_ai"],
        embed[embed["source_type"] == "kaggle_ai"],
        "text_content",
        "text",
    )
    print(f"  kaggle_ai: {len(merged_kai)} rows")

    print("Joining google_real on text_content = webpage_content...")
    merged_gr = _join_on_text(
        ctrl[ctrl["source_type"] == "google_real"],
        embed[embed["source_type"] == "google_real"],
        "text_content",
        "webpage_content",
    )
    print(f"  google_real: {len(merged_gr)} rows")

    merged = pd.concat([merged_kai, merged_gr], ignore_index=True)

    print(f"\nctrl rows:   {len(ctrl)}")
    print(f"embed rows:  {len(embed)}")
    print(f"merged rows: {len(merged)}")
    print(f"merged cols: {len(merged.columns)}")
    print(f"source_type counts:\n{merged['source_type'].value_counts()}")
    print(f"date range: {merged['date'].min()} → {merged['date'].max()}")
    print(
        f"null counts (non-embed):\n"
        f"{merged[['next_day_return','sentiment_score','return_1d','lda_topic_0']].isnull().sum()}"
    )

    if len(merged) < 2000:
        raise ValueError(
            f"Merged row count ({len(merged)}) is below 2000 — "
            "join keys are likely misaligned. Investigate before saving."
        )

    out_path = os.path.join(DATASETS, "master_final.parquet")
    merged.to_parquet(out_path, index=False)
    print(f"\nSaved {len(merged)} rows, {len(merged.columns)} columns → {out_path}")


if __name__ == "__main__":
    main()
