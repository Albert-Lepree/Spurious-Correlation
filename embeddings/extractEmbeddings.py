import os
import pandas as pd
from sentence_transformers import SentenceTransformer

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = os.path.join(_HERE, "..", "datasets")


def main():
    df = pd.read_parquet(os.path.join(_DATASETS, "master_with_lda.parquet"))

    df["encode_text"] = df["text"].where(df["source_type"] == "kaggle_ai", df["webpage_content"])

    to_encode = df[df["encode_text"].notna()].copy()

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    vectors = model.encode(
        to_encode["encode_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
    )

    embed_df = pd.DataFrame(
        vectors,
        index=to_encode.index,
        columns=[f"embed_{i}" for i in range(768)],
    )

    df = df.join(embed_df)
    df = df.drop(columns=["encode_text"])

    df.to_parquet(os.path.join(_DATASETS, "master_with_embeddings.parquet"), index=False)

    print(df.shape)
    print(df[["embed_0", "embed_767"]].head())
    print(df[["embed_0", "embed_767"]].isnull().sum())


if __name__ == "__main__":
    main()
