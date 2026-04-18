import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = os.path.join(_HERE, "..", "datasets")


def main():
    spurious = pd.read_csv(os.path.join(_DATASETS, "spurious_news.csv"))
    spurious["source_type"] = "kaggle_ai"
    spurious["text"] = spurious["Headline"]

    real = pd.read_csv(os.path.join(_DATASETS, "real_news.csv"))
    real["source_type"] = "google_real"
    real["text"] = real["webpage_content"]

    df = pd.concat([spurious, real], ignore_index=True)

    df = df[df["text"].notna() & (df["text"].str.strip() != "")].reset_index(drop=True)

    vectorizer = CountVectorizer(max_features=1000, stop_words="english")
    dtm = vectorizer.fit_transform(df["text"])

    lda = LatentDirichletAllocation(n_components=10, random_state=42)
    topic_distributions = lda.fit_transform(dtm)

    for i in range(10):
        df[f"lda_topic_{i}"] = topic_distributions[:, i]

    out_path = os.path.join(_DATASETS, "master_with_lda.parquet")
    df.to_parquet(out_path, index=False)

    print(df.shape)
    print(df[["lda_topic_0", "lda_topic_9"]].head())


if __name__ == "__main__":
    main()
