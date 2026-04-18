import json
import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def run_ingestion():
    """Pull data from Databricks and save to data/."""
    from databricksIngestion import databricks_ingest

    seed_path = os.path.join(_HERE, "databricksIngestion", "databricks_seed.json")
    with open(seed_path, "r") as f:
        queries = json.load(f)

    spurious_csv = os.path.join(_HERE, "datasets", "spurious_news.csv")
    kaggle_df = pd.read_csv(spurious_csv)
    print(f"  Spurious rows: {len(kaggle_df)}")

    conn = databricks_ingest.init_databricks()

    output_dir = os.path.join(_HERE, "data")
    os.makedirs(output_dir, exist_ok=True)

    google_df = databricks_ingest.query_databricks(conn, queries["google_news"])
    google_df.to_csv(os.path.join(output_dir, "google_news.csv"), index=False)
    print(f"  Google rows: {len(google_df)}")

    conn.close()


def run_sentiment():
    """Run binary BULLISH/BEARISH/NEUTRAL sentiment extraction."""
    from sentimentAnalysis import extractSentiment
    real_csv = os.path.join(_HERE, "datasets", "real_news.csv")
    extractSentiment.main(csv_path=real_csv)


def run_sentiment_score():
    """Run 0-100 numeric sentiment scoring."""
    from sentimentAnalysis import extractSentimentScore
    real_csv = os.path.join(_HERE, "datasets", "real_news.csv")
    extractSentimentScore.main(csv_path=real_csv)


def run_spurious_sentiment_score():
    """Run 0-100 numeric sentiment scoring."""
    from sentimentAnalysis import extractSpuriousSentiment
    spurious_csv = os.path.join(_HERE, "datasets", "spurious_news.csv")
    extractSpuriousSentiment.main(csv_path=spurious_csv)


def run_compile_master():
    """Build master_base.parquet then master_with_control.parquet."""
    from compileMasterData import build_master_base, add_control_features
    build_master_base.main()
    add_control_features.main()


def run_lda_topics():
    """Fit 10-topic LDA on all article text and save master_with_lda.parquet."""
    from lda import extractTopics
    extractTopics.main()


def run_embeddings():
    """Encode article text with all-mpnet-base-v2 and save master_with_embeddings.parquet."""
    from embeddings import extractEmbeddings
    extractEmbeddings.main()


if __name__ == "__main__":
    # run_ingestion()
    # run_sentiment()
    # run_sentiment_score()
    # run_spurious_sentiment_score()
    # run_compile_master()
    # run_compile_master()
    pd.set_option('display.max_columns', None)
    # df = pd.read_parquet('./datasets/master_with_control.parquet')
    # print(df)
    # df = pd.read_parquet('./datasets/master_base.parquet')
    # print(df)
    # run_lda_topics()
    df = pd.read_parquet('./datasets/master_with_lda.parquet')
    print(df)

