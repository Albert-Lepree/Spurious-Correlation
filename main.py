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


STAGES = [
    "ingest",
    "spurious-sentiment",
    "sentiment",
    "sentiment-score",
    "compile",
    "lda",
    "embeddings",
]

STAGE_FNS = {
    "ingest": run_ingestion,
    "spurious-sentiment": run_spurious_sentiment_score,
    "sentiment": run_sentiment,
    "sentiment-score": run_sentiment_score,
    "compile": run_compile_master,
    "lda": run_lda_topics,
    "embeddings": run_embeddings,
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Spurious-Correlation data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Stages (must be run in this order):",
            "  ingest              Pull news from Databricks → datasets/real_news.csv",
            "  spurious-sentiment  Score spurious headlines  → datasets/spurious_news_scored.csv",
            "  sentiment           Classify real news        → PostgreSQL sentiment_results",
            "  sentiment-score     Score real news 0-100     → PostgreSQL sentiment_score_results",
            "  compile             Join all sources          → datasets/master_base.parquet + master_with_control.parquet",
            "  lda                 Topic modelling           → datasets/master_with_lda.parquet",
            "  embeddings          Encode text               → datasets/master_with_embeddings.parquet",
        ]),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--stage",
        choices=STAGES,
        metavar="STAGE",
        help=f"Run a single stage: {{{', '.join(STAGES)}}}",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all stages in order",
    )
    args = parser.parse_args()

    if args.all:
        for stage in STAGES:
            print(f"\n=== {stage} ===")
            STAGE_FNS[stage]()
    else:
        STAGE_FNS[args.stage]()
