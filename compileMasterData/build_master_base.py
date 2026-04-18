import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATASETS = os.path.join(_ROOT, "datasets")


def get_db():
    return psycopg2.connect(
        dbname="spuriousCorrelationdb",
        user=os.getenv("db_user"),
        password=os.getenv("db_pass"),
        host="localhost",
    )


def load_spurious_news():
    print("Loading spurious news...")
    df = pd.read_csv(os.path.join(DATASETS, "spurious_news_scored.csv"))
    df = df.rename(columns={"Date": "date", "Headline": "text_content"})
    df = df[["date", "text_content", "sentiment_score"]].copy()
    df["source_type"] = "kaggle_ai"
    print(f"  {len(df)} spurious news rows")
    return df


def load_real_news():
    print("Loading real news...")
    df = pd.read_csv(os.path.join(DATASETS, "real_news.csv"), usecols=["id", "webpage_content", "created_at"])

    print("  Fetching sentiment scores from postgres...")
    conn = get_db()
    scores = pd.read_sql("SELECT news_id, sentiment_score FROM sentiment_score_results", conn)
    conn.close()
    print(f"  {len(scores)} sentiment score rows fetched")

    df = df.merge(scores, left_on="id", right_on="news_id", how="inner")
    df = df.rename(columns={"created_at": "date", "webpage_content": "text_content"})
    df = df[["date", "text_content", "sentiment_score"]].copy()
    df["source_type"] = "google_real"
    print(f"  {len(df)} real news rows after join")
    return df


def load_market_data():
    print("Loading market data...")
    spx = pd.read_csv(os.path.join(DATASETS, "spx_data.csv"), parse_dates=["date"])
    vix = pd.read_csv(os.path.join(DATASETS, "vix_data.csv"), parse_dates=["date"])
    print(f"  SPX: {len(spx)} rows, VIX: {len(vix)} rows")
    return spx, vix


def compute_next_day_return(spx):
    spx = spx.sort_values("date").reset_index(drop=True)
    spx["next_day_return"] = spx["close"].shift(-1) / spx["close"] - 1
    spx = spx.iloc[:-1]  # drop last row — no next trading day
    return spx


def main():
    spurious_df = load_spurious_news()
    real_df = load_real_news()

    print("Concatenating news sources...")
    news = pd.concat([spurious_df, real_df], ignore_index=True)
    news["date"] = pd.to_datetime(news["date"], format="mixed", utc=True).dt.tz_convert(None).dt.normalize()
    news["article_id"] = news.index
    print(f"  Combined news: {len(news)} rows")

    spx, vix = load_market_data()
    spx = compute_next_day_return(spx)

    spx_cols = [
        "date", "open", "high", "low", "close", "volume",
        "macd", "signal", "Upper_Band", "Lower_Band", "BB_Width",
        "RSI_14", "RSI_5", "RSI_9", "EMA_50", "EMA_200",
        "average_true_range", "adx", "bull_market", "drawdown",
        "next_day_return",
    ]

    print("Joining news onto market data...")
    merged = news.merge(spx[spx_cols], on="date", how="left")
    merged = merged.merge(vix[["date", "vix_close"]], on="date", how="left")

    before = len(merged)
    merged = merged.dropna(subset=["next_day_return"])
    print(f"  Dropped {before - len(merged)} rows without market data")

    out_path = os.path.join(DATASETS, "master_base.parquet")
    merged.to_parquet(out_path, index=False)

    print(f"\nSaved {len(merged)} rows to {out_path}")
    print(f"Columns: {list(merged.columns)}")


if __name__ == "__main__":
    main()
