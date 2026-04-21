#!/usr/bin/env python3
"""Re-score articles with sentinel sentiment_score values (0 or 50) in postgres."""

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

from sentimentAnalysis.extractSentimentScore import get_score_batch, BATCH_SIZE, SKILL_PATH

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

CSV_FILE = os.path.join(_ROOT, "datasets", "real_news.csv")


def get_db():
    return psycopg2.connect(
        dbname="spuriousCorrelationdb",
        user=os.getenv("db_user"),
        password=os.getenv("db_pass"),
        host="localhost",
    )


def main():
    conn = get_db()

    before = pd.read_sql(
        "SELECT sentiment_score, COUNT(*) as count FROM sentiment_score_results GROUP BY sentiment_score ORDER BY sentiment_score",
        conn,
    )
    print("Before distribution:")
    print(before)

    bad = pd.read_sql(
        "SELECT news_id, sentiment_score FROM sentiment_score_results WHERE sentiment_score IN (0, 50)",
        conn,
    )
    print(f"Found {len(bad)} rows to re-score")

    if len(bad) == 0:
        print("No rows to re-score")
        conn.close()
        return

    real_news = pd.read_csv(CSV_FILE)
    to_rescore = bad.merge(real_news[["id", "webpage_content"]], left_on="news_id", right_on="id", how="inner")
    to_rescore = to_rescore.dropna(subset=["webpage_content"])
    print(f"{len(to_rescore)} rows after joining to real_news.csv")

    with open(SKILL_PATH, "r") as f:
        skill_prompt = f.read()

    rows = list(zip(to_rescore["news_id"], to_rescore["webpage_content"]))
    results = []
    total = len(rows)
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        print(f"  Batch {i // BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} articles)...")
        results.extend(get_score_batch(batch, skill_prompt))

    cursor = conn.cursor()
    cursor.executemany(
        "UPDATE sentiment_score_results SET sentiment_score = %s WHERE news_id = %s",
        [(score, news_id) for news_id, score in results],
    )
    conn.commit()
    print(f"Updated {len(results)} rows")

    after = pd.read_sql(
        "SELECT sentiment_score, COUNT(*) FROM sentiment_score_results GROUP BY sentiment_score ORDER BY sentiment_score",
        conn,
    )
    print(after)

    conn.close()
