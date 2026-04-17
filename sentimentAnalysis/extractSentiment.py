#!/usr/bin/env python3
"""Process news sentiment in batches using BULLISH/BEARISH/NEUTRAL classification."""

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import requests
from sentimentAnalysis import normalise_llm_output

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

BATCH_SIZE = 50
CSV_FILE = os.path.join(_ROOT, "datasets", "real_news.csv")
SKILL_PATH = os.path.join(_HERE, "financial-sentiment-skill.md")
VLLM_URL = "http://localhost:8001/v1/completions"


def get_db():
    return psycopg2.connect(
        dbname="spuriousCorrelationdb",
        user=os.getenv("db_user"),
        password=os.getenv("db_pass"),
        host="localhost"
    )


def get_sentiment_batch(texts: list, skill_prompt: str) -> list:
    """
    Send up to BATCH_SIZE article texts in a single vLLM request.

    Returns a list of floats (1.0 BULLISH / -1.0 BEARISH / 0.0 NEUTRAL),
    in the same order as the input list.
    """
    prompts = [
        f"{skill_prompt}\n\nArticle: {t[:5000]}\n\nSentiment:"
        for t in texts
    ]
    response = requests.post(VLLM_URL, json={
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "prompt": prompts,
        "max_tokens": 16000,
        "temperature": 0
    })
    choices = sorted(response.json()["choices"], key=lambda c: c["index"])
    results = []
    for choice in choices:
        raw = normalise_llm_output(choice["text"].strip().upper())
        if "BULLISH" in raw:
            results.append(1.0)
        elif "BEARISH" in raw:
            results.append(-1.0)
        else:
            results.append(0.0)
    return results


def save_batch(conn, batch_data):
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO sentiment_results (news_id, sentiment) VALUES (%s, %s)",
        batch_data
    )
    conn.commit()
    cursor.close()


def main(csv_path: str = None):
    with open(SKILL_PATH, "r") as f:
        skill_prompt = f.read()

    df = pd.read_csv(csv_path or CSV_FILE)
    df = df.dropna(subset=['webpage_content'])
    total = len(df)

    print(f"Total articles: {total}")

    conn = get_db()

    for i in range(0, total, BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]

        print(f"Processing batch {i//BATCH_SIZE + 1} ({i+1}-{min(i+BATCH_SIZE, total)})...")

        texts = batch['webpage_content'].tolist()
        ids = batch['id'].tolist()

        try:
            sentiments = get_sentiment_batch(texts, skill_prompt)
            results = list(zip(ids, sentiments))
        except Exception as e:
            print(f"  Batch error: {e}")
            continue

        if results:
            save_batch(conn, results)
            print(f"  Saved {len(results)} results")

    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
