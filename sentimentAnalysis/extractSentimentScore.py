#!/usr/bin/env python3
"""Process news sentiment in batches using a 0-100 numeric score (0=bearish, 100=bullish)."""

import os
import re
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
SKILL_PATH = os.path.join(_HERE, "financial-sentiment-score-skill.md")
VLLM_URL = "http://localhost:8001/v1/completions"


def get_db():
    return psycopg2.connect(
        dbname="spuriousCorrelationdb",
        user=os.getenv("db_user"),
        password=os.getenv("db_pass"),
        host="localhost"
    )


def ensure_table(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_score_results (
            news_id        INTEGER,
            sentiment_score INTEGER
        )
    """)
    conn.commit()
    cursor.close()


def parse_score(raw: str) -> int:
    """
    Extract a 0-100 integer from LLM output.

    Normalises first (strips think tags and fences), then finds the first
    integer in the cleaned string. Clamps result to [0, 100].
    Falls back to 50 (neutral) if no integer is found.
    """
    cleaned = normalise_llm_output(raw)
    matches = re.findall(r'\d+', cleaned)
    if not matches:
        return 50
    return max(0, min(100, int(matches[0])))


def get_score_batch(texts: list, skill_prompt: str) -> list:
    """
    Send up to BATCH_SIZE article texts in a single vLLM request.

    Returns a list of integers in [0, 100], in the same order as the input list.
    """
    prompts = [
        f"{skill_prompt}\n\nArticle: {t[:5000]}\n\nScore:"
        for t in texts
    ]
    response = requests.post(VLLM_URL, json={
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "prompt": prompts,
        "max_tokens": 16000,
        "temperature": 0
    })
    choices = sorted(response.json()["choices"], key=lambda c: c["index"])
    return [parse_score(c["text"]) for c in choices]


def save_batch(conn, batch_data):
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO sentiment_score_results (news_id, sentiment_score) VALUES (%s, %s)",
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

    print(f"Total articles (score): {total}")

    conn = get_db()
    ensure_table(conn)

    for i in range(0, total, BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]

        print(f"Processing batch {i//BATCH_SIZE + 1} ({i+1}-{min(i+BATCH_SIZE, total)})...")

        texts = batch['webpage_content'].tolist()
        ids = batch['id'].tolist()

        try:
            scores = get_score_batch(texts, skill_prompt)
            results = list(zip(ids, scores))
        except Exception as e:
            print(f"  Batch error: {e}")
            continue

        if results:
            save_batch(conn, results)
            print(f"  Saved {len(results)} results")

    conn.close()
    print("Done (score)!")


if __name__ == "__main__":
    main()
