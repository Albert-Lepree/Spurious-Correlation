#!/usr/bin/env python3
"""Process news sentiment in batches using a 0-100 numeric score (0=bearish, 100=bullish)."""

import asyncio
import os
import re
import pandas as pd
import psycopg2
import httpx
from dotenv import load_dotenv
from sentimentAnalysis import normalise_llm_output

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

BATCH_SIZE = 50
CSV_FILE = os.path.join(_ROOT, "datasets", "real_news.csv")
SKILL_PATH = os.path.join(_HERE, "financial-sentiment-score-skill.md")
VLLM_URL = "http://localhost:8001/v1/completions"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


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


async def _get_score_async(
    article_id: int,
    text: str,
    skill_prompt: str,
    client: httpx.AsyncClient,
) -> tuple:
    """Single async LLM call. Returns (article_id, score, reason)."""
    prompt = f"{skill_prompt}\n\nArticle: {text[:5000]}\n\nScore:"
    try:
        resp = await client.post(VLLM_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 32000,
            "temperature": 0,
        }, timeout=120.0)
        score = parse_score(resp.json()["choices"][0]["text"])
        return (article_id, score, "ok")
    except Exception as e:
        return (article_id, 50, f"request_failed: {e}")


async def _run_batch_async(
    rows: list,
    skill_prompt: str,
) -> list:
    """
    Two-pass async batch.

    Pass 1: Fire all articles concurrently via asyncio.gather.
    Pass 2: Retry only articles where reason starts with 'request_failed'
            (network/parse errors) — legitimate scores are NOT retried.
    """
    async with httpx.AsyncClient() as client:
        tasks = [
            _get_score_async(aid, text, skill_prompt, client)
            for aid, text in rows
        ]
        results = list(await asyncio.gather(*tasks))

        retry_indices = [
            i for i, (_, _, reason) in enumerate(results)
            if reason.startswith("request_failed")
        ]
        if retry_indices:
            print(f"    Retrying {len(retry_indices)} failed requests...")
            retry_tasks = [
                _get_score_async(rows[i][0], rows[i][1], skill_prompt, client)
                for i in retry_indices
            ]
            for idx, res in zip(retry_indices, await asyncio.gather(*retry_tasks)):
                results[idx] = res

        return results


def get_score_batch(rows: list, skill_prompt: str) -> list:
    """Run 50 concurrent async LLM requests, return (news_id, score) pairs."""
    results = asyncio.run(_run_batch_async(rows, skill_prompt))
    still_failed = sum(1 for _, _, reason in results if reason.startswith("request_failed"))
    if still_failed:
        print(f"    {still_failed}/{len(results)} articles failed after retry")
    return [(aid, score) for aid, score, _ in results]


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

    for i in range(700, total, BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]

        print(f"Processing batch {i//BATCH_SIZE + 1} ({i+1}-{min(i+BATCH_SIZE, total)})...")

        rows = list(zip(batch['id'].tolist(), batch['webpage_content'].tolist()))

        try:
            results = get_score_batch(rows, skill_prompt)
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
