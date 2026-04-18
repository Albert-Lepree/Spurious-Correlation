#!/usr/bin/env python3
"""Process spurious news headlines in batches using a 0-100 numeric score (0=bearish, 100=bullish)."""

import asyncio
import os
import re
import pandas as pd
import httpx
from dotenv import load_dotenv
from sentimentAnalysis import normalise_llm_output

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

BATCH_SIZE = 50
CSV_IN = os.path.join(_ROOT, "datasets", "spurious_news.csv")
CSV_OUT = os.path.join(_ROOT, "datasets", "spurious_news_scored.csv")
SKILL_PATH = os.path.join(_HERE, "spurious-sentiment-score-skill.md")
VLLM_URL = "http://localhost:8001/v1/completions"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def parse_score(raw: str, flag: str = "") -> int:
    """
    Extract a 0-100 integer from LLM output.

    Normalises first (strips think tags and fences), then finds the first
    integer in the cleaned string. Clamps result to [0, 100].
    Falls back to flag-specific default if no integer is found.
    """
    cleaned = normalise_llm_output(raw)
    matches = re.findall(r'\d+', cleaned)
    if not matches:
        if flag.lower() == "positive":
            return 65
        if flag.lower() == "negative":
            return 35
        return 50
    return max(0, min(100, int(matches[0])))


async def _get_score_async(
    row_idx: int,
    headline: str,
    sentiment_flag: str,
    skill_prompt: str,
    client: httpx.AsyncClient,
) -> tuple:
    """Single async LLM call. Returns (row_idx, score, reason)."""
    prompt = f"{skill_prompt}\n\nHeadline: {headline}\nSentiment flag: {sentiment_flag}\n\nScore:"
    try:
        resp = await client.post(VLLM_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 32000,
            "temperature": 0,
        }, timeout=120.0)
        score = parse_score(resp.json()["choices"][0]["text"], sentiment_flag)
        return (row_idx, score, "ok")
    except Exception as e:
        return (row_idx, parse_score("", sentiment_flag), f"request_failed: {e}")


async def _run_batch_async(
    rows: list,
    skill_prompt: str,
) -> list:
    """
    Two-pass async batch.

    Pass 1: Fire all rows concurrently via asyncio.gather.
    Pass 2: Retry only rows where reason starts with 'request_failed'
            (network/parse errors) — legitimate scores are NOT retried.
    """
    async with httpx.AsyncClient() as client:
        tasks = [
            _get_score_async(idx, headline, flag, skill_prompt, client)
            for idx, headline, flag in rows
        ]
        results = list(await asyncio.gather(*tasks))

        retry_indices = [
            i for i, (_, _, reason) in enumerate(results)
            if reason.startswith("request_failed")
        ]
        if retry_indices:
            print(f"    Retrying {len(retry_indices)} failed requests...")
            retry_tasks = [
                _get_score_async(rows[i][0], rows[i][1], rows[i][2], skill_prompt, client)
                for i in retry_indices
            ]
            for idx, res in zip(retry_indices, await asyncio.gather(*retry_tasks)):
                results[idx] = res

        return results


def get_score_batch(rows: list, skill_prompt: str) -> list:
    """Run 50 concurrent async LLM requests, return (row_idx, score) pairs."""
    results = asyncio.run(_run_batch_async(rows, skill_prompt))
    still_failed = sum(1 for _, _, reason in results if reason.startswith("request_failed"))
    if still_failed:
        print(f"    {still_failed}/{len(results)} rows failed after retry")
    return [(idx, score) for idx, score, _ in results]


def main(csv_path=None):
    with open(SKILL_PATH) as f:
        skill_prompt = f.read()

    df = pd.read_csv(csv_path or CSV_IN)
    df = df.dropna(subset=['Headline'])
    df = df[df['Headline'].str.strip() != '']
    df = df.reset_index(drop=True)
    total = len(df)

    print(f"Total rows to score: {total}")

    scores = {}
    for i in range(0, total, BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE]

        print(f"Processing batch {i // BATCH_SIZE + 1} ({i + 1}-{min(i + BATCH_SIZE, total)})...")

        rows = [
            (idx, row['Headline'], str(row['Sentiment']) if pd.notna(row['Sentiment']) else '')
            for idx, row in batch.iterrows()
        ]

        try:
            results = get_score_batch(rows, skill_prompt)
        except Exception as e:
            print(f"  Batch error: {e}")
            continue

        for idx, score in results:
            scores[idx] = score

    df['sentiment_score'] = df.index.map(scores)
    df.to_csv(CSV_OUT, index=False)
    print(f"Saved {len(df)} rows to {CSV_OUT}")
    print("Done (spurious score)!")


if __name__ == "__main__":
    main()
