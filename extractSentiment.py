#!/usr/bin/env python3
"""Process news sentiment in batches."""

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import requests
import re

load_dotenv()

# Config
BATCH_SIZE = 50
CSV_FILE = "real_news.csv"
VLLM_URL = "http://localhost:8001/v1/completions"

def normalise_llm_output(raw: str) -> str:
    """
    Strip chain-of-thought blocks, markdown fences, and excess whitespace.

    Processing order:
    1. Strip <think>...</think> block (keeps content after </think>)
    2. Strip loose </think> tag with no opening block
    3. Strip ```json or ``` fences
    4. Strip leading/trailing whitespace

    Parameters:
    - raw: Raw string returned by the LLM.

    Returns: Cleaned content string.
    """
    if '<think>' in raw and '</think>' in raw:
        raw = raw.split('</think>', 1)[-1]
    elif '</think>' in raw:
        raw = raw.replace('</think>', '')

    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)

    return raw.strip()

# DB connection
def get_db():
    return psycopg2.connect(
        dbname="spuriousCorrelationdb",
        user=os.getenv("db_user"),
        password=os.getenv("db_pass"),
        host="localhost"
    )

# Get sentiment from vLLM
def get_sentiment(text, skill_prompt):
    prompt = f"""{skill_prompt}

Article: {text[:5000]}

Sentiment:"""
    
    response = requests.post(VLLM_URL, json={
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "prompt": prompt,
        "max_tokens": 16000,
        "temperature": 0
    })
    
    result = response.json()["choices"][0]["text"].strip().upper()

    result = normalise_llm_output(result)

    if "BULLISH" in result:
        # print(result)
        # print("bull")
        return 1.0
    elif "BEARISH" in result:
        # print(result)
        # print("bear")
        return -1.0
    else:
        return 0.0

# Save batch to DB
def save_batch(conn, batch_data):
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO sentiment_results (news_id, sentiment) VALUES (%s, %s)",
        batch_data
    )
    conn.commit()
    cursor.close()

# Main
def main():
    # Load skill
    with open("financial-sentiment-skill.md", "r") as f:
        skill_prompt = f.read()
    
    # Load CSV
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=['webpage_content'])
    total = len(df)
    
    print(f"Total articles: {total}")
    
    conn = get_db()
    
    # Process in batches
    for i in range(0, total, BATCH_SIZE):
        batch = df.iloc[i:i+BATCH_SIZE]
        
        print(f"Processing batch {i//BATCH_SIZE + 1} ({i+1}-{min(i+BATCH_SIZE, total)})...")
        
        results = []
        for _, row in batch.iterrows():
            try:
                sentiment = get_sentiment(row['webpage_content'], skill_prompt)
                results.append((row['id'], sentiment))
            except Exception as e:
                print(f"  Error on article {row['id']}: {e}")
                continue
        
        # Save batch
        if results:
            save_batch(conn, results)
            print(f"  Saved {len(results)} results")
    
    conn.close()
    print("Done!")

if __name__ == "__main__":
    main()