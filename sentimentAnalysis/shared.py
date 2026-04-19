"""Shared utilities for async LLM batch processing and database access."""

import asyncio
import os

import httpx
import psycopg2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

DATASETS = os.path.join(_ROOT, "datasets")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8001/v1/completions")
MODEL = os.getenv("VLLM_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")


def get_db():
    """Open a PostgreSQL connection using env-configured credentials."""
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "spuriousCorrelationdb"),
        user=os.getenv("db_user"),
        password=os.getenv("db_pass"),
        host=os.getenv("DB_HOST", "localhost"),
    )


def save_batch(conn, batch_data: list, table: str, columns: tuple):
    """Generic batch INSERT into `table` with the given column names."""
    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)
    cursor = conn.cursor()
    cursor.executemany(
        f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})",
        batch_data,
    )
    conn.commit()
    cursor.close()


async def run_batch_async(rows: list, skill_prompt: str, async_fn) -> list:
    """
    Two-pass async batch processor.

    Pass 1: fires all rows concurrently via asyncio.gather using `async_fn`.
    Pass 2: retries only rows whose reason starts with 'request_failed'.

    `async_fn` must accept (*row, skill_prompt, client) and return
    (id, value, reason).  `rows` may be 2- or 3-element tuples; they are
    unpacked with * so any arity works.
    """
    async with httpx.AsyncClient() as client:
        tasks = [async_fn(*row, skill_prompt, client) for row in rows]
        results = list(await asyncio.gather(*tasks))

        retry_indices = [
            i for i, (*_, reason) in enumerate(results)
            if reason.startswith("request_failed")
        ]
        if retry_indices:
            print(f"    Retrying {len(retry_indices)} failed requests...")
            retry_tasks = [async_fn(*rows[i], skill_prompt, client) for i in retry_indices]
            for idx, res in zip(retry_indices, await asyncio.gather(*retry_tasks)):
                results[idx] = res

        return results
