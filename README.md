# Spurious Correlation

## Purpose

This project investigates whether spurious or sensationalist news headlines carry a measurable sentiment signal that correlates with subsequent equity market returns. The pipeline ingests real financial news from a Databricks SQL warehouse and a Kaggle dataset of spurious headlines, scores both corpora for sentiment using a locally-hosted vLLM inference server (NVIDIA Nemotron), extracts latent topic features via Latent Dirichlet Allocation, and encodes article text as dense sentence embeddings. All features are joined with daily SPX, NDX, and VIX market data to produce a final analysis-ready Parquet dataset.

---

## Directory Structure

```
Spurious-Correlation/
├── main.py                          # CLI pipeline runner (--stage / --all)
├── requirements.txt                 # Python dependencies
├── .env.example                     # Required environment variable template
├── compileMasterData/
│   ├── build_master_base.py         # Joins news + market data into master_base.parquet
│   └── add_control_features.py      # Adds lagged returns and rolling volatility features
├── databricksIngestion/
│   ├── databricks_ingest.py         # Databricks SQL connector
│   └── databricks_seed.json         # SQL query definitions
├── sentimentAnalysis/
│   ├── shared.py                    # Shared async batch processor, DB helpers, constants
│   ├── extractSentiment.py          # Binary BULLISH/BEARISH/NEUTRAL → PostgreSQL
│   ├── extractSentimentScore.py     # Numeric 0-100 score for real news → PostgreSQL
│   ├── extractSpuriousSentiment.py  # Numeric 0-100 score for spurious news → CSV
│   ├── financial-sentiment-skill.md       # LLM prompt: binary classification
│   ├── financial-sentiment-score-skill.md # LLM prompt: numeric scoring
│   └── spurious-sentiment-score-skill.md  # LLM prompt: spurious headline scoring
├── lda/
│   └── extractTopics.py             # Scikit-learn LDA (10 topics) → master_with_lda.parquet
├── embeddings/
│   └── extractEmbeddings.py         # Sentence-transformers encoding → master_with_embeddings.parquet
└── datasets/                        # Generated data files (git-ignored)
    ├── spurious_news.csv            # Kaggle spurious headlines (input)
    ├── real_news.csv                # News from Databricks (input)
    ├── spurious_news_scored.csv     # Spurious news with sentiment scores
    ├── spx_data.csv                 # S&P 500 daily OHLCV
    ├── ndx_data.csv                 # Nasdaq-100 daily OHLCV
    ├── vix_data.csv                 # CBOE VIX daily values
    ├── master_base.parquet          # Combined news + market data
    ├── master_with_control.parquet  # + lagged features
    ├── master_with_lda.parquet      # + 10 LDA topic columns
    └── master_with_embeddings.parquet  # + 768-dim sentence embeddings (final)
```

---

## Setup

### Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Sentence-transformers will download the `all-mpnet-base-v2` model (~420 MB) on first run. A GPU is strongly recommended for the embeddings stage.

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABRICKS_SERVER_HOSTNAME` | Databricks workspace hostname | — (required) |
| `DATABRICKS_HTTP_PATH` | SQL warehouse HTTP path | — (required) |
| `DATABRICKS_TOKEN` | Databricks personal access token | — (required) |
| `db_user` | PostgreSQL username | — (required) |
| `db_pass` | PostgreSQL password | — (required) |
| `DB_HOST` | PostgreSQL host | `localhost` |
| `DB_NAME` | PostgreSQL database name | `spuriousCorrelationdb` |
| `VLLM_URL` | vLLM completions endpoint | `http://localhost:8001/v1/completions` |
| `VLLM_MODEL` | Model name served by vLLM | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| `EMBEDDING_MODEL` | Sentence-transformer model | `sentence-transformers/all-mpnet-base-v2` |
| `DATABRICKS_END_DATE` | End date for Databricks news ingestion | `2026-03-15` |

### External Services Required

- **PostgreSQL** — local or remote instance with the `spuriousCorrelationdb` database created.
- **vLLM server** — running `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` on port 8001 (or override with `VLLM_URL`/`VLLM_MODEL`). Example launch:
  ```bash
  vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --port 8001
  ```
- **Databricks workspace** — SQL warehouse accessible with the credentials above (required for the `ingest` stage only).

---

## Running the Pipeline

Stages must be run in order. Each stage can be run individually or all at once:

```bash
# Run a single stage
python main.py --stage ingest
python main.py --stage spurious-sentiment
python main.py --stage sentiment
python main.py --stage sentiment-score
python main.py --stage compile
python main.py --stage lda
python main.py --stage embeddings

# Run the full pipeline end-to-end
python main.py --all

# See all options
python main.py --help
```

---

## Data Flow

Each stage consumes the output of the previous stage.

| Stage | Consumes | Produces |
|-------|----------|----------|
| `ingest` | Databricks SQL warehouse | `datasets/real_news.csv` |
| `spurious-sentiment` | `datasets/spurious_news.csv` | `datasets/spurious_news_scored.csv` |
| `sentiment` | `datasets/real_news.csv` | PostgreSQL `sentiment_results` table |
| `sentiment-score` | `datasets/real_news.csv` | PostgreSQL `sentiment_score_results` table |
| `compile` | `datasets/spurious_news_scored.csv`<br>`datasets/real_news.csv` + PostgreSQL scores<br>`datasets/spx_data.csv`, `datasets/vix_data.csv` | `datasets/master_base.parquet`<br>`datasets/master_with_control.parquet` |
| `lda` | `datasets/spurious_news.csv`, `datasets/real_news.csv` | `datasets/master_with_lda.parquet` |
| `embeddings` | `datasets/master_with_lda.parquet` | `datasets/master_with_embeddings.parquet` |

> **Note:** Market data CSVs (`spx_data.csv`, `ndx_data.csv`, `vix_data.csv`) are static inputs and must be present in `datasets/` before running the `compile` stage. They are not generated by the pipeline.

---

## Known Issues / Technical Debt

See the codebase review notes for a full list. High-priority items:

- `sentimentAnalysis/extractSentimentScore.py` line 148: loop starts at row 700 due to a hardcoded debug offset — first 700 articles are never scored.
- All three sentiment scripts use a hardcoded vLLM URL and model name; override via `VLLM_URL` / `VLLM_MODEL` env vars after the planned refactor.
- `databricksIngestion/databricks_seed.json` contains a hardcoded end date — override via `DATABRICKS_END_DATE`.
- No `requirements.txt` was present originally; pinned versions in the current file are minimum bounds and may need tightening for reproducibility.
