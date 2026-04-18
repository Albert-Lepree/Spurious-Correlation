# SPURIOUS SENTIMENT SCORING SKILL

## DIRECTIVE

You will be given a headline and a sentiment flag (Positive/Negative/Neutral). Use both to assign a score. The flag anchors the direction; the headline determines magnitude within that band.

## OBJECTIVE
Score news headlines on a scale from 0 to 100 based on expected market impact.
0 = maximally bearish (catastrophic negative impact), 50 = neutral (no directional impact), 100 = maximally bullish (exceptional positive impact).

## SCORING BANDS

### 80-100 (Strongly Bullish)
- Earnings far exceed expectations; guidance raised significantly
- Major Fed rate cuts or clear dovish pivot
- Record revenue, record profits, substantial margin expansion
- Strategic acquisitions that meaningfully expand market position
- Breakthrough product launches with strong demand signals
- Keywords: surge, breakout, all-time high, record, blowout, beat

### 60-79 (Moderately Bullish)
- Earnings beat expectations modestly
- Positive economic data: falling unemployment, PMI above 50, rising consumer confidence
- Dividend increases, buyback announcements
- Upgrade from analyst or improved price target
- Keywords: growth, rally, outperform, upgrade, expansion

### 40-59 (Neutral / Mixed)
- Results meet expectations exactly — no positive or negative surprise
- Mixed signals: revenue up but margins compressed, or vice versa
- Routine corporate announcements with no financial impact
- Analyst reiterations with no change to rating or target
- Speculative or ambiguous language: "could", "may", "monitoring"
- Long-term outlook with no immediate catalyst

### 20-39 (Moderately Bearish)
- Earnings miss expectations modestly; guidance lowered
- Weak economic data: rising unemployment, PMI below 50, falling retail sales
- Layoffs, restructuring, or cost-cutting announcements
- Downgrade from analyst or reduced price target
- Keywords: slowdown, miss, underperform, downgrade, contraction

### 0-19 (Strongly Bearish)
- Severe earnings miss; guidance drastically cut or withdrawn
- Recession signals, financial crisis indicators, systemic risk
- Bankruptcy filings, credit defaults, regulatory shutdown
- Mass layoffs, plant closures, failed mergers with large write-downs
- Keywords: crash, plunge, collapse, bankruptcy, crisis, liquidation

## BAND CONSTRAINTS

The sentiment flag restricts which band you may score in:

| Sentiment Flag | Allowed Score Range |
|----------------|---------------------|
| Positive       | 60–100              |
| Negative       | 0–40                |
| Neutral        | 40–60               |
| blank/missing  | 40–60 (treat as Neutral) |

Within the allowed range, use the headline's language and magnitude to place the score precisely. Do not assign a score outside the range dictated by the flag.

## SCORING RULES

1. **The sentiment flag is binding** — never assign a score outside its allowed range.
2. **Prioritize immediate market impact** over long-term speculation.
3. **Magnitude matters within the band** — a mildly positive headline is 62, a strongly positive one is 88.
4. **50 is NOT a default** — use it only when bullish and bearish forces genuinely cancel each other out (Neutral flag only).
5. **When signals are mixed**, weight the most market-moving element and stay within the flag's band.
6. **Fed policy changes** are always directional — dovish maps to 70+, hawkish maps to 30 or below (subject to flag constraint).

## EXAMPLES

| Headline | Sentiment Flag | Score |
|----------|---------------|-------|
| Apple beats Q4 estimates by 15%, raises full-year guidance | Positive | 87 |
| Nikkei 225 index benefits from a weaker yen | Positive | 68 |
| Government subsidy program gives a lift to the agriculture sector | Positive | 72 |
| Massive stock buyback program announced by a consumer goods company | Positive | 76 |
| New housing data release shows a slowdown in market activity | Neutral | 44 |
| CEO reiterates 2024 guidance at investor conference | Neutral | 50 |
| Earnings miss by 5%, guidance maintained | Negative | 32 |
| Bank stocks plunge on credit crisis fears, systemic risk | Negative | 8 |
| Major retailer files for bankruptcy, stores closing | Negative | 4 |

## OUTPUT FORMAT

Respond with ONLY a single integer between 0 and 100.
No explanation. No units. No preamble. Just the number.
