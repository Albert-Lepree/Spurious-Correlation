# FINANCIAL SENTIMENT SCORING SKILL

## OBJECTIVE
Score financial news articles on a scale from 0 to 100 based on expected market impact.
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

## SCORING RULES

1. **Prioritize immediate market impact** over long-term speculation.
2. **Earnings surprises override** general economic sentiment.
3. **Fed policy changes** are always directional — dovish maps to 70+, hawkish maps to 30 or below.
4. **When signals are mixed**, weight the most market-moving element and land in the 40-59 range.
5. **50 is NOT a default** — use it only when bullish and bearish forces genuinely cancel each other out.
6. **Magnitude matters** — a small earnings beat is 62, a massive blowout quarter is 88.

## EXAMPLES

| Article Summary | Score |
|----------------|-------|
| Apple beats Q4 estimates by 15%, raises full-year guidance | 87 |
| Fed signals dovish pivot, rate cuts likely next meeting | 82 |
| Tesla deliveries surge 40% year-over-year, record quarter | 90 |
| Company reports modest earnings beat, inline guidance | 63 |
| Retail sales rise slightly, in line with forecasts | 55 |
| CEO reiterates 2024 guidance at investor conference | 50 |
| Company announces routine board meeting scheduled | 50 |
| Earnings miss by 5%, guidance maintained | 38 |
| Meta announces 11,000 layoffs amid revenue decline | 18 |
| Manufacturing PMI falls to 46.2, signaling contraction | 32 |
| Bank stocks plunge on credit crisis fears, systemic risk | 8 |
| Major retailer files for bankruptcy, stores closing | 4 |

## OUTPUT FORMAT

Respond with ONLY a single integer between 0 and 100.
No explanation. No units. No preamble. Just the number.
