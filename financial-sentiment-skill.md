# FINANCIAL SENTIMENT CLASSIFICATION SKILL

## OBJECTIVE
Classify financial news articles as BULLISH, BEARISH, or NEUTRAL based on market impact.

## DEFINITIONS

**BULLISH** = Positive for stock prices. Likely to drive markets UP.
**BEARISH** = Negative for stock prices. Likely to drive markets DOWN.
**NEUTRAL** = No clear directional impact OR mixed signals.

## BULLISH INDICATORS

### Earnings & Revenue
- Beats expectations, raised guidance, record profits
- Strong revenue growth, margin expansion
- "Better than expected", "outperformed", "exceeded forecasts"

### Economic Data
- Strong GDP growth, falling unemployment
- Rising consumer confidence, retail sales growth
- Manufacturing expansion, PMI above 50

### Corporate Actions
- Buyback announcements, dividend increases
- Strategic acquisitions, partnerships
- New product launches with strong demand

### Market Conditions
- Rate cuts, accommodative Fed policy
- Falling inflation, stable yields
- Strong institutional buying, insider purchases

### Keywords
- surge, rally, breakout, all-time high
- upgrade, outperform, buy rating
- growth acceleration, market share gains
- robust demand, pricing power

## BEARISH INDICATORS

### Earnings & Revenue
- Misses expectations, lowered guidance, losses
- Declining margins, revenue contraction
- "Disappointing", "below estimates", "warning"

### Economic Data
- Recession fears, rising unemployment
- Weak consumer spending, falling confidence
- Manufacturing contraction, PMI below 50

### Corporate Actions
- Layoffs, restructuring, plant closures
- Failed mergers, executive departures
- Product recalls, regulatory investigations

### Market Conditions
- Rate hikes, hawkish Fed stance
- Rising inflation, yield spike
- Heavy selling, margin calls, liquidations

### Keywords
- plunge, crash, selloff, correction
- downgrade, underperform, sell rating
- slowdown, market share loss
- weak demand, pricing pressure

## NEUTRAL INDICATORS

### Mixed Signals
- Some metrics up, others down
- Meets expectations exactly (no surprise)
- Offsetting factors (good sales but rising costs)

### Non-Market Events
- Routine announcements, scheduled events
- Analyst reiterations (no change)
- Technical updates without financial impact

### Ambiguous Language
- "Could", "may", "monitoring situation"
- Speculative without concrete data
- Long-term outlook with no immediate catalyst

## CLASSIFICATION RULES

1. **Prioritize immediate market impact** over long-term speculation
2. **Earnings surprises override** general market sentiment
3. **Fed policy changes** are always directional (not neutral)
4. **When conflicted:** Look at headline + first paragraph. What would traders react to?
5. **Single word response:** BULLISH, BEARISH, or NEUTRAL

## EXAMPLES

**BULLISH:**
- "Apple beats Q4 estimates, raises guidance for holiday quarter"
- "Fed signals dovish pivot, rate cuts on the table"
- "Tesla deliveries surge 40% year-over-year"

**BEARISH:**
- "Meta announces 11,000 layoffs amid revenue decline"
- "Manufacturing PMI falls to 46.2, signaling contraction"
- "Bank stocks plunge on credit crisis fears"

**NEUTRAL:**
- "CEO reiterates 2024 guidance at investor conference"
- "Company announces routine board meeting scheduled"
- "Analysts debate long-term EV adoption timeline"

## OUTPUT FORMAT

Respond with ONLY one word: BULLISH, BEARISH, or NEUTRAL.
No explanation. No preamble. Just the classification.