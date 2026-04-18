# SPURIOUS SENTIMENT SCORING SKILL

## DIRECTIVE
You will be given a headline and a sentiment flag (Positive/Negative/Neutral).
**The headline takes priority. Always.**
Use the flag as a tiebreaker only when the headline is genuinely ambiguous.
If the headline clearly conveys a bullish or bearish tone, score it accordingly — ignore the flag if it contradicts.
Every headline is different. Every score must reflect that.

## SCORING BANDS

| Range | Label | Meaning |
|-------|-------|---------|
| 80–100 | Strongly Bullish | Record results, massive beats, Fed cuts, breakthrough launches |
| 60–79 | Moderately Bullish | Modest beats, positive data, buybacks, upgrades |
| 40–59 | Neutral / Mixed | In-line results, routine news, ambiguous signals |
| 20–39 | Moderately Bearish | Misses, layoffs, weak data, downgrades |
| 0–19 | Strongly Bearish | Bankruptcy, collapse, systemic crisis, mass closures |

## BAND CONSTRAINTS

Use the flag's band only when the headline is ambiguous. When the headline clearly signals a direction, use the full scoring range (0–100) based on the headline.

| Flag | Use this band IF headline is ambiguous |
|------|----------------------------------------|
| Positive | 60–100 |
| Negative | 20–40 |
| Neutral or blank | 42–58 |

**Examples of contradiction — headline wins:**
- "New IPO oversubscribed" + Negative flag → score 70–80 (bullish headline overrides)
- "Mining company surges on commodity boom" + Negative flag → score 72–82 (bullish headline overrides)
- "Major oil spill sends energy stocks plummeting" + Positive flag → score 10–25 (bearish headline overrides)

**0–19 is reserved for genuine catastrophe (bankruptcy, collapse, systemic failure). Do not assign 0 unless the headline explicitly describes one of these.**

## VARIANCE RULE
This is critical: **do not output the same number repeatedly**. Each headline has a different magnitude. A mildly positive headline might be 63. A strongly positive one might be 84. A mildly negative one might be 37. A strongly negative one might be 22. Use the full width of the allowed band — don't anchor to a single value.

## MAGNITUDE GUIDANCE

**Positive band (60–100):**
- 60–65: Vague positive tone, no concrete catalyst ("sector expected to benefit")
- 66–74: Moderate concrete positive (buyback, subsidy, modest beat)
- 75–84: Strong positive (significant acquisition, major beat, rate cut signal)
- 85–100: Exceptional (record profits, massive guidance raise, crisis resolution)

**Negative band (20–40):**
- 36–40: Mild headwind, uncertainty, soft data
- 28–35: Concrete negative (layoffs, missed estimates, weak PMI)
- 20–27: Severe negative (major losses, regulatory shutdown, failed deal)
- 0–19: ONLY for bankruptcy, collapse, systemic crisis — use sparingly

**Neutral band (42–58):**
- 42–45: Slightly negative tilt within neutral
- 46–54: Genuinely mixed or routine
- 55–58: Slightly positive tilt within neutral

## RULES
1. **Headline takes priority** — if headline and flag contradict, score based on the headline
2. Flag is a tiebreaker only when the headline is genuinely ambiguous
3. No score of 0 unless the headline is genuinely catastrophic (bankruptcy, collapse, mass closure)
4. No score of 44 or 50 as a default — actually read the headline
5. Vary your scores — consecutive identical scores are wrong

## OUTPUT FORMAT
Single integer only. No explanation. No preamble. No units.