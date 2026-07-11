"""
Sentiment analysis service.

Uses VADER by default (fast, no GPU needed).
To upgrade to FinBERT, set USE_FINBERT=true in your .env —
it will lazy-load the model on first use (~3GB RAM, 30s startup).
"""
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()
_finbert = None

TICKER_ALIASES = {
    "apple": "AAPL", "iphone": "AAPL", "ipad": "AAPL", "tim cook": "AAPL",
    "tesla": "TSLA", "elon musk": "TSLA", "elon": "TSLA",
    "nvidia": "NVDA", "nvda": "NVDA", "jensen huang": "NVDA",
    "microsoft": "MSFT", "azure": "MSFT", "satya": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL", "gemini": "GOOGL",
    "amazon": "AMZN", "aws": "AMZN",
    "meta": "META", "facebook": "META", "instagram": "META",
}

VALID_TICKERS = {"AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"}

SOURCE_WEIGHTS = {
    "reuters": 1.0,
    "bloomberg": 1.0,
    "newsapi": 0.75,
    "reddit": 0.4,
    "twitter": 0.35,
}


def _load_finbert():
    global _finbert
    if _finbert is None:
        from transformers import pipeline
        print("Loading FinBERT model (first time, ~30s)...")
        _finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            return_all_scores=True,
        )
        print("FinBERT ready.")
    return _finbert


def extract_tickers(text: str) -> list[str]:
    found = set()
    lower = text.lower()

    # Match $AAPL style mentions
    for match in re.finditer(r"\$([A-Z]{1,5})", text):
        t = match.group(1)
        if t in VALID_TICKERS:
            found.add(t)

    # Match company name aliases
    for alias, ticker in TICKER_ALIASES.items():
        if alias in lower:
            found.add(ticker)

    return list(found)


def classify_sentiment(text: str) -> dict:
    use_finbert = os.getenv("USE_FINBERT", "false").lower() == "true"

    if use_finbert:
        model = _load_finbert()
        results = model(text[:512])[0]
        scores = {r["label"]: r["score"] for r in results}
        raw_score = scores.get("positive", 0) - scores.get("negative", 0)
        label = max(scores, key=scores.get)
        confidence = scores[label]
    else:
        vs = _vader.polarity_scores(text)
        raw_score = vs["compound"]  # -1 to +1
        if raw_score >= 0.05:
            label = "positive"
        elif raw_score <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        confidence = abs(raw_score)

    return {
        "score": round(raw_score, 4),   # -1 to +1
        "label": label,
        "confidence": round(confidence, 4),
    }


def compute_composite(signals: list[dict]) -> float:
    """
    Takes a list of {score, source, age_hours} dicts.
    Returns a 0–100 composite sentiment score.
    """
    if not signals:
        return 50.0

    total_weight = 0.0
    weighted_sum = 0.0

    for s in signals:
        age_hours = s.get("age_hours", 0)
        decay = max(0.1, 1 - age_hours / 72)
        weight = SOURCE_WEIGHTS.get(s.get("source", "newsapi"), 0.5) * decay
        # score is -1 to +1, map to 0–100 for composite
        mapped = (s["score"] + 1) / 2 * 100
        weighted_sum += mapped * weight
        total_weight += weight

    return round(weighted_sum / total_weight, 1)
