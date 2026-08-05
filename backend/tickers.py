from fastapi import HTTPException

TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]


def validate_ticker(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not tracked")
    return ticker
