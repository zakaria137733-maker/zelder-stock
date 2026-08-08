import asyncio

from fastapi import APIRouter, Depends

from services.auth import get_current_user
from services.lstm_predictor import fetch_live_daily_context, predict_ensemble
from tickers import TICKERS, validate_ticker

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


def _predict_sync(ticker: str) -> dict:
    recent_data = fetch_live_daily_context(ticker)
    result = predict_ensemble(ticker, recent_data)
    result["ticker"] = ticker
    result["window_granularity"] = "daily"
    return result


@router.get("/{ticker}")
async def get_prediction(ticker: str, user=Depends(get_current_user)):
    ticker = validate_ticker(ticker)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _predict_sync, ticker)
    return result


@router.get("/")
async def get_all_predictions(user=Depends(get_current_user)):
    loop = asyncio.get_event_loop()
    results = await asyncio.gather(
        *(loop.run_in_executor(None, _predict_sync, ticker) for ticker in TICKERS)
    )
    return list(results)
