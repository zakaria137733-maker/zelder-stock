from fastapi import APIRouter
from services import influx
from services.lstm_predictor import predict

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

TICKERS = ["AAPL","TSLA","NVDA","MSFT","GOOGL","AMZN","META"]

@router.get("/{ticker}")
async def get_prediction(ticker: str):
    ticker = ticker.upper()

    price_history = influx.query_price_history(ticker, hours=12)
    sent_history = influx.query_sentiment_history(ticker, hours=12)

    sent_map = {s["time"][:13]: s["value"] for s in sent_history}

    recent_data = []
    for p in price_history[-8:]:
        hour_key = p["time"][:13]
        recent_data.append({
            "price": p["price"],
            "sentiment": sent_map.get(hour_key,50.0)
        })

    return predict(ticker,recent_data)


@router.get("/")
async def get_all_predictions():
    results = []
    for ticker in TICKERS:
        price_history = influx.query_price_history(ticker, hours=12)
        sent_history = influx.query_sentiment_history(ticker, hours=12)
        sent_map = {s["time"][:13]: s["value"] for s in sent_history}
        recent_data = [
            {"price": p["price"], "sentiment": sent_map.get(p["time"][:13], 50.0)}
            for p in price_history[-8:]
        ]
        result = predict(ticker, recent_data)
        result["ticker"] = ticker
        results.append(result)
    return results