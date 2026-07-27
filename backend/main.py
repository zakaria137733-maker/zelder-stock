from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import sentiment, customers, transactions, signals, prices, alerts, predictions
from services import mongo
from routers import auth
import os



@asynccontextmanager
async def lifespan(app: FastAPI):
    db = mongo.get_db()
    await db.customers.create_index("email", unique=True)
    await db.customers.create_index("watchlist")
    await db.customers.create_index([("sentiment_score", -1)])
    print("ZelderStock API ready")
    yield
    await mongo.close()


app = FastAPI(
    title="ZelderStock",
    description="Stock market sentiment analysis platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://zelder-stock-f.onrender.com",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sentiment.router)
app.include_router(customers.router)
app.include_router(transactions.router)
app.include_router(signals.router)
app.include_router(prices.router)
app.include_router(alerts.router)
app.include_router(predictions.router)
app.include_router(auth.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ZelderStock"}


@app.get("/api/tickers")
async def list_tickers():
    return ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
