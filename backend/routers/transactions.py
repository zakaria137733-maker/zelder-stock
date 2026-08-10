import asyncio

from fastapi import APIRouter, Depends, HTTPException

from models.schemas import TradeCreate
from services import influx
from services.auth import get_current_user
from services.mongo import get_db
from tickers import validate_ticker

router=APIRouter(prefix="/api/transactions",tags=["transactions"])


@router.get("/{ticker}")
async def get_trades(ticker:str, limit:int=20, user=Depends(get_current_user)):
    ticker=validate_ticker(ticker)
    db = get_db()
    customer = await db.customers.find_one({"email": user["email"]})
    if not customer:
        raise HTTPException(404, "Customer account not found")
    customer_id = str(customer["_id"])
    loop = asyncio.get_event_loop()
    trades = await loop.run_in_executor(
        None, influx.query_recent_trades, ticker, limit, customer_id
    )
    return {"ticker":ticker,"trades":trades}


@router.post("/")
async def record_trade(body:TradeCreate, user=Depends(get_current_user)):
    ticker = validate_ticker(body.ticker)
    side = body.side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(422, "side must be BUY or SELL")
    db = get_db()
    customer = await db.customers.find_one({"email": user["email"]})
    if not customer:
        raise HTTPException(404, "Customer account not found")
    customer_id = str(customer["_id"])

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, influx.write_trade,
        ticker, side, body.price, body.quantity, customer_id,
    )
    return {"ok":True,"total_usd":round(body.price*body.quantity,2),"customer_id":customer_id}
