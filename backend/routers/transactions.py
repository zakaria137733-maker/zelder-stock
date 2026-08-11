from fastapi import APIRouter
from services import influx
from models.schemas import TradeCreate

router=APIRouter(prefix="/api/transactions",tags=["transactions"])


@router.get("/{ticker}")
async def get_trades(ticker:str,limit:int=20):
    ticker=ticker.upper()
    trades=influx.query_recent_trades(ticker, limit)
    return {"ticker":ticker,"trades":trades}


@router.post("/")
async def record_trade(body:TradeCreate):
    influx.write_trade(
        ticker=body.ticker.upper(),
        side=body.side,
        price=body.price,
        quantity=body.quantity,
        customer_id=body.customer_id,
    )
    return {"ok":True,"total_usd":round(body.price*body.quantity,2)}
