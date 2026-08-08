from datetime import datetime

from bson import ObjectId
from pydantic import BaseModel, ConfigDict


class PyObjectId(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)


class AlertPreferences(BaseModel):
    score_threshold:float=30.0
    channels:list[str]=["email"]


class CustomerCreate(BaseModel):
    email:str
    name:str
    portfolio_value:float=0.0
    watchlist: list[str]=["AAPL","TSLA","NVDA"]
    risk_profile:str="moderate"
    password: str | None = None


class CustomerOut(BaseModel):
    id:str
    email:str
    name:str
    portfolio_value:float
    watchlist:list[str]
    risk_profile:str
    sentiment_score:float
    created_at:datetime

    model_config = ConfigDict(from_attributes=True)


class SignalOut(BaseModel):
    ticker:str
    source:str
    source_name:str
    title:str
    url:str
    published_at:str
    score:float
    label:str
    confidence:float
    age_hours:float


class SentimentOut(BaseModel):
    ticker:str
    composite:float
    label:str
    signal_count:int
    breakdown:dict
    history:list[dict]


class TradeCreate(BaseModel):
    ticker:str
    side:str
    price:float
    quantity:int
