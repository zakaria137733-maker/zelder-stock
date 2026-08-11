from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from bson import ObjectId
from services.mongo import get_db
from models.schemas import CustomerCreate

router = APIRouter(prefix="/api/customers", tags=["customers"])


def _serialize(doc:dict)->dict:
    doc["id"] = str(doc.pop("_id"))
    return doc


@router.get("/")
async def list_customers(limit:int=50):
    db = get_db()
    docs = await db.customers.find().sort("sentiment_score",-1).limit(limit).to_list(limit)
    return [_serialize(d) for d in docs]


@router.post("/")
async def create_customer(body: CustomerCreate):
    db = get_db()
    existing = await db.customers.find_one({"email": body.email})
    if existing:
        raise HTTPException(400, "Email already registered")

    doc = {
        **body.model_dump(),
        "sentiment_score":50.0,
        "created_at":datetime.now(timezone.utc),
    }
    result=await db.customers.insert_one(doc)
    doc["_id"]=result.inserted_id
    return _serialize(doc)


@router.get("/{customer_id}")
async def get_customer(customer_id:str):
    db=get_db()
    doc=await db.customers.find_one({"_id":ObjectId(customer_id)})
    if not doc:
        raise HTTPException(404,"Customer not found")
    return _serialize(doc)


@router.patch("/{customer_id}/watchlist")
async def update_watchlist(customer_id: str,tickers:list[str]):
    db = get_db()
    await db.customers.update_one(
        {"_id": ObjectId(customer_id)},
        {"$set": {"watchlist": [t.upper() for t in tickers]}}
    )
    return {"ok":True}


@router.delete("/{customer_id}")
async def delete_customer(customer_id:str):
    db=get_db()
    await db.customers.delete_one({"_id":ObjectId(customer_id)})
    return {"ok":True}
