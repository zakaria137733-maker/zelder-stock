import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException

from models.schemas import CustomerCreate
from services.auth import get_current_user, hash_password, require_admin
from services.mongo import get_db

router = APIRouter(prefix="/api/customers", tags=["customers"])


def _serialize(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    doc.pop("password_hash", None)
    return doc


async def _get_own_customer(user: dict) -> dict:
    db = get_db()
    doc = await db.customers.find_one({"email": user["email"]})
    if not doc:
        raise HTTPException(404, "Customer account not found")
    return doc


@router.get("/")
async def list_customers(_admin=Depends(require_admin), limit:int=50):
    db = get_db()
    docs = await db.customers.find().sort("sentiment_score",-1).limit(limit).to_list(limit)
    return [_serialize(d) for d in docs]


@router.post("/")
async def create_customer(body: CustomerCreate, _admin=Depends(require_admin)):
    db = get_db()
    existing = await db.customers.find_one({"email": body.email})
    if existing:
        raise HTTPException(400, "Email already registered")

    password = body.password or secrets.token_urlsafe(12)
    doc = {
        **body.model_dump(exclude={"password"}),
        "password_hash": hash_password(password),
        "sentiment_score": 50.0,
        "created_at": datetime.now(UTC),
    }
    result = await db.customers.insert_one(doc)
    doc["_id"] = result.inserted_id
    response = _serialize(doc)
    if not body.password:
        response["generated_password"] = password
    return response


@router.get("/{customer_id}")
async def get_customer(customer_id:str, user=Depends(get_current_user)):
    return _serialize(await _get_own_customer(user))


@router.patch("/{customer_id}/watchlist")
async def update_watchlist(customer_id: str, tickers:list[str], user=Depends(get_current_user)):
    db = get_db()
    own = await _get_own_customer(user)
    await db.customers.update_one(
        {"_id": own["_id"]},
        {"$set": {"watchlist": [t.upper() for t in tickers]}}
    )
    return {"ok":True}


@router.delete("/{customer_id}")
async def delete_customer(customer_id:str, user=Depends(get_current_user)):
    db=get_db()
    own = await _get_own_customer(user)
    await db.customers.delete_one({"_id": own["_id"]})
    return {"ok":True}
