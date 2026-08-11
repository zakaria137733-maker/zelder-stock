from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.auth import hash_password, verify_password, create_token
from services.mongo import get_db
from datetime import datetime, timezone

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterBody(BaseModel):
    name: str
    email: str
    password: str
    portfolio_value: float = 0.0
    risk_profile: str = "moderate"
    watchlist: list[str] = ["AAPL", "NVDA", "TSLA"]


class LoginBody(BaseModel):
    email: str
    password: str


@router.post("/register")
async def register(body: RegisterBody):
    db = get_db()
    existing = await db.customers.find_one({"email": body.email})
    if existing:
        raise HTTPException(400, "Email already registered")

    doc = {
        "name": body.name,
        "email": body.email,
        "password_hash": hash_password(body.password),
        "portfolio_value": body.portfolio_value,
        "risk_profile": body.risk_profile,
        "watchlist": [t.upper() for t in body.watchlist],
        "sentiment_score": 50.0,
        "created_at": datetime.now(timezone.utc),
    }
    result = await db.customers.insert_one(doc)
    token = create_token({"sub": body.email, "name": body.name})
    return {"token": token, "name": body.name, "email": body.email}


@router.post("/login")
async def login(body: LoginBody):
    db = get_db()
    user = await db.customers.find_one({"email": body.email})
    if not user:
        raise HTTPException(401, "Invalid email or password")
    if not user.get("password_hash"):
        raise HTTPException(401, "Account has no password — register first")
    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")

    token = create_token({"sub": user["email"], "name": user["name"]})
    return {
        "token": token,
        "name": user["name"],
        "email": user["email"],
        "watchlist": user.get("watchlist", []),
        "risk_profile": user.get("risk_profile", "moderate"),
    }


@router.get("/me")
async def me(token: str):
    from services.auth import decode_token
    payload = decode_token(token)
    return {"email": payload.get("sub"), "name": payload.get("name")}