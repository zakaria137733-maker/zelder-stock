import secrets
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config import settings
from services.auth import create_token, get_current_user, hash_password, verify_password
from services.mongo import get_db

router = APIRouter(prefix="/api/auth", tags=["auth"])


class AdminLoginBody(BaseModel):
    username: str
    password: str


admin_router = APIRouter(prefix="/api/admin", tags=["admin"])


@admin_router.post("/login")
async def admin_login(body: AdminLoginBody):
    if not settings.admin_username or not settings.admin_password:
        raise HTTPException(503, "ADMIN_USERNAME / ADMIN_PASSWORD are not configured")
    if not secrets.compare_digest(body.username, settings.admin_username):
        raise HTTPException(401, "Invalid username or password")
    if not secrets.compare_digest(body.password, settings.admin_password):
        raise HTTPException(401, "Invalid username or password")

    token = create_token({"sub": "admin", "name": settings.admin_username, "role": "admin"})
    return {"token": token, "username": settings.admin_username, "role": "admin"}


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
        "created_at": datetime.now(UTC),
    }
    await db.customers.insert_one(doc)
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
async def me(user=Depends(get_current_user)):
    return user
