import secrets
from datetime import UTC, datetime, timedelta

import bcrypt
from fastapi import Depends, Header, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from config import settings

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
BCRYPT_ROUNDS = 12

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def _get_secret() -> str:
    if not settings.jwt_secret:
        raise RuntimeError("JWT_SECRET is not set in the environment or .env — refusing to start with an empty secret")
    return settings.jwt_secret


def hash_password(password: str) -> str:
    """bcrypt directly (passlib is unmaintained and breaks with bcrypt>=4.1)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(UTC) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    return jwt.encode(payload, _get_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _get_secret(), algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from None


async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    email = payload.get("sub")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"email": email, "name": payload.get("name")}


async def require_admin(
    x_admin_key: str = Header(default=""),
    authorization: str = Header(default=""),
):
    key_configured = bool(settings.admin_secret)
    password_configured = bool(settings.admin_username and settings.admin_password)
    if not key_configured and not password_configured:
        raise HTTPException(status_code=503, detail="No admin credentials configured (ADMIN_SECRET or ADMIN_USERNAME/ADMIN_PASSWORD)")

    if key_configured and secrets.compare_digest(x_admin_key, settings.admin_secret):
        return True

    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            payload = decode_token(token)
            if payload.get("role") == "admin":
                return True

    raise HTTPException(status_code=401, detail="Invalid admin credentials")
