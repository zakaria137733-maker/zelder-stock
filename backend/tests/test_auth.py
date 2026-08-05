"""Authentication: register, login, /me, and the admin login endpoint."""

from config import settings
from services.auth import create_token


def test_register_returns_token(client):
    r = client.post("/api/auth/register", json={"name": "A", "email": "a@x.com", "password": "pw12345"})
    assert r.status_code == 200
    body = r.json()
    assert body["token"]
    assert body["email"] == "a@x.com"
    assert client.db.customers.docs[0]["password_hash"].startswith("$2")


def test_register_duplicate_email(client):
    client.post("/api/auth/register", json={"name": "A", "email": "a@x.com", "password": "pw12345"})
    r = client.post("/api/auth/register", json={"name": "B", "email": "a@x.com", "password": "pw12345"})
    assert r.status_code == 400


def test_login_success_and_bad_password(client):
    client.post("/api/auth/register", json={"name": "A", "email": "a@x.com", "password": "pw12345"})
    ok = client.post("/api/auth/login", json={"email": "a@x.com", "password": "pw12345"})
    assert ok.status_code == 200
    assert ok.json()["token"]
    bad = client.post("/api/auth/login", json={"email": "a@x.com", "password": "wrong"})
    assert bad.status_code == 401


def test_me_requires_token(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 401

    client.post("/api/auth/register", json={"name": "A", "email": "a@x.com", "password": "pw12345"})
    token = client.post("/api/auth/login", json={"email": "a@x.com", "password": "pw12345"}).json()["token"]
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["email"] == "a@x.com"


def test_me_rejects_invalid_token(client):
    r = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_admin_login_success(client):
    r = client.post("/api/admin/login", json={"username": "admin", "password": "admin-pass"})
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "admin"
    # token must be accepted as a normal JWT too
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {body['token']}"})
    assert me.status_code == 200


def test_admin_login_wrong_password(client):
    r = client.post("/api/admin/login", json={"username": "admin", "password": "nope"})
    assert r.status_code == 401


def test_admin_login_wrong_username(client):
    r = client.post("/api/admin/login", json={"username": "root", "password": "admin-pass"})
    assert r.status_code == 401


def test_admin_login_unconfigured(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_password", "")
    r = client.post("/api/admin/login", json={"username": "admin", "password": "admin-pass"})
    assert r.status_code == 503


def test_register_uses_uppercase_watchlist(client):
    r = client.post("/api/auth/register", json={
        "name": "A", "email": "b@x.com", "password": "pw12345", "watchlist": ["aapl", "nvda"],
    })
    assert r.status_code == 200
    assert client.db.customers.docs[0]["watchlist"] == ["AAPL", "NVDA"]


def test_create_token_requires_secret(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", "")
    try:
        create_token({"sub": "x"})
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
