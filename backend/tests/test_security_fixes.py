"""Regression tests for the security hardening pass."""

import pytest

from config import settings
from services.auth import hash_password


def _seed(client, email="a@x.com", name="A"):
    client.db.customers.docs.append({
        "_id": 1, "email": email, "name": name,
        "password_hash": hash_password("pw12345"), "watchlist": [],
    })


def test_list_does_not_leak_password_hash(client):
    _seed(client)
    r = client.get("/api/customers/", headers={"X-Admin-Key": "test-admin-secret"})
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert "password_hash" not in r.json()[0]


def test_get_own_customer_does_not_leak_password_hash(client):
    from services.auth import create_token

    _seed(client)
    token = {"Authorization": f"Bearer {create_token({'sub': 'a@x.com', 'name': 'A'})}"}
    r = client.get("/api/customers/1", headers=token)
    assert r.status_code == 200
    assert "password_hash" not in r.json()


def test_admin_created_customer_can_log_in(client):
    created = client.post("/api/customers/", json={"name": "A", "email": "pw@x.com"},
                          headers={"X-Admin-Key": "test-admin-secret"})
    assert created.status_code == 200
    body = created.json()
    assert "password_hash" not in body
    assert body["generated_password"]
    assert client.db.customers.docs[0]["password_hash"].startswith("$2")

    login = client.post("/api/auth/login", json={"email": "pw@x.com", "password": body["generated_password"]})
    assert login.status_code == 200
    assert login.json()["token"]


def test_login_rate_limited(monkeypatch):
    from fastapi import HTTPException

    from routers import auth as auth_router

    monkeypatch.setattr(settings, "auth_rate_limit", 2)
    # Direct unit check with a unique key, so the shared in-process bucket
    # (per-IP state from other tests) cannot affect this test.
    auth_router._check_rate_limit("unit", "key-a")
    auth_router._check_rate_limit("unit", "key-a")
    with pytest.raises(HTTPException) as exc:
        auth_router._check_rate_limit("unit", "key-a")
    assert exc.value.status_code == 429
