"""record_trade validates ticker + side before writing to Influx.

Local TestClient wiring only the transactions router against a FakeDB, so no
live Mongo / Influx is needed. Mirrors the auth pattern from
test_security_fixes.py (create_token({'sub': email})).
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import settings
from services.auth import create_token

from .conftest import FakeDB


def _client(monkeypatch, writes):
    from routers import transactions

    monkeypatch.setattr(settings, "jwt_secret", "test-jwt-secret")

    db = FakeDB(customers=[{"_id": 1, "email": "a@x.com", "name": "A"}])
    monkeypatch.setattr(transactions, "get_db", lambda: db)

    def fake_write_trade(*args, **kwargs):
        writes.append((args, kwargs))

    monkeypatch.setattr("services.influx.write_trade", fake_write_trade)

    app = FastAPI()
    app.include_router(transactions.router)
    client = TestClient(app)
    client.db = db
    return client


def _auth(email="a@x.com"):
    return {"Authorization": f"Bearer {create_token({'sub': email})}"}


def test_valid_trade_uppercases_and_writes(monkeypatch):
    writes = []
    client = _client(monkeypatch, writes)
    r = client.post("/api/transactions/", json={
        "ticker": "aapl", "side": "buy", "price": 150.0, "quantity": 4,
    }, headers=_auth())
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert len(writes) == 1
    args, _kwargs = writes[0]
    assert args[0] == "AAPL"
    assert args[1] == "BUY"


def test_unknown_ticker_404_no_write(monkeypatch):
    writes = []
    client = _client(monkeypatch, writes)
    r = client.post("/api/transactions/", json={
        "ticker": "BTC", "side": "buy", "price": 150.0, "quantity": 4,
    }, headers=_auth())
    assert r.status_code == 404
    assert writes == []


def test_invalid_side_422_no_write(monkeypatch):
    writes = []
    client = _client(monkeypatch, writes)
    r = client.post("/api/transactions/", json={
        "ticker": "AAPL", "side": "HODL", "price": 150.0, "quantity": 4,
    }, headers=_auth())
    assert r.status_code == 422
    assert writes == []
