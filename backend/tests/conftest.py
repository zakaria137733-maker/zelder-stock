"""Shared fixtures for the ZelderStock test suite.

Tests run WITHOUT live Mongo / Influx / Redis. Routers call ``get_db()`` directly
(not via Depends), so we patch the router module references to point at an
in-memory fake database.
"""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import settings


class FakeCollection:
    """Minimal async Mongo-collection stand-in covering the ops used by the app."""

    def __init__(self, docs=None):
        self.docs = list(docs or [])
        self._next_id = 1
        self._limit = None

    async def find_one(self, filt):
        for d in self.docs:
            if all(d.get(k) == v for k, v in filt.items()):
                return dict(d)
        return None

    def find(self, *args, **kwargs):
        return self

    def sort(self, *args, **kwargs):
        return self

    def limit(self, n):
        self._limit = n
        return self

    async def to_list(self, n=None):
        return [dict(d) for d in self.docs[:n]]

    async def insert_one(self, doc):
        doc["_id"] = self._next_id
        self._next_id += 1
        self.docs.append(dict(doc))
        return SimpleNamespace(inserted_id=doc["_id"])

    async def update_one(self, filt, update, *args, **kwargs):
        for d in self.docs:
            if all(d.get(k) == v for k, v in filt.items()):
                for k, v in update.get("$set", {}).items():
                    d[k] = v
                return SimpleNamespace(modified_count=1)
        return SimpleNamespace(modified_count=0)

    async def delete_one(self, filt):
        for d in list(self.docs):
            if all(d.get(k) == v for k, v in filt.items()):
                self.docs.remove(d)
                return SimpleNamespace(deleted_count=1)
        return SimpleNamespace(deleted_count=0)


class FakeDB:
    def __init__(self, customers=None):
        self.customers = FakeCollection(customers or [])


def make_test_client(monkeypatch) -> TestClient:
    """Build a FastAPI app with only the routers under test and a fake DB."""
    from routers import auth, customers

    monkeypatch.setattr(settings, "jwt_secret", "test-jwt-secret")
    monkeypatch.setattr(settings, "admin_secret", "test-admin-secret")
    monkeypatch.setattr(settings, "admin_username", "admin")
    monkeypatch.setattr(settings, "admin_password", "admin-pass")

    db = FakeDB()
    monkeypatch.setattr(auth, "get_db", lambda: db)
    monkeypatch.setattr(customers, "get_db", lambda: db)

    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(auth.admin_router)
    app.include_router(customers.router)

    client = TestClient(app)
    client.db = db
    return client


@pytest.fixture
def client(monkeypatch):
    return make_test_client(monkeypatch)
