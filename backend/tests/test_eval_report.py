"""Admin eval-report endpoint: serves the committed eval_report.json behind admin auth."""

import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import settings


def _make_client(monkeypatch) -> TestClient:
    from routers import admin
    from services.auth import create_token

    monkeypatch.setattr(settings, "jwt_secret", "test-jwt-secret")
    monkeypatch.setattr(settings, "admin_username", "admin")
    monkeypatch.setattr(settings, "admin_password", "admin-pass")

    app = FastAPI()
    app.include_router(admin.router)
    client = TestClient(app)
    client.create_token = lambda: create_token({"sub": "admin", "name": "admin", "role": "admin"})
    return client


def test_eval_report_requires_admin(monkeypatch):
    client = _make_client(monkeypatch)
    r = client.get("/api/admin/eval/report")
    assert r.status_code == 401


def test_eval_report_returns_committed_report(monkeypatch):
    client = _make_client(monkeypatch)
    token = client.create_token()
    r = client.get("/api/admin/eval/report", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "eval_report.json"
    report = body["report"]
    assert report is not None
    for key in ("ticker", "windows", "accuracy", "baseline"):
        assert key in report, f"report missing {key}"
    assert isinstance(report["accuracy"], (int, float))
    assert body["caveats"], "expected caveat text alongside the report"


def test_eval_report_falls_back_when_missing(monkeypatch):
    client = _make_client(monkeypatch)
    from routers import admin

    monkeypatch.setattr(admin, "_models_dir", lambda: str(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "missing_dir")))
    token = client.create_token()
    r = client.get("/api/admin/eval/report", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert body["report"] is None
    assert body["source"] is None
    assert body["caveats"], "caveats should be returned even without a committed report"
