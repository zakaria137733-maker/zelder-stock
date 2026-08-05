"""Admin authorization: X-Admin-Key header and admin-JWT Bearer both work; regular users are rejected."""

from services.auth import create_token, hash_password


def _seed(client, email="a@x.com", name="A"):
    client.db.customers.docs.append({
        "_id": 1, "email": email, "name": name,
        "password_hash": hash_password("pw12345"), "watchlist": [],
    })


def _user_token(email="a@x.com"):
    return create_token({"sub": email, "name": "A"})


def test_list_requires_admin(client):
    _seed(client)
    assert client.get("/api/customers/").status_code == 401


def test_list_with_x_admin_key(client):
    _seed(client)
    r = client.get("/api/customers/", headers={"X-Admin-Key": "test-admin-secret"})
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_list_with_wrong_admin_key(client):
    _seed(client)
    assert client.get("/api/customers/", headers={"X-Admin-Key": "wrong"}).status_code == 401


def test_list_with_admin_bearer_token(client):
    _seed(client)
    admin_token = create_token({"sub": "admin", "role": "admin"})
    r = client.get("/api/customers/", headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == 200


def test_list_rejects_regular_user_token(client):
    _seed(client)
    r = client.get("/api/customers/", headers={"Authorization": f"Bearer {_user_token()}"})
    assert r.status_code == 401


def test_list_rejects_bogus_bearer(client):
    _seed(client)
    r = client.get("/api/customers/", headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 401


def test_admin_unconfigured_returns_503(client, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "admin_secret", "")
    monkeypatch.setattr(settings, "admin_username", "")
    monkeypatch.setattr(settings, "admin_password", "")
    assert client.get("/api/customers/").status_code == 503


def test_create_customer_requires_admin(client):
    r = client.post("/api/customers/", json={"name": "A", "email": "a@x.com"})
    assert r.status_code == 401
