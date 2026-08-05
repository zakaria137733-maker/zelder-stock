"""IDOR regression: customer routes must operate on the caller's OWN record and
ignore any customer_id in the path."""

from services.auth import create_token, hash_password


def _seed_two(client):
    client.db.customers.docs.extend([
        {"_id": 1, "email": "a@x.com", "name": "Alice", "watchlist": ["AAPL"], "password_hash": hash_password("pw")},
        {"_id": 2, "email": "b@x.com", "name": "Bob", "watchlist": ["TSLA"], "password_hash": hash_password("pw")},
    ])


def _token(email):
    return {"Authorization": f"Bearer {create_token({'sub': email, 'name': email})}"}


def test_get_returns_own_customer_ignoring_path_id(client):
    _seed_two(client)
    # Alice requests Bob's id (2) but must receive her OWN record (id 1)
    r = client.get("/api/customers/2", headers=_token("a@x.com"))
    assert r.status_code == 200
    assert r.json()["id"] == "1"
    assert r.json()["email"] == "a@x.com"


def test_patch_updates_only_own_watchlist(client):
    _seed_two(client)
    r = client.patch("/api/customers/2/watchlist", json=["NVDA", "MSFT"], headers=_token("a@x.com"))
    assert r.status_code == 200
    alice = client.db.customers.docs[0]
    bob = client.db.customers.docs[1]
    assert alice["watchlist"] == ["NVDA", "MSFT"]
    assert bob["watchlist"] == ["TSLA"]


def test_delete_deletes_only_own_customer(client):
    _seed_two(client)
    r = client.delete("/api/customers/2", headers=_token("a@x.com"))
    assert r.status_code == 200
    emails = [d["email"] for d in client.db.customers.docs]
    assert emails == ["b@x.com"]


def test_unknown_own_customer_404(client):
    _seed_two(client)
    # A token for someone not in the DB
    r = client.get("/api/customers/1", headers=_token("ghost@x.com"))
    assert r.status_code == 404
