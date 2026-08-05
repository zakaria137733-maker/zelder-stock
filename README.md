# SentimentIQ — Stock Sentiment Analysis Platform

Real-time stock sentiment analysis with a machine-learning price-direction signal,
served through a FastAPI backend and a Next.js dashboard.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        Data collection (scripts/)                  │
│   free_collect.py        fetch_historical.py    train_ensemble.py  │
│   Google News RSS +      Yahoo Finance daily    trains 5-seed LSTM │
│   Yahoo Finance prices   bars + market indices   ensemble → /app/models
└──────────┬──────────────────────────┬──────────────────────────────┘
           ▼                          ▼
   ┌───────────────┐          ┌───────────────┐          ┌──────────┐
   │   InfluxDB    │          │    MongoDB    │          │   Redis  │
   │ sentiment TS  │          │  customers    │          │  cache / │
   │ prices_daily  │          │  profiles     │          │  dedup / │
   │ market_index  │          │  watchlists   │          │  pubsub  │
   │ trades        │          └───────┬───────┘          └──────────┘
   └───────┬───────┘                  │
           ▼                          ▼
   ┌──────────────────────────────────────────┐      ┌──────────────────┐
   │              FastAPI (api)               │      │  Temporal worker │
   │  /api/auth  /api/customers  /api/sentiment│◄────►│  scheduled jobs  │
   │  /api/prices  /api/predictions  /api/admin│      │  (Google News +  │
   └──────────────────┬───────────────────────┘      │   demo trades)   │
                      ▼                              └──────────────────┘
            Next.js dashboard (frontend)
```

## Tech stack

| Layer | Choice |
|-------|--------|
| API | FastAPI + Uvicorn, JWT auth (python-jose), bcrypt |
| Time-series | InfluxDB 2.x (sentiment, prices, market indices, trades) |
| Persistence | MongoDB (customers, profiles, watchlists) |
| Cache / dedup | Redis (cache, URL dedup, pub/sub) |
| Orchestration | Temporal (scheduled collection workflows) |
| ML | PyTorch LSTM ensemble (5 seeds), XGBoost, VADER / FinBERT sentiment |
| Frontend | Next.js 16 + React Query + Recharts |

## Getting started

### 1. Configure environment

```bash
cp .env.example .env
# Fill in JWT_SECRET, ADMIN_SECRET, ADMIN_USERNAME, ADMIN_PASSWORD, NEWS_API_KEY
# Generate secrets with:
#   python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### 2. Start the stack

```bash
docker-compose up -d
docker-compose ps   # wait for mongo/influx/redis to be healthy
```

- API + Swagger docs: http://localhost:8000/docs
- InfluxDB UI: http://localhost:8086
- Frontend: `cd frontend && npm install && npm run dev` → http://localhost:3000

### 3. Seed demo data

```bash
docker-compose exec api python scripts/seed.py          # 8 customers + 48h sentiment
docker-compose exec api python scripts/fetch_historical.py  # daily prices + market data
```

### 4. Trigger live collection

```bash
docker-compose exec api python scripts/free_collect.py
# or trigger the collection through the admin API:
curl -X POST http://localhost:8000/api/admin/collect \
     -H "X-Admin-Key: <ADMIN_SECRET>"
```

### 5. Train the prediction model

```bash
docker-compose exec api python scripts/train_ensemble.py
```

Trains a 5-seed LSTM ensemble on daily bars and writes `scaler.json` plus
`lstm_model_{seed}.pt` to `/app/models` (bind-mounted from `backend/models/`).

## Authentication

| Mechanism | How it works |
|-----------|--------------|
| Customer | `POST /api/auth/register` / `/api/auth/login` → JWT (`Authorization: Bearer <token>`) |
| Admin key | `X-Admin-Key: <ADMIN_SECRET>` header on admin endpoints |
| Admin login | `POST /api/admin/login` with `{username, password}` → JWT with `role: "admin"` |

Customer routes are scoped to the **authenticated user's own record** — the
`customer_id` in the path is ignored (IDOR-hardened).

## API endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/auth/register` | – | Create customer account |
| POST | `/api/auth/login` | – | Customer login |
| GET | `/api/auth/me` | customer | Current user |
| POST | `/api/admin/login` | – | Admin login (username/password) |
| GET | `/api/sentiment/` | – | All tickers overview |
| GET | `/api/sentiment/{ticker}` | – | Composite score + history |
| GET | `/api/sentiment/{ticker}/history` | – | Raw sentiment time-series |
| GET | `/api/signals` | – | Latest signals per ticker |
| GET | `/api/prices/` | – | Latest prices |
| GET | `/api/prices/{ticker}/history` | – | Price history |
| GET | `/api/prices/{ticker}` | – | Price detail |
| GET | `/api/predictions/` | customer | Predictions for all tracked tickers |
| GET | `/api/predictions/{ticker}` | customer | Prediction for one ticker |
| GET | `/api/transactions/{ticker}` | – | Recent trades |
| POST | `/api/transactions/` | customer | Record a trade (stamped with caller) |
| GET | `/api/customers/` | **admin** | List all customers |
| POST | `/api/customers/` | **admin** | Create customer |
| GET | `/api/customers/{id}` | customer | Own profile |
| PATCH | `/api/customers/{id}/watchlist` | customer | Update own watchlist |
| DELETE | `/api/customers/{id}` | customer | Delete own account |
| GET | `/api/alerts/` | customer | Cached alerts |
| POST | `/api/alerts/refresh` | customer | Recompute alerts |
| POST | `/api/admin/collect` | **admin** | Trigger news collection |
| GET | `/api/admin/collect/status` | **admin** | Collection status |
| POST | `/api/admin/seed` | **admin** | Seed demo data |
| GET | `/health` | – | Health check |

## Machine learning: what it actually does (honest evaluation)

**Model.** A binary LSTM (sentiment + 11 technical features over a 10-day window)
predicts whether the next-day return is positive. An ensemble of 5 independently
seeded models is averaged at serving time; a per-feature min-max scaler is applied
to raw features before the forward pass.

**Serving parity.** The prediction endpoint (`routers/predictions.py`) runs the
**exact same code path** used for evaluation — raw features → `scaler.json` →
ensemble forward — so what you measure in evaluation is what you serve.
`scripts/test_baselines.py` reports accuracy of the *deployed* artifact by calling
`evaluate_deployed()`, not a freshly retrained model.

**Caveats (read this before quoting accuracy numbers anywhere):**

- Sentiment-driven short-horizon price direction is a **hard and noisy problem**;
  single-digit edge over 50% is realistic even for much larger systems.
- The dataset is short (90 days of daily bars per ticker) and daily-aggregated,
  so sample counts are small and results are not statistically robust.
- `scripts/test_ensemble.py`, `test_walkforward.py`, `test_permutation.py`, and
  `test_generalization.py` train **their own** models with their own hyperparameters
  and do **not** measure the deployed artifact — treat their numbers separately.
- The ensemble is meant to power a dashboard signal, not autonomous trading.

## Testing & CI

```bash
cd backend
pip install -r requirements.txt pytest ruff
pytest -q          # 27 unit tests: auth, admin auth, IDOR, ticker validation, ML pipeline
ruff check .       # lint
```

Tests run against an in-memory fake DB — no live Mongo/Influx/Redis needed. GitHub
Actions (`.github/workflows/ci.yml`) runs pytest + ruff on the backend and
`tsc --noEmit` + eslint on the frontend on every push/PR.

## Project layout

```
backend/
  main.py               FastAPI app, admin endpoints
  config.py             Settings from .env (pydantic-settings)
  tickers.py            Shared ticker list + validation
  routers/              API route modules
  services/             auth, influx, mongo, redis, sentiment, alerts, lstm_predictor
  workers/              Temporal worker, activities, workflows
  tests/                pytest suite (offline, fake DB)
  scripts/              One-off / operational scripts (seed, collect, train, backtest)
  models/               Trained artifacts (lstm_model*.pt, scaler.json)
frontend/
  app/                  Next.js app router (login, dashboard)
  components/           UI components (charts, signal feed, customers table)
  lib/api.ts            Axios client with JWT interceptor
```

## Security notes

- `JWT_SECRET`, `ADMIN_SECRET`, `ADMIN_USERNAME`, `ADMIN_PASSWORD` live in `.env`
  (gitignored); `.env.example` documents them. Rotating `JWT_SECRET` invalidates all
  outstanding tokens.
- Admin endpoints require either `X-Admin-Key` or an admin-scoped JWT.
- Customer records are only reachable by their owner; the customer DB list is
  admin-only.
- Ticker inputs are validated against a fixed allowlist before entering Flux queries.

## Useful commands

```bash
docker-compose logs -f api          # API logs
docker-compose logs -f worker       # scheduler logs
docker-compose exec api python scripts/free_collect.py   # manual collection
docker-compose exec api python scripts/train_ensemble.py # retrain model
docker-compose restart api          # restart API after code changes
docker-compose down -v              # wipe all data and start fresh
```
