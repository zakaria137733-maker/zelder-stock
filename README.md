# SentimentIQ — Stock Sentiment Analysis Platform

Real-time stock sentiment analysis with a machine-learning price-direction signal,
served through a FastAPI backend and a Next.js dashboard.

## What this repo demonstrates

- **Leak-free evaluation.** The walk-forward harness (`scripts/eval_walkforward.py`)
  evaluates predictors strictly time-ordered — training only on past windows —
  against coin-flip, majority-up and momentum baselines, so reported accuracy
  isn't contaminated by lookahead.
- **Serving parity.** `/api/predictions` runs the exact raw-features → scaler →
  ensemble forward pass that the evaluation scripts measure, and
  `scripts/eval_deployed.py` scores the committed artifact itself.
- **Security hardening.** No secrets in the tree (generated demo credentials, admin
  passwords never logged), per-user IDOR scoping, ticker allowlists, and
  Redis-backed auth rate limiting with an in-process fallback.
- **CI.** GitHub Actions runs ruff + pytest (with Mongo/Influx/Redis containers),
  `tsc --noEmit`, eslint and vitest on every push/PR.

Honest caveats about signal quality are in the ML section below.

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
   │  /api/prices  /api/predictions  /api/admin│      │  (Google News)   │
   └──────────────────┬───────────────────────┘      │                  │
                      ▼                              └──────────────────┘
            Next.js dashboard (frontend)
```

The news collector runs as a **Temporal** scheduled workflow (every 5 min) rather
than a bare cron or an in-process timer. Temporal gives the schedule durable
state — if the worker or the server restarts, the schedule survives and the next
run is never lost or double-fired — plus built-in retry semantics for the
collect activity, so a transient news-feed failure is retried instead of silently
dropping a ticker's daily composite. A cron would need an external supervisor and
an operator to notice missed runs; Temporal makes the cadence part of the system
itself.

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
| GET | `/api/admin/eval/report` | **admin** | Honest eval report (deployed-model accuracy vs baselines + caveats) |
| GET | `/health` | – | Health check |

## Machine learning: what it actually does (honest evaluation)

**Model.** A binary LSTM (sentiment + 11 technical features over a 10-day window)
predicts whether the next-day return is positive. An ensemble of 5 independently
seeded models is averaged at serving time; a per-feature min-max scaler is applied
to raw features before the forward pass.

**Serving parity.** The prediction endpoint (`routers/predictions.py`) runs the
**exact same code path** used for evaluation — raw features → `scaler.json` →
ensemble forward — so what you measure in evaluation is what you serve.
`scripts/eval_deployed.py AAPL --days 120` reports accuracy of the *deployed*
artifact by calling `evaluate_deployed()`, not a freshly retrained model, and can
write a JSON report:

```bash
docker-compose exec api python scripts/eval_deployed.py AAPL --days 120 --json-out models/eval_report.json
```

The `test_deployed_artifact_scores_on_synthetic_data` pytest also loads the
committed 5-seed ensemble (`lstm_model_42.pt` … `lstm_model_2024.pt`) plus the
single `lstm_model.pt` fallback and `scaler.json` and asserts the serving
pipeline produces sane probabilities — so the shipped artifact is under test,
not just some freshly-trained twin.

The committed report (`backend/models/eval_report.json`) is also served to the
dashboard: `GET /api/admin/eval/report` returns the report plus the caveats
verbatim, and the **Model Evaluation** page (sidebar → System, admin-gated)
renders accuracy vs. the majority and coin-flip baselines, the per-ticker
table, and the caveats. The honest numbers live in the product, not just the
README.

**Walk-forward harness.** `scripts/eval_walkforward.py` (offline by default,
`--ticker` for real data, `--thresholds` for a label-threshold sweep) evaluates any
predictor time-ordered — training only on past windows — against the three "no
signal" baselines: coin flip (0.5), the majority-up prior, and trailing momentum
continuation. Causal logistic, XGBoost and MLP models are included as reference
models:

```bash
python scripts/eval_walkforward.py                    # synthetic, fully offline
docker-compose exec api python scripts/eval_walkforward.py --ticker AAPL --days 1900 --thresholds 0.5,1.0,1.5,2.0
```

With 5 years of daily bars backfilled (`scripts/fetch_historical.py --period 5y`,
~1,257 bars/ticker), the causal **logistic** model averages **~54%** across
AAPL/NVDA/SPY/MSFT/TSLA at the 1% threshold — tied with the majority-up prior
and ~2 pts above momentum — reaching 55.4–57.7% on AAPL, NVDA and SPY.
XGBoost (~53%) barely edges momentum (~52%), adding nothing durable over it, and
MLP *loses* signal (46%, overfits the small folds). The edge is real but thin
and ticker-dependent (TSLA shows none). The LSTM's one unique input, sentiment,
still only covers ~90 days, so it cannot be judged over the 5-year window yet.
The harness lives in `services/walkforward.py` with its own offline pytest suite.

**Caveats (read this before quoting accuracy numbers anywhere):**

- Sentiment-driven short-horizon price direction is a **hard and noisy problem**;
  single-digit edge over 50% is realistic even for much larger systems.
- The dataset is short (90 days of daily bars per ticker) and daily-aggregated,
  so sample counts are small and results are not statistically robust.
- A recent run of `eval_deployed.py` on AAPL (120 days, 59 windows) scored
  **0.42 accuracy vs a 0.69 up-majority baseline** (balanced accuracy 0.45,
  AUC 0.35) — honest, and it shows why this powers a dashboard signal rather
  than autonomous trading.
- `scripts/test_ensemble.py`, `test_walkforward.py`, `test_permutation.py`, and
  `test_generalization.py` train **their own** models with their own hyperparameters
  and do **not** measure the deployed artifact — treat their numbers separately.
- The ensemble is meant to power a dashboard signal, not autonomous trading.

## Testing & CI

```bash
# backend
cd backend
pip install -r requirements.txt pytest ruff
pytest -q          # unit + integration: auth, admin auth, IDOR, ticker validation, ML pipeline
ruff check .       # lint

# frontend
cd frontend
npm ci
npm run test       # vitest: auth redirect flow, login, components
npx tsc --noEmit   # type-check
npm run lint       # eslint
```

- Unit tests run against an in-memory fake DB — no live services needed.
- `tests/test_integration_services.py` exercises the **real** Mongo / Influx /
  Redis modules (`services/mongo.py`, `redis_client.py`, `influx.py`) end-to-end.
  They auto-skip when the services are down and run for real in CI (which starts
  them as service containers) or locally with `docker-compose up -d`.
- Frontend tests (vitest + Testing Library) cover the login/admin auth flow, the
  `/login` redirect gate, and component rendering/error states.
- GitHub Actions (`.github/workflows/ci.yml`) runs ruff + pytest (with Mongo /
  Influx / Redis service containers) on the backend, and `tsc --noEmit` + eslint
  + vitest on the frontend, on every push/PR.

## Project layout

```
backend/
  main.py               FastAPI app wiring (routers only)
  config.py             Settings from .env (pydantic-settings)
  tickers.py            Shared ticker list + validation
  routers/              API route modules (admin: collect/seed/eval report)
  services/             auth, influx, mongo, redis, sentiment, alerts, lstm_predictor
  workers/              Temporal worker, activities, workflows
  tests/                pytest suite (unit: offline fake DB; integration: real services)
  scripts/              One-off / operational scripts (seed, collect, train, backtest, eval)
  models/               Trained artifacts (lstm_model*.pt, scaler.json, eval_report.json)
frontend/
  app/                  Next.js app router (login, dashboard)
  components/           UI components (charts, signal feed, customers, evaluation)
  lib/api.ts            Axios client with JWT interceptor
  __tests__/            Vitest + Testing Library (auth flow, components)
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
