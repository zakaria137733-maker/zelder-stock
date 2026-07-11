# SentimentIQ — Stock Sentiment Analysis Platform

## Saturday setup (do this now)

### 1. Prerequisites

Make sure you have installed:
- Docker Desktop (docker.com/products/docker-desktop)
- Node.js 18+ (nodejs.org)
- Python 3.11+ (optional for local dev, not needed if using Docker)

### 2. Get your NewsAPI key

Go to https://newsapi.org/register — takes 60 seconds, free tier gives 100 requests/day.

### 3. Clone / enter this project

```bash
cd sentimentiq
cp .env.example .env
# Edit .env and paste your NewsAPI key
```

### 4. Start all services

```bash
docker-compose up -d
```

Wait about 30 seconds for all services to become healthy. Check with:

```bash
docker-compose ps
```

All four services (mongodb, influxdb, redis, api, worker) should show "healthy" or "running".

### 5. Seed the databases

```bash
docker-compose exec api python seed.py
```

This writes 8 demo customers to MongoDB and 48 hours of sentiment history + 200 trades to InfluxDB.
Your charts will have real data immediately.

### 6. Trigger the first news collection

```bash
docker-compose exec api python -c "
import asyncio
from services.news_collector import collect_and_score_all
asyncio.run(collect_and_score_all())
"
```

This fetches live headlines from NewsAPI and scores them. Takes ~10 seconds.

### 7. Verify everything works

```bash
# Health check
curl http://localhost:8000/health

# All sentiment scores
curl http://localhost:8000/api/sentiment/

# AAPL detail
curl http://localhost:8000/api/sentiment/AAPL

# Customer list (MongoDB)
curl http://localhost:8000/api/customers/

# Recent AAPL trades (InfluxDB)
curl http://localhost:8000/api/transactions/AAPL
```

Auto-generated API docs: http://localhost:8000/docs

---

## Architecture

```
Data Sources (NewsAPI, Reddit, SEC)
        ↓
Redis Streams (dedup + queue)
        ↓
NLP Pipeline (VADER / FinBERT)
        ↓
┌─────────────────┬──────────────────┐
│   MongoDB       │   InfluxDB       │
│   Customers     │   Sentiment TS   │
│   Profiles      │   Trade data     │
│   Watchlists    │   Price history  │
└─────────────────┴──────────────────┘
        ↓
FastAPI (REST + WebSocket)
        ↓
Next.js Dashboard
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/sentiment/ | All tickers overview |
| GET | /api/sentiment/{ticker} | Composite score + history |
| GET | /api/sentiment/{ticker}/history | InfluxDB timeseries |
| GET | /api/signals | Latest sentiment signals |
| WS  | /ws/signals | Live WebSocket feed |
| GET | /api/customers/ | MongoDB customer list |
| POST | /api/customers/ | Create customer |
| GET | /api/transactions/{ticker} | Recent trades |
| POST | /api/transactions/ | Record a trade |

## Upgrading to FinBERT

Once the demo is working, set `USE_FINBERT=true` in your `.env` and restart:

```bash
docker-compose restart api worker
```

FinBERT gives significantly better accuracy on financial text. Requires ~3GB RAM and
30-60 seconds to load on first startup.

## Build schedule

- **Saturday**: Docker up, seed data, API working ✓
- **Sunday**: Connect NewsAPI, tune scoring, add Reddit collector
- **Monday**: Finalize all API endpoints, Postman test suite
- **Tuesday**: Next.js frontend — charts, signal feed, customer table
- **Wednesday morning**: Polish, seed more data, demo rehearsal

## Useful commands

```bash
# View API logs
docker-compose logs -f api

# View worker/scheduler logs
docker-compose logs -f worker

# Open MongoDB shell
docker-compose exec mongodb mongosh sentimentiq

# Open InfluxDB UI
open http://localhost:8086
# Login: admin / password123

# Restart just the API (after code changes)
docker-compose restart api

# Run collection manually
docker-compose exec api python -c "
import asyncio; from services.news_collector import collect_and_score_all
asyncio.run(collect_and_score_all())"

# Stop everything
docker-compose down

# Wipe all data and start fresh
docker-compose down -v
```
