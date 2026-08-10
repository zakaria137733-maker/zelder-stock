# Scripts index

The `scripts/` directory holds the **operational** scripts you actually run to
collect data, seed the demo, train the deployed artifact, and evaluate it.
Throwaway / one-off dev experiments live in `devtools/` so the main directory
stays a clear, runnable surface.

All scripts assume they run from `backend/` (or via
`docker compose exec api python scripts/<name>.py`), which they handle
themselves by inserting the repo root on `sys.path`.

## Operational scripts (`scripts/`)

| Script | What it does |
|--------|--------------|
| `seed_demo.py` | One-command offline demo: wipes + seeds prices, sentiment, market indices, trades, Mongo customers and Redis cache, then self-checks. `--no-wipe` / `--days` flags. **This is the quickstart path.** |
| `seed.py` | Legacy seed: 8 demo customers (random passwords) + 48h of demo-tagged Influx sentiment/trades. Superseded by `seed_demo.py`. |
| `fetch_historical.py` | Backfill 5y of daily OHLCV bars (`prices_daily`) + market indices (`market_index`) from Yahoo Finance. |
| `free_collect.py` | One-off manual collection: Google News sentiment + Yahoo prices for all tracked tickers into Influx/Redis. |
| `check_influx.py` | Dump what's currently in the Influx `sentiment_scores` bucket (measurements, tags, counts). |
| `check_yf.py` | Sanity-check the Yahoo Finance API surface (columns, index type) the collector depends on. |
| `backfill_sentiment.py` | Backfill 5 years of daily mean news tone from GDELT `timelinetone` as `sentiment` points (source `gdelt`). |
| `train_ensemble.py` | Train the deployed 5-seed LSTM ensemble; writes `scaler.json` + `lstm_model_<seed>.pt` to `models/`. |
| `eval_deployed.py` | Score the **deployed** artifact (ensemble + scaler) on recent held-out daily data via the serving pipeline. `--json-out models/eval_report.json`. |
| `eval_lstm_signal.py` | Causal walk-forward LSTM eval per ticker vs momentum/majority baselines; writes `predict_thresholds.json` + `lstm_signal_report.json`. |
| `eval_walkforward.py` | Time-ordered walk-forward harness (offline by default, `--ticker` for real data) vs coin-flip / majority / momentum baselines. |
| `build_relative_signal.py` | Experiment: can a model predict 5-day relative outperformance vs SPY? |
| `test_ensemble.py` | Cross-ticker generalization test (train 5, test 2). Trains its own model — not the deployed artifact. |
| `test_walkforward.py` | Walk-forward stability test across time periods. Trains its own model. |
| `test_permutation.py` | Permutation test: shuffled-label accuracy vs real-label accuracy. Trains its own model. |

## Dev / throwaway scripts (`devtools/`)

Moved here so the operational surface stays clean. They are kept for reference
but are not part of the quickstart, the collect/train/eval pipeline, or CI.

| Script | What it does |
|--------|--------------|
| `force_collect.py` | Force a synchronous sentiment collect for all tickers (ignores caching). |
| `seed_alerts.py` | Seed scripted sentiment trajectories to exercise alert thresholds. |
| `test_alerts.py` | Manual check of the alert engine against real sentiment history. |
| `test_baselines.py` | Compare feature/baseline stats from training data. |
| `test_generalization.py` | Cross-sector generalization test (train tech+fin, test healthcare+energy+consumer). |
| `test_price_write.py` | Verify Yahoo price points write into Influx correctly. |
| `debug_features.py` | Dump training feature shapes/statistics before normalization. |
| `debug_predict.py` | Inspect what the prediction endpoint sees (price + sentiment history). |
| `train_lstm.py` | Single-model LSTM training entrypoint (superseded by `train_ensemble.py`). |
| `train_xgboost.py` | Legacy XGBoost training experiment (superseded by the walk-forward harness). |

## Running

```bash
# local (from backend/)
python scripts/seed_demo.py --days 40
python scripts/eval_deployed.py AAPL --days 120 --json-out models/eval_report.json

# in docker
docker compose exec api python scripts/seed_demo.py
docker compose exec api python scripts/train_ensemble.py
```

`devtools/` scripts are excluded from ruff (throwaway quality bar); the
operational scripts are not.
