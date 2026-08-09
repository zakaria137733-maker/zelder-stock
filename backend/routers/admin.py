import asyncio
import json
import os
from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from services import lstm_predictor as lsp
from services.auth import require_admin
from tickers import TICKERS

router = APIRouter(prefix="/api/admin", tags=["admin"])

_last_collect = {"time": None, "status": "idle"}

# Verbatim from the README "Caveats" section — the honest numbers behind this
# dashboard belong next to the model, not just in the docs.
EVAL_CAVEATS = [
    "Sentiment-driven short-horizon price direction is a hard and noisy problem; "
    "single-digit edge over 50% is realistic even for much larger systems.",
    "The dataset is short (90 days of daily bars per ticker) and daily-aggregated, "
    "so sample counts are small and results are not statistically robust.",
    "A recent run of eval_deployed.py on AAPL (120 days, 59 windows) scored "
    "0.42 accuracy vs a 0.69 up-majority baseline (balanced accuracy 0.45, "
    "AUC 0.35) — honest, and it shows why "
    "this powers a dashboard signal rather than autonomous trading.",
    "scripts/test_ensemble.py, test_walkforward.py, test_permutation.py, and "
    "test_generalization.py train their own models with their own hyperparameters "
    "and do not measure the deployed artifact — treat their numbers separately.",
    "The ensemble is meant to power a dashboard signal, not autonomous trading.",
]


def _run_lightweight_collect():
    try:
        _last_collect["status"] = "running"
        from services.news_collector import collect_google_sentiment

        for ticker in TICKERS:
            try:
                result = collect_google_sentiment(ticker)
                print(f"  {ticker}: {result['composite']:.1f} ({result['signal_count']} signals)")
            except Exception as e:
                print(f"  {ticker} error: {e}")

        _last_collect["time"] = datetime.now(UTC).isoformat()
        _last_collect["status"] = "idle"
        print("Collection completed")
    except Exception as e:
        _last_collect["status"] = f"error: {e}"
        print(f"Collection error: {e}")
        import traceback
        traceback.print_exc()


def _load_report_file(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Eval report load error ({path}): {e}")
        return None


def _models_dir() -> str:
    """The artifacts dir: MODEL_DIR when it points at real models, else the
    repo's backend/models so the committed report is found outside Docker too."""
    if os.path.isdir(lsp.MODEL_DIR):
        return lsp.MODEL_DIR
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))


@router.get("/eval/report")
async def eval_report(_admin=Depends(require_admin)):
    """The committed evaluation report: deployed-model accuracy vs baselines.

    Serves backend/models/eval_report.json (written by scripts/eval_deployed.py
    --json-out), falling back to a committed walk-forward summary if present, so
    the honest numbers are visible in the product rather than only in the README.
    """
    models_dir = _models_dir()
    report_path = os.path.join(models_dir, "eval_report.json")
    walkforward_path = os.path.join(models_dir, "walkforward_report.json")

    report = _load_report_file(report_path)
    source = "eval_report.json" if report is not None else None
    if report is None:
        report = _load_report_file(walkforward_path)
        source = "walkforward_report.json" if report is not None else None

    return {
        "report": report,
        "source": source,
        "caveats": EVAL_CAVEATS,
    }


@router.post("/collect")
async def trigger_collect(_admin=Depends(require_admin)):
    asyncio.get_event_loop().run_in_executor(None, _run_lightweight_collect)
    return {"ok": True, "message": "Collection started"}


@router.get("/collect/status")
async def collect_status(_admin=Depends(require_admin)):
    return _last_collect


@router.post("/seed")
async def seed_all(_admin=Depends(require_admin)):
    from services import seeding

    errors = []
    customer_count = 0
    influx_count = 0
    demo_credentials = []

    try:
        demo_credentials = await seeding.seed_customers()
        customer_count = len(demo_credentials)
    except Exception as e:
        errors.append(f"MongoDB: {str(e)}")

    try:
        influx_count = seeding.seed_influx()
    except Exception as e:
        errors.append(f"InfluxDB: {str(e)}")

    return {
        "ok": len(errors) == 0,
        "customers": customer_count,
        "influx_points": influx_count,
        "demo_credentials": demo_credentials,
        "errors": errors,
    }
