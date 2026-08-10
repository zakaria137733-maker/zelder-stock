"""Facade re-exporting the split ML modules (ml_features / ml_training / ml_serving).

Kept so operational scripts (scripts/*.py) and the walk-forward harness keep
importing from services.lstm_predictor unchanged. Real consumers (routers,
tests, services/walkforward.py) import the concrete modules directly so
monkeypatching in tests targets the module that owns the code.
"""

from services.ml_features import (  # noqa: F401
    DROPOUT,
    ENSEMBLE_SEEDS,
    FEATURES,
    HIDDEN_SIZE,
    HORIZON,
    LABEL_THRESHOLD,
    LOOKBACK,
    NUM_LAYERS,
    SEQUENCE_LEN,
    TRACKED_TICKERS,
    SentimentLSTM,
    _default_indicators,
    _feature_row,
    _ind_value,
    _window_indicators,
    apply_scaler,
    build_eval_sequences,
    build_raw_features,
    build_sequences,
    classification_metrics,
    compute_indicators,
    fit_scaler,
    persistence_baseline,
    prediction_evidence,
    signal_from_prob,
    youden_threshold,
)
from services.ml_serving import (  # noqa: F401
    _ENSEMBLE_CACHE,
    _SCALER_CACHE,
    _THRESHOLDS_CACHE,
    MODEL_DIR,
    MODEL_PATH,
    PRED_THRESHOLDS_PATH,
    SCALER_PATH,
    _daily_returns,
    _path_signature,
    ensemble_forward,
    evaluate_deployed,
    fetch_live_daily_context,
    load_ensemble_models,
    load_predict_thresholds,
    load_scaler,
    predict_ensemble,
)
from services.ml_training import (  # noqa: F401
    EPOCHS,
    GRAD_CLIP,
    LR,
    MIN_MARKET_COVERAGE,
    MIN_SENTIMENT_COVERAGE,
    WEIGHT_DECAY,
    coverage_report,
    fetch_training_data,
    print_coverage_report,
    require_coverage,
    train,
)
