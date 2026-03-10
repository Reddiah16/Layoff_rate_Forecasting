
import os, logging, traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import config as cfg
from config import get_config, apply_config
from data_preprocessing import (
    load_data,
    clean_data,
    engineer_features,
    encode_and_scale,
    prepare_time_series,
)
from random_forest_model import (
    train_random_forest,
    evaluate_model,
    save_model,
    load_model,
    predict_single,
)
from arima_model import (
    fit_arima,
    forecast,
    evaluate_arima,
    make_stationary,
    auto_select_order,
    plot_forecast,
    plot_diagnostics,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

for d in ["data", "outputs", "models", "templates"]:
    os.makedirs(d, exist_ok=True)

app = FastAPI(
    title="LayoffLens API",
    description="Layoff Rate Forecasting — Random Forest + ARIMA",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

ALLOWED_EXTS = {"csv", "xlsx"}


def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


# ── Persistent state ──────────────────────────────────────────
STATE_FILE = "models/state.json"


def _find_default_dataset():
    candidates = ["data/Layoffs_Dataset.csv", "data/layoffs.csv"]
    if os.path.exists("data"):
        for f in os.listdir("data"):
            if f.endswith(".csv") or f.endswith(".xlsx"):
                p = os.path.join("data", f)
                if p not in candidates:
                    candidates.append(p)
    for path in candidates:
        if os.path.exists(path):
            logger.info(f"Auto-detected dataset: {path}")
            return path
    return None


def _load_state() -> dict:
    import json

    default = {
        "dataset_path": None,
        "rf_trained": False,
        "arima_trained": False,
        "rf_metrics": {},
        "arima_metrics": {},
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                saved = json.load(f)
            if saved.get("dataset_path") and not os.path.exists(saved["dataset_path"]):
                saved["dataset_path"] = None
            if saved.get("rf_trained") and not os.path.exists(
                "models/random_forest.pkl"
            ):
                saved["rf_trained"] = False
                saved["rf_metrics"] = {}
            default.update(saved)
            logger.info(
                f"Restored: dataset={saved.get('dataset_path')} rf={saved.get('rf_trained')} arima={saved.get('arima_trained')}"
            )
        except Exception:
            pass
    # Auto-load dataset from data/ folder if not already set
    if not default["dataset_path"]:
        default["dataset_path"] = _find_default_dataset()
    return default


def _save_state():
    import json

    os.makedirs("models", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(STATE, f, indent=2)


STATE = _load_state()


# ── Pydantic Schemas ─────────────────────────────────────────


class RFConfig(BaseModel):
    n_estimators: int = Field(200, ge=10, le=1000)
    max_depth: str = Field("20")
    min_samples_split: int = Field(5, ge=2, le=20)
    min_samples_leaf: int = Field(2, ge=1, le=10)
    max_features: str = Field("sqrt")
    positive_threshold: float = Field(0.20, ge=0.0, le=1.0)


class ARIMAConfig(BaseModel):
    auto_order: bool = Field(True)
    p: int = Field(1, ge=0, le=5)
    d: int = Field(1, ge=0, le=2)
    q: int = Field(1, ge=0, le=5)
    forecast_steps: int = Field(12, ge=1, le=60)


class TrainingConfig(BaseModel):
    test_size: float = Field(0.20, ge=0.1, le=0.4)
    random_state: int = Field(42)


class RiskThresholds(BaseModel):
    low_max: float = Field(0.10, ge=0.0, le=1.0)
    medium_max: float = Field(0.25, ge=0.0, le=1.0)
    high_prob: float = Field(0.70, ge=0.0, le=1.0)
    medium_prob: float = Field(0.40, ge=0.0, le=1.0)


class FullConfig(BaseModel):
    random_forest: RFConfig = RFConfig()
    arima: ARIMAConfig = ARIMAConfig()
    training: TrainingConfig = TrainingConfig()
    risk_thresholds: RiskThresholds = RiskThresholds()


class TrainRequest(BaseModel):
    random_forest: bool = True
    arima: bool = True


class PredictRequest(BaseModel):
    Industry: str = Field("Fintech", description="Industry sector")
    Funding_Status: str = Field("Private", description="Public / Private / Acquired")
    City: str = Field("Bengaluru", description="City of HQ")
    Layoff_Count: float = Field(500.0, description="Number of employees laid off")
    year: int = Field(2024, description="Year of layoff")
    month: int = Field(6, ge=1, le=12)
    quarter: int = Field(2, ge=1, le=4)


class PredictResponse(BaseModel):
    probability: float
    risk_label: str
    prediction: int


# ── Routes ───────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def serve_ui():
    html_path = os.path.join("templates", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(
            status_code=404, detail="index.html not found in templates/"
        )
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/config", tags=["Configuration"])
async def get_configuration():
    """View current model configuration."""
    return get_config()


@app.post("/config", tags=["Configuration"])
async def update_configuration(body: FullConfig):
    """Update model configuration. Changes apply to next training run."""
    apply_config(body.model_dump())
    return {"message": "Configuration updated ✅", "applied": get_config()}


@app.post("/upload", tags=["Data"])
async def upload_dataset(file: UploadFile = File(...)):
    """Upload your layoffs CSV dataset."""
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400, detail="Only .csv and .xlsx files supported"
        )
    filepath = os.path.join("data", file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())
    try:
        df = load_data(filepath)
        df = clean_data(df)
        df = engineer_features(df)
        STATE["dataset_path"] = filepath
        STATE["rf_trained"] = STATE["arima_trained"] = False
        _save_state()
        return {
            "message": "Dataset uploaded ✅",
            "filename": file.filename,
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "class_balance": {
                str(k): int(v) for k, v in df["layoffs_occurred"].value_counts().items()
            },
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", tags=["Models"])
async def train_models(body: TrainRequest):
    """Train Random Forest and/or ARIMA models."""
    if not STATE["dataset_path"]:
        raise HTTPException(
            status_code=400, detail="Upload a dataset first via POST /upload"
        )
    try:
        df = load_data(STATE["dataset_path"])
        df = clean_data(df)
        df = engineer_features(df)
        results = {}

        if body.random_forest:
            logger.info("Training Random Forest...")
            X_train, X_test, y_train, y_test, scaler, features, encoders = (
                encode_and_scale(df)
            )
            model = train_random_forest(X_train, y_train)
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test, features)
            save_model(model, scaler, encoders, features)
            STATE["rf_trained"] = True
            STATE["rf_metrics"] = metrics
            _save_state()
            results["random_forest"] = {
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "cv_f1_mean": metrics["cv_f1_mean"],
            }

        if body.arima:
            logger.info("Training ARIMA...")
            ts = prepare_time_series(df, freq=cfg.TIME_SERIES_FREQ)
            _, d = make_stationary(ts)
            order = (
                auto_select_order(ts, d=d)
                if cfg.ARIMA_AUTO_ORDER
                else (cfg.ARIMA_P, cfg.ARIMA_D, cfg.ARIMA_Q)
            )
            am = evaluate_arima(ts, order)
            fitted, order = fit_arima(ts, order=order)
            df_fc = forecast(fitted, steps=cfg.ARIMA_FORECAST_STEPS)
            plot_forecast(ts, df_fc)
            plot_diagnostics(fitted)
            STATE["arima_trained"] = True
            STATE["arima_metrics"] = am
            _save_state()
            results["arima"] = {
                "order": str(order),
                "RMSE": am["RMSE"],
                "MAE": am["MAE"],
                "MAPE": am["MAPE"],
            }

        return {"message": "Training complete ✅", "results": results}
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(body: PredictRequest):
    """Predict layoff risk for a single company."""
    if not STATE["rf_trained"]:
        raise HTTPException(
            status_code=400, detail="Train the model first via POST /train"
        )
    try:
        model, scaler, encoders, features = load_model()
        input_data = {
            "Industry": body.Industry,
            "Funding Status": body.Funding_Status,
            "City": body.City,
            "Layoff Count Numeric": body.Layoff_Count,
            "log_layoff_count": __import__("numpy").log1p(body.Layoff_Count),
            "is_public": 1 if body.Funding_Status.lower() == "public" else 0,
            "year": body.year,
            "month": body.month,
            "quarter": body.quarter,
        }
        return predict_single(model, scaler, encoders, features, input_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Models"])
async def get_metrics():
    """Get current model evaluation metrics."""
    return {
        "random_forest": STATE["rf_metrics"] if STATE["rf_trained"] else {},
        "arima": STATE["arima_metrics"] if STATE["arima_trained"] else {},
        "status": {
            "dataset_loaded": STATE["dataset_path"] is not None,
            "rf_trained": STATE["rf_trained"],
            "arima_trained": STATE["arima_trained"],
        },
    }


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    print(f"\n  LayoffLens — FastAPI")
    print(f"  App      → http://127.0.0.1:{cfg.PORT}")
    print(f"  API Docs → http://127.0.0.1:{cfg.PORT}/docs\n")
    uvicorn.run("app:app", host="127.0.0.1", port=cfg.PORT, reload=True)
                    