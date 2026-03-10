
import os
from dotenv import load_dotenv
load_dotenv()

# Random Forest
RF_N_ESTIMATORS      = int(os.getenv("RF_N_ESTIMATORS",      200))
RF_MAX_DEPTH         = os.getenv("RF_MAX_DEPTH",              "20")
RF_MIN_SAMPLES_SPLIT = int(os.getenv("RF_MIN_SAMPLES_SPLIT",  5))
RF_MIN_SAMPLES_LEAF  = int(os.getenv("RF_MIN_SAMPLES_LEAF",   2))
RF_MAX_FEATURES      = os.getenv("RF_MAX_FEATURES",           "sqrt")
RF_POSITIVE_THRESHOLD= float(os.getenv("RF_POSITIVE_THRESHOLD", 0.20))

# ARIMA
ARIMA_AUTO_ORDER     = os.getenv("ARIMA_AUTO_ORDER", "true").lower() == "true"
ARIMA_P              = int(os.getenv("ARIMA_P",  1))
ARIMA_D              = int(os.getenv("ARIMA_D",  1))
ARIMA_Q              = int(os.getenv("ARIMA_Q",  1))
ARIMA_FORECAST_STEPS = int(os.getenv("ARIMA_FORECAST_STEPS", 12))

# Training
TEST_SIZE            = float(os.getenv("TEST_SIZE",   0.20))
RANDOM_STATE         = int(os.getenv("RANDOM_STATE",  42))
TIME_SERIES_FREQ     = os.getenv("TIME_SERIES_FREQ",  "ME")

# Risk Thresholds
RISK_LOW_MAX         = float(os.getenv("RISK_LOW_MAX",    0.10))
RISK_MEDIUM_MAX      = float(os.getenv("RISK_MEDIUM_MAX", 0.25))
PRED_HIGH_PROB       = float(os.getenv("PRED_HIGH_PROB",  0.70))
PRED_MEDIUM_PROB     = float(os.getenv("PRED_MEDIUM_PROB",0.40))

# Server
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))

def get_config() -> dict:
    return {
        "random_forest":   {"n_estimators": RF_N_ESTIMATORS, "max_depth": RF_MAX_DEPTH,
                            "min_samples_split": RF_MIN_SAMPLES_SPLIT, "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
                            "max_features": RF_MAX_FEATURES, "positive_threshold": RF_POSITIVE_THRESHOLD},
        "arima":           {"auto_order": ARIMA_AUTO_ORDER, "p": ARIMA_P, "d": ARIMA_D,
                            "q": ARIMA_Q, "forecast_steps": ARIMA_FORECAST_STEPS},
        "training":        {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "time_series_freq": TIME_SERIES_FREQ},
        "risk_thresholds": {"low_max": RISK_LOW_MAX, "medium_max": RISK_MEDIUM_MAX,
                            "high_prob": PRED_HIGH_PROB, "medium_prob": PRED_MEDIUM_PROB},
        "server":          {"host": HOST, "port": PORT},
    }

def apply_config(new_config: dict):
    global RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_SPLIT, RF_MIN_SAMPLES_LEAF
    global RF_MAX_FEATURES, RF_POSITIVE_THRESHOLD, ARIMA_AUTO_ORDER, ARIMA_P, ARIMA_D
    global ARIMA_Q, ARIMA_FORECAST_STEPS, TEST_SIZE, RISK_LOW_MAX, RISK_MEDIUM_MAX
    global PRED_HIGH_PROB, PRED_MEDIUM_PROB
    rf = new_config.get("random_forest", {})
    if "n_estimators"       in rf: RF_N_ESTIMATORS       = int(rf["n_estimators"])
    if "max_depth"          in rf: RF_MAX_DEPTH          = str(rf["max_depth"])
    if "min_samples_split"  in rf: RF_MIN_SAMPLES_SPLIT  = int(rf["min_samples_split"])
    if "min_samples_leaf"   in rf: RF_MIN_SAMPLES_LEAF   = int(rf["min_samples_leaf"])
    if "max_features"       in rf: RF_MAX_FEATURES       = str(rf["max_features"])
    if "positive_threshold" in rf: RF_POSITIVE_THRESHOLD = float(rf["positive_threshold"])
    ar = new_config.get("arima", {})
    if "auto_order"     in ar: ARIMA_AUTO_ORDER     = bool(ar["auto_order"])
    if "p"              in ar: ARIMA_P              = int(ar["p"])
    if "d"              in ar: ARIMA_D              = int(ar["d"])
    if "q"              in ar: ARIMA_Q              = int(ar["q"])
    if "forecast_steps" in ar: ARIMA_FORECAST_STEPS = int(ar["forecast_steps"])
    tr = new_config.get("training", {})
    if "test_size" in tr: TEST_SIZE = float(tr["test_size"])
    rt = new_config.get("risk_thresholds", {})
    if "low_max"     in rt: RISK_LOW_MAX     = float(rt["low_max"])
    if "medium_max"  in rt: RISK_MEDIUM_MAX  = float(rt["medium_max"])
    if "high_prob"   in rt: PRED_HIGH_PROB   = float(rt["high_prob"])
    if "medium_prob" in rt: PRED_MEDIUM_PROB = float(rt["medium_prob"])
