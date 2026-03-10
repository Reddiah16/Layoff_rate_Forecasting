

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, logging, os, itertools

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
os.makedirs("outputs", exist_ok=True)


def check_stationarity(series):
    if len(series) < 5:
        logger.warning("Too few points for ADF test — assuming non-stationary")
        return False
    result = adfuller(series.dropna(), autolag="AIC")
    logger.info(f"ADF p-value: {result[1]:.4f} | Stationary: {result[1] < 0.05}")
    return result[1] < 0.05


def make_stationary(series):
    d, ts = 0, series.copy()
    while d < 2:
        if len(ts.dropna()) < 4:
            break
        try:
            if adfuller(ts.dropna(), autolag="AIC")[1] < 0.05:
                return ts, d
        except Exception:
            break
        ts = ts.diff().dropna()
        d += 1
    return ts, d


def auto_select_order(series, d=1):
    logger.info("Auto-selecting ARIMA order via AIC...")
    best_aic, best_order = np.inf, (1, d, 1)
    # Smaller search space for small datasets
    p_range = range(3) if len(series) < 20 else range(4)
    q_range = range(3) if len(series) < 20 else range(4)
    for p, q in itertools.product(p_range, q_range):
        try:
            res = ARIMA(series, order=(p, d, q)).fit()
            if res.aic < best_aic:
                best_aic, best_order = res.aic, (p, d, q)
        except Exception:
            continue
    logger.info(f"Best order: {best_order} | AIC: {best_aic:.2f}")
    return best_order


def fit_arima(series, order=None):
    _, d = make_stationary(series)
    if order is None:
        order = auto_select_order(series, d=d)
    logger.info(f"Fitting ARIMA{order}...")
    result = ARIMA(series, order=order).fit()
    return result, order


def forecast(fitted_result, steps=12):
    import pandas as pd
    pred  = fitted_result.get_forecast(steps=steps)
    mean  = pred.predicted_mean
    ci    = pred.conf_int(alpha=0.05)
    df_fc = pd.DataFrame({
        "forecast": mean.values.clip(min=0),
        "lower_ci": ci.iloc[:, 0].values.clip(min=0),
        "upper_ci": ci.iloc[:, 1].values,
    }, index=mean.index)
    logger.info(f"Forecast ({steps} periods):\n{df_fc.round(0).to_string()}")
    return df_fc


def evaluate_arima(series, order, test_size=0.2):
    # For small series, use at least 2 test points
    n_test = max(2, int(len(series) * test_size))
    split  = len(series) - n_test
    train, test = series[:split], series[split:]

    if len(train) < 4:
        logger.warning("Too few training points for walk-forward — skipping evaluation")
        return {"RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0}

    history, preds = list(train), []
    for t in range(len(test)):
        try:
            res  = ARIMA(history, order=order).fit()
            preds.append(res.forecast(steps=1)[0])
        except Exception:
            preds.append(np.mean(history))
        history.append(test.iloc[t])

    preds   = np.array(preds).clip(min=0)
    actuals = test.values
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100

    metrics = {"RMSE": round(float(rmse),2), "MAE": round(float(mae),2), "MAPE": round(float(mape),2)}
    logger.info(f"RMSE:{rmse:.2f} | MAE:{mae:.2f} | MAPE:{mape:.2f}%")

    with open("outputs/arima_metrics.txt","w") as f:
        f.write(f"ARIMA Order: {order}\n")
        for k,v in metrics.items(): f.write(f"{k}: {v}\n")

    _plot_walk_forward(test, preds)
    return metrics


def plot_forecast(series, df_fc):
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0e1318"); ax.set_facecolor("#0e1318")
    ax.plot(series.index, series.values, color="#00e5ff", lw=2, marker="o",
            markersize=4, label="Historical Layoff Count")
    ax.plot(df_fc.index, df_fc["forecast"], color="#ff4d6d", lw=2,
            linestyle="--", label="ARIMA Forecast")
    ax.fill_between(df_fc.index, df_fc["lower_ci"], df_fc["upper_ci"],
                    alpha=0.2, color="#ff4d6d", label="95% Confidence Interval")
    ax.axvline(series.index[-1], color="#4a6380", lw=1.5, linestyle=":")
    ax.set_title("Monthly Layoff Count — ARIMA Forecast", fontsize=14,
                 fontweight="bold", color="#eaf4ff", pad=14)
    ax.set_xlabel("Month", color="#4a6380")
    ax.set_ylabel("Total Layoffs", color="#4a6380")
    ax.tick_params(colors="#4a6380"); ax.spines[:].set_color("#1e2d3d")
    ax.grid(color="#1e2d3d", linewidth=0.5)
    ax.legend(facecolor="#151c24", edgecolor="#1e2d3d", labelcolor="#c8d8e8", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/arima_forecast.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved → outputs/arima_forecast.png")


def plot_diagnostics(fitted_result):
    try:
        fig = fitted_result.plot_diagnostics(figsize=(13, 8))
        fig.suptitle("ARIMA — Model Diagnostics", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("outputs/arima_diagnostics.png", dpi=150)
        plt.close()
        logger.info("Saved → outputs/arima_diagnostics.png")
    except Exception as e:
        logger.warning(f"Could not plot diagnostics: {e}")


def _plot_walk_forward(test, preds):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test.values, color="#00e5ff", lw=2, marker="o",
            markersize=5, label="Actual")
    ax.plot(test.index, preds, color="#ff4d6d", lw=2, linestyle="--",
            marker="s", markersize=5, label="Predicted")
    ax.set_title("Walk-Forward Validation", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Layoff Count")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/arima_walk_forward.png", dpi=150)
    plt.close()
    logger.info("Saved → outputs/arima_walk_forward.png")


if __name__ == "__main__":
    import sys
    from data_preprocessing import load_data, clean_data, engineer_features, prepare_time_series
    path  = sys.argv[1] if len(sys.argv) > 1 else "data/Layoffs_Dataset.csv"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    df = load_data(path); df = clean_data(df); df = engineer_features(df)
    ts = prepare_time_series(df, freq="ME")
    print(f"\nTime series ({len(ts)} months):\n{ts}")
    _, d = make_stationary(ts); order = auto_select_order(ts, d=d)
    metrics = evaluate_arima(ts, order)
    fitted, order = fit_arima(ts, order=order)
    df_fc = forecast(fitted, steps=steps)
    plot_forecast(ts, df_fc); plot_diagnostics(fitted)
    for k, v in metrics.items(): print(f"  {k}: {v}")