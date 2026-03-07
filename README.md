# LayoffLens — ML Layoff Rate Forecasting Platform

Supervised machine learning platform using **Random Forest** (classification)
and **ARIMA** (time series forecasting) to predict company layoff risk.

---

## Project Structure

```
layoff_forecasting/
│
├── app.py                  # Flask web backend (all API routes)
├── data_preprocessing.py   # Load, clean, encode, engineer features
├── random_forest_model.py  # RF training, evaluation, prediction
├── arima_model.py          # ARIMA fitting, forecasting, evaluation
├── requirements.txt        # Python dependencies
│
├── data/                   # ← Put your CSV dataset here
│   └── layoffs.csv
│
├── models/                 # Saved model files (auto-created)
│   └── random_forest.pkl
│
├── outputs/                # Generated charts (auto-created)
│   ├── rf_confusion_matrix.png
│   ├── rf_roc_curve.png
│   ├── rf_feature_importance.png
│   ├── arima_forecast.png
│   ├── arima_diagnostics.png
│   └── arima_metrics.txt
│
├── templates/
│   └── index.html          # Flask HTML template (full UI)
│
└── .vscode/
    ├── launch.json         # Debug configurations
    ├── settings.json       # Editor settings
    └── extensions.json     # Recommended extensions
```

---

## CSV Dataset Format

Your CSV should include these columns (extra columns are ignored):

| Column           | Type    | Description                          |
|------------------|---------|--------------------------------------|
| `company`        | string  | Company name                         |
| `industry`       | string  | Industry sector                      |
| `date`           | date    | Record date (YYYY-MM-DD)             |
| `employees`      | int     | Employee headcount                   |
| `revenue_growth` | float   | YoY revenue growth (%)               |
| `profit_margin`  | float   | Net profit margin (%)                |
| `debt_equity`    | float   | Debt-to-equity ratio                 |
| `stock_change`   | float   | 6-month stock price change (%)       |
| `macro_env`      | string  | stable / slowdown / recession / recovery |
| `layoff_rate`    | float   | Layoff rate % (used by ARIMA)        |
| `layoffs_occurred`| int    | 0 or 1 (used by Random Forest)       |

> If `layoffs_occurred` is missing but `layoff_rate` is present,
> it will be auto-created: `layoff_rate > 3.0` → 1 (Layoff occurred).

---

## Setup in VS Code

### 1. Open Project
```bash
# Open the project folder in VS Code
code layoff_forecasting/
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Recommended VS Code Extensions
- Open Command Palette: `Ctrl+Shift+P`
- Type: `Extensions: Show Recommended Extensions`
- Install all suggestions

---

## Running the App

### Option A — Flask Web App (Recommended)
```bash
python app.py
```
Open browser → `http://127.0.0.1:5000`

**Then use the UI to:**
1. Upload your CSV → Train Models → Predict

### Option B — Run Scripts Directly
```bash
# Preprocessing only
python data_preprocessing.py data/layoffs.csv

# Train Random Forest
python random_forest_model.py data/layoffs.csv

# Run ARIMA (12-period forecast)
python arima_model.py data/layoffs.csv 12
```

### Option C — VS Code Debugger
- Press `F5` → select configuration:
  - `Run Flask App`
  - `Train Random Forest`
  - `Run ARIMA`
  - `Preprocess Data`

---

## API Endpoints

| Method | Route              | Description                        |
|--------|--------------------|------------------------------------|
| GET    | `/`                | Web dashboard                      |
| POST   | `/upload`          | Upload CSV dataset                 |
| POST   | `/train`           | Train RF + ARIMA models            |
| POST   | `/predict`         | Single company prediction (JSON)   |
| POST   | `/batch`           | Batch CSV prediction               |
| GET    | `/forecast`        | ARIMA forecast data (JSON)         |
| GET    | `/metrics`         | Model evaluation metrics (JSON)    |
| GET    | `/outputs/<file>`  | Serve chart images                 |

### Example: Single Prediction via API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "industry": "Technology",
    "employees": 5000,
    "revenue_growth": -15,
    "profit_margin": 3.2,
    "debt_equity": 2.1,
    "stock_change": -28,
    "macro_env": "recession"
  }'
```

**Response:**
```json
{
  "probability": 84.3,
  "risk_label": "High",
  "prediction": 1
}
```

---

## Model Details

### Random Forest
- **Type:** Ensemble classifier (bagging of decision trees)
- **Target:** `layoffs_occurred` (binary 0/1)
- **Features:** Revenue growth, profit margin, debt/equity, stock change, industry, macro environment + engineered features
- **Outputs:** Confusion matrix, ROC curve, feature importance, accuracy/F1/AUC

### ARIMA
- **Type:** AutoRegressive Integrated Moving Average
- **Target:** `layoff_rate` (continuous time series)
- **Auto-selection:** Grid search over (p,d,q) via AIC minimization
- **Outputs:** Forecast chart with confidence intervals, walk-forward validation, RMSE/MAE/MAPE

---

## Output Charts

| File                          | Description                        |
|-------------------------------|------------------------------------|
| `rf_confusion_matrix.png`     | TP/TN/FP/FN breakdown              |
| `rf_roc_curve.png`            | ROC curve with AUC score           |
| `rf_feature_importance.png`   | Top 15 most predictive features    |
| `arima_forecast.png`          | Historical + forecast + CI band    |
| `arima_diagnostics.png`       | Residuals, ACF, PACF, QQ plot      |
| `arima_walk_forward.png`      | Walk-forward validation comparison |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in activated venv |
| `FileNotFoundError: layoffs.csv` | Place your CSV in the `data/` folder |
| ARIMA takes too long | Reduce `max_combinations` in `auto_select_order()` |
| `layoffs_occurred` column missing | Add it to CSV, or ensure `layoff_rate` column exists |
| Charts not showing in UI | Train models first via `/train` route |
