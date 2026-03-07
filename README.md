# Layoff Rate Forecasting Using Supervised Machine Learning Models

A web application that predicts company layoff risk using **Random Forest** classification and **ARIMA** time series forecasting, built with **FastAPI** as the backend framework.

---

## 🎯 Objective

To forecast and classify layoff risk for companies using supervised machine learning models trained on real-world layoff data from 2024–2025.

---

## 📊 Dataset

- **39 real companies** including Paytm, Byju's, Intel, Meta, Microsoft, Tesla, Flipkart, Swiggy and more
- **Time period:** January 2024 – March 2025
- **Source:** Manually collected from news reports and business publications
- **Features:** Company Name, Industry, Layoff Count, Date, City, Funding Status

---

## 🤖 Machine Learning Models

### Random Forest (Classification)
- Predicts whether a company is at **High / Medium / Low** layoff risk
- Binary target: `layoffs_occurred` (1 if layoff count ≥ 500)
- Handles imbalanced classes using `class_weight="balanced"`

### ARIMA (Time Series Forecasting)
- Forecasts **monthly layoff counts** for the next 12 months
- Auto-selects best (p, d, q) order using AIC minimization
- Walk-forward validation for evaluation

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI |
| ML Models | Scikit-learn, Statsmodels |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Frontend | HTML, CSS, JavaScript |
| Server | Uvicorn |

---

## 📁 Project Structure

```
layoff_forecasting/
├── app.py                  # FastAPI backend
├── config.py               # Model configuration
├── data_preprocessing.py   # Data cleaning & feature engineering
├── random_forest_model.py  # Random Forest training & evaluation
├── arima_model.py          # ARIMA forecasting
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
├── data/                   # Dataset folder
├── templates/
│   └── index.html          # Web interface
├── models/                 # Saved trained models
└── outputs/                # Generated charts
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Reddiah16/layoff_rate_Forecasting.git
cd layoff_rate_Forecasting
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:8000
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web dashboard |
| GET | `/docs` | Swagger API documentation |
| POST | `/train` | Train RF + ARIMA models |
| POST | `/predict` | Single company prediction |
| GET | `/metrics` | Model evaluation metrics |
| GET | `/config` | View configuration |
| POST | `/config` | Update configuration |

---

## 📈 Results

- **Random Forest Accuracy:** Trained on real layoff data
- **Charts Generated:** Confusion Matrix, ROC Curve, Feature Importance, ARIMA Forecast, Walk-Forward Validation

---

## 💡 Why FastAPI?

- **Faster than Flask** — built on Starlette and Pydantic
- **Auto-generates API docs** at `/docs` (Swagger UI)
- **Async support** — handles multiple requests efficiently
- **Type validation** — automatic request/response validation using Pydantic models
- **Production ready** — runs on Uvicorn ASGI server

---

## 👩‍💻 Author

**Reddiah16**  
Project: Layoff Rate Forecasting Using Supervised Machine Learning Models
