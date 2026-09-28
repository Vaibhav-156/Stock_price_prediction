# Predictor - Stock Price Prediction Model

A machine learning project that predicts next-day stock price movement from historical market data. The model uses seven-day lagged features and a stacking classifier built from Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM) estimators.

## Project Highlights

- Built a stacking classification model to predict next-day stock price movement.
- Created seven-day lagged features from historical stock data.
- Combined Logistic Regression, KNN, and SVM in the classifier ensemble.
- Evaluated predictive reliability with Accuracy, ROC-AUC, and a Confusion Matrix.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn

## Model Workflow

1. Load and prepare historical stock price data.
2. Generate lagged features using the previous seven trading days.
3. Train a stacking classifier using Logistic Regression, KNN, and SVM.
4. Predict the direction of the next day's stock price movement.
5. Evaluate the model using Accuracy, ROC-AUC, and a Confusion Matrix.

## Repository Structure

```text
backend/
|-- app/
|   |-- api/          # API routes for the prediction service
|   |-- ml/           # Feature engineering, models, training, and inference
|   |-- services/     # Market data services
|   `-- main.py       # FastAPI application entry point
|-- requirements.txt  # Python dependencies
`-- saved_models/     # Persisted trained model artifacts

frontend/             # Next.js interface for interacting with the model
```

## Getting Started

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`. Interactive API documentation is available at `http://localhost:8000/docs`.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

The dashboard is available at `http://localhost:3000`.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified price movements. |
| ROC-AUC | Measures how well the model distinguishes between upward and downward movement. |
| Confusion Matrix | Shows the counts of correct and incorrect predictions by class. |

## Disclaimer

This project is for educational and research purposes. It is not financial advice, and model predictions do not guarantee investment returns.
