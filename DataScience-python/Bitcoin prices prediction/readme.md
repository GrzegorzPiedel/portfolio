# Bitcoin Price Prediction

This project explores predicting Bitcoin prices using historical data and various regression models in Python. It demonstrates feature engineering for time series data, model comparison, and visualizations of predictions versus real prices.

## Dataset

The dataset (`bitcoin.csv`) contains daily Bitcoin prices with columns:

- `date` – date of observation (used as index)
- `value` – Bitcoin price

> **Note:** Ensure the CSV uses `,` as decimal separator or adjust preprocessing accordingly.

## Data Preprocessing

- Convert the `value` column to numeric
- Extract date features: year, month, day, weekday
- Encode cyclical features for month and weekday using sine and cosine transformations
- Create lag features and rolling mean features (lags: 1, 3, 7, 14 days)
- Drop rows with missing values after feature engineering

## Models Used

The following regression models are trained and evaluated:

- Linear Regression
- Lasso Regression (`alpha=0.1`)
- Support Vector Regression (SVR)
- K-Nearest Neighbors Regression
- Random Forest Regression
- XGBoost Regression

## Model Evaluation

- Time Series Split (5 splits) is used for validation
- Metrics:
  - R²
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- Predictions for each model are plotted against actual prices

## Visualization

- Time series plot of Bitcoin prices
- Model predictions versus real prices for each model

## How to Run

1. Clone the repository:
   ```bash
   git clone <YOUR_REPO_URL>
   cd <REPO_FOLDER>
   ```
2. Run the Python script or Jupyter notebook to see the analysis, model evaluation, and visualizations.

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib

You can install dependencies using:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

## Notes

- This project demonstrates basic time series feature engineering and regression model comparison.
- Can be extended with hyperparameter tuning, more advanced models, or additional features such as trading volume or sentiment data.
- Designed for educational purposes and portfolio demonstration.

