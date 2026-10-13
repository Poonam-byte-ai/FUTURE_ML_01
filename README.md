# Sales & Demand Forecasting

## Project Overview
This project uses Machine Learning to analyze historical retail sales data and predict sales using time-based features.

## Objectives
- Analyze historical sales trends.
- Prepare and clean sales data.
- Train a Random Forest Regression model.
- Evaluate predictions using MAE and RMSE.
- Visualize actual sales against predicted sales.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn

## Dataset
The project uses the Store Sales Time Series Forecasting dataset from Kaggle.

Dataset: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

## Project Files
- `sales_forecasting.py` — Data analysis, model training, and evaluation.
- `sales_trend.png` — Historical sales visualization.
- `sales_forecast.png` — Actual vs. predicted sales visualization.
- `forecast_results.csv` — Actual and predicted sales results.

## Model Evaluation
The model is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). The resulting values are printed when the script runs.

## How to Run
1. Install the required Python libraries.
2. Place `train.csv` in the project folder.
3. Run `python sales_forecasting.py`.

## Disclaimer
The model's predictions are experimental and should be validated before being used for real business decisions.