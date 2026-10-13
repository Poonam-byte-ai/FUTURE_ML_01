import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv("train.csv")
df["date"] = pd.to_datetime(df["date"])

# Aggregate sales by date
daily = df.groupby("date")["sales"].sum().sort_index()

# Create forecasting features
def make_features(date, history):
    return [
        date.day,
        date.month,
        date.year,
        date.dayofweek,
        date.dayofyear,
        history[-1],
        history[-7],
        history[-14],
        np.mean(history[-7:])
    ]

X, y = [], []

for i in range(14, len(daily)):
    date = daily.index[i]
    history = daily.iloc[:i].tolist()
    X.append(make_features(date, history))
    y.append(daily.iloc[i])

feature_names = [
    "day", "month", "year", "day_of_week", "day_of_year",
    "lag_1", "lag_7", "lag_14", "rolling_mean_7"
]

X = pd.DataFrame(X, columns=feature_names)
y = np.array(y)

# Chronological split
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y[:split], y[split:]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Recursive evaluation on future test period
history = daily.iloc[:split + 14].tolist()
test_dates = daily.index[split + 14:]
predictions = []

for date in test_dates:
    features = pd.DataFrame(
        [make_features(date, history)],
        columns=feature_names
    )
    pred = max(0, model.predict(features)[0])
    predictions.append(pred)
    history.append(pred)

# Evaluation
actual = daily.loc[test_dates].to_numpy()
mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

results = pd.DataFrame({
    "Date": test_dates,
    "Actual_Sales": actual,
    "Predicted_Sales": predictions
})
results.to_csv("forecast_results.csv", index=False)

# Plot evaluation results
plt.figure(figsize=(12, 5))
plt.plot(test_dates, actual, label="Actual Sales")
plt.plot(test_dates, predictions, label="Forecast Sales")
plt.title("Actual vs Forecast Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("sales_forecast.png")
plt.close()

# Historical sales plot
plt.figure(figsize=(12, 5))
plt.plot(daily.index, daily.values)
plt.title("Historical Daily Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("sales_trend.png")
plt.close()

# Retrain on all available historical data
model.fit(X, y)

# Forecast next 16 days
history = daily.tolist()
future_dates = pd.date_range(
    start=daily.index[-1] + pd.Timedelta(days=1),
    periods=16,
    freq="D"
)
future_predictions = []

for date in future_dates:
    features = pd.DataFrame(
        [make_features(date, history)],
        columns=feature_names
    )
    pred = max(0, model.predict(features)[0])
    future_predictions.append(pred)
    history.append(pred)

future = pd.DataFrame({
    "Date": future_dates,
    "Forecast_Sales": future_predictions
})
future.to_csv("future_forecast.csv", index=False)

plt.figure(figsize=(12, 5))
plt.plot(daily.index[-60:], daily.values[-60:], label="Historical Sales")
plt.plot(future_dates, future_predictions, marker="o", label="Future Forecast")
plt.title("Sales Forecast for the Next 16 Days")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("future_forecast.png")
plt.close()

print("All results and charts saved successfully!")
print(future)