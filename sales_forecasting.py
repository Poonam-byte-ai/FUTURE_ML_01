import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Total sales per day
daily_sales = df.groupby("date")["sales"].sum().reset_index()

print(daily_sales.head())
print("Total days:", len(daily_sales))

# Plot historical sales
plt.figure(figsize=(12, 5))
plt.plot(daily_sales["date"], daily_sales["sales"])
plt.title("Historical Daily Sales")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("sales_trend.png")
plt.show()


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Create time-based features
daily_sales["day"] = daily_sales["date"].dt.day
daily_sales["month"] = daily_sales["date"].dt.month
daily_sales["year"] = daily_sales["date"].dt.year
daily_sales["day_of_week"] = daily_sales["date"].dt.dayofweek
daily_sales["lag_7"] = daily_sales["sales"].shift(7)
daily_sales["lag_14"] = daily_sales["sales"].shift(14)

daily_sales = daily_sales.dropna().reset_index(drop=True)

# Chronological train-test split
split = int(len(daily_sales) * 0.8)
train = daily_sales.iloc[:split]
test = daily_sales.iloc[split:]

features = ["day", "month", "year", "day_of_week", "lag_7", "lag_14"]

X_train = train[features]
y_train = train["sales"]
X_test = test[features]
y_test = test["sales"]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Mean Absolute Error (MAE):", round(mae, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))

# Plot actual vs predicted sales
plt.figure(figsize=(12, 5))
plt.plot(test["date"], y_test, label="Actual Sales")
plt.plot(test["date"], predictions, label="Predicted Sales")
plt.title("Actual vs Predicted Daily Sales")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.legend()
plt.tight_layout()
plt.savefig("sales_forecast.png")
plt.show()

results = pd.DataFrame({
    "Date": test["date"].values,
    "Actual_Sales": y_test.values,
    "Predicted_Sales": predictions
})

results.to_csv("forecast_results.csv", index=False)

print("Forecast results saved successfully!")