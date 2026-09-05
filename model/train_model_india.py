import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

print("Loading Indore ride data...")
df = pd.read_csv("data/indore_ola_clean.csv")
df["pickup_datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df["pickup_window"] = df["pickup_datetime"].dt.floor("15min")

print("Encoding zone names...")
le = LabelEncoder()
df["zone_id"] = le.fit_transform(df["Pickup Location"])

print("Aggregating demand per zone per window...")
demand = df.groupby(["pickup_window", "zone_id"]).size().reset_index(name="trip_count")

all_windows = pd.date_range(demand["pickup_window"].min(), demand["pickup_window"].max(), freq="15min")
all_zones = demand["zone_id"].unique()
full_index = pd.MultiIndex.from_product([all_windows, all_zones], names=["pickup_window", "zone_id"])
demand = demand.set_index(["pickup_window", "zone_id"]).reindex(full_index, fill_value=0).reset_index()

demand = demand.sort_values(["zone_id", "pickup_window"])

print("Engineering features...")
demand["hour"] = demand["pickup_window"].dt.hour
demand["day_of_week"] = demand["pickup_window"].dt.dayofweek
demand["lag_1"] = demand.groupby("zone_id")["trip_count"].shift(1)
demand["lag_2"] = demand.groupby("zone_id")["trip_count"].shift(2)
demand["rolling_mean_4"] = demand.groupby("zone_id")["trip_count"].shift(1).rolling(4).mean().reset_index(level=0, drop=True)

demand = demand.dropna()

demand["target"] = demand.groupby("zone_id")["trip_count"].shift(-1)
demand = demand.dropna()

feature_cols = ["zone_id", "hour", "day_of_week", "lag_1", "lag_2", "rolling_mean_4"]
X = demand[feature_cols]
y = demand["target"]

split_time = demand["pickup_window"].quantile(0.8)
train_mask = demand["pickup_window"] <= split_time
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

print("Training XGBoost model...")
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=2)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

joblib.dump(model, "model/demand_model_india.pkl")
joblib.dump(le, "model/zone_encoder_india.pkl")
print("Model and zone encoder saved.")
