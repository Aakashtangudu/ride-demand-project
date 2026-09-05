import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="NYC Ride Demand Predictor", layout="wide")

st.title("NYC Ride Demand Predictor")

@st.cache_data
def load_zone_lookup():
    return pd.read_csv("../data/taxi_zone_lookup.csv")

@st.cache_data
def load_demand_data():
    df = pd.read_parquet("../data/yellow_tripdata_2024-01.parquet", columns=["tpep_pickup_datetime", "PULocationID"])
    df = df[(df["tpep_pickup_datetime"] >= "2024-01-01") & (df["tpep_pickup_datetime"] < "2024-01-15")]
    df["pickup_window"] = df["tpep_pickup_datetime"].dt.floor("15min")
    demand = df.groupby(["pickup_window", "PULocationID"]).size().reset_index(name="trip_count")
    demand = demand.rename(columns={"PULocationID": "zone_id"})
    return demand

zones = load_zone_lookup()
demand = load_demand_data()

available_windows = sorted(demand["pickup_window"].unique())

st.sidebar.header("Select a time")
selected_window = st.sidebar.selectbox(
    "Pickup window",
    available_windows,
    index=len(available_windows) - 1,
    format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M")
)

st.sidebar.write(f"Predicting demand for the window after {pd.Timestamp(selected_window).strftime('%Y-%m-%d %H:%M')}")

if st.sidebar.button("Predict demand for all zones"):
    results = []
    hour = pd.Timestamp(selected_window).hour
    day_of_week = pd.Timestamp(selected_window).dayofweek

    for zone_id in demand["zone_id"].unique():
        zone_history = demand[(demand["zone_id"] == zone_id) & (demand["pickup_window"] <= selected_window)].sort_values("pickup_window")
        if len(zone_history) < 4:
            continue
        lag_1 = zone_history.iloc[-1]["trip_count"]
        lag_2 = zone_history.iloc[-2]["trip_count"]
        rolling_mean_4 = zone_history.iloc[-4:]["trip_count"].mean()

        payload = {
            "zone_id": int(zone_id),
            "hour": int(hour),
            "day_of_week": int(day_of_week),
            "lag_1": float(lag_1),
            "lag_2": float(lag_2),
            "rolling_mean_4": float(rolling_mean_4)
        }
        try:
            resp = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=5)
            pred = resp.json()["predicted_trip_count"]
        except Exception:
            pred = None

        results.append({"zone_id": zone_id, "predicted_demand": pred})

    results_df = pd.DataFrame(results).dropna()
    results_df = results_df.merge(zones, left_on="zone_id", right_on="LocationID", how="left")
    results_df = results_df.sort_values("predicted_demand", ascending=False)

    st.subheader("Predicted demand by zone (top 20)")
    st.dataframe(results_df[["zone_id", "Zone", "Borough", "predicted_demand"]].head(20), use_container_width=True)

    st.bar_chart(results_df.set_index("Zone")["predicted_demand"].head(15))
else:
    st.info("Select a time window and click 'Predict demand for all zones' in the sidebar.")
