import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Indore Ride Demand Predictor", layout="wide")

st.title("Indore Ride Demand Predictor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "indore_ola_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "demand_model_india.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "..", "model", "zone_encoder_india.pkl")

@st.cache_resource
def load_model_and_encoder():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

@st.cache_data
def load_demand_data():
    df = pd.read_csv(DATA_PATH)
    df["pickup_datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df["pickup_window"] = df["pickup_datetime"].dt.floor("15min")
    demand = df.groupby(["pickup_window", "Pickup Location"]).size().reset_index(name="trip_count")
    demand = demand.rename(columns={"Pickup Location": "zone_name"})
    return demand

model, encoder = load_model_and_encoder()
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

    for zone_name in demand["zone_name"].unique():
        zone_history = demand[(demand["zone_name"] == zone_name) & (demand["pickup_window"] <= selected_window)].sort_values("pickup_window")
        if len(zone_history) < 4:
            continue
        lag_1 = zone_history.iloc[-1]["trip_count"]
        lag_2 = zone_history.iloc[-2]["trip_count"]
        rolling_mean_4 = zone_history.iloc[-4:]["trip_count"].mean()

        zone_id = int(encoder.transform([zone_name])[0])
        features = pd.DataFrame([{
            "zone_id": zone_id,
            "hour": int(hour),
            "day_of_week": int(day_of_week),
            "lag_1": float(lag_1),
            "lag_2": float(lag_2),
            "rolling_mean_4": float(rolling_mean_4)
        }])
        pred = model.predict(features)[0]
        results.append({"zone_name": zone_name, "predicted_demand": float(pred)})

    results_df = pd.DataFrame(results).dropna()
    results_df = results_df.sort_values("predicted_demand", ascending=False).reset_index(drop=True)
    results_df.index = results_df.index + 1

    st.subheader("Predicted demand by zone (top 20)")
    st.dataframe(results_df.head(20), use_container_width=True)

    st.bar_chart(results_df.set_index("zone_name")["predicted_demand"].head(15))
else:
    st.info("Select a time window and click 'Predict demand for all zones' in the sidebar.")
