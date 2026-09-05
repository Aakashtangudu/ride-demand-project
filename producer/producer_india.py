import pandas as pd
from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

df = pd.read_csv("../data/indore_sample_100.csv")
df["pickup_datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.sort_values("pickup_datetime")

print(f"Streaming {len(df)} bookings...")

for i, row in df.iterrows():
    event = {
        "pickup_datetime": str(row["pickup_datetime"]),
        "pickup_zone": str(row["Pickup Location"]),
        "drop_zone": str(row["Drop Location"]),
        "booking_status": str(row["Booking Status"]),
        "vehicle_type": str(row["Vehicle Type"]) if pd.notna(row["Vehicle Type"]) else "unknown",
        "ride_distance": float(row["Ride Distance"]) if pd.notna(row["Ride Distance"]) else 0.0,
        "booking_value": float(row["Booking Value"]) if pd.notna(row["Booking Value"]) else 0.0,
    }
    producer.send("trip-events-india", event)

    if i % 5000 == 0:
        print(f"Sent {i} events so far...")

    time.sleep(0.01)

producer.flush()
print("Done streaming all bookings.")
