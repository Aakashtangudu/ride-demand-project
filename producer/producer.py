import pandas as pd
from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

df = pd.read_parquet("../data/sample_200.parquet")
df = df.sort_values("tpep_pickup_datetime")

print(f"Streaming {len(df)} trips...")

for i, row in df.iterrows():
    event = {
        "pickup_datetime": str(row["tpep_pickup_datetime"]),
        "dropoff_datetime": str(row["tpep_dropoff_datetime"]),
        "pickup_zone_id": int(row["PULocationID"]),
        "dropoff_zone_id": int(row["DOLocationID"]),
        "passenger_count": float(row["passenger_count"]) if pd.notna(row["passenger_count"]) else 1.0,
        "trip_distance": float(row["trip_distance"]),
        "fare_amount": float(row["fare_amount"]),
        "total_amount": float(row["total_amount"]),
    }
    producer.send("trip-events", event)

    if i % 5000 == 0:
        print(f"Sent {i} events so far...")

    time.sleep(0.01)  # controls playback speed - lower = faster stream

producer.flush()
print("Done streaming all trips.")
