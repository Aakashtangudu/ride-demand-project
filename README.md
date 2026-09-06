# Ride Demand Predictor

A real-time ride-demand prediction pipeline built end-to-end: streaming ingestion, distributed stream processing, ML-based demand forecasting, a serving API, and a live dashboard.

**Live demo:** _add your Streamlit Cloud URL here_

Built and validated on two independent datasets to prove the pipeline is data-source agnostic:
- **NYC Yellow Taxi** (TLC trip records, 2.9M+ trips, numeric zone IDs)
- **Indore, India** (Ola ride bookings, 100K bookings, 50 named zones)

## Architecture

    Data Source (CSV/Parquet)
            |
            v
       Kafka Producer  ---->  Redpanda (Kafka-compatible broker)
                                    |
                                    v
                        Spark Structured Streaming
                        (windowed demand aggregation)
                                    |
                                    v
                       XGBoost Model (offline training)
                                    |
                                    v
                        FastAPI Serving Layer
                                    |
                                    v
                        Streamlit Dashboard (deployed)

## Tech Stack

- **Streaming:** Redpanda (Kafka API), kafka-python
- **Processing:** Apache Spark Structured Streaming (PySpark)
- **ML:** XGBoost, scikit-learn, pandas
- **Serving:** FastAPI, Uvicorn
- **Dashboard:** Streamlit
- **Infra:** Docker Compose, WSL2/Ubuntu

## Project Structure

    producer/    -> Streams historical trip data into Kafka/Redpanda as if live
    spark/       -> Structured Streaming job: windowed demand aggregation by zone
    model/       -> Feature engineering + XGBoost training scripts
    api/         -> FastAPI serving layer for live predictions
    dashboard/   -> Streamlit app (deployed version predicts directly from the model file)
    data/        -> Cleaned datasets used for training/dashboard

## Running Locally

    docker compose up -d              # start Redpanda + Postgres
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python producer/producer_india.py # stream data into Redpanda
    python spark/spark_stream_india.py # run the streaming aggregation
    python model/train_model_india.py  # train the demand model
    cd dashboard && streamlit run app_india.py

## Modeling Notes (honest findings, not just results)

- **Demand is defined as all booking requests**, not just completed rides — this reflects real-world resource-planning use cases (predicting request volume), not just fulfilled trips.
- **Adding an `is_weekend` feature made zero difference** to model accuracy. Since it's a deterministic function of `day_of_week`, XGBoost's tree splits could already represent the same decision boundary — a good reminder that engineered features need to add genuinely new information, not just restate existing features in a different form.
- **More training data (2 weeks vs. 1 week) slightly worsened MAE/RMSE** on the NYC dataset. This wasn't a bug — a longer window introduced more weekday/weekend variability that simple lag-based features couldn't fully capture, illustrating that "more data" isn't automatically better without matching feature complexity.
- **Indore's model (MAE 0.688) meaningfully outperformed NYC's (MAE ~1.2)** — smaller, lower-volume zones are easier to predict accurately in absolute terms than NYC's higher-variance, high-volume zones.

## Datasets

- NYC: [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Indore: [Indore Ola Dataset (Kaggle)](https://www.kaggle.com/datasets/vinaykumarpanika/indore-ola-dataset) — note: this dataset is community-created/simulated, not official government or company data.
