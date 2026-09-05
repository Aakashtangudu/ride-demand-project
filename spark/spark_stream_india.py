from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

spark = SparkSession.builder \
    .appName("RideDemandStreamingIndia") \
    .master("local[2]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("pickup_datetime", StringType()),
    StructField("pickup_zone", StringType()),
    StructField("drop_zone", StringType()),
    StructField("booking_status", StringType()),
    StructField("vehicle_type", StringType()),
    StructField("ride_distance", DoubleType()),
    StructField("booking_value", DoubleType()),
])

raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "trip-events-india") \
    .option("startingOffsets", "earliest") \
    .load()

parsed = raw.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

parsed = parsed.withColumn("pickup_ts", col("pickup_datetime").cast(TimestampType()))

windowed = parsed \
    .withWatermark("pickup_ts", "30 minutes") \
    .groupBy(
        window(col("pickup_ts"), "15 minutes"),
        col("pickup_zone")
    ) \
    .agg(count("*").alias("booking_count"))

query = windowed.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query.awaitTermination()
