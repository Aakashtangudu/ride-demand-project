from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, count
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType

spark = SparkSession.builder \
    .appName("RideDemandStreaming") \
    .master("local[2]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("pickup_datetime", StringType()),
    StructField("dropoff_datetime", StringType()),
    StructField("pickup_zone_id", IntegerType()),
    StructField("dropoff_zone_id", IntegerType()),
    StructField("passenger_count", DoubleType()),
    StructField("trip_distance", DoubleType()),
    StructField("fare_amount", DoubleType()),
    StructField("total_amount", DoubleType()),
])

raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "trip-events") \
    .option("startingOffsets", "earliest") \
    .load()

parsed = raw.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")

parsed = parsed.withColumn("pickup_ts", col("pickup_datetime").cast(TimestampType()))

windowed = parsed \
    .withWatermark("pickup_ts", "10 minutes") \
    .groupBy(
        window(col("pickup_ts"), "5 minutes"),
        col("pickup_zone_id")
    ) \
    .agg(count("*").alias("trip_count"))

query = windowed.writeStream \
    .outputMode("update") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query.awaitTermination()
