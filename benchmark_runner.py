import time
from pyspark.sql import SparkSession
import psutil
from src.feature_extraction import run_full_pipeline  # adjust this if needed

def main():
    start = time.time()
    spark = SparkSession.builder \
        .appName("EEG Feature Benchmark") \
        .config("spark.master", "local[8]") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()

    print(f"Memory usage before: {psutil.virtual_memory().percent}%")

   # TODO: make this run_full_pipeline
    run_full_pipeline(spark)  # Your main entry point for feature extraction

    end = time.time()
    print(f"Total runtime: {(end - start) / 60:.2f} minutes")
    print(f"Memory usage after: {psutil.virtual_memory().percent}%")

    spark.stop()

if __name__ == "__main__":
    main()
