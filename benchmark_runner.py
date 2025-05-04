import time
from pyspark.sql import SparkSession
import psutil
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, MapType
import pandas as pd
from functools import reduce
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import col
from pyspark import StorageLevel



#from src.feature_extraction import run_full_pipeline  # adjust this if needed

# importing costum module , don't need both but i have both lol 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./src/")))

ROOT_DIR = "."
SRC_DIR = ROOT_DIR + "/src"
sys.path.append("/Users/admin/projectst/EEG-Feature-Benchmark")

from config_handler import initiate_config, load_config


def main():
    print("start")
    start = time.time()
    # change the spar ksettings ! 

    if SparkContext._active_spark_context:
        print("Stopping existing Spark context...")
        SparkContext._active_spark_context.stop()
        print("Previous Spark context stopped successfully")


#     os.environ["_JAVA_OPTIONS"] = "-Xmx12g -Xms4g" #done automatically

   
    spark = (
        SparkSession.builder
        .appName("EEG_Analysis_HPC")

        # Target ~12 executors across the cluster (safe parallelism)
        .config("spark.executor.instances", "12")
        .config("spark.executor.cores", "1")             # 1 core per executor
        .config("spark.executor.memory", "4g")           # 12 × 4g = 48 GB total
        .config("spark.driver.memory", "6g")             # leave room for driver ops

        # Parallelism and shuffle tuning
        .config("spark.default.parallelism", "12")
        .config("spark.sql.shuffle.partitions", "12")

        # UDF / pandas optimization
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "500")

        # JVM GC tuning
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:+UseStringDeduplication")

        # Optional: Raise broadcast size if EEG config or schema is large
        .config("spark.sql.autoBroadcastJoinThreshold", "20MB")

        .getOrCreate()
)


        
    
    # Enable Apache Arrow (huge for pandas UDFs)
    
    
    print("New Spark session created successfully")
    print("=== Spark Configuration & Runtime info ===")
    print(f"App ID: {spark.sparkContext.applicationId}")
    print(f"Master: {spark.sparkContext.master}")
    print(f"Default Parallelism: {spark.sparkContext.defaultParallelism}")
    print(f"Total Executors: {spark.sparkContext._jsc.sc().getExecutorMemoryStatus().size()}")

    for item in spark.sparkContext.getConf().getAll():
        print(f"{item[0]} = {item[1]}")
    
    
    print(f"Memory usage before: {psutil.virtual_memory().percent}%")


    # load / create config file 
    print("initiate config")
    initiate_config()
    print("load config")
    config = load_config()
    print("config loaded: ", config)

    # Making all the modules files available to spark
    from populate_schemas import load_subjects_df, extract_features_udtf
    # from populate_schemas import extract_features_udtf
    from feature_extraction import processEpoch, processSub
    from schema_definition import get_feature_schema, get_subject_schema
    from preprocess_sets import load_subjects_spark, join_epochs_with_metadata


    # def join_epochs_with_metadata(df_epochs: DataFrame, df_metadata: DataFrame) -> DataFrame:

    sc = spark.sparkContext
    try:
        sc.addPyFile(os.path.join(SRC_DIR, "feature_extraction.py"))
        print("Added feature_extraction.py to the pyspark context")
        sc.addPyFile(os.path.join(SRC_DIR, "feature_extraction_helper.py"))
        print("Added feature_extraction_helper.py to the pyspark context")
        sc.addPyFile(os.path.join(SRC_DIR, "preprocess_sets.py"))
        print("Added preprocess_sets to the pyspark context")
        sc.addPyFile(os.path.join(SRC_DIR, "schema_definition.py"))
        print("Added schema_definition.py to the pyspark context")
        sc.addPyFile(os.path.join(SRC_DIR, "config_handler.py"))
        print("Added config_handler.py to the pyspark context")
        sc.addPyFile(os.path.join(SRC_DIR, "populate_schemas.py"))
        print("Added populate_schemas.py to the pyspark context")
        sc.addPyFile(os.path.join(SRC_DIR, "preprocess_sets.py"))
        print("Added preprocess_sets.py to the pyspark context")


    except Exception as e:
        print(f"Error adding files to SparkContext: {e}")
    

    # Load subject group info (e.g., labels for A/C/F)
    subject_df = load_subjects_df(spark)

    # Load data for target subjects
    subject_ids = ['sub-001', 'sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-008', 'sub-009', 'sub-010']
    # subject_ids = ['sub-003'] #, 'sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-008', 'sub-009', 'sub-010']
    print("loading subjects")
    df_epochs, df_metadata = load_subjects_spark(spark, subject_ids)
    
    print("got epochs and metadata")
    
    print("df_epochs")
    df_epochs.show(2)
    print("df_metadata")
    df_metadata.show(2)
    # Join to add metadata (sfreq, ch_names) to each epoch
    print("joining epoch table and metatable ")
    df = join_epochs_with_metadata(df_epochs, df_metadata)

    # Basic preview
    df.printSchema()
    df.show(2)
    print(f"Loaded data shape: ({df.count()}, {len(df.columns)})")

    # Optimize
    df = df.repartition(1000).persist(StorageLevel.MEMORY_AND_DISK)
    
    start = time.time()
    # Apply the UDTF: One subject group at a time
    #
    print("[SPARK] Applying extract_features_udtf...")
    subs = (
        df.groupBy("SubjectID").applyInPandas(
            extract_features_udtf,
            schema="""
                SubjectID string,
                EpochID string,
                Electrode string,
                WaveBand string,
                FeatureName string,
                FeatureValue double,
                table_type string
            """
        )
    )
    
    subs = subs.repartition(1000).persist(StorageLevel.MEMORY_AND_DISK)
    rows = subs.count()
    end = time.time()
    cols = len(subs.columns)
    print(f"✅ Shape of extracted features for subject(s): ({rows}, {cols})")
    print("unique subjects")
    df.select("SubjectID").distinct().show()
    # Benchmark time
    total_time = end - start
    print(f"✅ Total runtime: {total_time / 60:.2f} minutes")
    print(f"Memory usage after: {psutil.virtual_memory().percent}%")

    subs.show(3)
    spark.stop()
    print("=== FINISHED ===")
    print("elapsed time of the udtf: ", )
   
    spark.stop()

if __name__ == "__main__":
    main()
