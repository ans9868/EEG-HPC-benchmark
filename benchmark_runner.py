import time
from pyspark.sql import SparkSession
import psutil
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, MapType
import pandas as pd
from functools import reduce
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
#from src.feature_extraction import run_full_pipeline  # adjust this if needed

# importing costum module 
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


    os.environ["_JAVA_OPTIONS"] = "-Xmx12g -Xms4g"

# Build Spark session with memory, parallelism, and network settings
    spark = (
        SparkSession.builder 
        # Application name shown in Spark UI
        .appName("EEG_Analysis") 
    
        # Use all available logical cores or specify a number
        # "local[*]" uses all available cores, "local[12]" limits to 12 threads
        .config("spark.master", "local[12]") \
    
        # Executor memory: how much memory each Spark worker can use
        .config("spark.executor.memory", "8g") \
    
        # Driver memory: memory available to the Spark driver (main Python process)
        .config("spark.driver.memory", "8g") \
    
        # Number of shuffle partitions (e.g., after groupBy, join, etc.)
        # Lower this in local mode to reduce overhead (default is 200)
        .config("spark.sql.shuffle.partitions", "12") \
    
        # Default number of partitions in operations like parallelize
        .config("spark.default.parallelism", "12") \
    
        # Maximum size (in MB) allowed for any RPC message (e.g., large UDF closures or data broadcasts)
        .config("spark.rpc.message.maxSize", "256") \
        
        # to not reach 100% CPU utilizatoin and get stuck
        .config("spark.master", "local[8]")
    
        # Required for avoiding binding issues on some MacOS environments
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") 
        .getOrCreate()
    )
    # Enable Apache Arrow for pandas UDF performance boost
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    
    
    print("New Spark session created successfully")

    print(f"Memory usage before: {psutil.virtual_memory().percent}%")


    # load / create config file 
    print("initiate config")
    initiate_config()
    print("load config")
    config = load_config()
    print("config loaded: ", config)

    # Making all the modules files available to spark
    from populate_schemas import load_subjects_df, extract_features_udtf
    from feature_extraction import processEpoch, processSub
    from schema_definition import get_feature_schema, get_subject_schema
    
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
    except Exception as e:
        print(f"Error adding files to SparkContext: {e}")
    

    # load in a single subject
    subject_df = load_subjects_df(spark) #this is the .tsv with the information of all the participantsfname
    subject_df = subject_df.repartition(200, "SubjectID")  # tweak 200 based on cluster resources
    # we need to give the path of our  data directory to process the EEG data from
    from preprocess_sets import get_data_path
    
    # set_data_path("/Users/user/eeg-ds004504") !!! this doesn't work! so we need to do it manually in preprocess_sets.py! or else won't work!
    print(get_data_path())
    
    #Example below is how to get a single subject  and extract its features
    sub1 = (
        subject_df
        .filter((subject_df.SubjectID == "sub-003"))
        .groupBy("SubjectID")
        .apply(extract_features_udtf)
    )
    sub1.persist()
    # sub1.persist(StorageLevel.MEMORY_AND_DISK)
    print("3 tables, one for each layer of abraction, specific --> general (band, electrode, epoch) ")
    sub1.filter((sub1.table_type=="band")).show(3)
    sub1.filter((sub1.table_type=="electrode")).show(3)
    sub1.filter((sub1.table_type=="epoch")).show(3)
    sub1.unpersist()


    end = time.time()
    print(f"Total runtime: {(end - start) / 60:.2f} minutes")
    print(f"Memory usage after: {psutil.virtual_memory().percent}%")

    # print single subject 

    spark.stop()

if __name__ == "__main__":
    main()
