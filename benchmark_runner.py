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
import psutil 


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


    os.environ["_JAVA_OPTIONS"] = "-Xmx12g -Xms4g"

# Build Spark session with memory, parallelism, and network settings
    # spark = (
    #     SparkSession.builder
    #     .appName("EEG_Analysis")
    # 
    #     # Use 8 threads for better CPU balance on an M4 (assume 8–10 efficiency/performance cores)
    #     .config("spark.master", "local[8]")
    # 
    #     # Optimize memory use (you can likely push this higher depending on RAM)
    #     .config("spark.executor.memory", "18g")
    #     .config("spark.driver.memory", "18g")
    # 
    #     # Reduce shuffle overhead in local mode
    #     .config("spark.sql.shuffle.partitions", "100")
    #     .config("spark.sql.adaptive.enabled", "true")
    #     # Match parallelism to CPU threads
    #     .config("spark.default.parallelism", "8")
    #     .config("spark.sql.adaptive.shuffle.targetPostShuffleInputSize", "5120000")  # 5MB
    #     # Larger RPC size helps with big EEG rows
    #     .config("spark.rpc.message.maxSize", "256")
    #     .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    #     .config("spark.sql.execution.arrow.maxRecordsPerBatch", "500")
    #     # Ensure no duplicate/conflicting config keys (you had spark.master twice!)
    #     .config("spark.driver.bindAddress", "127.0.0.1")
    #     .config("spark.driver.host", "127.0.0.1")
    #     
    #     .getOrCreate()
    # )
   
# Path to your external SSD
    external_ssd_path = "/Volumes/CrucialX6/spark_temp"
    
    # Calculate available system memory
    system_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"Total system memory: {system_memory_gb:.2f} GB")
    
    # Reserve memory for OS and other processes (25% or at least 4GB)
    reserve_memory_gb = max(4, system_memory_gb * 0.25)
    available_memory_gb = system_memory_gb - reserve_memory_gb
    
    # Calculate reasonable memory settings (rounded down)
    driver_memory_gb = int(available_memory_gb * 0.6)  # 60% for driver
    executor_memory_gb = int(available_memory_gb * 0.3)  # 30% for executor
    
    print(f"Setting driver memory: {driver_memory_gb}g")
    print(f"Setting executor memory: {executor_memory_gb}g")
    
    # Create SparkSession with optimized memory configuration
    spark = (SparkSession.builder
        .appName("EEG-Analysis")
        .config("spark.driver.memory", f"{driver_memory_gb}g")
        .config("spark.executor.memory", f"{executor_memory_gb}g")
        .config("spark.memory.fraction", "0.7")  # Fraction of heap used for execution and storage
        .config("spark.memory.storageFraction", "0.5")  # Fraction used for storage (caching)
        .config("spark.shuffle.spill.numElementsForceSpillThreshold", "5000")
        # Add this to force more aggressive spillage
        .config("spark.sql.shuffle.partitions", "200")  # Higher than default
        .config("spark.driver.maxResultSize", f"{int(driver_memory_gb * 0.5)}g")  # Limit result collection size
        
        # Storage locations on external SSD
        .config("spark.local.dir", external_ssd_path)
        .config("spark.worker.dir", external_ssd_path)
        .config("spark.shuffle.service.index.cache.entries", "4096")
        .config("spark.sql.warehouse.dir", f"{external_ssd_path}/warehouse")
        
        # Temporary file locations and GC settings
        .config("spark.driver.extraJavaOptions", 
                f"-Djava.io.tmpdir={external_ssd_path}/tmp -XX:+UseG1GC") # -XX:+PrintGCDetails") this is to print the memorry details 
        .config("spark.executor.extraJavaOptions", 
                f"-Djava.io.tmpdir={external_ssd_path}/tmp -XX:+UseG1GC")
                
        # Optimize disk spill performance
        .config("spark.shuffle.file.buffer", "1m")
        .config("spark.shuffle.spill.compress", "true")
        .config("spark.shuffle.compress", "true")
        
        .getOrCreate()
    )
    
    # Create necessary directories if they don't exist
    os.makedirs(f"{external_ssd_path}/tmp", exist_ok=True)
    os.makedirs(f"{external_ssd_path}/warehouse", exist_ok=True)
    
    # Verify configuration
    print(f"Driver memory configured: {spark.conf.get('spark.driver.memory')}")
    print(f"Storage directory: {spark.conf.get('spark.local.dir')}")
    
        
    
    # Enable Apache Arrow (huge for pandas UDFs)
    
    
    print("New Spark session created successfully")

    print(f"Memory usage before: {psutil.virtual_memory().percent}%")


    # load / create config file 
    print("initiate config")
    initiate_config()
    print("load config")
    config = load_config()
    print("config loaded: ", config)

    # Making all the modules files available to spark
    from populate_schemas import load_subjects_df, process_subjects_parallel, extract_features_udtf
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
    # subject_ids = ['sub-001', 'sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-008', 'sub-009', 'sub-010']
    subject_ids = [f'sub-{i:03d}' for i in range(1, 89)]
    # subject_ids = ['sub-003'] #, 'sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007', 'sub-008', 'sub-009', 'sub-010']
    print("loading subjects")
    abs_start = time.time()
    df_epochs, df_metadata = load_subjects_spark(spark, subject_ids)
    
    print("got epochs and metadata")
    
    print("df_epochs")
    # df_epochs.show(2)
    print("df_metadata")
    # df_metadata.show(2)
    # Join to add metadata (sfreq, ch_names) to each epoch
    print("joining epoch table and metatable ")
    df = join_epochs_with_metadata(df_epochs, df_metadata)

    # Basic preview
    df.printSchema()
    df.show(2)
    print(f"Loaded data shape: ({df.count()}, {len(df.columns)})")


    '''
    total_subjects = df.select("SubjectID").distinct().count()
    available_executors = 5  # Replace with your cluster's executor count
    partitions_per_executor = 2  # Adjust based on memory availability

    num_partitions = min(total_subjects, available_executors * partitions_per_executor)
    '''
    num_partitions = 88 #this can and should be optimised later

    # Optimize - might break since trying to get all subjects bruh
    # df = df.repartition(num_partitions , "SubjectID").persist(StorageLevel.MEMORY_AND_DISK)
   
    print("repartition finished")
    start = time.time()
    # Apply the UDTF: One subject group at a time
    #
    print("[SPARK] Applying extract_features_udtf...")
    
    subs = process_subjects_parallel(spark, df, output_base_dir=external_ssd_path)
    # subs = (
    #     df.groupBy("SubjectID").applyInPandas(
    #         extract_features_udtf,
    #         schema="""
    #             SubjectID string,
    #             EpochID string,
    #             Electrode string,
    #             WaveBand string,
    #             FeatureName string,
    #             FeatureValue double,
    #             table_type string
    #         """
    #     )
    # )
    
    subs = subs.repartition("SubjectID")
    rows = subs.count()
    
    end = time.time()

    print("[SPARK] Saving data partitioned by SubjectID...")
    subs.write.partitionBy("SubjectID").parquet("output_path/features_by_subject")

    cols = len(subs.columns)
    print(f"✅ Shape of extracted features for subject(s): ({rows}, {cols})")
    print("unique subjects")
    df.select("SubjectID").distinct().show()
    # Benchmark time
    total_time = end - start
    print(f"✅ Total runtime: {total_time / 60:.2f} minutes")
    print(f"✅ absolute Total runtime: {end - abs_start / 60:.2f} minutes")
    print(f"Memory usage after: {psutil.virtual_memory().percent}%")

    subs.show(3)
    spark.stop()
    print("=== FINISHED ===")
    print("elapsed time of the udtf: ", )
   
    spark.stop()

if __name__ == "__main__":
    main()
