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

# ROOT_DIR = "."
# SRC_DIR = ROOT_DIR + "/src"
# sys.path.append("/Users/admin/projectst/EEG-Feature-Benchmark")

from config_handler import initiate_config, load_config


def main():
    print("start")
    start = time.time()
    # change the spar ksettings ! 



    # Cluster configuration for PySpark - 2 nodes (16 cores, 64GB RAM total)
    import psutil
    import os
    
    # Define storage paths (modify as needed for your cluster)
    external_ssd_path = "/scratch/ans9868/spark_temp"
    os.makedirs(external_ssd_path, exist_ok=True)
    
    # Calculate cluster resources
    total_cores = 16
    total_memory_gb = 64
    nodes = 2
    
    # Reserve memory for OS and system processes (15% per node)
    reserve_memory_per_node = total_memory_gb / nodes * 0.15
    available_memory_gb = total_memory_gb - (reserve_memory_per_node * nodes)
    
    # Allocate resources
    cores_per_executor = 2  # Recommended size for good parallelism without excessive overhead
    num_executors = nodes * (total_cores // cores_per_executor) // nodes - 1  # Reserve 1 core per node for overhead
    
    # Memory settings (account for overhead)
    executor_memory_gb = int((available_memory_gb * 0.8) / num_executors)  # 80% for executors
    driver_memory_gb = int(available_memory_gb * 0.2)  # 20% for driver
    
    # Calculate executor overhead (roughly 10% of executor memory)
    executor_overhead_mb = int(executor_memory_gb * 1024 * 0.1)
    

    
    print(f"Cluster configuration:")
    print(f"  - Total nodes: {nodes}")
    print(f"  - Total cores: {total_cores}")
    print(f"  - Total memory: {total_memory_gb}GB")
    print(f"  - Number of executors: {num_executors}")
    print(f"  - Cores per executor: {cores_per_executor}")
    print(f"  - Driver memory: {driver_memory_gb}GB")
    print(f"  - Executor memory: {executor_memory_gb}GB")
    print(f"  - Executor overhead: {executor_overhead_mb}MB")
    
    # Create SparkSession with distributed cluster configuration
    master_url = os.getenv('SPARK_URL')
    spark = (SparkSession.builder
        .appName("EEG-Analysis-Cluster")
        .master(master_url)  # Set to your cluster's master URL
        
        # Resource allocation
        .config("spark.executor.instances", str(num_executors))
        .config("spark.executor.cores", str(cores_per_executor))
        .config("spark.executor.memory", f"{executor_memory_gb}g")
        .config("spark.driver.memory", f"{driver_memory_gb}g")
        .config("spark.executor.memoryOverhead", f"{executor_overhead_mb}m")
        
        # Performance tuning
        .config("spark.default.parallelism", str(total_cores * 2))  # 2x total cores
        .config("spark.sql.shuffle.partitions", str(total_cores * 4))  # 4x total cores
        .config("spark.memory.fraction", "0.75")  # Higher for data processing workloads
        .config("spark.memory.storageFraction", "0.4")  # Balanced for compute/storage
        .config("spark.driver.maxResultSize", f"{driver_memory_gb // 2}g")
        
        # Network and shuffle optimizations
        .config("spark.reducer.maxSizeInFlight", "96m")  # Larger for faster network transfers
        .config("spark.shuffle.file.buffer", "2m")  # Increased for better I/O
        .config("spark.shuffle.io.maxRetries", "10")  # More resilient in distributed environment
        .config("spark.shuffle.io.retryWait", "30s")  # Wait longer between retries
        .config("spark.executor.heartbeatInterval", "10s")  # Set a much shorter heartbeat interval
        .config("spark.network.timeout", "800s")  # Prevent timeouts on larger operations
        
        # Storage locations
        .config("spark.local.dir", external_ssd_path)
        .config("spark.worker.dir", external_ssd_path)
        .config("spark.sql.warehouse.dir", f"{external_ssd_path}/warehouse")
        
        # Memory management and GC
        .config("spark.driver.extraJavaOptions", 
                f"-Djava.io.tmpdir={external_ssd_path}/tmp -XX:+UseG1GC -XX:G1HeapRegionSize=16m -XX:+PrintFlagsFinal -XX:+PrintReferenceGC -XX:+HeapDumpOnOutOfMemoryError")
        .config("spark.executor.extraJavaOptions", 
                f"-Djava.io.tmpdir={external_ssd_path}/tmp -XX:+UseG1GC -XX:G1HeapRegionSize=16m -XX:+UnlockDiagnosticVMOptions -XX:+G1SummarizeConcMark")
        
        # Data compression - snappy is balanced for speed/compression ratio
        .config("spark.io.compression.codec", "snappy")
        .config("spark.shuffle.compress", "true")
        .config("spark.shuffle.spill.compress", "true")
        .config("spark.broadcast.compress", "true")
        
        # Dynamic allocation (optional, remove if you want fixed allocation)
        #.config("spark.dynamicAllocation.enabled", "true")
        #.config("spark.dynamicAllocation.minExecutors", str(num_executors // 2))
        #.config("spark.dynamicAllocation.maxExecutors", str(num_executors + 4))
        
        .getOrCreate()
)

    # Optional: Set log level to reduce console output
    spark.sparkContext.setLogLevel("WARN")    
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
