import time
import os
import sys
import zipfile
import psutil
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, MapType
import pandas as pd
from pyspark import SparkContext
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import col

# Update sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./src")))
sys.path.append("/home/ans9868/EEG-HPC-benchmark/src")

from eeg_hpc_benchmark.config_handler import initiate_config, load_config, load_config_file_only

def main():
    print("start")
    start = time.time()

    os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"
    os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3"

    external_ssd_path = "/scratch/ans9868/spark_temp"
    os.makedirs(external_ssd_path, exist_ok=True)

    total_cores = 16
    total_memory_gb = 64
    nodes = 2
    reserve_memory_per_node = total_memory_gb / nodes * 0.15
    available_memory_gb = total_memory_gb - (reserve_memory_per_node * nodes)
    cores_per_executor = 2
    num_executors = nodes * (total_cores // cores_per_executor) // nodes - 1
    executor_memory_gb = int((available_memory_gb * 0.8) / num_executors)
    driver_memory_gb = int(available_memory_gb * 0.2)
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

    master_url = os.getenv('SPARK_URL')
    spark = (SparkSession.builder
        .appName("EEG-Analysis-Cluster")
        .master(master_url)
        .config("spark.executor.instances", str(num_executors))
        .config("spark.executor.cores", str(cores_per_executor))
        .config("spark.executor.memory", f"{executor_memory_gb}g")
        .config("spark.driver.memory", f"{driver_memory_gb}g")
        .config("spark.executor.memoryOverhead", f"{executor_overhead_mb}m")
        .config("spark.default.parallelism", str(total_cores * 2))
        .config("spark.sql.shuffle.partitions", str(total_cores * 4))
        .config("spark.memory.fraction", "0.75")
        .config("spark.memory.storageFraction", "0.4")
        .config("spark.driver.maxResultSize", f"{driver_memory_gb // 2}g")
        .config("spark.reducer.maxSizeInFlight", "96m")
        .config("spark.shuffle.file.buffer", "2m")
        .config("spark.shuffle.io.maxRetries", "10")
        .config("spark.shuffle.io.retryWait", "30s")
        .config("spark.executor.heartbeatInterval", "10s")
        .config("spark.network.timeout", "800s")
        .config("spark.local.dir", external_ssd_path)
        .config("spark.worker.dir", external_ssd_path)
        .config("spark.sql.warehouse.dir", f"{external_ssd_path}/warehouse")
        .config("spark.driver.extraJavaOptions",
                f"-Djava.io.tmpdir={external_ssd_path}/tmp -XX:+UseG1GC -XX:G1HeapRegionSize=16m -XX:+PrintFlagsFinal -XX:+PrintReferenceGC -XX:+HeapDumpOnOutOfMemoryError")
        .config("spark.executor.extraJavaOptions",
                f"-Djava.io.tmpdir={external_ssd_path}/tmp -XX:+UseG1GC -XX:G1HeapRegionSize=16m -XX:+UnlockDiagnosticVMOptions -XX:+G1SummarizeConcMark")
        .config("spark.io.compression.codec", "snappy")
        .config("spark.shuffle.compress", "true")
        .config("spark.shuffle.spill.compress", "true")
        .config("spark.broadcast.compress", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(f"{external_ssd_path}/tmp", exist_ok=True)
    os.makedirs(f"{external_ssd_path}/warehouse", exist_ok=True)

    print(f"Driver memory configured: {spark.conf.get('spark.driver.memory')}")
    print(f"Storage directory: {spark.conf.get('spark.local.dir')}")

    try:
        config = load_config()
    except RuntimeError:
        config = initiate_config()
    config = load_config_file_only("config.yaml")
    broadcast_config = spark.sparkContext.broadcast(config)
    config = broadcast_config.value

    print("New Spark session created successfully")
    print(f"Memory usage before: {psutil.virtual_memory().percent}%")

    ROOT_DIR = "/home/ans9868/EEG-HPC-benchmark"
    SRC_DIR = os.path.join(ROOT_DIR, "src")

    # Create and distribute the module
    sc = spark.sparkContext
    module_dir = os.path.join(SRC_DIR, "eeg_hpc_benchmark")
    zip_path = os.path.join(external_ssd_path, "eeg_hpc_benchmark.zip")

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_name in [
                "__init__.py",
                "feature_extraction.py",
                "feature_extraction_helper.py",
                "preprocess_sets.py",
                "schema_definition.py",
                "config_handler.py",
                "populate_schemas.py",
                # Add dimensionality_reduction.py if needed
                # "dimensionality_reduction.py"
            ]:
                file_path = os.path.join(module_dir, file_name)
                if os.path.exists(file_path):
                    zipf.write(file_path, os.path.join("eeg_hpc_benchmark", file_name))
                    print(f"Added {file_name} to {zip_path}")
                else:
                    print(f"WARNING: File {file_path} does not exist")
        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
            sc.addPyFile(zip_path)
            print(f"Added {zip_path} to SparkContext ({os.path.getsize(zip_path)} bytes)")
        else:
            print(f"ERROR: Failed to create or add {zip_path}")
            raise RuntimeError(f"Failed to create {zip_path}")
    except Exception as e:
        print(f"ERROR creating or adding {zip_path}: {e}")
        raise

    # Import from the module
    from eeg_hpc_benchmark.populate_schemas import load_subjects_df, process_subjects_parallel, extract_features_udtf
    from eeg_hpc_benchmark.feature_extraction import processEpoch, processSub
    from eeg_hpc_benchmark.schema_definition import get_feature_schema, get_subject_schema
    from eeg_hpc_benchmark.preprocess_sets import load_subjects_spark, join_epochs_with_metadata

    subject_ids = ['sub-001']
    print("loading subjects")
    abs_start = time.time()
    df_epochs, df_metadata = load_subjects_spark(spark, subject_ids, config=config, output_base_dir=external_ssd_path)

    print("got epochs and metadata")
    print("df_epochs")
    print("df_metadata")
    print("joining epoch table and metatable")
    df = join_epochs_with_metadata(df_epochs, df_metadata)

    df.printSchema()
    df.show(2)
    print(f"Loaded data shape: ({df.count()}, {len(df.columns)})")

    num_partitions = 1
    print("repartition finished")
    start = time.time()

    print("[SPARK] Applying process_subjects_parallel...")
    subs = process_subjects_parallel(spark=spark, epochs_df=df_epochs, metadata_df=df_metadata, output_base_dir=external_ssd_path)

    subs = subs.repartition("SubjectID")
    rows = subs.count()
    end = time.time()

    print("[SPARK] Saving data partitioned by SubjectID...")
    cols = len(subs.columns)
    print(f"✅ Shape of extracted features for subject(s): ({rows}, {cols})")
    print("unique subjects")
    df.select("SubjectID").distinct().show()
    total_time = end - start
    print(f"✅ Total runtime: {total_time / 60:.2f} minutes")
    print(f"✅ absolute Total runtime: {end - abs_start / 60:.2f} minutes")
    print(f"Memory usage after: {psutil.virtual_memory().percent}%")

    subs.show(3)
    spark.stop()
    print("=== FINISHED ===")

if __name__ == "__main__":
    main()