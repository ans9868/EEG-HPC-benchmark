from pyspark.sql import SparkSession, Row, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.storagelevel import StorageLevel
import pandas as pd
import numpy as np
import os
import gc
import time
import traceback
import glob

try:
    from src.feature_extraction import processEpoch
    from src.schema_definition import get_feature_schema, get_subject_schema
    from src.preprocess_sets import subPath, processSubPSDs, processSub
    from src.config_handler import load_config, initiate_config
except ImportError:
    from feature_extraction import processEpoch
    from schema_definition import get_feature_schema, get_subject_schema
    from preprocess_sets import subPath, processSubPSDs, processSub
    from config_handler import load_config, initiate_config

try:
    config = load_config()
except RuntimeError:
    config = initiate_config()

def participantsInfoPath(config):
    return os.path.join(config['data_path'], 'ds004504', 'participants.tsv')

def load_subjects_df(spark: SparkSession, config, participants_path: str = "") -> DataFrame:
    path = participants_path or participantsInfoPath(config)
    participantsInfo = pd.read_table(path)
    records = []
    for group_code in ["A", "C", "F"]:
        group_subjects = participantsInfo[participantsInfo["Group"] == group_code]["participant_id"].tolist()
        for sub in group_subjects:
            records.append((sub, group_code))
    return spark.createDataFrame(records, schema=get_subject_schema())

def extract_features_udtf(pdf: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in pdf.iterrows():
        try:
            eeg_data = np.stack(np.array(row["EEG"]))
            min_len = min(len(ch) for ch in eeg_data)
            eeg_data = [ch[:min_len] for ch in eeg_data]
            feature_rows = processEpoch(
                row["SubjectID"], row["EpochID"], eeg_data,
                channelNames=row["ChannelNames"], sfreq=float(row["SFreq"])
            )
            results.extend(feature_rows)
        except Exception as e:
            print(f"[ERROR] {row['SubjectID']}:{row['EpochID']} - {e}")
    return pd.DataFrame([r.asDict() for r in results])

def process_subjects_parallel(spark: SparkSession, config, df: DataFrame, output_base_dir="/Volumes/CrucialX6/spark_data", force_recompute=False):
    input_path = f"{output_base_dir}/tmp_input_data"
    output_path = f"{output_base_dir}/tmp_epochs_processed"
    os.makedirs(output_path, exist_ok=True)

    if not os.path.exists(input_path) or force_recompute:
        print(f"Saving input DataFrame to {input_path}...")
        df.write.mode("overwrite").parquet(input_path)

    def should_process_subject(subject_id):
        if force_recompute:
            return True
        subject_dir = f"{output_path}/SubjectID={subject_id}"
        return not os.path.exists(subject_dir) or not glob.glob(f"{subject_dir}/*.parquet")

    def process_partition(subject_ids_iter):
        worker_spark = SparkSession.builder.getOrCreate()
        results = []
        for subject_id in subject_ids_iter:
            try:
                start = time.time()
                subject_df = worker_spark.read.parquet(input_path).filter(f"SubjectID = '{subject_id}'")
                subject_result = subject_df.groupBy("SubjectID").applyInPandas(
                    extract_features_udtf,
                    schema=get_feature_schema()
                )
                subject_result.coalesce(1).write.mode("overwrite").parquet(f"{output_path}/SubjectID={subject_id}")
                results.append({"subject_id": subject_id, "status": "success", "count": subject_result.count(), "duration": time.time() - start, "error": None})
                subject_result.unpersist()
                subject_df.unpersist()
                worker_spark.catalog.clearCache()
                gc.collect()
            except Exception as e:
                results.append({"subject_id": subject_id, "status": "failed", "count": 0, "duration": 0, "error": str(e)})
        return results

    subject_ids = [row.SubjectID for row in df.select("SubjectID").distinct().collect()]
    subjects_to_process = [sid for sid in subject_ids if should_process_subject(sid)]
    if not subjects_to_process:
        return spark.read.parquet(output_path)

    partitions = min(spark.sparkContext.defaultParallelism * 2, len(subjects_to_process))
    rdd = spark.sparkContext.parallelize(subjects_to_process, numSlices=partitions)
    all_results = rdd.mapPartitions(process_partition).flatMap(lambda x: x).collect()

    for result in all_results:
        status = result['status']
        print(f"{status.upper()}: {result['subject_id']} ({result['count']} rows, {result['duration']:.2f}s)" if status == 'success' else f"FAILED: {result['subject_id']} - {result['error']}")

    spark.catalog.clearCache()
    gc.collect()

    return spark.read.parquet(output_path)