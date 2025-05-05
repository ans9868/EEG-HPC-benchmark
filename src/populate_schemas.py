from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, MapType
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
import gc
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
import time
import os

try:
    from src.feature_extraction import processEpoch, processSub
    from src.schema_definition import get_feature_schema, get_subject_schema
    from src.preprocess_sets import subPath, participantsInfoPath, processSubPSDs, processSub
    from src.config_handler import load_config, initiate_config
except ImportError:
    from feature_extraction import processEpoch, processSub
    from schema_definition import get_feature_schema, get_subject_schema
    from preprocess_sets import subPath, participantsInfoPath, processSubPSDs, processSub
    from config_handler import load_config, initiate_config


try:
    config = load_config()
except RuntimeError:
    config = initiate_config()



def load_subjects_df(spark: SparkSession, participants_path: str="") -> DataFrame:
    if len(participants_path) == 0:
        participantsInfo = pd.read_table(participantsInfoPath())
    else:
        participantsInfo = pd.read_table(participants_path)

    records = []
    for group_code in ["A", "C", "F"]:
        group_subjects = participantsInfo[participantsInfo["Group"] == group_code]["participant_id"].tolist()
        for sub in group_subjects:
            records.append((sub, group_code))
    return spark.createDataFrame(records, schema=get_subject_schema())




import numpy as np
import pandas as pd
from pyspark.sql import Row

from feature_extraction import processEpoch  # This function returns a list of Rows



# Maybe should flatMap/map through this and store each subject to disk like we did for preprocess_sets process_subjects_spark

# @pandas_udf(get_feature_schema(), PandasUDFType.GROUPED_MAP)
# def extract_features_udtf(pdf: pd.DataFrame) -> pd.DataFrame:
def extract_features_udtf(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a pandas DataFrame with:
    - SubjectID (str)
    - EpochID (str)
    - EEG (2D array: n_channels x n_times)
    - ChannelNames (list of str)
    - SFreq (float)

    Returns a long-format DataFrame with:
    - SubjectID, EpochID, Electrode, WaveBand, FeatureName, FeatureValue, table_type
    """
    results = []

    for _, row in pdf.iterrows():
        subject_id = row["SubjectID"]
        epoch_id = row["EpochID"]
        eeg_data = np.array(row["EEG"])
        ch_names = row["ChannelNames"]
        sfreq = float(row["SFreq"])
        
        # here we are formatting the rows to numpy arrays and ensuring they are the same size (across channels ie electrodes)
        # * This can be a bottleneck since we are using numpy for calculating PSD its necessary
        #   Solutions : pyarray, using pandas/numpy for raw eeg data
        eeg_data = np.array(row["EEG"])  # infer dtype and shape
        eeg_data = np.stack(eeg_data, axis=0)
       


        lens = [len(ch) for ch in eeg_data]                                   
        min_len = min(lens) 
        max_len = max(lens) 
        if min_len != max_len:                                                
            print(f"[WARN] Epoch {row['EpochID']} has uneven channel lengths")
            print(f"       → Shortest = {min_len}, Longest = {max_len}")      
            print(f"       → Trimming all to {min_len}")                      
            eeg_data = [ch[:min_len] for ch in eeg_data]


       
        try:
            feature_rows = processEpoch(
                subject_id,
                epoch_id,
                eeg_data,
                channelNames=ch_names,
                sfreq=sfreq
            )
            results.extend(feature_rows)
        except Exception as e:
            print(f"[ERROR] {subject_id}:{epoch_id} - {e}")

    return pd.DataFrame([r.asDict() for r in results])

def process_subjects_parallel(df, output_path="/Volumes/CrucialX6/spark_data", force_recompute=False):
    """
    Process subjects in parallel using Spark's RDD transformations
    to maintain both parallelism and memory safety.
    """
    import gc
    import os
    import time
    from pyspark.sql import SparkSession

    output_path=f"{output_path}/tmp_epochs_processed"

    os.makedirs(output_path, exist_ok=True)
    
    # Get the SparkSession
    spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    
    # Function to check if a subject has already been processed
    def should_process_subject(subject_id):
        if force_recompute:
            return True
            
        # Check if the subject directory exists with parquet files
        subject_dir = f"{output_path}/SubjectID={subject_id}"
        if not os.path.exists(subject_dir):
            return True
            
        # Check if directory contains parquet files
        import glob
        parquet_files = glob.glob(f"{subject_dir}/*.parquet")
        if not parquet_files:
            return True
            
        return False
    
    # Function to process a single subject
    def process_single_subject(subject_id):
        try:
            print(f"Processing subject: {subject_id}")
            start_time = time.time()
            
            # Filter to just this subject - this is done on the driver
            # and distributed for processing
            subject_df = df.filter(f"SubjectID = '{subject_id}'")
            
            # Process this subject - groupBy + applyInPandas
            subject_result = subject_df.groupBy("SubjectID").applyInPandas(
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
            
            # Make sure output directory exists
            subject_path = f"{output_path}/SubjectID={subject_id}"
            os.makedirs(os.path.dirname(subject_path), exist_ok=True)
            
            # Write to a single file for this subject
            subject_result = subject_result.coalesce(1)
            subject_result.write.mode("overwrite").parquet(subject_path)
            
            # Count rows for reporting
            result_count = subject_result.count()
            
            duration = time.time() - start_time
            
            # Return success metadata ie this goes to results :)
            return {
                "subject_id": subject_id,
                "status": "success",
                "count": result_count,
                "duration": duration,
                "error": None
            }
        
        except Exception as e:
            import traceback
            error_msg = str(e)
            trace = traceback.format_exc()
            print(f"Error processing subject {subject_id}: {error_msg}")
            print(trace)
            
            # Return failure metadata 
            return {
                "subject_id": subject_id,
                "status": "failed",
                "count": 0,
                "duration": 0,
                "error": error_msg
            }
    
    # Get all unique subject IDs
    subject_ids = [row.SubjectID for row in df.select("SubjectID").distinct().collect()]
    print(f"Found {len(subject_ids)} unique subjects")
    
    # Filter to only subjects that need processing
    subjects_to_process = []
    for subject_id in subject_ids:
        if should_process_subject(subject_id):
            subjects_to_process.append(subject_id)
            print(f"Subject {subject_id} will be processed")
        else:
            print(f"Subject {subject_id} already exists, skipping")
    
    if not subjects_to_process:
        print("No subjects need processing, using existing output")
        return spark.read.parquet(output_path)
    
    print(f"Processing {len(subjects_to_process)} subjects in parallel")
    
    # Calculate optimal number of partitions
    num_cores = spark.sparkContext.defaultParallelism
    num_subjects = len(subjects_to_process)
    
    # For memory safety, choose a partition count that won't process too many subjects at once
    # But still maintains good parallelism
    recommended_partitions = min(num_cores * 2, num_subjects)
    
    print(f"Using {recommended_partitions} partitions (based on {num_cores} cores)")
    
    # Create RDD of subject IDs
    rdd = spark.sparkContext.parallelize(subjects_to_process, numSlices=recommended_partitions)
    
    # Process each subject in parallel
    results_rdd = rdd.map(process_single_subject)
    
    # Collect results of the process ie if the process was successful or not (small metadata, not the full data)
    results = results_rdd.collect()
    
    # Report results
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print(f"\nProcessing complete: {len(successful)} successful, {len(failed)} failed")
    
    if successful:
        total_rows = sum(r["count"] for r in successful)
        avg_duration = sum(r["duration"] for r in successful) / len(successful)
        print(f"Generated {total_rows} total feature rows")
        print(f"Average processing time per subject: {avg_duration:.2f}s")
    
    if failed:
        print("\nFailed subjects:")
        for f in failed:
            print(f"  - {f['subject_id']}: {f['error']}")
    
    # Clear caches to free memory
    spark.catalog.clearCache()
    gc.collect()
    
    # Return reference to the output data
    return spark.read.parquet(output_path)
