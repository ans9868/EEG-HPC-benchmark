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

from pyspark.sql import SparkSession
def process_subjects_parallel(spark: SparkSession, df, output_base_dir="/Volumes/CrucialX6/spark_data", 
                             force_recompute=False):
    """
    Process subjects in parallel, using your existing extract_features_udtf.
    
    Parameters:
    - spark: SparkSession
    - df: DataFrame containing EEG data
    - extract_features_udtf: Your existing UDTF that calls processEpoch
    - output_base_dir: Base directory for I/O
    - force_recompute: Whether to force recomputation
    """
    import gc
    import os
    import time
    import glob
    from pyspark.sql import SparkSession

    # Define paths
    input_path = f"{output_base_dir}/tmp_input_data"
    output_path = f"{output_base_dir}/tmp_epochs_processed"
    os.makedirs(output_path, exist_ok=True)
    
    # Check if input data exists
    input_exists = os.path.exists(input_path) and len(glob.glob(f"{input_path}/*.parquet")) > 0
    if not input_exists or force_recompute:
        print(f"Saving input DataFrame to {input_path}...")
        df.write.mode("overwrite").parquet(input_path)
    else:
        print(f"Using existing input data from {input_path}")
    
    # Function to check if a subject is already processed
    def should_process_subject(subject_id):
        if force_recompute:
            return True
        subject_dir = f"{output_path}/SubjectID={subject_id}"
        if not os.path.exists(subject_dir):
            return True
        parquet_files = glob.glob(f"{subject_dir}/*.parquet")
        if not parquet_files:
            return True
        return False
    
    # Define worker function to process a partition of subjects
    def process_partition(subject_ids_iter):
        """Process a partition of subject IDs - runs on the executor"""
        from pyspark.sql import SparkSession
        import traceback
        
        # Create a worker SparkSession
        worker_spark = SparkSession.builder.getOrCreate()
        
        # Process each subject in this partition
        results = []
        for subject_id in subject_ids_iter:
            try:
                print(f"Processing subject: {subject_id}")
                start_time = time.time()
                
                # Read subject data
                subject_df = worker_spark.read.parquet(input_path).filter(f"SubjectID = '{subject_id}'")
                
                # Process using your existing UDTF
                subject_result = subject_df.groupBy("SubjectID").applyInPandas(
                    extract_features_udtf,  # Your existing UDTF that calls processEpoch
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
                
                # Write results
                subject_path = f"{output_path}/SubjectID={subject_id}"
                os.makedirs(os.path.dirname(subject_path), exist_ok=True)
                subject_result.coalesce(1).write.mode("overwrite").parquet(subject_path)
                
                # Get stats for reporting
                result_count = subject_result.count()
                duration = time.time() - start_time
                
                # Record success
                results.append({
                    "subject_id": subject_id,
                    "status": "success",
                    "count": result_count,
                    "duration": duration,
                    "error": None
                })

                    # Clean up memory for this subject
                subject_result.unpersist()
                subject_df.unpersist()
                # Force Python garbage collection
                import gc
                gc.collect()
                # Clear Spark cache for this subject
                worker_spark.catalog.clearCache()
                print(f"Cleaned up memory after processing subject {subject_id}")
                
            except Exception as e:
                error_msg = str(e)
                trace = traceback.format_exc()
                print(f"Error processing subject {subject_id}: {error_msg}")
                print(trace)
                
                # Record failure
                results.append({
                    "subject_id": subject_id,
                    "status": "failed",
                    "count": 0,
                    "duration": 0, 
                    "error": error_msg
                })

                try:
                    import gc
                    gc.collect()
                    worker_spark.catalog.clearCache()
                except:
                    pass
        
        return results
    
    # Get subjects to process
    subject_ids = [row.SubjectID for row in df.select("SubjectID").distinct().collect()]
    print(f"Found {len(subject_ids)} unique subjects")
    
    subjects_to_process = [sid for sid in subject_ids if should_process_subject(sid)]
    if not subjects_to_process:
        print("No subjects need processing, using existing output")
        return spark.read.parquet(output_path)
    
    print(f"Processing {len(subjects_to_process)} subjects in parallel")
    
    # Calculate partitions
    num_cores = spark.sparkContext.defaultParallelism
    num_subjects = len(subjects_to_process)
    recommended_partitions = min(num_cores * 2, num_subjects)
    print(f"Using {recommended_partitions} partitions (based on {num_cores} cores)")
    
    # Create RDD and process
    rdd = spark.sparkContext.parallelize(subjects_to_process, numSlices=recommended_partitions)
    results_rdd = rdd.mapPartitions(process_partition)
    all_results = results_rdd.flatMap(lambda x: x).collect()
    
    # Report results
    successful = [r for r in all_results if r["status"] == "success"]
    failed = [r for r in all_results if r["status"] == "failed"]
    
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
    
    # Clean up
    spark.catalog.clearCache()
    gc.collect()
    
    # Return reference to output data
    return spark.read.parquet(output_path)
