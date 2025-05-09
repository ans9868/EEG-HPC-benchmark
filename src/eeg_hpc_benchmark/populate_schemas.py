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


from eeg_hpc_benchmark.feature_extraction import processEpoch, processSub
from eeg_hpc_benchmark.schema_definition import get_feature_schema, get_subject_schema
from eeg_hpc_benchmark.preprocess_sets import subPath, participantsInfoPath, processSubPSDs, processSub
from eeg_hpc_benchmark.config_handler import load_config, initiate_config



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




# Maybe should flatMap/map through this and store each subject to disk like we did for preprocess_sets process_subjects_spark

# @pandas_udf(get_feature_schema(), PandasUDFType.GROUPED_MAP)
# def extract_features_udtf(pdf: pd.DataFrame) -> pd.DataFrame:
def extract_features_udtf(pdf: pd.DataFrame, config) -> pd.DataFrame:
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

    # Extract config values
    freqBands = config['freqBands']
    windowLength = config['windowLength']
    stepSize = config['stepSize']
    method = config['method']
    
    for _, row in pdf.iterrows():
        subject_id = row["SubjectID"]
        epoch_id = row["EpochID"]
        eeg_data = np.array(row["EEG"])
        ch_names = row["ChannelNames"]
        sfreq = float(row["SFreq"])
        
        # Ensure channels are consistent
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
                sfreq=sfreq,
                freqBands=freqBands,
                windowLength=windowLength,
                stepSize=stepSize,
                method=method,
                n_jobs=1  # Default value, can be adjusted from config if needed
            )
            results.extend(feature_rows)
        except Exception as e:
            print(f"[ERROR] {subject_id}:{epoch_id} - {e}")

    return pd.DataFrame([r.asDict() for r in results])




from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType, MapType
from pyspark.sql import DataFrame
import pandas as pd
import numpy as np
import gc
import time
import os
import glob
from eeg_hpc_benchmark.feature_extraction import processEpoch
from eeg_hpc_benchmark.schema_definition import get_feature_schema, get_subject_schema
from eeg_hpc_benchmark.preprocess_sets import participantsInfoPath

def process_subjects_parallel(spark: SparkSession, epochs_df, metadata_df, config, output_base_dir="/Volumes/CrucialX6/spark_data", 
                             force_recompute=False):
    """
    Process subjects in parallel using separate epochs and metadata DataFrames.
    
    Parameters:
    - spark: SparkSession
    - epochs_df: DataFrame containing EEG epoch data
    - metadata_df: DataFrame containing metadata (SFreq, ChannelNames, etc.)
    - output_base_dir: Base directory for I/O
    - force_recompute: Whether to force recomputation

    freqBands = config['freqBands']
    windowLength = config['windowLength']
    stepSize = config['stepSize']
    method = config['method']
    """
    print("In process_subjects_parallel")

    
    # Define paths with correct directory structure
    epochs_path = f"{output_base_dir}/tmp_epochs"
    metadata_path = f"{output_base_dir}/tmp_metadata"
    output_path = f"{output_base_dir}/tmp_epochs_processed"
    os.makedirs(output_path, exist_ok=True)
    print(f"[INFO] Paths → epochs: {epochs_path}\n metadata: {metadata_path}\n output: {output_path}")
    print(f"config {config}")

    freqBands = config['freqBands']
    windowLength = config['windowLength']
    stepSize = config['stepSize']
    method = config['method']
    
    
    # Improved check for partitioned parquet data
    def check_parquet_exists(path):
        if not os.path.exists(path):
            return False
        all_parquet_files = glob.glob(f"{path}/**/*.parquet", recursive=True)
        return len(all_parquet_files) > 0
    
    # Check if epochs data exists
    epochs_exists = check_parquet_exists(epochs_path)
    if not epochs_exists or force_recompute:
        print(f"Saving epochs DataFrame to {epochs_path}...")
        epochs_df.write.mode("overwrite").parquet(epochs_path)
    else:
        print(f"Using existing epochs data from {epochs_path}")
    
    # Similarly for metadata
    metadata_exists = check_parquet_exists(metadata_path)
    if not metadata_exists or force_recompute:
        print(f"Saving metadata DataFrame to {metadata_path}...")
        metadata_df.write.mode("overwrite").parquet(metadata_path)
    else:
        print(f"Using existing metadata from {metadata_path}")
    
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
    
    # Create broadcast variables for the paths
    epochs_path_b = spark.sparkContext.broadcast(epochs_path)
    metadata_path_b = spark.sparkContext.broadcast(metadata_path)
    output_path_b = spark.sparkContext.broadcast(output_path)
    
    def process_partition(subject_ids_iter):
        """Process a partition of subject IDs with extensive debugging"""
        import os
        import sys
        import time
        import traceback
        import uuid
        
        # Generate a unique ID for this partition execution
        partition_id = str(uuid.uuid4())[:8]
        
        try:
            # Get paths from broadcast variables
            epochs_path = epochs_path_b.value
            metadata_path = metadata_path_b.value
            output_path = output_path_b.value
            
            # Create debug directory for logs
            debug_dir = f"{output_path}/debug_logs"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create a unique log file
            try:
                hostname = os.uname().nodename
            except:
                hostname = "unknown_host"
            
            timestamp = time.time()
            log_file = f"{debug_dir}/worker_{hostname}_{partition_id}_{timestamp}.log"
            
            # Function to log both to console and file
            def log(message):
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                formatted = f"[{timestamp}] [{partition_id}] {message}"
                print(formatted)
                try:
                    with open(log_file, 'a') as f:
                        f.write(formatted + "\n")
                except Exception as e:
                    print(f"ERROR WRITING TO LOG FILE: {e}")
            
            # Start logging
            log(f"=== STARTING PARTITION EXECUTION {partition_id} ===")
            log(f"Worker hostname: {hostname}")
            log(f"Log file: {log_file}")
            log(f"Python version: {sys.version}")
            log(f"Paths → epochs: {epochs_path}")
            log(f"       metadata: {metadata_path}")
            log(f"       output: {output_path}")
            
            # Log system info
            log("=== SYSTEM INFO ===")
            log(f"Current working directory: {os.getcwd()}")
            log(f"User: {os.environ.get('USER', 'unknown')}")
            log(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
            log(f"Python sys.path: {sys.path}")
            
            # Check for zip file
            zip_path = "./eeg_hpc_benchmark.zip"
            if os.path.exists(zip_path):
                log(f"Found eeg_hpc_benchmark.zip at: {zip_path}")
                try:
                    import zipfile
                    with zipfile.ZipFile(zip_path, 'r') as zipf:
                        zip_contents = zipf.namelist()
                        log(f"Zip contents: {zip_contents}")
                except Exception as e:
                    log(f"Error reading zip file: {e}")
            else:
                log(f"eeg_hpc_benchmark.zip NOT found at: {zip_path}")
            
            # Import required packages
            log("=== IMPORTING REQUIRED PACKAGES ===")
            try:
                log("Importing pandas...")
                import pandas as pd
                log("✓ pandas import successful")
                
                log("Importing numpy...")
                import numpy as np
                log("✓ numpy import successful")
                
                log("Importing mne...")
                import mne
                log("✓ mne import successful")
                
                log("Importing feature_extraction module...")
                from eeg_hpc_benchmark.feature_extraction import processEpoch
                log("✓ Successfully imported processEpoch from eeg_hpc_benchmark.feature_extraction")
            except Exception as e:
                log(f"✗ Failed to import: {e}")
                log(f"Traceback: {traceback.format_exc()}")
                return [{"subject_id": "import_error", "status": "failed", "count": 0, "duration": 0, "error": str(e)}]
            
            # Process subject data
            log("=== STARTING SUBJECT PROCESSING ===")
            subjects = list(subject_ids_iter)
            log(f"Processing {len(subjects)} subjects: {subjects[:5]}..." + ("" if len(subjects) <= 5 else f" (and {len(subjects)-5} more)"))
            
            results = []
            for subject_idx, subject_id in enumerate(subjects):
                log(f"[{subject_idx+1}/{len(subjects)}] Processing subject: {subject_id}")
                start_time = time.time()
                
                try:
                    # Load subject data files
                    subject_epochs_path = f"{epochs_path}/SubjectID={subject_id}"
                    subject_metadata_path = f"{metadata_path}/SubjectID={subject_id}"
                    
                    log(f"Looking for epochs at: {subject_epochs_path}")
                    log(f"Looking for metadata at: {subject_metadata_path}")
                    
                    # Verify directories exist
                    if not os.path.exists(subject_epochs_path):
                        log(f"ERROR: Epochs directory does not exist: {subject_epochs_path}")
                        raise FileNotFoundError(f"Epochs directory not found: {subject_epochs_path}")
                    
                    if not os.path.exists(subject_metadata_path):
                        log(f"ERROR: Metadata directory does not exist: {subject_metadata_path}")
                        raise FileNotFoundError(f"Metadata directory not found: {subject_metadata_path}")
                    
                    # Get parquet files
                    epoch_files = glob.glob(f"{subject_epochs_path}/*.parquet")
                    if not epoch_files:
                        log(f"ERROR: No epoch files found in {subject_epochs_path}")
                        log(f"Directory contents: {os.listdir(subject_epochs_path) if os.path.exists(subject_epochs_path) else 'Directory not found'}")
                        raise FileNotFoundError(f"No epoch files found for {subject_id}")
                    
                    log(f"Found {len(epoch_files)} epoch files")
                    
                    # Read epochs data
                    log(f"Reading epoch files...")
                    epochs_dfs = []
                    for i, f in enumerate(epoch_files):
                        log(f"Reading epoch file {i+1}/{len(epoch_files)}: {os.path.basename(f)}")
                        df = pd.read_parquet(f)
                        log(f"Successfully read file with shape: {df.shape}")
                        epochs_dfs.append(df)
                    
                    epochs_df = pd.concat(epochs_dfs) if len(epochs_dfs) > 1 else epochs_dfs[0]
                    log(f"Combined epochs DataFrame shape: {epochs_df.shape}")
                    log(f"Epochs DataFrame columns: {epochs_df.columns.tolist()}")
                    
                    # Add SubjectID to epochs_df if it doesn't exist
                    if 'SubjectID' not in epochs_df.columns:
                        log("Adding SubjectID column to epochs_df")
                        epochs_df['SubjectID'] = subject_id
                    
                    # Read metadata
                    meta_files = glob.glob(f"{subject_metadata_path}/*.parquet")
                    if not meta_files:
                        log(f"ERROR: No metadata files found in {subject_metadata_path}")
                        log(f"Directory contents: {os.listdir(subject_metadata_path) if os.path.exists(subject_metadata_path) else 'Directory not found'}")
                        raise FileNotFoundError(f"No metadata files found for {subject_id}")
                    
                    log(f"Found {len(meta_files)} metadata files")
                    log(f"Reading metadata file: {os.path.basename(meta_files[0])}")
                    metadata_df = pd.read_parquet(meta_files[0])
                    log(f"Metadata DataFrame shape: {metadata_df.shape}")
                    log(f"Metadata DataFrame columns: {metadata_df.columns.tolist()}")
                    
                    # Add SubjectID to metadata_df if it doesn't exist
                    if 'SubjectID' not in metadata_df.columns:
                        log("Adding SubjectID column to metadata_df")
                        metadata_df['SubjectID'] = subject_id
                    
                    # Get SFreq and ChannelNames from metadata
                    log("Extracting SFreq and ChannelNames from metadata")
                    sfreq = float(metadata_df['SFreq'].iloc[0])
                    channel_names = metadata_df['ChannelNames'].iloc[0]
                    log(f"SFreq: {sfreq}")
                    log(f"Channel names (first few): {channel_names[:5]}...")
                    
                    # Process each epoch
                    log(f"Processing {len(epochs_df)} epochs")
                    feature_rows = []
                    for i, (_, row) in enumerate(epochs_df.iterrows()):
                        if i % 10 == 0:
                            log(f"Processing epoch {i+1}/{len(epochs_df)}")
                        
                        try:
                            subject_id = row["SubjectID"]
                            epoch_id = row["EpochID"]
                            eeg_data = np.array(row["EEG"])
                            
                            # Make sure channels are consistent
                            eeg_data = np.stack(eeg_data, axis=0)
                            lens = [len(ch) for ch in eeg_data]
                            min_len = min(lens)
                            max_len = max(lens)
                            if min_len != max_len:
                                log(f"[WARN] Epoch {epoch_id} has uneven channel lengths")
                                log(f"       → Shortest = {min_len}, Longest = {max_len}")
                                log(f"       → Trimming all to {min_len}")
                                eeg_data = [ch[:min_len] for ch in eeg_data]
                            
                            # Process the epoch
                            epoch_features = processEpoch(
                                subject_id,
                                epoch_id,
                                eeg_data,
                                channelNames=channel_names,
                                sfreq=sfreq,
                                freqBands=freqBands,
                                windowLength=windowLength,
                                stepSize=stepSize,
                                method=method
                            )
                            feature_rows.extend(epoch_features)
                            if i % 10 == 0:
                                log(f"Successfully processed epoch {epoch_id} ({len(epoch_features)} features)")
                        
                        except Exception as e:
                            log(f"[ERROR] Processing epoch {epoch_id}: {e}")
                            continue
                    
                    # Convert rows to DataFrame
                    log(f"Creating result DataFrame from {len(feature_rows)} feature rows")
                    result_df = pd.DataFrame([r.asDict() for r in feature_rows])
                    log(f"Result DataFrame shape: {result_df.shape}")
                    log(f"Result DataFrame columns: {result_df.columns.tolist()}")
                    
                    # Save to parquet
                    subject_output_path = f"{output_path}/SubjectID={subject_id}"
                    log(f"Creating output directory: {subject_output_path}")
                    os.makedirs(subject_output_path, exist_ok=True)
                    result_file = f"{subject_output_path}/features.parquet"
                    log(f"Saving results to: {result_file}")
                    result_df.to_parquet(result_file, index=False)
                    log(f"✓ Successfully saved results to {result_file}")
                    
                    # Record success
                    duration = time.time() - start_time
                    log(f"✓ Successfully processed subject {subject_id} in {duration:.2f} seconds")
                    results.append({
                        "subject_id": subject_id,
                        "status": "success",
                        "count": len(result_df),
                        "duration": duration,
                        "error": None
                    })
                    
                    # Clean up memory
                    log("Cleaning up memory")
                    import gc
                    del result_df, epochs_df, metadata_df, feature_rows
                    gc.collect()
                
                except Exception as e:
                    error_msg = str(e)
                    trace = traceback.format_exc()
                    log(f"Error processing subject {subject_id}: {error_msg}")
                    log(f"Traceback: {trace}")
                    results.append({
                        "subject_id": subject_id,
                        "status": "failed",
                        "count": 0,
                        "duration": 0,
                        "error": error_msg
                    })
            
            # Summarize results
            success_count = sum(1 for r in results if r["status"] == "success")
            fail_count = len(results) - success_count
            log(f"=== PARTITION SUMMARY ===")
            log(f"Total subjects processed: {len(results)}")
            log(f"Successful: {success_count}")
            log(f"Failed: {fail_count}")
            log("=== PARTITION EXECUTION COMPLETE ===")
            return results
        
        except Exception as e:
            error_message = str(e)
            stack_trace = traceback.format_exc()
            try:
                with open(f"{output_path_b.value}/partition_error_{partition_id}.log", 'w') as f:
                    f.write(f"CRITICAL ERROR in partition {partition_id}: {error_message}\n")
                    f.write(stack_trace)
            except:
                pass
            print(f"CRITICAL ERROR in partition {partition_id}: {error_message}")
            print(stack_trace)
            return [{"subject_id": "partition_error", "status": "failed", "count": 0, "duration": 0, "error": error_message}]
    
    # Get subjects to process
    subject_ids = [row.SubjectID for row in epochs_df.select("SubjectID").distinct().collect()]
    print(f"Found {len(subject_ids)} unique subjects")
    
    subjects_to_process = [sid for sid in subject_ids if should_process_subject(sid)]
    if not subjects_to_process:
        print("No subjects need processing, using existing output")
        parquet_files = glob.glob(f"{output_path}/**/*.parquet", recursive=True)
        if not parquet_files:
            print(f"WARNING: No Parquet files found in {output_path}")
            return spark.createDataFrame([], schema=get_feature_schema())
        return spark.read.parquet(*parquet_files)
    
    print(f"Processing {len(subjects_to_process)} subjects in parallel")
    
    # Calculate partitions
    num_cores = spark.sparkContext.defaultParallelism
    num_subjects = len(subjects_to_process)
    recommended_partitions = min(num_cores * 2, num_subjects)
    print(f"Using {recommended_partitions} partitions (based on {num_cores} cores)")
    
    # Create RDD and process
    rdd = spark.sparkContext.parallelize(subjects_to_process, numSlices=recommended_partitions)
    results_rdd = rdd.mapPartitions(process_partition)
    all_results = results_rdd.collect()
    
    # Report results
    successful = []
    failed = []
    for r in all_results:
        print(f"Result: {r}")
        if isinstance(r, dict) and "status" in r:
            if r["status"] == "success":
                successful.append(r)
            elif r["status"] == "failed":
                failed.append(r)
        else:
            print(f"Warning: Unexpected result format: {type(r).__name__} - {r}")
            failed.append({
                "subject_id": "unknown",
                "status": "failed",
                "error": f"Invalid result format: {r}"
            })
    
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
    parquet_files = glob.glob(f"{output_path}/**/*.parquet", recursive=True)
    if not parquet_files:
        print(f"WARNING: No Parquet files found in {output_path}")
        return spark.createDataFrame([], schema=get_feature_schema())
    return spark.read.parquet(*parquet_files)










# def process_subjects_parallel(spark: SparkSession, epochs_df, metadata_df, output_base_dir="/Volumes/CrucialX6/spark_data", 
#                              force_recompute=False):
#     """
#     Process subjects in parallel using separate epochs and metadata DataFrames.
    
#     Parameters:
#     - spark: SparkSession
#     - epochs_df: DataFrame containing EEG epoch data
#     - metadata_df: DataFrame containing metadata (SFreq, ChannelNames, etc.)
#     - output_base_dir: Base directory for I/O
#     - force_recompute: Whether to force recomputation
#     """
#     import gc
#     import os
#     import time
#     import glob
#     from pyspark.sql import SparkSession
#     print("In process_subjects_parallel")
    
#     # Define paths with correct directory structure
#     epochs_path = f"{output_base_dir}/tmp_epochs"
#     metadata_path = f"{output_base_dir}/tmp_metadata"
#     output_path = f"{output_base_dir}/tmp_epochs_processed"
#     os.makedirs(output_path, exist_ok=True)
    
#     # Improved check for partitioned parquet data
#     def check_parquet_exists(path):
#         # First check if the directory exists
#         if not os.path.exists(path):
#             return False
            
#         # Then check if there are any parquet files in any subdirectory
#         import glob
#         all_parquet_files = glob.glob(f"{path}/**/*.parquet", recursive=True)
#         return len(all_parquet_files) > 0
    
#     # Check if epochs data exists with the improved function
#     epochs_exists = check_parquet_exists(epochs_path)
#     if not epochs_exists or force_recompute:
#         print(f"Saving epochs DataFrame to {epochs_path}...")
#         epochs_df.write.mode("overwrite").parquet(epochs_path)
#     else:
#         print(f"Using existing epochs data from {epochs_path}")
    
#     # Similarly for metadata
#     metadata_exists = check_parquet_exists(metadata_path)
#     if not metadata_exists or force_recompute:
#         print(f"Saving metadata DataFrame to {metadata_path}...")
#         metadata_df.write.mode("overwrite").parquet(metadata_path)
#     else:
#         print(f"Using existing metadata from {metadata_path}")
    
#     # Function to check if a subject is already processed
#     def should_process_subject(subject_id):
#         if force_recompute:
#             return True
#         subject_dir = f"{output_path}/SubjectID={subject_id}"
#         if not os.path.exists(subject_dir):
#             return True
#         parquet_files = glob.glob(f"{subject_dir}/*.parquet")
#         if not parquet_files:
#             return True
#         return False
    
#     # Define worker function to process a partition of subjects
   
#     # Create broadcast variables for the paths
#     epochs_path_b = spark.sparkContext.broadcast(epochs_path)
#     metadata_path_b = spark.sparkContext.broadcast(metadata_path)
#     output_path_b = spark.sparkContext.broadcast(output_path)
    
#     # Modify the worker function to avoid SparkSession creation
#     def process_partition(subject_ids_iter):
#         """Process a partition of subject IDs without creating a new SparkSession"""
#         import traceback
#         import pandas as pd
        
#         # Get paths from broadcast variables
#         epochs_path = epochs_path_b.value
#         metadata_path = metadata_path_b.value
#         output_path = output_path_b.value
        
#         # Import the necessary functions for processing
#         try:
#             # For local pandas processing
#             from pandas import DataFrame
#             import numpy as np
            
#             # For loading EEG data format
#             import mne

#             # Import the feature extraction function
#             try:
#                 from eeg_hpc_benchmark.feature_extraction import processEpoch
#             except ImportError:
#                 from src.feature_extraction import processEpoch
#         except Exception as e:
#             return [{"subject_id": "import_error", "status": "failed", "error": str(e)}]
        
#         # Process each subject in this partition
#         results = []
#         for subject_id in subject_ids_iter:
#             try:
#                 print(f"Processing subject: {subject_id}")
#                 start_time = time.time()
                
#                 # Load subject data files
#                 subject_epochs_path = f"{epochs_path}/SubjectID={subject_id}"
#                 subject_metadata_path = f"{metadata_path}/SubjectID={subject_id}"
                
#                 # Read the parquet files directly into pandas 
#                 import glob
#                 import pandas as pd
                
#                 # Read epochs data
#                 epoch_files = glob.glob(f"{subject_epochs_path}/*.parquet")
#                 if not epoch_files:
#                     raise FileNotFoundError(f"No epoch files found for {subject_id}")


#                 epochs_dfs = [pd.read_parquet(f) for f in epoch_files]
#                 epochs_df = pd.concat(epochs_dfs) if len(epochs_dfs) > 1 else epochs_dfs[0]
                
#                 # Add SubjectID to epochs_df if it doesn't exist
#                 if 'SubjectID' not in epochs_df.columns:
#                     epochs_df['SubjectID'] = subject_id
                
#                 # Read metadata
#                 meta_files = glob.glob(f"{subject_metadata_path}/*.parquet")
#                 if not meta_files:
#                     raise FileNotFoundError(f"No metadata files found for {subject_id}")
                
#                 metadata_df = pd.read_parquet(meta_files[0])
               
                
#                 # Add SubjectID to metadata_df if it doesn't exist
#                 if 'SubjectID' not in metadata_df.columns:
#                     metadata_df['SubjectID'] = subject_id

#                 # Get SFreq and ChannelNames from metadata
#                 sfreq = float(metadata_df['SFreq'].iloc[0])
#                 channel_names = metadata_df['ChannelNames'].iloc[0]
                
#                 # Process each epoch
#                 feature_rows = []
#                 for _, row in epochs_df.iterrows():
#                     subject_id = row["SubjectID"]
#                     epoch_id = row["EpochID"]


#                     eeg_data = np.array(row["EEG"])
#                     # Make sure channels are consistent
#                     eeg_data = np.stack(eeg_data, axis=0)
                   
#                     # this is debatebly safer , will need to check this out 
#                     # if isinstance(eeg_data[0], (list, np.ndarray)):  # If data is already 2D
#                     #     eeg_data = np.stack(eeg_data, axis=0)
#                     #

#                     lens = [len(ch) for ch in eeg_data]
#                     min_len = min(lens)
#                     max_len = max(lens)
#                     if min_len != max_len:
#                         print(f"[WARN] Epoch {epoch_id} has uneven channel lengths")
#                         print(f"       → Shortest = {min_len}, Longest = {max_len}")
#                         print(f"       → Trimming all to {min_len}")
#                         eeg_data = [ch[:min_len] for ch in eeg_data]
                    
#                     # Process the epoch
#                     try:
#                         epoch_features = processEpoch(
#                             subject_id,
#                             epoch_id,
#                             eeg_data,
#                             channelNames=channel_names,
#                             sfreq=sfreq
#                         )
#                         feature_rows.extend(epoch_features)
#                     except Exception as e:
#                         print(f"[ERROR] {subject_id}:{epoch_id} - {e}")
                
#                 # Convert rows to DataFrame
#                 result_df = pd.DataFrame([r.asDict() for r in feature_rows])
                
#                 # Save to parquet
#                 subject_output_path = f"{output_path}/SubjectID={subject_id}"
#                 os.makedirs(subject_output_path, exist_ok=True)
#                 result_df.to_parquet(f"{subject_output_path}/features.parquet", index=False)
                
#                 # Record success
#                 duration = time.time() - start_time
#                 results.append({
#                     "subject_id": subject_id,
#                     "status": "success", 
#                     "count": len(result_df),
#                     "duration": duration,
#                     "error": None
#                 })
                
#                 # Clean up memory
#                 import gc
#                 del result_df, epochs_df, metadata_df, feature_rows
#                 gc.collect()
                
#             except Exception as e:
#                 error_msg = str(e)
#                 trace = traceback.format_exc()
#                 print(f"Error processing subject {subject_id}: {error_msg}")
#                 print(trace)
                
#                 results.append({
#                     "subject_id": subject_id,
#                     "status": "failed",
#                     "count": 0,
#                     "duration": 0,
#                     "error": error_msg
#                 })
        
#         return results   



#     # Get subjects to process (from epochs_df which contains all subjects)
#     subject_ids = [row.SubjectID for row in epochs_df.select("SubjectID").distinct().collect()]
#     print(f"Found {len(subject_ids)} unique subjects")
    
#     subjects_to_process = [sid for sid in subject_ids if should_process_subject(sid)]
#     if not subjects_to_process:
#         print("No subjects need processing, using existing output")
#         return spark.read.parquet(output_path)
    
#     print(f"Processing {len(subjects_to_process)} subjects in parallel")
    
#     # Calculate partitions
#     num_cores = spark.sparkContext.defaultParallelism
#     num_subjects = len(subjects_to_process)
#     recommended_partitions = min(num_cores * 2, num_subjects)
#     print(f"Using {recommended_partitions} partitions (based on {num_cores} cores)")
    
#     # Create RDD and process
#     rdd = spark.sparkContext.parallelize(subjects_to_process, numSlices=recommended_partitions)
#     results_rdd = rdd.mapPartitions(process_partition)
#     all_results = results_rdd.flatMap(lambda x: x).collect()
    
#     # Report results
#     successful = []
#     failed = []

#     for r in all_results:
#         print(f"r : {r}")
#         if isinstance(r, dict) and "status" in r:
#             if r["status"] == "success":
#                 successful.append(r)
#             elif r["status"] == "failed":
#                 failed.append(r)
#         else:
#             print(f"Warning: Unexpected result format: {type(r).__name__} - {r}")
#             # Treat as failure if not a proper result dict
#             failed.append({
#                 "subject_id": "unknown",
#                 "status": "failed",
#                 "error": f"Invalid result format: {r}"
#             })
    
#     print(f"\nProcessing complete: {len(successful)} successful, {len(failed)} failed")
    
#     if successful:
#         total_rows = sum(r["count"] for r in successful)
#         avg_duration = sum(r["duration"] for r in successful) / len(successful)
#         print(f"Generated {total_rows} total feature rows")
#         print(f"Average processing time per subject: {avg_duration:.2f}s")
    
#     if failed:
#         print("\nFailed subjects:")
#         for f in failed:
#             print(f"  - {f['subject_id']}: {f['error']}")
    
#     # Clean up
#     spark.catalog.clearCache()
#     gc.collect()
    
#     # Return reference to output data
#     # Return reference to output data
#     parquet_files = glob.glob(f"{output_path}/**/*.parquet", recursive=True)
#     if not parquet_files:
#         print(f"WARNING: No Parquet files found in {output_path}")
#         return spark.createDataFrame([], schema=get_feature_schema())
#     return spark.read.parquet(*parquet_files)
