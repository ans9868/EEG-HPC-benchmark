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

def process_subjects_parallel(spark: SparkSession, epochs_df, metadata_df, output_base_dir="/Volumes/CrucialX6/spark_data", 
                             force_recompute=False):
    """
    Process subjects in parallel using separate epochs and metadata DataFrames.
    
    Parameters:
    - spark: SparkSession
    - epochs_df: DataFrame containing EEG epoch data
    - metadata_df: DataFrame containing metadata (SFreq, ChannelNames, etc.)
    - output_base_dir: Base directory for I/O
    - force_recompute: Whether to force recomputation
    """
    import gc
    import os
    import time
    import glob
    from pyspark.sql import SparkSession
    print("In process_subjects_parallel")
    
    # Define paths with correct directory structure
    epochs_path = f"{output_base_dir}/tmp_epochs"
    metadata_path = f"{output_base_dir}/tmp_metadata"
    output_path = f"{output_base_dir}/tmp_epochs_processed"
    os.makedirs(output_path, exist_ok=True)
    
    # Improved check for partitioned parquet data
    def check_parquet_exists(path):
        # First check if the directory exists
        if not os.path.exists(path):
            return False
            
        # Then check if there are any parquet files in any subdirectory
        import glob
        all_parquet_files = glob.glob(f"{path}/**/*.parquet", recursive=True)
        return len(all_parquet_files) > 0
    
    # Check if epochs data exists with the improved function
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
    
    # Define worker function to process a partition of subjects
   
    # Create broadcast variables for the paths
    epochs_path_b = spark.sparkContext.broadcast(epochs_path)
    metadata_path_b = spark.sparkContext.broadcast(metadata_path)
    output_path_b = spark.sparkContext.broadcast(output_path)
    
    # Modify the worker function to avoid SparkSession creation
    def process_partition(subject_ids_iter):
        """Process a partition of subject IDs without creating a new SparkSession"""
        import traceback
        import pandas as pd
        
        # Get paths from broadcast variables
        epochs_path = epochs_path_b.value
        metadata_path = metadata_path_b.value
        output_path = output_path_b.value
        
        # Import the necessary functions for processing
        try:
            # For local pandas processing
            from pandas import DataFrame
            import numpy as np
            
            # For loading EEG data format
            import mne

            # Import the feature extraction function
            try:
                from src.feature_extraction import processEpoch
            except ImportError:
                from feature_extraction import processEpoch
        except Exception as e:
            return [{"subject_id": "import_error", "status": "failed", "error": str(e)}]
        
        # Process each subject in this partition
        results = []
        for subject_id in subject_ids_iter:
            try:
                print(f"Processing subject: {subject_id}")
                start_time = time.time()
                
                # Load subject data files
                subject_epochs_path = f"{epochs_path}/SubjectID={subject_id}"
                subject_metadata_path = f"{metadata_path}/SubjectID={subject_id}"
                
                # Read the parquet files directly into pandas 
                import glob
                import pandas as pd
                
                # Read epochs data
                epoch_files = glob.glob(f"{subject_epochs_path}/*.parquet")
                if not epoch_files:
                    raise FileNotFoundError(f"No epoch files found for {subject_id}")


                epochs_dfs = [pd.read_parquet(f) for f in epoch_files]
                epochs_df = pd.concat(epochs_dfs) if len(epochs_dfs) > 1 else epochs_dfs[0]
                
                # Add SubjectID to epochs_df if it doesn't exist
                if 'SubjectID' not in epochs_df.columns:
                    epochs_df['SubjectID'] = subject_id
                
                # Read metadata
                meta_files = glob.glob(f"{subject_metadata_path}/*.parquet")
                if not meta_files:
                    raise FileNotFoundError(f"No metadata files found for {subject_id}")
                
                metadata_df = pd.read_parquet(meta_files[0])
               
                
                # Add SubjectID to metadata_df if it doesn't exist
                if 'SubjectID' not in metadata_df.columns:
                    metadata_df['SubjectID'] = subject_id

                # Get SFreq and ChannelNames from metadata
                sfreq = float(metadata_df['SFreq'].iloc[0])
                channel_names = metadata_df['ChannelNames'].iloc[0]
                
                # Process each epoch
                feature_rows = []
                for _, row in epochs_df.iterrows():
                    subject_id = row["SubjectID"]
                    epoch_id = row["EpochID"]


                    eeg_data = np.array(row["EEG"])
                    # Make sure channels are consistent
                    eeg_data = np.stack(eeg_data, axis=0)
                   
                    # this is debatebly safer , will need to check this out 
                    # if isinstance(eeg_data[0], (list, np.ndarray)):  # If data is already 2D
                    #     eeg_data = np.stack(eeg_data, axis=0)
                    #

                    lens = [len(ch) for ch in eeg_data]
                    min_len = min(lens)
                    max_len = max(lens)
                    if min_len != max_len:
                        print(f"[WARN] Epoch {epoch_id} has uneven channel lengths")
                        print(f"       → Shortest = {min_len}, Longest = {max_len}")
                        print(f"       → Trimming all to {min_len}")
                        eeg_data = [ch[:min_len] for ch in eeg_data]
                    
                    # Process the epoch
                    try:
                        epoch_features = processEpoch(
                            subject_id,
                            epoch_id,
                            eeg_data,
                            channelNames=channel_names,
                            sfreq=sfreq
                        )
                        feature_rows.extend(epoch_features)
                    except Exception as e:
                        print(f"[ERROR] {subject_id}:{epoch_id} - {e}")
                
                # Convert rows to DataFrame
                result_df = pd.DataFrame([r.asDict() for r in feature_rows])
                
                # Save to parquet
                subject_output_path = f"{output_path}/SubjectID={subject_id}"
                os.makedirs(os.path.dirname(subject_output_path), exist_ok=True)
                result_df.to_parquet(f"{subject_output_path}/features.parquet", index=False)
                
                # Record success
                duration = time.time() - start_time
                results.append({
                    "subject_id": subject_id,
                    "status": "success", 
                    "count": len(result_df),
                    "duration": duration,
                    "error": None
                })
                
                # Clean up memory
                import gc
                del result_df, epochs_df, metadata_df, feature_rows
                gc.collect()
                
            except Exception as e:
                error_msg = str(e)
                trace = traceback.format_exc()
                print(f"Error processing subject {subject_id}: {error_msg}")
                print(trace)
                
                results.append({
                    "subject_id": subject_id,
                    "status": "failed",
                    "count": 0,
                    "duration": 0,
                    "error": error_msg
                })
        
        return results   



    # Get subjects to process (from epochs_df which contains all subjects)
    subject_ids = [row.SubjectID for row in epochs_df.select("SubjectID").distinct().collect()]
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
