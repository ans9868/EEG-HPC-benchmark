import os
import time
import glob
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from joblib import Parallel, delayed

from pyspark.sql import Row, SparkSession, DataFrame
from pyspark import StorageLevel
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType

# Config loader
try:
    from src.config_handler import load_config_file_only
except ImportError:
    from config_handler import load_config_file_only



def ensure_config_passed(config):
    if config is None:
        caller = inspect.stack()[1].function
        raise ValueError(f"[ERROR] Missing `config` in function '{caller}' — make sure to pass it explicitly.")

def subPath(sub, config):
    ensure_config_passed(config)
    data_path = config['data_path']
    derivatives = config['derivatives']

    sub_id = sub.replace('sub-', '') if isinstance(sub, str) and sub.startswith('sub-') else sub

    if derivatives:
        path = os.path.join(data_path, 'ds004504', 'derivatives', f'sub-{sub_id}', 'eeg', f'sub-{sub_id}_task-eyesclosed_eeg.set')
    else:
        path = os.path.join(data_path, 'ds004504', f'sub-{sub_id}', 'eeg', f'sub-{sub_id}_task-eyesclosed_eeg.set')

    if not os.path.exists(path):
        raise FileNotFoundError(f'The path was not found for {sub}, path: {path}')
    return path

def participantsInfoPath(config):
    ensure_config_passed(config)
    return os.path.join(config['data_path'], 'ds004504', 'participants.tsv')

def processSub(sub, config):
    ensure_config_passed(config)
    windowLength = config['windowLength']
    stepSize = config['stepSize']

    raw = mne.io.read_raw_eeglab(subPath(sub, config), preload=True)

    for i, desc in enumerate(raw.annotations.description):
        if 'boundary' in desc:
            raw.annotations.description[i] = 'BAD_boundary'

    sfreq = raw.info['sfreq']
    start_times = np.arange(0, raw.times[-1] - windowLength, stepSize)
    events = np.array([[int(t * sfreq), 0, 1] for t in start_times])

    epochs = mne.Epochs(
        raw, events, event_id=1, tmin=0, tmax=windowLength,
        baseline=None, detrend=1, preload=True, verbose=False,
        reject_by_annotation=True
    )
    return epochs

def processSubPSDs(sub, config, mode='generator', n_jobs=1):
    ensure_config_passed(config)
    method = config['stepSize']
    windowLength = config['windowLength']
    stepSize = config['stepSize']
    freqBands = config['freqBands']

    raw = mne.io.read_raw_eeglab(subPath(sub, config), preload=True)
    sfreq = raw.info['sfreq']

    start_times = np.arange(0, raw.times[-1] - windowLength, stepSize)
    events = np.array([[int(t * sfreq), 0, 1] for t in start_times])

    epochs = mne.Epochs(
        raw, events, event_id=1, tmin=0, tmax=windowLength,
        baseline=None, detrend=1, preload=True, verbose=False
    )

    freqLow = freqBands['Delta'][0]
    freqHigh = freqBands['Beta'][1]

    def compute_psd(epoch):
        return epoch.compute_psd(fmin=freqLow, fmax=freqHigh, method=method, verbose=False)

    if mode == 'generator':
        return (compute_psd(epochs[i]) for i in range(len(epochs)))
    elif mode == 'sequential':
        return [compute_psd(epochs[i]) for i in range(len(epochs))]
    elif mode == 'parallel':
        return Parallel(n_jobs=n_jobs)(
            delayed(compute_psd)(epochs[i]) for i in range(len(epochs))
        )
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'generator', 'sequential', or 'parallel'.")

def load_epochs(config, subjects=None):
    ensure_config_passed(config)
    data_path = config['data_path']
    windowLength = config['windowLength']
    stepSize = config['stepSize']

    if subjects is None:
        import glob
        subjects = [os.path.basename(p) for p in glob.glob(os.path.join(data_path, 'ds004504', 'sub-*'))]
        subjects = [s.replace('sub-', '') for s in subjects]

    all_epochs_data = []

    for sub in subjects:
        try:
            epochs = processSub(sub, config)
            epochs_data = epochs.get_data()
            all_epochs_data.append(epochs_data)
        except Exception as e:
            print(f"Error processing subject {sub}: {e}")
            continue

    if not all_epochs_data:
        raise ValueError("No valid epochs data found for any subjects")

    X = np.concatenate(all_epochs_data, axis=0)
    return X



def join_epochs_with_metadata(df_epochs: DataFrame, df_metadata: DataFrame) -> DataFrame:
    df_metadata = df_metadata.drop("NumEpochs")
    joined_df = df_epochs.join(df_metadata, on="SubjectID", how="left")
    return joined_df



def load_subjects_spark(spark: SparkSession, subject_ids: list, config=None, output_base_dir="/Volumes/CrucialX6/spark_data", force_recompute=False):
    if not config:
        print("NO CONFIG IN LOAD_SUBJECTS_SPARK@!")
    print(f"[DEBUG] load_subjects_spark")
    windowLength = config['windowLength']
    stepSize = config['stepSize']
    def processSubSpark(sub_id):
        try:
            sub_path = subPath(sub_id, config)
            print(f"[DEBUG] load_subjects_spark --> processSubSpark {sub_id}")
            print(f"[DEBUG] processSubSpark {sub_id} path: {sub_path}")
            raw = mne.io.read_raw_eeglab(sub_path, preload=True)

            # Mark boundary annotations
            for i, desc in enumerate(raw.annotations.description):
                if 'boundary' in desc:
                    raw.annotations.description[i] = 'BAD_boundary'

            sfreq = raw.info['sfreq']
            ch_names = raw.info['ch_names']
            start_times = np.arange(0, raw.times[-1] - windowLength, stepSize)
            events = np.array([[int(t * sfreq), 0, 1] for t in start_times])

            epochs = mne.Epochs(
                raw, events, event_id=1, tmin=0, tmax=windowLength,
                baseline=None, detrend=1, preload=True, verbose=False,
                reject_by_annotation=True
            )
            data = epochs.get_data()

            epoch_rows = [
                Row(
                    SubjectID=sub_id,
                    EpochID=f"ep-{ep_idx}",
                    EEG=data[ep_idx].astype(float).tolist(),
                )
                for ep_idx in range(data.shape[0])
            ]

            metadata_row = Row(
                SubjectID=sub_id,
                ChannelNames=ch_names,
                SFreq=float(sfreq),
                NumEpochs=int(data.shape[0])
            )

            return (epoch_rows, metadata_row)

        except Exception as e:
            print(f"[ERROR] {sub_id}: {e}")
            return ([], None)

    def should_process_subject(sub_id):
        """Check if a subject needs processing based on existing parquet files"""
        if force_recompute:
            return True
            
        # Check if the subject directories exist
        epoch_dir = f"{output_base_dir}/tmp_epochs/SubjectID={sub_id}"
        meta_dir = f"{output_base_dir}/tmp_metadata/SubjectID={sub_id}"
        
        if not (os.path.exists(epoch_dir) and os.path.exists(meta_dir)):
            return True
            
        # Check if directories contain parquet files
        import glob
        epoch_files = glob.glob(f"{epoch_dir}/*.parquet")
        meta_files = glob.glob(f"{meta_dir}/*.parquet")
        
        if not (epoch_files and meta_files):
            return True
            
        return False

    '''
    The strategy here is to multithread getting the subjects information, then putting the subjects on disk (on external hard drive) and keeping a reference of these subjects in the dataframe
    New: Now we check which subjects already exist and only process the new ones
    '''
    print(sorted(subject_ids))
    num_subjects = len(set(subject_ids))
    print(f"Unique: {num_subjects}, Total: {len(subject_ids)}")
    
    # Check if parquet path exists
    epochs_path = f"{output_base_dir}/tmp_epochs"
    metadata_path = f"{output_base_dir}/tmp_metadata"
    
    # Create directories if they don't exist
    os.makedirs(epochs_path, exist_ok=True)
    os.makedirs(metadata_path, exist_ok=True)


    # Filter out subjects that don't need processing
    subjects_to_process = []
    for sub_id in subject_ids:
        if should_process_subject(sub_id):
            subjects_to_process.append(sub_id)
            print(f"Subject {sub_id} will be processed")
        else:
            print(f"Subject {sub_id} already exists, skipping processing")
    
    # If we have subjects to process, do the processing
    if subjects_to_process and len(subjects_to_process) > 0:
        print(f"Processing {len(subjects_to_process)} subjects: {subjects_to_process}")
        
        # Calculate optimal number of partitions
        # Rule of thumb: aim for partitions that are 100-200MB each
        # Another approach: 2-3 partitions per core in your cluster
        num_cores = spark.sparkContext.defaultParallelism
        num_subjects_to_process = len(subjects_to_process)
        recommended_partitions = max(num_cores * 2, num_subjects_to_process)
        
        print(f"Using {recommended_partitions} partitions (based on {num_cores} cores and {num_subjects_to_process} subjects)")
        
        # ⚙️ RDD of subject IDs with improved partitioning
        rdd = spark.sparkContext.parallelize(subjects_to_process, numSlices=recommended_partitions)

        # Run per-subject processing
        result_rdd = rdd.map(processSubSpark)

        # Unpack into two RDDs and control partitioning
        epoch_rows_rdd = result_rdd.flatMap(lambda x: x[0]).repartition(recommended_partitions)
        metadata_rows_rdd = result_rdd.map(lambda x: x[1]).filter(lambda x: x is not None).repartition(min(recommended_partitions, num_subjects_to_process))

        # Convert to DataFrames
        df_epochs = spark.createDataFrame(epoch_rows_rdd)
        df_metadata = spark.createDataFrame(metadata_rows_rdd)

        # Count to materialize (forces evaluation)
        epoch_count = df_epochs.count()
        metadata_count = df_metadata.count()
        print(f"Created {epoch_count} epoch rows and {metadata_count} metadata rows")

        # Repartition by SubjectID for optimal parquet write performance
        # This creates one file per subject per partition
        df_epochs = df_epochs.repartition(recommended_partitions, "SubjectID")
        df_metadata = df_metadata.repartition(min(recommended_partitions, num_subjects_to_process), "SubjectID")
        
        print("Writing epochs to parquet...")
        df_epochs.write.mode("append").partitionBy("SubjectID").option("maxRecordsPerFile", 1000000).parquet(f"{output_base_dir}/tmp_epochs/")
        
        print("Writing metadata to parquet...")
        df_metadata.write.mode("append").partitionBy("SubjectID").parquet(f"{output_base_dir}/tmp_metadata/")
        
        # Explicitly force garbage collection to free memory
        import gc
        # Remove references to the DataFrames
        df_epochs = None
        df_metadata = None
        epoch_rows_rdd = None
        metadata_rows_rdd = None
        result_rdd = None
        rdd = None
        # Force garbage collection
        gc.collect()
        spark.catalog.clearCache()  # Clear any cached data
    else:
        print("No subjects need processing, using existing parquet files")

    
    # Create references to the parquet files - these don't load data into memory yet
    print("Creating references to parquet files...")
    parquet_epochs = spark.read.parquet(f"{output_base_dir}/tmp_epochs/")
    parquet_metadata = spark.read.parquet(f"{output_base_dir}/tmp_metadata/")
    
    # Return the references to the parquet files
    return parquet_epochs, parquet_metadata