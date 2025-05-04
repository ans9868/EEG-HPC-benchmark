import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import mne
import os
import time
from joblib import Parallel, delayed 

try:
    # When run as part of a package (local scripts, Jupyter, etc.)
    from src.config_handler import initiate_config, load_config
except ImportError:
    # When run inside Spark workers (which get flat files via sc.addPyFile)
    from config_handler import initiate_config, load_config


try:
    config = load_config()
except RuntimeError:
    print("Config not found in feature_extraction.py")
    config = initiate_config()


config = load_config()
data_path = config['data_path']
derivatives = config['derivatives']
freqBands = config['freqBands']
windowLength = config['windowLength']
stepSize = config['stepSize']
method = config['stepSize']


def get_data_path():
    return data_path

def subPath(sub, derivatives=derivatives):
    print("subPath", sub)
    print("subPath data_path", data_path)
    
    sub_id = sub.replace('sub-', '') if isinstance(sub, str) and sub.startswith('sub-') else sub
    print(f"Derivatives: {derivatives}") 
    print(f"Derivatives from config: {config['derivatives']}") 

    if derivatives:
       path = os.path.join(data_path, 'ds004504', 'derivatives', f'sub-{sub_id}', 'eeg', f'sub-{sub_id}_task-eyesclosed_eeg.set')
    else:
       path = os.path.join(data_path, 'ds004504', f'sub-{sub_id}', 'eeg', f'sub-{sub_id}_task-eyesclosed_eeg.set')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f'The path was not found for {sub}, path: {path}')
    print(f"Path handed: {path}")
    return path

def participantsInfoPath():
    return os.path.join(data_path, 'ds004504', 'participants.tsv')

'''
ProcessSub gets the power density from a subject. It has 3 modes. Generator, sequential or parallel.
Ironically the parallel mode seems to be the slowest by about 15% and the other two are about tied.
'''
def _psd_generator(epochs, compute_psd):
    for i in range(len(epochs)):
        yield compute_psd(epochs[i])

def processSubPSDs(sub, derivatives=derivatives, method=method, windowLength=windowLength, stepSize=stepSize, mode='generator', n_jobs=1):
    raw = mne.io.read_raw_eeglab(subPath(sub, derivatives), preload=True)
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
        return _psd_generator(epochs, compute_psd)

    elif mode == 'sequential':
        return [compute_psd(epochs[i]) for i in range(len(epochs))]

    elif mode == 'parallel':
        return Parallel(n_jobs=n_jobs)(
            delayed(compute_psd)(epochs[i]) for i in range(len(epochs))
        )

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'generator', 'sequential', or 'parallel'.")


'''
This is for processing the subject without getting the psd's. It gets all the epochs for the subject.
'''
def processSub(sub, derivatives=derivatives, windowLength=windowLength, stepSize=stepSize):
    print("processSub", sub)
    print("processSub: derivatives", derivatives)
    print("processSub: windowLength", windowLength)
    print("processSub: windowLength", stepSize)
    raw = mne.io.read_raw_eeglab(subPath(sub, derivatives), preload=True)
    
    # removing the boundary events by adding BAD_boundary here and 'reject by anotaion' in mne.Epochs( .... )
    for i, desc in enumerate(raw.annotations.description):
        if 'boundary' in desc:
            raw.annotations.description[i] = 'BAD_boundary'

    sfreq = raw.info['sfreq']


    start_times = np.arange(0, raw.times[-1] - windowLength, stepSize)
    events = np.array([[int(t * sfreq), 0, 1] for t in start_times]) # [sample_index, previous_event_1d, current_event_id], note, 0 -> means we don't have transitions between events,
    
    epochs = mne.Epochs(
        raw, events, event_id=1, tmin=0, tmax=windowLength,
        baseline=None, detrend=1, preload=True, verbose=False,
        reject_by_annotation=True
    )
    
    return epochs



def load_epochs(subjects=None, derivatives=derivatives, windowLength=windowLength, stepSize=stepSize):
    """
    Load and combine epochs from multiple subjects into the format expected by mne-features
    
    Parameters
    ----------
    subjects : list of str or None
        List of subject IDs to process. If None, process all available subjects.
    derivatives : bool, default=True
        Whether to use the derivatives data path.
    windowLength : float
        Length of each epoch window in seconds.
    stepSize : float
        Step size between consecutive epochs in seconds.
    
    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        Combined epochs data from all subjects.
    """
    # If no subjects specified, could get them from a directory listing
    if subjects is None:
        # Example: get all subject directories
        import glob
        subjects = [os.path.basename(p) for p in 
                   glob.glob(os.path.join(data_path, 'ds004504', 'sub-*'))]
        subjects = [s.replace('sub-', '') for s in subjects]
    
    # List to hold all epochs
    all_epochs_data = []
    
    print("load_epochs, first subject path: ", subjects[0])
    # Process each subject
    for sub in subjects:
        print(f"Processing subject {sub}")
        try:
            # Get epochs for this subject
            epochs = processSub(sub, derivatives, windowLength, stepSize)
            
            # Convert epochs to numpy array - shape (n_epochs, n_channels, n_times)
            epochs_data = epochs.get_data()
            
            # Append to our collection
            all_epochs_data.append(epochs_data)
            
        except Exception as e:
            print(f"Error processing subject {sub}: {e}")
            continue
    
    # Combine all subjects' epochs into one array
    if not all_epochs_data:
        raise ValueError("No valid epochs data found for any subjects")
    
    # Concatenate along the first dimension (epochs)
    X = np.concatenate(all_epochs_data, axis=0)
    
    print(f"Loaded {X.shape[0]} epochs from {len(subjects)} subjects")
    print(f"Data shape: {X.shape}")
    
    return X


from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark import StorageLevel







from pyspark.sql import Row
from pyspark.sql import SparkSession
import numpy as np
from preprocess_sets import processSub
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
import os


from pyspark import StorageLevel

from pyspark.sql import Row
from pyspark import StorageLevel
import mne
import numpy as np
from src.preprocess_sets import subPath

def load_subjects_spark(spark: SparkSession, subject_ids: list):
    def processSubSpark(sub_id):
        try:
            raw = mne.io.read_raw_eeglab(subPath(sub_id), preload=True)

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
                    ChannelNames=ch_names,
                    SFreq=float(sfreq)
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



    print(sorted(subject_ids))
    print(f"Unique: {len(set(subject_ids))}, Total: {len(subject_ids)}")
    # ⚙️ RDD of subject IDs
    rdd = spark.sparkContext.parallelize(subject_ids, numSlices=2)

    # Run per-subject processing
    result_rdd = rdd.map(processSubSpark)

    # Unpack into two RDDs
    epoch_rows_rdd = result_rdd.flatMap(lambda x: x[0])
    metadata_rows_rdd = result_rdd.map(lambda x: x[1]).filter(lambda x: x is not None)

    # Convert to DataFrames
    df_epochs = spark.createDataFrame(epoch_rows_rdd)
    df_metadata = spark.createDataFrame(metadata_rows_rdd)

    # Optional: partition/cache/write
    df_epochs = df_epochs.repartition("SubjectID").persist(StorageLevel.MEMORY_AND_DISK)
    df_metadata = df_metadata.repartition("SubjectID").persist(StorageLevel.MEMORY_AND_DISK)

    df_epochs.write.mode("overwrite").partitionBy("SubjectID").parquet("tmp_epochs/")
    df_metadata.write.mode("overwrite").partitionBy("SubjectID").parquet("tmp_metadata/")

    return df_epochs, df_metadata




    all_epoch_dfs = []
    all_metadata_dfs = []

    for subject_id in subject_ids:
        print(f"[START] {subject_id}")
        try:
            epochs = processSub(subject_id)
            data = epochs.get_data()
            ch_names = epochs.info['ch_names']
            sfreq = epochs.info['sfreq']

            epoch_rows = [
                Row(
                    SubjectID=subject_id,
                    EpochID=f"ep-{ep_idx}",
                    EEG=data[ep_idx].astype(float).tolist()
                )
                for ep_idx in range(data.shape[0])
            ]

            metadata_rows = [
                Row(
                    SubjectID=subject_id,
                    ChannelNames=ch_names,
                    SFreq=float(sfreq)
                )
            ]

            df_epochs = spark.createDataFrame(epoch_rows)
            df_metadata = spark.createDataFrame(metadata_rows)

            # Partition by subject, optionally cache
            df_epochs = df_epochs.repartition("SubjectID").persist(StorageLevel.MEMORY_AND_DISK)
            df_metadata = df_metadata.repartition("SubjectID").persist(StorageLevel.MEMORY_AND_DISK)

            # Save partitioned Parquet (optional)
            df_epochs.write.mode("append").partitionBy("SubjectID").parquet("tmp_epochs/")
            df_metadata.write.mode("append").partitionBy("SubjectID").parquet("tmp_metadata/")

            all_epoch_dfs.append(df_epochs)
            all_metadata_dfs.append(df_metadata)

            print(f"[DONE] {subject_id} with {data.shape[0]} epochs")

        except Exception as e:
            print(f"[ERROR] Failed to load subject {subject_id}: {e}")
            continue

    # Union all cached DataFrames
    full_df_epochs = all_epoch_dfs[0].unionByName(*all_epoch_dfs[1:]) if all_epoch_dfs else spark.createDataFrame([], schema=None)
    full_df_metadata = all_metadata_dfs[0].unionByName(*all_metadata_dfs[1:]) if all_metadata_dfs else spark.createDataFrame([], schema=None)

    return full_df_epochs, full_df_metadata

# def load_subjects_spark(spark: SparkSession, subject_ids: list):
#     epoch_rows = []
#     metadata_rows = []
#
#     for subject_id in subject_ids:
#         print(f"[START] {subject_id}")
#         try:
#             epochs = processSub(subject_id)
#             data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
#             ch_names = epochs.info['ch_names']
#             sfreq = epochs.info['sfreq']
#             
#             # **easy errors here looping over epochs **
#             for ep_idx in range(data.shape[0]):
#                 epoch_rows.append(Row(
#                     SubjectID=subject_id,
#                     EpochID=f"ep-{ep_idx}",
#                     EEG=data[ep_idx].astype(float).tolist()
#                     # EEG=data[ep_idx].tolist()  # shape: (n_channels, n_times)
#                 ))
#
#             metadata_rows.append(Row(
#                 SubjectID=subject_id,
#                 ChannelNames=ch_names,
#                 SFreq=float(sfreq)
#             ))
#
#             if epoch_rows:
#                 # print(f"[BUILDINGsPARK DATAFRAME for] {subject_id}")
#                 # sub_df_epochs = spark.createDataFrame(epoch_rows)
#                 # sub_df_metadata =spark.createDataFrame(metadata_rows)
#                 # sub_df_epochs.write.mode("append").parquet("tmp_epochs/")
#                 # sub_df_metadata.write.mode("append").parquet("tmp_metadata/")
#                 # epoch_rows = []
#                 # metadata_rows = []
#             #
#             #     print(f"[BUILDING SPARK DATAFRAME for] {subject_id}")
#     
#                 print(f"[BUILDING SPARK DATAFRAME for] {subject_id}")
#                 # epoch_rows.                
#                 sub_df_epochs = spark.createDataFrame(epoch_rows)
#                 sub_df_metadata = spark.createDataFrame(metadata_rows)
#             
#                 # Add partitioning (optional, good if querying later by SubjectID)
#                 sub_df_epochs.write \
#                     .mode("append") \
#                     .parquet("tmp_epochs/")
#             
#                 sub_df_metadata.write \
#                     .mode("append") \
#                     .parquet("tmp_metadata/")
#             
#                 # Reset rows for next subject
#                 epoch_rows.clear()
#                 metadata_rows.clear()
#
#
#
#             print(f"[DONE] {subject_id} with {data.shape[0]} epochs")
#
#         except Exception as e:
#             print(f"[ERROR] Failed to load subject {subject_id}: {e}")
#             continue
#
#     df_epochs = spark.createDataFrame(epoch_rows)
#     df_metadata = spark.createDataFrame(metadata_rows)
#
#     return df_epochs, df_metadata
#

from pyspark.sql import DataFrame
def join_epochs_with_metadata(df_epochs: DataFrame, df_metadata: DataFrame) -> DataFrame:
    joined_df = df_epochs.join(df_metadata, on="SubjectID", how="left")
    return joined_df

