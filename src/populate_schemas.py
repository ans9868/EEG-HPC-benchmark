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


