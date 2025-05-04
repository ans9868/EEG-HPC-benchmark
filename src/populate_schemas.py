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
def extract_features_udtf(key: pd.Series, pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a pandas DataFrame with columns:
    - SubjectID (str)
    - EpochID (str)
    - EEG (array): shape = (n_channels, n_times)

    Returns a long-format pandas DataFrame with:
    - SubjectID, EpochID, Electrode, WaveBand, FeatureName, FeatureValue, table_type
    """
    results = []

    for _, row in pdf.iterrows():
        subject_id = row["SubjectID"]
        epoch_id = row["EpochID"]
        eeg_data = np.array(row["EEG"])

        # You might want to reconstruct a fake MNE Epoch if needed, or pass to a local wrapper
        # But assuming processEpoch can handle this NumPy input:
        # #TODO need to pass tei epoch info here ! Need to do a join on the table Per subject for there epoch info
        feature_rows = processEpoch(subject_id, epoch_id, eeg_data)

        results.extend(feature_rows)

    # Convert list of Row objects to pandas DataFrame
    return pd.DataFrame([r.asDict() for r in results])

'''
@pandas_udf(get_feature_schema(), PandasUDFType.GROUPED_MAP)
def extract_features_udtf(pdf):
    from feature_extraction import processEpoch, processSub
    import time

    all_rows = []

    for _, row in pdf.iterrows():
        subject_id = row["SubjectID"]
        print(f"[START] {subject_id}")
        start = time.time()

        try:
            epochs = processSub(subject_id, config['derivatives'])

            def safe_process(i, ep):
                try:
                    return processEpoch(subject_id, f"ep-{i}", ep)
                except Exception as e:
                    print(f"[ERROR] {subject_id}:ep-{i}: {e}")
                    return []

            #It's possible to optimize this with joblib, but it can cause problems with threads not releasing semlocks when CPU reaches 100%
            results = [safe_process(i, epochs[i]) for i in range(len(epochs))]

            subject_rows = []
            for res in results:
                if isinstance(res, list):
                    subject_rows.extend(res)
                else:
                    subject_rows.append(res)

            all_rows.extend(subject_rows)

        except Exception as e:
            print(f"[ERROR] Failed to process {subject_id}: {e}")
            continue

        print(f"[FINISHED] {subject_id} in {time.time() - start:.2f}s")

    return pd.DataFrame([r.asDict() for r in all_rows])

'''
