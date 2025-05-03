import time
import psutil
import numpy as np
from mne_features.feature_extraction import FeatureExtractor


#iporting 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./src/")))
#
# ROOT_DIR = "."
# SRC_DIR = ROOT_DIR + "/src"
# sys.path.append("/Users/admin/projectst/EEG-Feature-Benchmark")
#


from preprocess_sets import load_epochs  # Import the function we just created
from config_handler import initiate_config, load_config
# TODO benchmark extract_feature too ! 

# Simulate loading epoched EEG data (replace with your actual data loading code)
# Shape: (n_epochs, n_channels, n_times)

def benchmark_feature_extractor(X, freq_bands, sfreq=500.0 ):
    """Benchmark FeatureExtractor class"""
    start = time.time()
    
    print(f"Memory usage before: {psutil.virtual_memory().percent}%")
   
    fe = FeatureExtractor(
        sfreq=sfreq,
        selected_funcs = [
            "mean",                 # np.mean
            "std",                  # np.std
            "rms",                  # Root Mean Square
            "hjorth_mobility",      # Hjorth mobility
            "hjorth_complexity",    # Hjorth complexity
            "pow_freq_bands"        # Band Power for Delta, Theta, Alpha, Beta
        ],
        params={
            "pow_freq_bands__freq_bands": freq_bands,
            "pow_freq_bands__psd_method": "welch",
            "pow_freq_bands__normalize": False
        },
        n_jobs=-1
    )
    
    features = fe.fit_transform(X)
    
    end = time.time()
    print(f"FeatureExtractor: Features shape: {features.shape}")
    print(f"FeatureExtractor: Total runtime: {(end - start) / 60:.2f} minutes")
    print(f"Memory usage after: {psutil.virtual_memory().percent}%")
    
    return features


'''
def benchmark_extract_features(X, sfreq=500.0):
    """Benchmark extract_features function"""
    start = time.time()
    
    print(f"Memory usage before: {psutil.virtual_memory().percent}%")
    
    freq_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100)
    }
    
    features = extract_features(
        X=X,
        sfreq=sfreq,
        selected_funcs=["mean", "std", "skewness", "kurtosis", "rms", "pow_freq_bands"],
        funcs_params={
            "pow_freq_bands__freq_bands": freq_bands,
            "pow_freq_bands__psd_method": "welch",
            "pow_freq_bands__normalize": False
        },
        n_jobs=-1
    )
    
    end = time.time()
    print(f"extract_features: Features shape: {features.shape}")
    print(f"extract_features: Total runtime: {(end - start) / 60:.2f} minutes")
    print(f"Memory usage after: {psutil.virtual_memory().percent}%")
    
    return features

'''

def main():


    # load_epochs(subjects=None, derivatives=derivatives, windowLength=windowLength, stepSize=stepSize):
    # Load data once to use for both benchmarks
    print("start")
    start = time.time()
    subjects = ['sub-003']
    X = load_epochs(subjects)
    

    print("initiate config")
    initiate_config()
    print("load config")
    config = load_config()
    print("config loaded: ", config)
    
    freq_bands = config["freqBands"]
 
    # Run both benchmarks
    print("Running benchmark for FeatureExtractor class...")
    fe_features = benchmark_feature_extractor(X, freq_bands)
    
    # print("\nRunning benchmark for extract_features function...")
    # ef_features = benchmark_extract_features(X)
    # 
    # # Compare results to ensure they're identical
    # print("\nComparing results:")
    # if np.allclose(fe_features, ef_features):
    #     print("Results are identical!")
    # else:
    #     print("Warning: Results differ between methods!")
    end = time.time()
    print(f"Total runtime: {(end - start) / 60:.2f} minutes")
    print(f"Memory usage after: {psutil.virtual_memory().percent}%")
    print(type(fe_features))
    print(fe_features.shape)


if __name__ == "__main__":
    main()

