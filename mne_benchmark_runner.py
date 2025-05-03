import time
import psutil
import numpy as np
from mne_features.feature_extraction import FeatureExtractor

# Simulate loading epoched EEG data (replace with your actual data loading code)
# Shape: (n_epochs, n_channels, n_times)
def load_data():
    return np.random.randn(5000, 19, 1500)  # Example shape: 5000 epochs, 19 channels, 3s @ 500Hz

def main():
    X = load_data()

    start = time.time()
    print(f"Memory usage before: {psutil.virtual_memory().percent}%")

    freq_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100)
    }

    fe = FeatureExtractor(
        sfreq=500.0,
        selected_funcs=["mean", "std", "skewness", "kurtosis", "rms", "pow_freq_bands"],
        params={
            "pow_freq_bands__freq_bands": freq_bands,
            "pow_freq_bands__psd_method": "welch",
            "pow_freq_bands__normalize": False
        },
        n_jobs=-1
    )

    features = fe.fit_transform(X)

    end = time.time()
    print(f"Features shape: {features.shape}")
    print(f"Total runtime: {(end - start) / 60:.2f} minutes")
    print(f"Memory usage after: {psutil.virtual_memory().percent}%")

if __name__ == "__main__":
    main()
