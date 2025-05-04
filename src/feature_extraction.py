import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import mne
import os
import time
from joblib import Parallel, delayed 
from pyspark.sql import Row
from mne_features.univariate import (
    compute_app_entropy,
    compute_samp_entropy,
    compute_higuchi_fd,
    compute_katz_fd,
    compute_hjorth_mobility,
    compute_hjorth_complexity,
    compute_rms,
    compute_skewness,
    compute_kurtosis,
    compute_std,
    compute_mean
)
try:
    # When run as part of a package (local scripts, Jupyter, etc.)
    from src.config_handler import load_config, initiate_config
    from src.preprocess_sets import processSub, participantsInfoPath
    from src.feature_extraction_helper import *
except ImportError:
    # When run inside Spark workers (which get flat files via sc.addPyFile)
    from preprocess_sets import processSub, participantsInfoPath
    from config_handler import load_config, initiate_config
    from feature_extraction_helper import *

try:
    config = load_config()
except RuntimeError:
    print("Config not found in feature_extraction.py")
    config = initiate_config()


config = load_config()
freqBands = config['freqBands']
windowLength = config['windowLength']
stepSize = config['stepSize']
method = config['method']
print(f"Config using in feature Extraction.py {config}")




method = 'welch'
windowLength = 3
stepSize = 1.5
from mne.filter import filter_data
from mne.time_frequency import psd_array_welch
from pyspark.sql import Row


def debug_epoch_psd(epoch_id, raw_data, sfreq):
    print(f"\n[DEBUG] --- Analyzing Epoch: {epoch_id} ---")
    
    # 1. Shape & Type
    print("Raw EEG shape (should be 2D: channels x time):", raw_data.shape)
    print("Raw EEG type:", type(raw_data))
    print("First channel type:", type(raw_data[0]))
    
    if raw_data.ndim != 2:
        print("[ERROR] raw_data is not 2D!")
        return
    
    n_channels, n_times = raw_data.shape
    print(f"Channels: {n_channels}, Timepoints: {n_times}")
    
    # 2. Signal Stats
    print("Check for NaNs:", np.isnan(raw_data).any())
    print("Min/Max value:", np.min(raw_data), np.max(raw_data))
    print("Mean per channel (first 3):", [np.mean(raw_data[i]) for i in range(min(3, n_channels))])
    print("Std per channel (first 3):", [np.std(raw_data[i]) for i in range(min(3, n_channels))])

    # 3. Safe n_fft setup
    fft_len = min(2048, n_times)
    print(f"Using n_fft = {fft_len}")

    try:
        psds, freqs = psd_array_welch(
            raw_data,
            sfreq=sfreq,
            fmin=0.5,
            fmax=40.0,
            n_fft=fft_len,
            n_per_seg=fft_len,  # force match to avoid overlap issues
            n_overlap=int(fft_len * 0.5),
            remove_dc=True,
            average='mean',
            window='hamming',
            verbose=True
        )
        print("[SUCCESS] PSD computed.")
        print("PSD shape:", psds.shape)
        print("Freqs shape:", freqs.shape)
        print("First 5 freqs:", freqs[:5])
        print("Mean PSD value for first channel:", np.mean(psds[0]))
    
    except Exception as e:
        print(f"[ERROR] PSD computation failed: {e}")

    print("[DEBUG DONE]")




def processEpoch(subjectID, epochID, rawData, channelNames, sfreq, freqBands=freqBands, method=method, windowLength=windowLength, stepSize=stepSize, n_jobs=1):
    fmin = min(band_range[0] for band_range in freqBands.values())
    fmax = max(band_range[1] for band_range in freqBands.values())

    # PSD based features , welch , need to add functionality for other later
    # ** might need to tune the fft, seg and n_overlap settings 
    rawData = np.array(rawData)

    # debug_epoch_psd(epochID, rawData, sfreq)
    
    n_times = rawData.shape[1]
    fft_len = min(int(sfreq * windowLength), n_times)
    
    psds, freqs = psd_array_welch(
        rawData,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        # n_fft=fft_len, set to default since when I play with it goes all weird on me
        # n_per_seg=fft_len,
        # n_overlap=int(fft_len * (1 - stepSize / windowLength)),
        verbose=False
    )

    
    normalPsds = psds / np.sum(psds, axis=-1, keepdims=True)

    # Time domain EEG data for this epoch
    rows = []

    for channel_idx, channel_name in enumerate(channelNames):
        # Electrode-level features (non-band-specific)
        electrode_features = [
            # ("TotalPower", totalBandPower(normalPsds, freqs, channel_idx)),
            # ("TotalEnergy", totalEnergy(normalPsds, freqs, channel_idx)),
            # ("SpectralEntropy", spectral_entropy_from_psd(normalPsds[channel_idx, :])),
            # ("HjorthActivity", np.var(data[channel_idx, :])),
            # ("HjorthIndex", compute_hjorth_index(data[channel_idx:channel_idx+1])[0])
            # ("Skewness", scipy.stats.skew(data[channel_idx])),
            # ("Kurtosis", scipy.stats.kurtosis(data[channel_idx])),
            
            ("Mean", np.mean(rawData[channel_idx])),
            ("Std", np.std(rawData[channel_idx])),
            ("RMS", np.sqrt(np.mean(rawData[channel_idx] ** 2))),
            ("HjorthMobility", compute_hjorth_mobility(rawData[channel_idx:channel_idx+1])[0]),
            ("HjorthComplexity", compute_hjorth_complexity(rawData[channel_idx:channel_idx+1])[0])
        ]

        for fname, val in electrode_features:
            rows.append(Row(
                SubjectID=subjectID,
                EpochID=epochID,
                Electrode=channel_name,
                WaveBand=None,
                FeatureName=fname,
                FeatureValue=float(val),
                # table_type="electrode"
                table_type="band" # PUTTING BAND FOR BENCHMARK ONLY
            ))

        
        for band_name, (band_fmin, band_fmax) in freqBands.items():
            band_mask = (freqs >= band_fmin) & (freqs < band_fmax)
            psd_band = normalPsds[channel_idx, band_mask]

            # Band Power from PSD
            band_power = psd_band.mean()


            mobility = compute_hjorth_mobility(rawData[channel_idx:channel_idx+1])[0]
            complexity = compute_hjorth_complexity(rawData[channel_idx:channel_idx+1])[0]
            
            # spectral_entropy = spectral_entropy_from_psd(psd_band)
            # hjorth_index = compute_hjorth_index(data[channel_idx:channel_idx+1])[0]

            # Band-level features
            for fname, val in [
                ("Power", band_power),
                # ("SpectralEntropy", spectral_entropy),
                # ("HjorthActivity", activit),
                ("HjorthMobility", mobility),
                ("HjorthComplexity", complexity),
                # ("HjorthIndex", hjorth_index)
            ]:
                rows.append(Row(
                    SubjectID=subjectID,
                    EpochID=epochID,
                    Electrode=channel_name,
                    WaveBand=band_name,
                    FeatureName=fname,
                    FeatureValue=float(val),
                    table_type="band"
                ))    


    # Epoch-level features: averaged across all channels
    # epoch_feature_list = [
    #     ("Mean", np.mean(compute_mean(data))),
    #     ("Std", np.mean(compute_std(data))),
    #     ("Variance", np.mean(compute_variance(data))),
    #     ("RMS", np.mean(compute_rms(data))),
    #     ("HjorthMobility", np.mean(compute_hjorth_mobility(data))),
    # ]
    # 
    # for fname, val in epoch_feature_list:
    #     rows.append(Row(
    #         SubjectID=subjectID,
    #         EpochID=epochID,
    #         Electrode=None,
    #         WaveBand=None,
    #         FeatureName=fname,
    #         FeatureValue=float(val),
    #         table_type="epoch"
    #     ))
    
    return rows



'''
Process's a specific subject
'''
def processSubject(subject, n_jobs=-1, freqBands=freqBands):
    start = time.time()    


    epochs = processSub(subject)
    epochResults = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(processEpoch)(epochs[x], method=method) for x in range(len(epochs)))
   
    print(f"processSubject {subject}:", time.time()-start)
    return epochResults



'''
Put in a list of subjects, and it will process them in a dataframe with numpy arrays for data
'''
def processSubjects(subjectList, n_jobs=-1):
    allResults = {}

    for subject in subjectList:
        result = processSubject(subject, n_jobs=n_jobs)
        allResults[subject] = result

    return allResults
processEpoch
