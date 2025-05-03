#!/bin/bash
#SBATCH --job-name=mnefeatures-benchmark
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=mne_benchmark_output.log

module purge
# Uncomment and load MNE-features dependencies, or activate your conda environment:
# source activate your_mne_env

python mne_benchmark_runner.py
