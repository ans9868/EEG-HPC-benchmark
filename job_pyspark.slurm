#!/bin/bash
#SBATCH --job-name=pyspark-benchmark
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=benchmark_output.log

module purge
module load spark/3.1.2

# Activate your conda environment if needed:
# source activate your_env_name

spark-submit --master local[8] benchmark_runner.py
