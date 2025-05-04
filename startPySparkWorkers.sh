#!/bin/bash
#SBATCH --job-name=start_spark
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1         # One task per node
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --output=worker-%j.out

# Load Spark environment (if available)
module load spark 2>/dev/null || echo "Spark module not found — continuing without"

# Start a Spark worker on each allocated node
srun --ntasks=2 --nodes=2 --exclusive bash -c '$SPARK_HOME/sbin/start-worker.sh spark://cm001:13049'

