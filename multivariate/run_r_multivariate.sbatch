#!/bin/bash
#SBATCH --job-name=r_multivariate
#SBATCH --output=logs/r_multivariate_%j.log
#SBATCH --error=logs/r_multivariate_%j.err
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=shared

# Navigate to script directory
cd /n/home00/msong300/conservation

# Load R module (uncomment and adjust if needed)
# module load R  # or module load R/4.x

# Create logs directory
mkdir -p logs

# Run the R model using the shell script
./multivariate/run_r.sh

