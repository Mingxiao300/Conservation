#!/bin/bash
#SBATCH --job-name=python_multivariate
#SBATCH --output=logs/python_multivariate_%j.log
#SBATCH --error=logs/python_multivariate_%j.err
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=shared

# Navigate to script directory
cd /n/home00/msong300/conservation

# Load modules if needed (uncomment if required)
# module load python/3.x  # Adjust version
# module load python3/3.12  # Example

# Create logs directory
mkdir -p logs

# Run the Python model using the shell script
./multivariate/run_python_model.sh

