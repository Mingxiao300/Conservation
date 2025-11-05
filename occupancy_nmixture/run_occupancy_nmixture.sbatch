#!/bin/bash
#SBATCH --job-name=occupancy_nmixture
#SBATCH --output=logs/occupancy_nmixture_%j.log
#SBATCH --error=logs/occupancy_nmixture_%j.err
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=shared  # Adjust based on your cluster's partitions

# Navigate to script directory
cd /n/home00/msong300/conservation

# Load modules if needed (uncomment if required)
# module load python/3.x  # Adjust version
# module load R/4.x       # If needed

# Run the script
./run_occupancy_nmixture.sh

