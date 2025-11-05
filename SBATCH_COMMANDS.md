# SBATCH Commands for Multivariate Models

## Submit Jobs Separately

### 1. Submit Python Multivariate Model

```bash
cd /n/home00/msong300/conservation
sbatch run_python_multivariate.sbatch
```

**Expected output:**
- Log file: `logs/python_multivariate_<JOBID>.log`
- Error file: `logs/python_multivariate_<JOBID>.err`
- Model outputs: `outputs/models/`, `outputs/diagnostics/`, `outputs/plots/`

**Check job status:**
```bash
squeue -u $USER
```

**View output while running:**
```bash
tail -f logs/python_multivariate_<JOBID>.log
```

**Cancel job if needed:**
```bash
scancel <JOBID>
```

---

### 2. Submit R Multivariate Model

```bash
cd /n/home00/msong300/conservation
sbatch run_r_multivariate.sbatch
```

**Expected output:**
- Log file: `logs/r_multivariate_<JOBID>.log`
- Error file: `logs/r_multivariate_<JOBID>.err`
- Model outputs: `outputs/models/`, `outputs/diagnostics/`, `outputs/plots/`

**Check job status:**
```bash
squeue -u $USER
```

**View output while running:**
```bash
tail -f logs/r_multivariate_<JOBID>.log
```

**Cancel job if needed:**
```bash
scancel <JOBID>
```

---

## Job Configuration

Both jobs use the same resource allocation:
- **Memory**: 16GB
- **Time**: 6 hours (360 minutes)
- **CPUs**: 1 core
- **Partition**: shared (adjust if your cluster uses different partitions)

### Adjust Resources if Needed

Edit the SBATCH files (`run_python_multivariate.sbatch` and `run_r_multivariate.sbatch`) and modify:

```bash
#SBATCH --mem=16G          # Increase if needed (e.g., 32G, 64G)
#SBATCH --time=6:00:00    # Increase if needed (e.g., 12:00:00 for 12 hours)
#SBATCH --partition=shared  # Change if your cluster uses different partitions
```

---

## Monitor Jobs

### Check all your jobs:
```bash
squeue -u $USER
```

### Check specific job:
```bash
squeue -j <JOBID>
```

### Check job details:
```bash
scontrol show job <JOBID>
```

### View recent job history:
```bash
sacct -u $USER --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,MaxRSS
```

---

## Expected Outputs

### Python Model Outputs:
- `outputs/models/inference_data.nc` - MCMC samples
- `outputs/models/pymc_model.pkl` - Model object
- `outputs/diagnostics/convergence_summary.csv` - Full summary
- `outputs/diagnostics/convergence_diagnostics.json` - R-hat and ESS stats
- `outputs/diagnostics/beta_coefficients.csv` - Coefficient estimates
- `outputs/diagnostics/posterior_summary.csv` - Posterior summaries
- `outputs/plots/*.png` - Visualizations

### R Model Outputs:
- `outputs/models/multivariate_model_brms_fitted.rds` - Fitted model
- `outputs/diagnostics/fixed_effects_summary.csv` - Fixed effects
- `outputs/diagnostics/coefficients_by_outcome.csv` - Coefficients by outcome
- `outputs/diagnostics/convergence_diagnostics.json` - R-hat and ESS stats
- `outputs/diagnostics/scaler_statistics.csv` - Scaling statistics
- `outputs/plots/*.pdf` - Visualizations

---

## Troubleshooting

### Job is pending for a long time:
```bash
# Check queue status
squeue -u $USER

# Check partition availability
sinfo

# Try different partition
# Edit .sbatch file: #SBATCH --partition=<other_partition>
```

### Job failed immediately:
```bash
# Check error file
cat logs/python_multivariate_<JOBID>.err
# or
cat logs/r_multivariate_<JOBID>.err

# Common issues:
# - Missing module: uncomment module load lines in .sbatch file
# - Wrong path: check that scripts exist in /n/home00/msong300/conservation
# - Permission denied: chmod +x run_python_model.sh run_r.sh
```

### Out of memory:
```bash
# Increase memory in .sbatch file
#SBATCH --mem=32G  # or 64G
```

### Time limit exceeded:
```bash
# Increase time limit in .sbatch file
#SBATCH --time=12:00:00  # or longer
```

---

## Quick Reference Commands

```bash
# Submit both jobs
sbatch run_python_multivariate.sbatch
sbatch run_r_multivariate.sbatch

# Check status
squeue -u $USER

# View latest Python log
ls -t logs/python_multivariate_*.log | head -1 | xargs tail -f

# View latest R log
ls -t logs/r_multivariate_*.log | head -1 | xargs tail -f

# Cancel all your jobs
scancel -u $USER

# Cancel specific job
scancel <JOBID>
```

