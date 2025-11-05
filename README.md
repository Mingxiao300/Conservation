# Multivariate Protein Consumption Model

This repository contains scripts for fitting multivariate Bayesian models to analyze protein consumption patterns using both PyMC (Python) and brms (R).

## Overview

The analysis models protein consumption across three categories:
- Fish
- Wild meat
- Domestic meat

The models include:
- Fixed effects: season, village, sex, education, age (ame), household size, non-hunting income, festivity, and lagged consumption variables
- Random effects: Household and week-level intercepts (correlated across outcomes)
- Temporal autocorrelation: Lagged consumption variables

## Directory Structure

```
conservation/
├── protein_full_data.csv          # Input data file
├── fit_model_pymc.py              # Python (PyMC) model fitting script
├── Multivariate_Modeling_1101.R   # R (brms) model fitting script
├── run_multivariate_models.sh     # Bash script to run both models
├── outputs/                        # Output directory (created automatically)
│   ├── models/                    # Saved model files
│   ├── diagnostics/               # Convergence diagnostics, summaries
│   └── plots/                     # Visualization plots
└── logs/                          # Execution logs (created automatically)
```

## Requirements

### Python (PyMC)
- Python 3.8+
- PyMC 5+
- ArviZ
- pandas, numpy, matplotlib, seaborn, scikit-learn

### R (brms)
- R 4.0+
- brms
- tidyverse
- bayesplot
- posterior

## Usage

### Run Both Models in Parallel (Recommended - Fastest)

Run both Python and R models **in parallel** using separate tmux sessions:

```bash
cd /n/home00/msong300/conservation
./run_multivariate_models.sh
```

This will:
- Create two separate tmux sessions (`pymc_model` and `brms_model`)
- Run both models simultaneously in parallel
- **Total time: ~2-3 hours** (instead of 4-6 hours sequentially)

**Monitor progress:**
```bash
# List active sessions
tmux ls

# View Python model
tmux attach -t pymc_model

# View R model
tmux attach -t brms_model
```

### Run Python Only

```bash
./run_multivariate_models.sh --python-only
```

### Run R Only

```bash
./run_multivariate_models.sh --r-only
```

### Run in tmux Session (Recommended for Long Runs)

```bash
# Create a new tmux session
tmux new -s model_fitting

# Run the script
cd /n/home00/msong300/conservation
./run_multivariate_models.sh

# Detach from session: Ctrl+B, then D
# Reattach to session: tmux attach -t model_fitting
```

### Run Python Script Directly

```bash
cd /n/home00/msong300/conservation
python3 fit_model_pymc.py
```

### Run R Script Directly

```bash
cd /n/home00/msong300/conservation
Rscript Multivariate_Modeling_1101.R
```

## Output Files

### Models
- **Python (PyMC)**:
  - `outputs/models/inference_data.nc` - ArviZ InferenceData (NetCDF)
  - `outputs/models/pymc_model.pkl` - PyMC model object
  - `outputs/models/encoder.pkl` - Feature encoder
  - `outputs/models/scaler.pkl` - Feature scaler

- **R (brms)**:
  - `outputs/models/multivariate_model_brms_fitted.rds` - Fitted brms model
  - `outputs/models/multivariate_model_brms.rds` - Model specification

### Diagnostics
- `outputs/diagnostics/convergence_summary.csv` - Full parameter summary
- `outputs/diagnostics/convergence_diagnostics.json` - R-hat and ESS statistics
- `outputs/diagnostics/beta_coefficients.csv` - Fixed effects coefficients
- `outputs/diagnostics/posterior_summary.csv` - Full posterior summary
- `outputs/diagnostics/fixed_effects_summary.csv` - Fixed effects (R)
- `outputs/diagnostics/coefficients_by_outcome.csv` - Coefficients by outcome (R)
- `outputs/diagnostics/posterior_samples.csv` - Posterior samples (R)
- `outputs/diagnostics/feature_correlation_matrix.csv` - Feature correlations

### Plots
- `outputs/plots/effects_by_meat_type.png` - Effect plots by meat type (horizontal bar charts)
- `outputs/plots/feature_correlation_matrix.png` - Feature correlation heatmap
- `outputs/plots/coefficient_comparison.png` - Coefficient comparison across meat types
- `outputs/plots/uncertainty_heatmap.png` - Uncertainty (CI width) heatmap
- `outputs/plots/coefficient_sign_heatmap.png` - Coefficient sign heatmap
- `outputs/plots/trace_plots.png` - MCMC trace plots
- Additional R plots: PDF files for various diagnostics

## Convergence Diagnostics

The scripts automatically check and report:
- **R-hat**: Target ≤ 1.01 for all parameters
- **ESS (Effective Sample Size)**: Target ≥ 400 for all parameters

If convergence is not achieved, consider:
- Increasing the number of draws (in script)
- Increasing the number of tuning iterations
- Checking for model specification issues

## Execution Time

### Parallel Mode (Default)
- **Total time: ~2-3 hours** (both models run simultaneously)
- Python model: 2-3 hours
- R model: 2-3 hours
- Models run in parallel, so total time = max(time for each model)

### Sequential Mode (`--sequential`)
- **Total time: 4-6 hours** (models run one after another)
- Python model: 2-3 hours
- R model: 2-3 hours

Time varies based on:
- Number of cores available
- System resources
- Model complexity

## Troubleshooting

### Python Issues

**ModuleNotFoundError**: Install missing packages
```bash
pip install pymc arviz pandas numpy matplotlib seaborn scikit-learn
```

**Memory errors**: Reduce number of chains or draws in script

### R Issues

**Package not found**: Install missing packages
```R
install.packages(c("brms", "tidyverse", "bayesplot", "posterior"))
```

**Stan compilation errors**: Check Stan installation
```R
install.packages("cmdstanr")
cmdstanr::check_cmdstan_toolchain()
```

### General Issues

**Script permission denied**: Make script executable
```bash
chmod +x run_multivariate_models.sh fit_model_pymc.py
```

**Data file not found**: Ensure `protein_full_data.csv` is in the conservation directory

**Logs location**: Check `logs/` directory for detailed execution logs

## Notes

- The scripts automatically detect available CPU cores and adjust parallelism
- On macOS, single-threaded mode is used to avoid multiprocessing issues
- All outputs are saved to `outputs/` subdirectories
- Logs are saved to `logs/` with timestamps
- The scripts include robust error handling for unattended execution

## Citation

If you use these scripts, please cite:
- PyMC: Salvatier et al. (2016) PyMC3: Python probabilistic programming framework
- brms: Bürkner (2017) brms: An R Package for Bayesian Multilevel Models using Stan

