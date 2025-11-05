#!/usr/bin/env python3
"""
Multivariate Protein Consumption Analysis - PyMC Model Fitting Script
Extracted from Multivariate_Modeling_1101.ipynb

This script:
1. Loads and preprocesses protein consumption data
2. Fits a multivariate Bayesian model using PyMC
3. Saves models, diagnostics, and visualizations
4. Handles errors robustly for unattended execution
"""

import os
import sys
import time
import pickle
import platform
import traceback
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Probabilistic modeling
try:
    import pymc as pm
    import arviz as az
    from scipy import stats
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    print("✅ Libraries imported successfully!")
except ImportError as e:
    print(f"❌ Error importing required libraries: {e}")
    sys.exit(1)

# Set random seed for reproducibility
np.random.seed(42)

# Compatibility layer for PyMC versions
if not hasattr(pm, 'MutableData'):
    def MutableData(name, value, dims=None):
        return pm.Data(name, value)

# =============================================================================
# Configuration
# =============================================================================

# Create output directories
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
DIAGNOSTICS_DIR = os.path.join(OUTPUT_DIR, 'diagnostics')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

for directory in [OUTPUT_DIR, MODELS_DIR, DIAGNOSTICS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

print(f"Output directories created:")
print(f"  Models: {MODELS_DIR}")
print(f"  Diagnostics: {DIAGNOSTICS_DIR}")
print(f"  Plots: {PLOTS_DIR}")

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_and_preprocess_data():
    """Load and preprocess protein consumption data"""
    print("\n" + "="*80)
    print("STEP 1: Loading and Preprocessing Data")
    print("="*80)
    
    try:
        # Load dataset
        data_file = os.path.join(os.path.dirname(__file__), 'protein_full_data.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_csv(data_file)
        print(f"✅ Dataset loaded: {df.shape}")
        
        # Convert recall.date to datetime
        df['recall.date'] = pd.to_datetime(df['recall.date'])
        df['year'] = df['recall.date'].dt.year
        df['month'] = df['recall.date'].dt.month
        df['month_num'] = df['month']
        
        # Drop NaN converted.mass
        df_clean = df.dropna(subset=['converted.mass'])
        print(f"✅ Cleaned dataset: {df_clean.shape}")
        
        # Organize into weekly level
        print("\n🔄 Organizing data into weekly level...")
        df_clean['week'] = df_clean['recall.date'].dt.isocalendar().week
        df_clean['year'] = df_clean['recall.date'].dt.year
        df_clean['year_week'] = df_clean['year'].astype(str) + '_' + df_clean['week'].astype(str)
        df_clean['iso_year'] = df_clean['recall.date'].dt.isocalendar().year
        df_clean['week_num'] = df_clean['recall.date'].dt.isocalendar().week
        
        # Count records per household per week
        record_counts = df_clean.groupby(['household_ID', 'year_week']).agg({
            'recall.date': 'count'
        }).reset_index()
        record_counts = record_counts.rename(columns={'recall.date': 'record.days'})
        
        # Group by household, year_week, and category for weekly sums
        weekly_data = df_clean.groupby(['household_ID', 'year_week', 'category']).agg({
            'converted.mass': 'sum',
            'year': 'first',
            'season': 'first',
            'village': 'first',
            'ame': 'first',
            'count.hunters': 'first',
            'sex': 'first',
            'edu': 'first',
            'non.hunt.income': 'first',
            'month': 'first',
            'month_num': 'first',
            'recall.date': 'first',
            'iso_year': 'first',
            'week_num': 'first'
        }).reset_index()
        
        # Merge record.days
        weekly_data = weekly_data.merge(record_counts, on=['household_ID', 'year_week'], how='left')
        weekly_data['record.days'] = weekly_data['record.days'].fillna(0).astype(int)
        weekly_data = weekly_data.dropna(subset=['converted.mass'])
        
        print(f"✅ Weekly data: {weekly_data.shape}")
        
        # Create lagged variables
        print("\n🔄 Creating lagged variables...")
        lagged_weekly_data = []
        for household in weekly_data['household_ID'].unique():
            hh_data = weekly_data[weekly_data['household_ID'] == household].copy()
            for category in ['Fish', 'Wild meat', 'Domestic meat']:
                cat_data = hh_data[hh_data['category'] == category].copy()
                if len(cat_data) > 0:
                    cat_data = cat_data.sort_values('recall.date')
                    cat_data['lagged.converted.mass'] = cat_data['converted.mass'].shift(1)
                    lagged_weekly_data.append(cat_data)
        
        df_with_lag = pd.concat(lagged_weekly_data, ignore_index=True)
        print(f"✅ Data with lagged variables: {df_with_lag.shape}")
        
        # Calculate festivity variable
        print("\n🔄 Calculating festivity variable...")
        from datetime import datetime, timedelta
        
        def calculate_easter(year):
            a = year % 19
            b = year // 100
            c = year % 100
            d = b // 4
            e = b % 4
            f = (b + 8) // 25
            g = (b - f + 1) // 3
            h = (19 * a + b - d - g + 15) % 30
            i = c // 4
            k = c % 4
            l = (32 + 2 * e + 2 * i - h - k) % 7
            m = (a + 11 * h + 22 * l) // 451
            month = (h + l - 7 * m + 114) // 31
            day = ((h + l - 7 * m + 114) % 31) + 1
            return datetime(year, month, day)
        
        holidays = {
            'Easter_2021': datetime(2021, 4, 4),
            'Easter_2022': calculate_easter(2022),
            'Christmas_2021': datetime(2021, 12, 25),
            'Christmas_2022': datetime(2022, 12, 25),
            'NewYear_2022': datetime(2022, 1, 1),
            'NewYear_2021': datetime(2021, 1, 1)
        }
        
        festive_periods = []
        for holiday_date in holidays.values():
            start_date = holiday_date - timedelta(days=5)
            end_date = holiday_date + timedelta(days=5)
            festive_periods.append((start_date.date(), end_date.date()))
        
        data_start = datetime(2021, 4, 1).date()
        data_end = datetime(2022, 3, 31).date()
        festive_periods_in_range = [(s, e) for s, e in festive_periods if s <= data_end and e >= data_start]
        
        festive_dates = set()
        for start, end in festive_periods_in_range:
            current = start
            while current <= end:
                festive_dates.add(current)
                current += timedelta(days=1)
        
        df_with_lag['date_only'] = pd.to_datetime(df_with_lag['recall.date']).dt.date
        df_with_lag['is_festive_date'] = df_with_lag['date_only'].isin(festive_dates)
        
        festive_weeks = df_with_lag[df_with_lag['is_festive_date']]['year_week'].unique().tolist()
        for week_key in festive_weeks[:]:
            parts = week_key.split('_')
            year_val = int(parts[0])
            week_val = int(parts[1])
            if week_val > 1:
                prev_week_key = f"{year_val}_{week_val - 1}"
            else:
                prev_week_key = f"{year_val - 1}_52"
            festive_weeks.append(prev_week_key)
            next_week_key = f"{year_val}_{week_val + 1}"
            festive_weeks.append(next_week_key)
        
        festive_weeks = list(set(festive_weeks))
        df_with_lag['festivity'] = (df_with_lag['year_week'].isin(festive_weeks)).astype(int)
        print(f"✅ Festivity variable created. Festive weeks: {len(festive_weeks)}")
        
        # Create outcome matrix
        print("\n🔄 Creating outcome matrix...")
        outcome_data = df_with_lag.pivot_table(
            index=['household_ID', 'recall.date', 'year', 'season', 'village', 'ame', 
                   'count.hunters', 'record.days', 'sex', 'edu', 'non.hunt.income', 
                   'month', 'month_num', 'week_num', 'iso_year', 'year_week', 'festivity'],
            columns='category',
            values=['converted.mass', 'lagged.converted.mass'],
            aggfunc='first',
            fill_value=0
        ).reset_index()
        
        outcome_data.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                               for col in outcome_data.columns.values]
        
        outcome_data = outcome_data.rename(columns={
            'converted.mass_Fish': 'fish_mass',
            'converted.mass_Wild meat': 'wild_mass',
            'converted.mass_Domestic meat': 'domestic_mass',
            'lagged.converted.mass_Fish': 'lagged_fish_mass',
            'lagged.converted.mass_Wild meat': 'lagged_wild_mass',
            'lagged.converted.mass_Domestic meat': 'lagged_domestic_mass'
        })
        
        for col in ['lagged_fish_mass', 'lagged_wild_mass', 'lagged_domestic_mass']:
            if col in outcome_data.columns:
                outcome_data[col] = outcome_data[col].fillna(0)
        
        print(f"✅ Outcome matrix created: {outcome_data.shape}")
        return outcome_data
    
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
        traceback.print_exc()
        raise

# =============================================================================
# Feature Engineering
# =============================================================================

def prepare_features(outcome_data):
    """Prepare features for modeling"""
    print("\n" + "="*80)
    print("STEP 2: Feature Engineering")
    print("="*80)
    
    try:
        # Prepare feature matrix
        categorical_vars = ['season', 'village', 'sex', 'edu']
        continuous_vars = ['ame', 'count.hunters', 'record.days', 'non.hunt.income', 
                          'festivity', 'lagged_fish_mass', 'lagged_wild_mass', 
                          'lagged_domestic_mass']
        
        X = outcome_data[categorical_vars + continuous_vars].copy()
        
        # One-hot encode categorical variables
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        categorical_encoded = encoder.fit_transform(X[categorical_vars])
        categorical_feature_names = encoder.get_feature_names_out(categorical_vars)
        
        # Scale continuous variables
        scaler = StandardScaler()
        continuous_scaled = scaler.fit_transform(X[continuous_vars])
        
        # Combine
        X_processed = np.hstack([categorical_encoded, continuous_scaled])
        feature_names = list(categorical_feature_names) + continuous_vars
        
        print(f"✅ Feature matrix: {X_processed.shape}")
        print(f"✅ Features: {len(feature_names)}")
        
        # Check colinearity
        print("\n🔄 Checking colinearity...")
        X_df = pd.DataFrame(X_processed, columns=feature_names)
        correlation_matrix = X_df.corr()
        
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))
        
        if high_corr_pairs:
            print("⚠️ High correlation pairs (|r| > 0.8):")
            for pair in high_corr_pairs:
                print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
        else:
            print("✅ No high correlations found.")
        
        # Prepare outcome variables
        Y = outcome_data[['fish_mass', 'wild_mass', 'domestic_mass']].values
        Y_log = np.log(Y + 1e-6)
        
        # Get indices for random effects
        households = outcome_data['household_ID'].astype('category').cat.codes.values
        weeks = (outcome_data['iso_year'].astype(str) + '_' + 
                outcome_data['week_num'].astype(str)).astype('category').cat.codes.values
        weeks_fixed = weeks - weeks.min()
        
        n_households = len(np.unique(households))
        n_weeks = len(np.unique(weeks))
        n_obs = len(Y_log)
        
        print(f"\n✅ Data prepared for modeling:")
        print(f"  Observations: {n_obs}")
        print(f"  Households: {n_households}")
        print(f"  Weeks: {n_weeks}")
        
        # Save correlation matrix
        correlation_matrix.to_csv(
            os.path.join(DIAGNOSTICS_DIR, 'feature_correlation_matrix.csv')
        )
        
        # Save feature scalers
        with open(os.path.join(MODELS_DIR, 'encoder.pkl'), 'wb') as f:
            pickle.dump(encoder, f)
        with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        return (X_processed, Y_log, households, weeks_fixed, feature_names, 
                n_households, n_weeks, n_obs, correlation_matrix)
    
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        traceback.print_exc()
        raise

# =============================================================================
# Model Building
# =============================================================================

def build_model(X_processed, Y_log, households, weeks_fixed, feature_names,
                n_households, n_weeks, n_obs):
    """Build PyMC multivariate model"""
    print("\n" + "="*80)
    print("STEP 3: Building Model")
    print("="*80)
    
    try:
        cats = ['Fish', 'Wild', 'Domestic']
        
        coords = {
            'obs': np.arange(n_obs),
            'outcome': cats,
            'fixed': feature_names,
            'household': np.arange(n_households),
            'week': np.arange(n_weeks)
        }
        
        print("🔧 Building Log-Normal Model with Random Intercepts...")
        print(f"Fixed effects: {len(feature_names)} features")
        print(f"Random effects: {n_households} households, {n_weeks} weeks")
        
        with pm.Model(coords=coords) as model:
            # Data containers
            X_d = pm.Data('X', X_processed, dims=('obs', 'fixed'))
            hh_idx = pm.Data('hh_idx', households, dims=('obs',))
            week_idx = pm.Data('week_idx', weeks_fixed, dims=('obs',))
            Yd = pm.Data('Y_log', Y_log, dims=('obs', 'outcome'))
            
            K = len(cats)
            
            # Fixed effects (with tighter priors to reduce divergences)
            beta = pm.Normal('beta', mu=0, sigma=0.5, dims=('fixed', 'outcome'))  # Tighter prior
            mu_fixed = pm.math.dot(X_d, beta)
            
            # Household random intercepts (correlated across outcomes)
            # Using higher eta (more regularized) to reduce divergences
            sd_hh, chol_hh, corr_hh = pm.LKJCholeskyCov(
                'L_hh', n=K, eta=3.0, sd_dist=pm.Exponential.dist(2.0)  # Higher eta, higher rate for Exponential
            )
            z_hh = pm.Normal('z_hh', 0, 1, dims=('household', 'outcome'))
            b_hh = pm.Deterministic(
                'b_hh', pm.math.dot(z_hh, chol_hh.T), dims=('household', 'outcome')
            )
            mu_hh = b_hh[hh_idx]
            
            # Week random intercepts (correlated across outcomes)
            # Using higher eta (more regularized) to reduce divergences
            sd_week, chol_week, corr_week = pm.LKJCholeskyCov(
                'L_week', n=K, eta=3.0, sd_dist=pm.Exponential.dist(2.0)  # Higher eta, higher rate for Exponential
            )
            z_week = pm.Normal('z_week', 0, 1, dims=('week', 'outcome'))
            b_week = pm.Deterministic(
                'b_week', pm.math.dot(z_week, chol_week.T), dims=('week', 'outcome')
            )
            mu_week = b_week[week_idx]
            
            # Residual correlation
            # Using higher eta (more regularized) to reduce divergences
            sd_eps, chol_eps, corr_eps = pm.LKJCholeskyCov(
                'L_eps', n=K, eta=3.0, sd_dist=pm.Exponential.dist(2.0)  # Higher eta, higher rate for Exponential
            )
            
            # Mean structure
            mu = pm.Deterministic(
                'mu', mu_fixed + mu_hh + mu_week, dims=('obs', 'outcome')
            )
            
            # Likelihood
            y = pm.MvNormal(
                'y', mu=mu, chol=chol_eps, observed=Yd, dims=('obs', 'outcome')
            )
        
        print("✅ Model built successfully!")
        return model
    
    except Exception as e:
        print(f"❌ Error building model: {e}")
        traceback.print_exc()
        raise

# =============================================================================
# Model Fitting
# =============================================================================

def fit_model(model):
    """Fit PyMC model with appropriate settings"""
    print("\n" + "="*80)
    print("STEP 4: Fitting Model")
    print("="*80)
    
    try:
        # Detect available cores
        # Some HPC systems have issues with multiprocessing, so we'll try but fall back
        if platform.system() == 'Darwin':  # macOS
            print("⚠️  Detected macOS - using single-threaded sampling")
            n_cores = 1
            n_chains = 2
        else:
            # Check available cores
            # For HPC systems, start conservatively and let the error handler catch issues
            try:
                import multiprocessing
                import os
                # Check if we're in a restricted environment (common on HPC)
                available_cores = multiprocessing.cpu_count()
                # Be conservative on HPC systems - start with 2 chains, 1 core
                # This avoids multiprocessing issues while still getting multiple chains
                n_cores = 1  # Use 1 core to avoid multiprocessing issues
                n_chains = 2  # Use 2 chains for convergence checking
                print(f"✅ Detected {available_cores} cores, using {n_chains} chains with {n_cores} core (conservative for stability)")
            except:
                n_cores = 1
                n_chains = 2
                print("⚠️  Could not detect cores, using single-threaded sampling")
        
        print(f"Using {n_chains} chains with {n_cores} core(s)")
        print("Target convergence: RHAT ~1.0, ESS ~500")
        print("⏱️  This may take 2-3 hours...")
        print("")
        print("="*80)
        print("STARTING MCMC SAMPLING")
        print("="*80)
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total iterations: {(1000 + 1000) * n_chains} (1000 tune + 1000 draws per chain)")
        print("")
        sys.stdout.flush()  # Ensure output is printed immediately
        
        start_time = time.time()
        last_progress_time = start_time
        
        # Progress callback function
        def progress_callback(iter, *args):
            global last_progress_time
            current_time = time.time()
            elapsed = current_time - start_time
            # Print progress every 30 seconds
            if current_time - last_progress_time >= 30:
                iter_per_chain = iter // n_chains if n_chains > 0 else iter
                total_iters = (1000 + 1000) * n_chains
                progress_pct = (iter / total_iters) * 100 if total_iters > 0 else 0
                elapsed_min = elapsed / 60
                print(f"⏳ Progress: {iter}/{total_iters} iterations ({progress_pct:.1f}%) | "
                      f"Elapsed: {elapsed_min:.1f} min | "
                      f"Time: {time.strftime('%H:%M:%S')}")
                sys.stdout.flush()
                last_progress_time = current_time
        
        with model:
            try:
                # Use higher target_accept to reduce divergences (0.99 is more conservative)
                # Also use different initialization strategies per chain to avoid all divergences
                # Note: progressbar=False to ensure output goes to log files properly
                print("📊 Starting MCMC sampling...")
                print("   Note: This may take 2-3 hours. Progress updates will be minimal during sampling.")
                print("   Check process status with: ps aux | grep python3 | grep fit_model_pymc")
                print("   Expected completion time: ~2-3 hours from start\n")
                sys.stdout.flush()
                idata = pm.sample(
                    target_accept=0.95,  # Standard target acceptance rate
                    draws=1000,
                    tune=1000,
                    chains=n_chains,
                    cores=n_cores,
                    random_seed=42,
                    init='adapt_diag',  # Use adapt_diag initialization (jitter is automatic)
                    idata_kwargs={"log_likelihood": True},
                    progressbar=False,  # Disable progressbar for better log file compatibility
                    compute_convergence_checks=True  # Enable convergence checks for output
                )
            except (EOFError, BrokenPipeError, OSError, pickle.PickleError) as e:
                print(f"\n⚠️  Multiprocessing error: {e}")
                print("Falling back to single-threaded sampling...")
                sys.stdout.flush()
                idata = pm.sample(
                    target_accept=0.95,  # Standard target acceptance rate
                    draws=1000,
                    tune=1000,
                    chains=2,
                    cores=1,
                    random_seed=42,
                    init='adapt_diag',  # Use adapt_diag initialization (jitter is automatic)
                    idata_kwargs={"log_likelihood": True},
                    progressbar=False,  # Disable progressbar for better log file compatibility
                    compute_convergence_checks=True  # Enable convergence checks for output
                )
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Model sampling completed in {elapsed_time/60:.1f} minutes!")
        
        # Check for divergences and warn user
        try:
            divergences = idata.sample_stats.diverging
            n_divergences = divergences.sum().values if hasattr(divergences.sum(), 'values') else divergences.sum()
            if hasattr(n_divergences, 'sum'):
                n_divergences = n_divergences.sum()
            
            if n_divergences > 0:
                print(f"\n⚠️  WARNING: {n_divergences} divergences detected!")
                print(f"  This may indicate geometric issues in the model.")
                print(f"  Consider reparameterization or adjusting target_accept.")
                
                # Check if all samples in a chain diverged
                for chain_idx in range(divergences.shape[0]):
                    chain_divs = divergences[chain_idx].sum().values if hasattr(divergences[chain_idx].sum(), 'values') else divergences[chain_idx].sum()
                    if chain_divs == divergences.shape[1]:  # All samples diverged
                        print(f"  ⚠️  Chain {chain_idx} has diverged completely - results may be unreliable!")
        except Exception as e:
            print(f"⚠️  Could not check divergences: {e}")
        
        # Save inference data with error handling
        try:
            idata_file = os.path.join(MODELS_DIR, 'inference_data.nc')
            idata.to_netcdf(idata_file)
            print(f"✅ Inference data saved to {idata_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save inference data: {e}")
            print("  Attempting to continue with analysis...")
        
        # Save model as pickle (for reference)
        try:
            model_file = os.path.join(MODELS_DIR, 'pymc_model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ Model saved to {model_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save model pickle: {e}")
        
        return idata
    
    except Exception as e:
        print(f"❌ Error fitting model: {e}")
        traceback.print_exc()
        raise

# =============================================================================
# Convergence Diagnostics
# =============================================================================

def check_convergence(idata):
    """Check and save convergence diagnostics"""
    print("\n" + "="*80)
    print("STEP 5: Convergence Diagnostics")
    print("="*80)
    
    try:
        # Get summary
        summary = az.summary(idata, var_names=['beta', 'L_hh', 'L_week', 'L_eps'])
        
        # Save full summary
        summary_file = os.path.join(DIAGNOSTICS_DIR, 'convergence_summary.csv')
        summary.to_csv(summary_file)
        print(f"✅ Full summary saved to {summary_file}")
        
        # Check R-hat
        rhat_vals = summary['r_hat']
        print(f"\n📊 RHAT statistics:")
        print(f"  Mean: {rhat_vals.mean():.4f}")
        print(f"  Min: {rhat_vals.min():.4f}")
        print(f"  Max: {rhat_vals.max():.4f}")
        print(f"  Parameters with RHAT > 1.01: {(rhat_vals > 1.01).sum()} / {len(rhat_vals)}")
        
        # Check ESS
        ess_vals = summary['ess_bulk']
        print(f"\n📊 ESS statistics:")
        print(f"  Mean: {ess_vals.mean():.0f}")
        print(f"  Min: {ess_vals.min():.0f}")
        print(f"  Max: {ess_vals.max():.0f}")
        print(f"  Parameters with ESS < 400: {(ess_vals < 400).sum()} / {len(ess_vals)}")
        
        # Save diagnostics
        diagnostics = {
            'rhat': {
                'mean': float(rhat_vals.mean()),
                'min': float(rhat_vals.min()),
                'max': float(rhat_vals.max()),
                'n_high': int((rhat_vals > 1.01).sum()),
                'n_total': int(len(rhat_vals))
            },
            'ess_bulk': {
                'mean': float(ess_vals.mean()),
                'min': float(ess_vals.min()),
                'max': float(ess_vals.max()),
                'n_low': int((ess_vals < 400).sum()),
                'n_total': int(len(ess_vals))
            }
        }
        
        import json
        diagnostics_file = os.path.join(DIAGNOSTICS_DIR, 'convergence_diagnostics.json')
        with open(diagnostics_file, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        print(f"✅ Diagnostics saved to {diagnostics_file}")
        
        # Check convergence status
        if (rhat_vals.max() > 1.01) or (ess_vals.min() < 400):
            print("\n⚠️ Convergence not fully achieved. Consider increasing draws/tunes/chains.")
        else:
            print("\n✅ Convergence achieved! RHAT ~1 and ESS >= 400")
        
        return summary
    
    except Exception as e:
        print(f"❌ Error in convergence diagnostics: {e}")
        traceback.print_exc()
        return None

# =============================================================================
# Extract and Save Results
# =============================================================================

def save_results(idata, feature_names):
    """Extract and save model results"""
    print("\n" + "="*80)
    print("STEP 6: Extracting Results")
    print("="*80)
    
    try:
        # Extract fixed effects
        beta_summary = az.summary(idata, var_names=['beta'])
        beta_summary['feature'] = [
            feature_names[i] for i in range(len(feature_names)) for _ in range(3)
        ]
        beta_summary['outcome'] = ['Fish', 'Wild', 'Domestic'] * len(feature_names)
        
        # Save beta summary
        beta_file = os.path.join(DIAGNOSTICS_DIR, 'beta_coefficients.csv')
        beta_summary.to_csv(beta_file)
        print(f"✅ Beta coefficients saved to {beta_file}")
        
        # Extract posterior samples for beta
        beta_samples = idata.posterior['beta']
        
        # Save posterior summaries
        posterior_summary = az.summary(idata)
        posterior_summary_file = os.path.join(DIAGNOSTICS_DIR, 'posterior_summary.csv')
        posterior_summary.to_csv(posterior_summary_file)
        print(f"✅ Posterior summary saved to {posterior_summary_file}")
        
        return beta_summary, beta_samples
    
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        traceback.print_exc()
        return None, None

# =============================================================================
# Create Visualizations
# =============================================================================

def create_visualizations(idata, feature_names, correlation_matrix):
    """Create and save visualizations"""
    print("\n" + "="*80)
    print("STEP 7: Creating Visualizations")
    print("="*80)
    
    try:
        # Get beta summary
        beta_summary = az.summary(idata, var_names=['beta'])
        
        # Extract fixed effects data
        fixed_effects_data = []
        for i, row in beta_summary.iterrows():
            var_name = row.name
            if 'beta[' in var_name:
                parts = var_name.replace('beta[', '').replace(']', '').split(', ')
                if len(parts) == 2:
                    predictor = parts[0]
                    outcome = parts[1]
                    fixed_effects_data.append({
                        'Predictor': predictor,
                        'Outcome': outcome,
                        'mean': row['mean'],
                        'hdi_3%': row['hdi_3%'],
                        'hdi_97%': row['hdi_97%']
                    })
        
        fixed_effects_df = pd.DataFrame(fixed_effects_data)
        
        # 1. Effect plots by meat type (matching example image)
        print("\n🔄 Creating effect plots by meat type...")
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        sns.set(style="whitegrid", context="talk")
        
        palette = {
            'Increase': '#1f77b4',
            'Uncertain': '#8c8c8c',
            'Decrease': '#d62728'
        }
        
        for idx, meat_type in enumerate(['Fish', 'Wild', 'Domestic']):
            meat_data = fixed_effects_df[fixed_effects_df['Outcome'] == meat_type].copy()
            meat_data['abs_effect'] = np.abs(meat_data['mean'])
            meat_data = meat_data.sort_values('abs_effect', ascending=True)
            
            def get_direction(row):
                if row['hdi_3%'] < 0 and row['hdi_97%'] > 0:
                    return 'Uncertain'
                elif row['mean'] > 0:
                    return 'Increase'
                else:
                    return 'Decrease'
            
            meat_data['Direction'] = meat_data.apply(get_direction, axis=1)
            
            sns.barplot(
                data=meat_data,
                x="mean",
                y="Predictor",
                hue="Direction",
                dodge=False,
                palette=palette,
                ax=axes[idx]
            )
            
            axes[idx].axvline(0, color="black", linestyle="--", alpha=0.6)
            axes[idx].set_xlabel("Change in expected consumption (log scale)")
            axes[idx].set_ylabel("")
            axes[idx].set_title(f"{meat_type} Meat Effects")
            axes[idx].legend().remove()
        
        handles, labels = axes[0].get_legend_handles_labels()
        order = ['Increase', 'Uncertain', 'Decrease']
        handles_ordered = [handles[labels.index(l)] for l in order]
        labels_ordered = [l for l in order]
        
        fig.legend(handles_ordered, labels_ordered, title="Effect direction",
                   loc='center right', bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        effect_plot_file = os.path.join(PLOTS_DIR, 'effects_by_meat_type.png')
        plt.savefig(effect_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Effect plots saved to {effect_plot_file}")
        
        # 2. Correlation matrix heatmap
        print("\n🔄 Creating correlation matrix...")
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        corr_plot_file = os.path.join(PLOTS_DIR, 'feature_correlation_matrix.png')
        plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Correlation matrix saved to {corr_plot_file}")
        
        # 3. Coefficient comparison across meat types
        print("\n🔄 Creating coefficient comparison plot...")
        predictors = fixed_effects_df['Predictor'].unique()
        meat_types = ['Fish', 'Wild', 'Domestic']
        
        # Prepare data for grouped bar chart
        comparison_data = []
        for predictor in predictors:
            for meat_type in meat_types:
                meat_data = fixed_effects_df[
                    (fixed_effects_df['Predictor'] == predictor) &
                    (fixed_effects_df['Outcome'] == meat_type)
                ]
                if len(meat_data) > 0:
                    comparison_data.append({
                        'Predictor': predictor,
                        'Meat Type': meat_type,
                        'Coefficient': meat_data['mean'].iloc[0],
                        'CI_Lower': meat_data['hdi_3%'].iloc[0],
                        'CI_Upper': meat_data['hdi_97%'].iloc[0],
                        'CI_Width': meat_data['hdi_97%'].iloc[0] - meat_data['hdi_3%'].iloc[0]
                    })
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(16, 8))
        x_pos = np.arange(len(predictors))
        width = 0.25
        
        colors = {'Fish': '#1f77b4', 'Wild': '#ff7f0e', 'Domestic': '#2ca02c'}
        
        for i, meat_type in enumerate(meat_types):
            meat_means = [
                comp_df[(comp_df['Predictor'] == p) & (comp_df['Meat Type'] == meat_type)]['Coefficient'].iloc[0]
                if len(comp_df[(comp_df['Predictor'] == p) & (comp_df['Meat Type'] == meat_type)]) > 0
                else 0
                for p in predictors
            ]
            meat_errors = [
                comp_df[(comp_df['Predictor'] == p) & (comp_df['Meat Type'] == meat_type)]['CI_Width'].iloc[0] / 2
                if len(comp_df[(comp_df['Predictor'] == p) & (comp_df['Meat Type'] == meat_type)]) > 0
                else 0
                for p in predictors
            ]
            
            ax.bar(x_pos + i*width, meat_means, width, yerr=meat_errors,
                  label=meat_type, alpha=0.8, capsize=5, color=colors[meat_type])
        
        ax.set_xlabel('Predictors')
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Coefficient Comparison Across Meat Types')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(predictors, rotation=45, ha='right')
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        comp_plot_file = os.path.join(PLOTS_DIR, 'coefficient_comparison.png')
        plt.savefig(comp_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Coefficient comparison saved to {comp_plot_file}")
        
        # 4. Uncertainty heatmap (CI width)
        print("\n🔄 Creating uncertainty heatmap...")
        uncertainty_data = comp_df.pivot(index='Predictor', columns='Meat Type', values='CI_Width')
        
        plt.figure(figsize=(10, 12))
        sns.heatmap(uncertainty_data, annot=True, cmap='YlOrRd', fmt='.2f',
                   cbar_kws={'label': 'Credible Interval Width'})
        plt.title('Uncertainty Heatmap (CI Width)')
        plt.tight_layout()
        
        uncert_plot_file = os.path.join(PLOTS_DIR, 'uncertainty_heatmap.png')
        plt.savefig(uncert_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Uncertainty heatmap saved to {uncert_plot_file}")
        
        # 5. Coefficient sign heatmap
        print("\n🔄 Creating coefficient sign heatmap...")
        sign_data = comp_df.pivot(index='Predictor', columns='Meat Type', values='Coefficient')
        
        plt.figure(figsize=(10, 12))
        sns.heatmap(sign_data, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
                   cbar_kws={'label': 'Coefficient Value'})
        plt.title('Coefficient Sign by Predictor and Outcome')
        plt.tight_layout()
        
        sign_plot_file = os.path.join(PLOTS_DIR, 'coefficient_sign_heatmap.png')
        plt.savefig(sign_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Coefficient sign heatmap saved to {sign_plot_file}")
        
        # 6. Trace plots for key parameters
        print("\n🔄 Creating trace plots...")
        try:
            fig = az.plot_trace(idata, var_names=['beta'], compact=True, backend_kwargs={'figsize': (12, 8)})
            trace_plot_file = os.path.join(PLOTS_DIR, 'trace_plots.png')
            plt.savefig(trace_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Trace plots saved to {trace_plot_file}")
        except Exception as e:
            print(f"⚠️  Could not create trace plots: {e}")
        
        print("\n✅ All visualizations created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        traceback.print_exc()

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("MULTIVARIATE PROTEIN CONSUMPTION MODEL - PyMC")
    print("="*80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Load and preprocess data
        outcome_data = load_and_preprocess_data()
        
        # Step 2: Prepare features
        (X_processed, Y_log, households, weeks_fixed, feature_names,
         n_households, n_weeks, n_obs, correlation_matrix) = prepare_features(outcome_data)
        
        # Step 3: Build model
        model = build_model(X_processed, Y_log, households, weeks_fixed, feature_names,
                           n_households, n_weeks, n_obs)
        
        # Step 4: Fit model
        idata = fit_model(model)
        
        # Step 5: Check convergence (with error handling)
        try:
            summary = check_convergence(idata)
        except Exception as e:
            print(f"\n⚠️  Warning: Error in convergence diagnostics: {e}")
            print("  Attempting to continue with remaining analysis...")
            traceback.print_exc()
            summary = None
        
        # Step 6: Save results (with error handling)
        try:
            beta_summary, beta_samples = save_results(idata, feature_names)
        except Exception as e:
            print(f"\n⚠️  Warning: Error saving results: {e}")
            print("  Attempting to continue with visualizations...")
            traceback.print_exc()
            beta_summary, beta_samples = None, None
        
        # Step 7: Create visualizations (with error handling)
        try:
            create_visualizations(idata, feature_names, correlation_matrix)
        except Exception as e:
            print(f"\n⚠️  Warning: Error creating visualizations: {e}")
            traceback.print_exc()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nAll outputs saved to: {OUTPUT_DIR}")
        print(f"  - Models: {MODELS_DIR}")
        print(f"  - Diagnostics: {DIAGNOSTICS_DIR}")
        print(f"  - Plots: {PLOTS_DIR}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

