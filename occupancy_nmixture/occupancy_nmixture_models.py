#!/usr/bin/env python3
"""
Occupancy and N-mixture Models for Protein Consumption Analysis

This script implements:
1. Dynamic Occupancy Model: Models binary consumption (0/1) with colonization/extinction
2. N-mixture Model: Models actual consumption mass with effects from other protein types
"""

import os
import sys
import time
import pickle
import platform
import traceback
import datetime
import re
from datetime import datetime as dt, timedelta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    import pymc as pm
    import arviz as az
    from scipy import stats
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    print("✅ Libraries imported successfully!")
except ImportError as e:
    print(f"❌ Error importing required libraries: {e}")
    sys.exit(1)

# Set random seed
np.random.seed(42)

# =============================================================================
# Configuration
# =============================================================================

# Get parent directory (conservation folder) for data file
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PARENT_DIR, 'outputs')
BASE_OCCUPANCY_DIR = os.path.join(OUTPUT_DIR, 'occupancy_models')
BASE_NMIXTURE_DIR = os.path.join(OUTPUT_DIR, 'nmixture_models')

# Create timestamped subdirectory for this run
TIMESTAMP = time.strftime('%Y%m%d_%H%M%S')
RUN_ID = f"run_{TIMESTAMP}"
OCCUPANCY_DIR = os.path.join(BASE_OCCUPANCY_DIR, RUN_ID)
NMIXTURE_DIR = os.path.join(BASE_NMIXTURE_DIR, RUN_ID)

for directory in [OUTPUT_DIR, BASE_OCCUPANCY_DIR, BASE_NMIXTURE_DIR, OCCUPANCY_DIR, NMIXTURE_DIR]:
    os.makedirs(directory, exist_ok=True)

print(f"Output directories created:")
print(f"  Occupancy models: {OCCUPANCY_DIR}")
print(f"  N-mixture models: {NMIXTURE_DIR}")
print(f"  Run ID: {RUN_ID}")

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_and_preprocess_data():
    """Load and preprocess data for occupancy/n-mixture models"""
    print("\n" + "="*80)
    print("STEP 1: Loading and Preprocessing Data")
    print("="*80)
    
    try:
        # Get parent directory (conservation folder) for data file
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_file = os.path.join(parent_dir, 'protein_full_data.csv')
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_csv(data_file)
        print(f"✅ Dataset loaded: {df.shape}")
        
        # Convert recall.date to datetime and derive calendar week (ISO week)
        df['recall.date'] = pd.to_datetime(df['recall.date'])
        df['year'] = df['recall.date'].dt.year
        df['month'] = df['recall.date'].dt.month
        # ISO calendar week and year
        iso = df['recall.date'].dt.isocalendar()
        df['iso_year'] = iso['year'].astype(int)
        df['week_num'] = iso['week'].astype(int)
        df['week_id'] = df['iso_year'].astype(str) + '_' + df['week_num'].astype(str)
        
        # Drop NaN converted.mass
        df_clean = df.dropna(subset=['converted.mass'])
        print(f"✅ Cleaned dataset: {df_clean.shape}")
        
        # Organize into daily level
        # NOTE: Data is kept at per household per day level
        # Each record represents consumption per household per day
        print("\n🔄 Organizing data into daily level (per household per day)...")
        df_clean['year'] = df_clean['recall.date'].dt.year
        
        # Group by household, date, and category for daily sums
        # (in case there are multiple records per household per day per category)
        daily_data = df_clean.groupby(['household_ID', 'recall.date', 'category']).agg({
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
            'iso_year': 'first',
            'week_num': 'first',
            'week_id': 'first'
        }).reset_index()
        
        # Count records per household per day (for record.days - usually 1 per day)
        record_counts = df_clean.groupby(['household_ID', 'recall.date']).size().reset_index(name='record.days')
        
        # Merge record.days
        daily_data = daily_data.merge(record_counts, on=['household_ID', 'recall.date'], how='left')
        daily_data['record.days'] = daily_data['record.days'].fillna(1).astype(int)
        daily_data = daily_data.dropna(subset=['converted.mass'])
        
        print(f"✅ Daily data: {daily_data.shape}")

        
        holidays = {
            'Easter_2021': dt(2021, 4, 4),
            'Easter_2022': dt(2022, 4, 24),
            'Christmas_2021': dt(2021, 12, 25),
            'Christmas_2022': dt(2022, 12, 25),
            'NewYear_2022': dt(2022, 1, 1),
            'NewYear_2021': dt(2021, 1, 1)
        }
        
        festive_periods = []
        for holiday_date in holidays.values():
            start_date = holiday_date - timedelta(days=5)
            end_date = holiday_date + timedelta(days=5)
            festive_periods.append((start_date.date(), end_date.date()))
        
        data_start = dt(2021, 4, 1).date()
        data_end = dt(2022, 3, 31).date()
        festive_periods_in_range = [(s, e) for s, e in festive_periods if s <= data_end and e >= data_start]
        
        festive_dates = set()
        for start, end in festive_periods_in_range:
            current = start
            while current <= end:
                festive_dates.add(current)
                current += timedelta(days=1)
        
        # Calculate festivity directly from dates (no need for week-based logic)
        daily_data['date_only'] = pd.to_datetime(daily_data['recall.date']).dt.date
        daily_data['is_festive_date'] = daily_data['date_only'].isin(festive_dates)
        
        # Create festivity as categorical (binary: 0=non-festive, 1=festive)
        daily_data['festivity'] = daily_data['is_festive_date'].astype(int)
        n_festive_days = daily_data['festivity'].sum()
        print(f"✅ Festivity variable created (categorical: 0=non-festive, 1=festive). Festive days: {n_festive_days}")
        
        # Create outcome matrix with all 4 protein types
        print("\n🔄 Creating outcome matrix...")
        outcome_data = daily_data.pivot_table(
            index=['household_ID', 'recall.date', 'year', 'season', 'village', 'ame', 
                   'count.hunters', 'record.days', 'sex', 'edu', 'non.hunt.income', 
                   'month', 'festivity', 'iso_year', 'week_num', 'week_id'],
            columns='category',
            values='converted.mass',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        # Handle column names
        outcome_data.columns.name = None
        outcome_data = outcome_data.rename(columns={
            'Fish': 'fish_mass',
            'Wild meat': 'wild_mass',
            'Domestic meat': 'domestic_mass',
            'Invertebrate': 'invertebrate_mass'
        })
        
        # Fill missing columns with zeros
        for col in ['fish_mass', 'wild_mass', 'domestic_mass', 'invertebrate_mass']:
            if col not in outcome_data.columns:
                outcome_data[col] = 0
        
        print(f"✅ Outcome matrix created: {outcome_data.shape}")
        
        # Sort by household and time for dynamic models
        outcome_data = outcome_data.sort_values(['household_ID', 'recall.date'])
        
        # Create binary indicators
        for col in ['fish_mass', 'wild_mass', 'domestic_mass', 'invertebrate_mass']:
            outcome_data[f'{col.replace("_mass", "_binary")}'] = (outcome_data[col] > 0).astype(int)
        
        print(f"✅ Binary indicators created")
        
        return outcome_data
    
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
        traceback.print_exc()
        raise

# =============================================================================
# Dynamic Occupancy Model
# =============================================================================

def prepare_occupancy_data(outcome_data, target_protein='Wild meat'):
    """Prepare data for dynamic occupancy model"""
    print(f"\n🔄 Preparing data for occupancy model (target: {target_protein})...")
    
    # Map target protein to column names
    protein_map = {
        'Wild meat': ('wild_mass', 'wild_binary'),
        'Fish': ('fish_mass', 'fish_binary'),
        'Domestic meat': ('domestic_mass', 'domestic_binary'),
        'Invertebrate': ('invertebrate_mass', 'invertebrate_binary')
    }
    
    target_col, target_binary_col = protein_map[target_protein]
    
    # Get other protein types
    other_proteins = {
        'fish': 'fish_binary',
        'domestic': 'domestic_binary',
        'invertebrate': 'invertebrate_binary'
    }
    # Remove the target from others
    if target_protein == 'Fish':
        other_proteins.pop('fish', None)
    elif target_protein == 'Domestic meat':
        other_proteins.pop('domestic', None)
    elif target_protein == 'Invertebrate':
        other_proteins.pop('invertebrate', None)
    else:  # Wild meat
        pass  # Keep all others
    
    # Prepare data - sort by household and time first
    model_data = outcome_data.copy().sort_values(['household_ID', 'recall.date'])
    model_data['target_binary'] = model_data[target_binary_col]
    
    # Get other protein binary indicators
    other_binary_cols = [col for col in other_proteins.values() if col in model_data.columns]
    
    # Prepare features
    # NOTE: festivity is categorical (binary) and should not be scaled
    categorical_vars = ['season', 'village', 'sex', 'edu', 'festivity']  # Include festivity as categorical
    continuous_vars = ['ame', 'count.hunters', 'record.days', 'non.hunt.income']
    
    # Create feature matrix
    X = model_data[categorical_vars + continuous_vars + other_binary_cols].copy()
    
    # One-hot encode categorical (includes festivity as binary indicator)
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    categorical_encoded = encoder.fit_transform(X[categorical_vars])
    categorical_feature_names = encoder.get_feature_names_out(categorical_vars)
    
    # Scale continuous + binary protein indicators (festivity is already encoded, so not included here)
    scaler = StandardScaler()
    continuous_plus_other = X[continuous_vars + other_binary_cols]
    continuous_scaled = scaler.fit_transform(continuous_plus_other)
    
    # Combine
    X_processed = np.hstack([categorical_encoded, continuous_scaled])
    feature_names = list(categorical_feature_names) + continuous_vars + other_binary_cols
    
    # Get binary outcome
    y_binary = model_data['target_binary'].values
    
    # Get household indices
    households = model_data['household_ID'].astype('category').cat.codes.values
    # Get calendar week indices (ISO week within ISO year)
    # Use week_id to avoid collisions across years
    weeks = model_data['week_id'].astype('category').cat.codes.values
    
    # Create time series per household and previous state indicator
    household_data = []
    prev_state = np.zeros(len(model_data))
    is_first = np.zeros(len(model_data), dtype=int)
    
    obs_idx = 0
    for hh_id in model_data['household_ID'].unique():
        hh_mask = model_data['household_ID'] == hh_id
        hh_data = model_data[hh_mask].copy().sort_values('recall.date').reset_index(drop=True)
        household_data.append(hh_data)
        
        hh_indices = np.where(hh_mask)[0]
        hh_y = y_binary[hh_indices]
        
        # Mark first observation
        if len(hh_indices) > 0:
            is_first[hh_indices[0]] = 1
        
        # Previous state for subsequent observations
        for i in range(1, len(hh_indices)):
            prev_state[hh_indices[i]] = hh_y[i-1]
    
    n_households = len(household_data)
    max_time = max(len(hh) for hh in household_data)
    
    print(f"✅ Prepared data for occupancy model:")
    print(f"  Households: {n_households}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Max time periods: {max_time}")
    print(f"  Total observations: {len(y_binary)}")
    
    return {
        'X_processed': X_processed,
        'y_binary': y_binary,
        'households': households,
        'weeks': weeks,
        'prev_state': prev_state,
        'is_first': is_first,
        'household_data': household_data,
        'feature_names': feature_names,
        'encoder': encoder,
        'scaler': scaler,
        'target_protein': target_protein
    }

def build_occupancy_model(data_dict):
    """Build dynamic occupancy model with colonization/extinction"""
    print("\n" + "="*80)
    print("STEP 2: Building Dynamic Occupancy Model")
    print("="*80)
    
    X_processed = data_dict['X_processed']
    y_binary = data_dict['y_binary']
    households = data_dict['households']
    household_data = data_dict['household_data']
    feature_names = data_dict['feature_names']
    
    n_households = len(household_data)
    n_features = len(feature_names)
    
    # Prepare time series data structure
    # For each household, get time-ordered data
    household_time_series = []
    time_indices = []
    
    for hh_idx, hh_data in enumerate(household_data):
        hh_y = hh_data['target_binary'].values
        hh_X = []
        for idx, row in hh_data.iterrows():
            # Find corresponding row in X_processed
            orig_idx = hh_data.index.get_loc(idx)
            if orig_idx < len(X_processed):
                hh_X.append(X_processed[orig_idx])
            else:
                # Fallback: use first available
                hh_X.append(X_processed[0])
        
        hh_X = np.array(hh_X)
        household_time_series.append({
            'y': hh_y,
            'X': hh_X,
            'n_times': len(hh_y)
        })
        time_indices.extend([i for i in range(len(hh_y))])
    
    max_time = max(hh['n_times'] for hh in household_time_series)
    
    print(f"🔧 Building Dynamic Occupancy Model...")
    print(f"  Households: {n_households}")
    print(f"  Features: {n_features}")
    print(f"  Max time periods: {max_time}")
    
    try:
        with pm.Model() as occupancy_model:
            # Fixed effects for initial occupancy
            beta_psi = pm.Normal('beta_psi', mu=0, sigma=0.5, shape=n_features)
            
            # Fixed effects for colonization
            beta_gamma = pm.Normal('beta_gamma', mu=0, sigma=0.5, shape=n_features)
            
            # Fixed effects for extinction
            beta_epsilon = pm.Normal('beta_epsilon', mu=0, sigma=0.5, shape=n_features)
            
            # Detection probability (can be fixed or estimated)
            p_det = pm.Beta('p_det', alpha=2, beta=2)  # Prior favoring p ~ 0.5
            
            # Latent occupancy states for each household-time
            # We'll model this using a loop over households
            z_list = []
            y_obs_list = []
            
            for hh_idx, hh_ts in enumerate(household_time_series):
                n_times = hh_ts['n_times']
                X_hh = hh_ts['X']
                y_hh = hh_ts['y']
                
                # Initial occupancy (t=1)
                psi_1 = pm.math.sigmoid(pm.math.dot(X_hh[0], beta_psi))
                z_1 = pm.Bernoulli(f'z_{hh_idx}_1', p=psi_1)
                z_list.append([z_1])
                
                # Subsequent time periods
                z_prev = z_1
                for t in range(1, n_times):
                    # Colonization/extinction probabilities
                    gamma_t = pm.math.sigmoid(pm.math.dot(X_hh[t], beta_gamma))
                    epsilon_t = pm.math.sigmoid(pm.math.dot(X_hh[t], beta_epsilon))
                    
                    # Persistence probability
                    phi_t = 1 - epsilon_t
                    
                    # State transition: if previously occupied, persist with prob phi_t
                    #                      if previously unoccupied, colonize with prob gamma_t
                    z_t = pm.Bernoulli(
                        f'z_{hh_idx}_{t+1}',
                        p=pm.math.switch(z_prev > 0.5, phi_t, gamma_t)
                    )
                    z_list[-1].append(z_t)
                    z_prev = z_t
                
                # Observations given latent state
                for t, y_t in enumerate(y_hh):
                    z_t = z_list[hh_idx][t]
                    y_obs = pm.Bernoulli(
                        f'y_{hh_idx}_{t+1}',
                        p=z_t * p_det,  # Only detect if occupied
                        observed=y_t
                    )
                    y_obs_list.append(y_obs)
            
            # Store reference to model
            occupancy_model.reference = {
                'beta_psi': beta_psi,
                'beta_gamma': beta_gamma,
                'beta_epsilon': beta_epsilon,
                'p_det': p_det
            }
        
        print("✅ Occupancy model built successfully!")
        return occupancy_model
    
    except Exception as e:
        print(f"❌ Error building occupancy model: {e}")
        traceback.print_exc()
        raise

def build_occupancy_model_vectorized(data_dict):
    """Build vectorized dynamic occupancy model (more efficient)"""
    print("\n" + "="*80)
    print("STEP 2: Building Dynamic Occupancy Model (Vectorized)")
    print("="*80)
    
    X_processed = data_dict['X_processed']
    y_binary = data_dict['y_binary']
    households = data_dict['households']
    weeks = data_dict['weeks']
    prev_state = data_dict['prev_state']
    is_first = data_dict['is_first']
    feature_names = data_dict['feature_names']
    
    n_obs = len(y_binary)
    n_features = len(feature_names)
    n_households = len(np.unique(households))
    n_weeks = len(np.unique(weeks))
    
    print(f"🔧 Building Vectorized Dynamic Occupancy Model...")
    print(f"  Observations: {n_obs}")
    print(f"  Households: {n_households}")
    print(f"  Weeks: {n_weeks}")
    print(f"  Features: {n_features}")
    print(f"  First observations: {is_first.sum()}")
    
    try:
        with pm.Model() as occupancy_model:
            # Fixed effects
            beta_psi = pm.Normal('beta_psi', mu=0, sigma=0.5, shape=n_features)
            beta_gamma = pm.Normal('beta_gamma', mu=0, sigma=0.5, shape=n_features)
            beta_epsilon = pm.Normal('beta_epsilon', mu=0, sigma=0.5, shape=n_features)

            # Random effects: household and week (calendar week)
            sigma_hh = pm.HalfNormal('sigma_hh_occ', sigma=0.5)
            sigma_week = pm.HalfNormal('sigma_week_occ', sigma=0.5)
            b_hh = pm.Normal('b_hh_occ', mu=0, sigma=sigma_hh, shape=n_households)
            b_week = pm.Normal('b_week_occ', mu=0, sigma=sigma_week, shape=n_weeks)
            
            # Detection probability (fixed at high value for weekly data, or estimated)
            # For weekly consumption data, we assume perfect detection (p=1)
            # But we can still estimate it for robustness
            p_det = pm.Beta('p_det', alpha=9, beta=1)  # Prior favoring high detection
            
            # Data containers
            X_d = pm.Data('X', X_processed)
            prev_state_d = pm.Data('prev_state', prev_state)
            is_first_d = pm.Data('is_first', is_first)
            hh_idx = pm.Data('hh_idx_occ', households)
            week_idx = pm.Data('week_idx_occ', weeks)
            
            # Linear predictors
            logit_psi = pm.math.dot(X_d, beta_psi) + b_hh[hh_idx] + b_week[week_idx]
            logit_gamma = pm.math.dot(X_d, beta_gamma) + b_hh[hh_idx] + b_week[week_idx]
            logit_epsilon = pm.math.dot(X_d, beta_epsilon) + b_hh[hh_idx] + b_week[week_idx]
            
            # Probabilities
            psi = pm.math.sigmoid(logit_psi)  # Initial occupancy
            gamma = pm.math.sigmoid(logit_gamma)  # Colonization
            epsilon = pm.math.sigmoid(logit_epsilon)  # Extinction
            phi = 1 - epsilon  # Persistence
            
            # Occupancy probability at each time
            # If first period: use psi (initial occupancy)
            # Else: if previously occupied, persist with prob phi, else colonize with prob gamma
            occup_prob_first = psi
            occup_prob_transition = prev_state_d * phi + (1 - prev_state_d) * gamma
            
            occup_prob = pm.math.switch(
                is_first_d,
                occup_prob_first,
                occup_prob_transition
            )
            
            # Observation probability (detection given occupancy)
            obs_prob = occup_prob * p_det
            
            # Likelihood
            y_obs = pm.Bernoulli('y_obs', p=obs_prob, observed=y_binary)
        
        print("✅ Vectorized occupancy model built successfully!")
        return occupancy_model
    
    except Exception as e:
        print(f"❌ Error building occupancy model: {e}")
        traceback.print_exc()
        raise

def fit_occupancy_model(model):
    """Fit occupancy model"""
    print("\n" + "="*80)
    print("STEP 3: Fitting Occupancy Model")
    print("="*80)
    
    try:
        # Check available cores
        if platform.system() == 'Darwin':
            n_cores = 1
            n_chains = 2
        else:
            try:
                import multiprocessing
                n_cores = 1
                n_chains = 2
            except:
                n_cores = 1
                n_chains = 2
        
        print(f"Using {n_chains} chains with {n_cores} core(s)")
        print("⏱️  Starting MCMC sampling...")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        with model:
            try:
                # Note: progressbar=False for better log file compatibility
                print("📊 Starting MCMC sampling...")
                print("   Note: Progress updates will be minimal during sampling.")
                print("   Check process status with: ps aux | grep python3 | grep occupancy")
                sys.stdout.flush()
                idata = pm.sample(
                    target_accept=0.95,
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
            except Exception as e:
                print(f"⚠️  Sampling error: {e}")
                print("Trying with reduced settings...")
                sys.stdout.flush()
                idata = pm.sample(
                    draws=500,
                    tune=500,
                    chains=2,
                    cores=1,
                    random_seed=42,
                    init='adapt_diag',  # Use adapt_diag initialization (jitter is automatic)
                    progressbar=False,  # Disable progressbar for better log file compatibility
                    compute_convergence_checks=True  # Enable convergence checks for output
                )
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Model sampling completed in {elapsed_time/60:.1f} minutes!")
        
        # Check convergence diagnostics
        print("\n" + "="*80)
        print("CONVERGENCE DIAGNOSTICS")
        print("="*80)
        try:
            summary = az.summary(idata)
            
            # Check R-hat values
            if 'r_hat' in summary.columns:
                rhat_vals = summary['r_hat'].dropna()
                max_rhat = rhat_vals.max()
                mean_rhat = rhat_vals.mean()
                n_high_rhat = (rhat_vals > 1.01).sum()
                
                print(f"\nR-hat Statistics:")
                print(f"  Mean R-hat: {mean_rhat:.4f}")
                print(f"  Max R-hat: {max_rhat:.4f}")
                print(f"  Parameters with R-hat > 1.01: {n_high_rhat} / {len(rhat_vals)}")
                
                if max_rhat <= 1.01:
                    print(f"  ✅ Convergence achieved (R-hat <= 1.01)")
                elif max_rhat <= 1.05:
                    print(f"  ⚠️  Marginal convergence (R-hat <= 1.05, but > 1.01)")
                else:
                    print(f"  ❌ Poor convergence (R-hat > 1.05)")
            
            # Check ESS values
            if 'ess_bulk' in summary.columns:
                ess_bulk = summary['ess_bulk'].dropna()
                min_ess_bulk = ess_bulk.min()
                mean_ess_bulk = ess_bulk.mean()
                n_low_ess = (ess_bulk < 400).sum()
                
                print(f"\nESS (Effective Sample Size) Statistics:")
                print(f"  Mean Bulk ESS: {mean_ess_bulk:.0f}")
                print(f"  Min Bulk ESS: {min_ess_bulk:.0f}")
                print(f"  Parameters with ESS < 400: {n_low_ess} / {len(ess_bulk)}")
                
                if min_ess_bulk >= 400:
                    print(f"  ✅ Sufficient ESS (>= 400)")
                elif min_ess_bulk >= 200:
                    print(f"  ⚠️  Marginal ESS (200-400)")
                else:
                    print(f"  ❌ Low ESS (< 200)")
            
            if 'ess_tail' in summary.columns:
                ess_tail = summary['ess_tail'].dropna()
                min_ess_tail = ess_tail.min()
                print(f"  Min Tail ESS: {min_ess_tail:.0f}")
            
            # Overall convergence assessment
            if 'r_hat' in summary.columns and 'ess_bulk' in summary.columns:
                rhat_vals = summary['r_hat'].dropna()
                ess_bulk = summary['ess_bulk'].dropna()
                max_rhat = rhat_vals.max()
                min_ess = ess_bulk.min()
                
                if max_rhat <= 1.01 and min_ess >= 400:
                    print(f"\n✅ Model converged successfully!")
                    print(f"   All R-hat <= 1.01 and all ESS >= 400")
                elif max_rhat <= 1.05 and min_ess >= 200:
                    print(f"\n⚠️  Model shows marginal convergence")
                    print(f"   Consider running more iterations for better convergence")
                else:
                    print(f"\n❌ Model shows poor convergence")
                    print(f"   Consider:")
                    print(f"   - Running more iterations")
                    print(f"   - Checking for model specification issues")
                    print(f"   - Reviewing divergences (if any)")
        
        except Exception as e:
            print(f"⚠️  Could not compute convergence diagnostics: {e}")
            traceback.print_exc()
        
        # Save inference data
        try:
            idata_file = os.path.join(OCCUPANCY_DIR, 'occupancy_inference_data.nc')
            idata.to_netcdf(idata_file)
            print(f"\n✅ Inference data saved to {idata_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save inference data: {e}")
        
        return idata
    
    except Exception as e:
        print(f"❌ Error fitting occupancy model: {e}")
        traceback.print_exc()
        raise

# =============================================================================
# N-mixture Model
# =============================================================================

def prepare_nmixture_data(outcome_data, target_protein='Wild meat'):
    """Prepare data for n-mixture model"""
    print(f"\n🔄 Preparing data for n-mixture model (target: {target_protein})...")
    
    protein_map = {
        'Wild meat': ('wild_mass', 'wild_binary'),
        'Fish': ('fish_mass', 'fish_binary'),
        'Domestic meat': ('domestic_mass', 'domestic_binary'),
        'Invertebrate': ('invertebrate_mass', 'invertebrate_binary')
    }
    
    target_col, target_binary_col = protein_map[target_protein]
    
    # Get other protein types (masses, not binary)
    other_proteins = {
        'fish': 'fish_mass',
        'domestic': 'domestic_mass',
        'invertebrate': 'invertebrate_mass'
    }
    
    if target_protein == 'Fish':
        other_proteins.pop('fish', None)
    elif target_protein == 'Domestic meat':
        other_proteins.pop('domestic', None)
    elif target_protein == 'Invertebrate':
        other_proteins.pop('invertebrate', None)
    
    # Prepare data
    model_data = outcome_data.copy()
    model_data['target_mass'] = model_data[target_col]
    
    # Get other protein masses
    other_mass_cols = [col for col in other_proteins.values() if col in model_data.columns]
    
    # Prepare features
    # NOTE: festivity is categorical (binary) and should not be scaled
    categorical_vars = ['season', 'village', 'sex', 'edu', 'festivity']  # Include festivity as categorical
    continuous_vars = ['ame', 'count.hunters', 'record.days', 'non.hunt.income']
    
    X = model_data[categorical_vars + continuous_vars + other_mass_cols].copy()
    
    # Log-transform other protein masses (with small epsilon)
    for col in other_mass_cols:
        X[f'log_{col}'] = np.log(X[col] + 1e-6)
    
    log_other_cols = [f'log_{col}' for col in other_mass_cols]
    
    # One-hot encode categorical (includes festivity as binary indicator)
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    categorical_encoded = encoder.fit_transform(X[categorical_vars])
    categorical_feature_names = encoder.get_feature_names_out(categorical_vars)
    
    # Scale continuous + log other protein masses (festivity is already encoded, so not included here)
    scaler = StandardScaler()
    continuous_plus_other = X[continuous_vars + log_other_cols]
    continuous_scaled = scaler.fit_transform(continuous_plus_other)
    
    # Combine
    X_processed = np.hstack([categorical_encoded, continuous_scaled])
    feature_names = list(categorical_feature_names) + continuous_vars + log_other_cols
    
    # Get mass outcome (log-transformed)
    y_mass = model_data['target_mass'].values
    y_log = np.log(y_mass + 1e-6)
    
    # Get household indices
    households = model_data['household_ID'].astype('category').cat.codes.values
    # Get calendar week indices via week_id
    weeks = model_data['week_id'].astype('category').cat.codes.values
    
    print(f"✅ Prepared data for n-mixture model:")
    print(f"  Observations: {len(y_log)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Households: {len(np.unique(households))}")
    
    return {
        'X_processed': X_processed,
        'y_log': y_log,
        'y_mass': y_mass,
        'households': households,
        'weeks': weeks,
        'feature_names': feature_names,
        'encoder': encoder,
        'scaler': scaler,
        'target_protein': target_protein
    }

def build_nmixture_model(data_dict):
    """Build n-mixture model for continuous mass data"""
    print("\n" + "="*80)
    print("STEP 4: Building N-mixture Model")
    print("="*80)
    
    X_processed = data_dict['X_processed']
    y_log = data_dict['y_log']
    y_mass = data_dict['y_mass']
    households = data_dict['households']
    weeks = data_dict['weeks']
    feature_names = data_dict['feature_names']
    
    n_obs = len(y_log)
    n_features = len(feature_names)
    n_households = len(np.unique(households))
    n_weeks = len(np.unique(weeks))
    
    print(f"🔧 Building N-mixture Model...")
    print(f"  Observations: {n_obs}")
    print(f"  Features: {n_features}")
    print(f"  Households: {n_households}")
    print(f"  Weeks: {n_weeks}")
    
    try:
        with pm.Model() as nmixture_model:
            # Fixed effects
            beta = pm.Normal('beta', mu=0, sigma=0.5, shape=n_features)
            
            # Random intercepts: household and week
            sigma_hh = pm.HalfNormal('sigma_hh', sigma=0.5)
            sigma_week = pm.HalfNormal('sigma_week', sigma=0.5)
            b_hh = pm.Normal('b_hh', mu=0, sigma=sigma_hh, shape=n_households)
            b_week = pm.Normal('b_week', mu=0, sigma=sigma_week, shape=n_weeks)
            
            # Data containers
            X_d = pm.Data('X', X_processed)
            hh_idx = pm.Data('hh_idx', households)
            week_idx = pm.Data('week_idx', weeks)
            y_d = pm.Data('y_log', y_log)
            
            # Mean structure
            mu_fixed = pm.math.dot(X_d, beta)
            mu_hh = b_hh[hh_idx]
            mu_week = b_week[week_idx]
            mu = pm.Deterministic('mu', mu_fixed + mu_hh + mu_week)
            
            # Residual variance
            sigma = pm.HalfNormal('sigma', sigma=0.5)
            
            # Likelihood (log-normal for mass data)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_d)
        
        print("✅ N-mixture model built successfully!")
        return nmixture_model
    
    except Exception as e:
        print(f"❌ Error building n-mixture model: {e}")
        traceback.print_exc()
        raise

def fit_nmixture_model(model):
    """Fit n-mixture model"""
    print("\n" + "="*80)
    print("STEP 5: Fitting N-mixture Model")
    print("="*80)
    
    try:
        if platform.system() == 'Darwin':
            n_cores = 1
            n_chains = 2
        else:
            try:
                import multiprocessing
                n_cores = 1
                n_chains = 2
            except:
                n_cores = 1
                n_chains = 2
        
        print(f"Using {n_chains} chains with {n_cores} core(s)")
        print("⏱️  Starting MCMC sampling...")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        with model:
            try:
                # Note: progressbar=False for better log file compatibility
                print("📊 Starting MCMC sampling...")
                print("   Note: Progress updates will be minimal during sampling.")
                print("   Check process status with: ps aux | grep python3 | grep occupancy")
                sys.stdout.flush()
                idata = pm.sample(
                    target_accept=0.95,
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
            except Exception as e:
                print(f"⚠️  Sampling error: {e}")
                print("Trying with reduced settings...")
                sys.stdout.flush()
                idata = pm.sample(
                    draws=500,
                    tune=500,
                    chains=2,
                    cores=1,
                    random_seed=42,
                    init='adapt_diag',  # Use adapt_diag initialization (jitter is automatic)
                    progressbar=False,  # Disable progressbar for better log file compatibility
                    compute_convergence_checks=True  # Enable convergence checks for output
                )
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Model sampling completed in {elapsed_time/60:.1f} minutes!")
        
        # Check convergence diagnostics
        print("\n" + "="*80)
        print("CONVERGENCE DIAGNOSTICS")
        print("="*80)
        try:
            summary = az.summary(idata)
            
            # Check R-hat values
            if 'r_hat' in summary.columns:
                rhat_vals = summary['r_hat'].dropna()
                max_rhat = rhat_vals.max()
                mean_rhat = rhat_vals.mean()
                n_high_rhat = (rhat_vals > 1.01).sum()
                
                print(f"\nR-hat Statistics:")
                print(f"  Mean R-hat: {mean_rhat:.4f}")
                print(f"  Max R-hat: {max_rhat:.4f}")
                print(f"  Parameters with R-hat > 1.01: {n_high_rhat} / {len(rhat_vals)}")
                
                if max_rhat <= 1.01:
                    print(f"  ✅ Convergence achieved (R-hat <= 1.01)")
                elif max_rhat <= 1.05:
                    print(f"  ⚠️  Marginal convergence (R-hat <= 1.05, but > 1.01)")
                else:
                    print(f"  ❌ Poor convergence (R-hat > 1.05)")
            
            # Check ESS values
            if 'ess_bulk' in summary.columns:
                ess_bulk = summary['ess_bulk'].dropna()
                min_ess_bulk = ess_bulk.min()
                mean_ess_bulk = ess_bulk.mean()
                n_low_ess = (ess_bulk < 400).sum()
                
                print(f"\nESS (Effective Sample Size) Statistics:")
                print(f"  Mean Bulk ESS: {mean_ess_bulk:.0f}")
                print(f"  Min Bulk ESS: {min_ess_bulk:.0f}")
                print(f"  Parameters with ESS < 400: {n_low_ess} / {len(ess_bulk)}")
                
                if min_ess_bulk >= 400:
                    print(f"  ✅ Sufficient ESS (>= 400)")
                elif min_ess_bulk >= 200:
                    print(f"  ⚠️  Marginal ESS (200-400)")
                else:
                    print(f"  ❌ Low ESS (< 200)")
            
            if 'ess_tail' in summary.columns:
                ess_tail = summary['ess_tail'].dropna()
                min_ess_tail = ess_tail.min()
                print(f"  Min Tail ESS: {min_ess_tail:.0f}")
            
            # Overall convergence assessment
            if 'r_hat' in summary.columns and 'ess_bulk' in summary.columns:
                rhat_vals = summary['r_hat'].dropna()
                ess_bulk = summary['ess_bulk'].dropna()
                max_rhat = rhat_vals.max()
                min_ess = ess_bulk.min()
                
                if max_rhat <= 1.01 and min_ess >= 400:
                    print(f"\n✅ Model converged successfully!")
                    print(f"   All R-hat <= 1.01 and all ESS >= 400")
                elif max_rhat <= 1.05 and min_ess >= 200:
                    print(f"\n⚠️  Model shows marginal convergence")
                    print(f"   Consider running more iterations for better convergence")
                else:
                    print(f"\n❌ Model shows poor convergence")
                    print(f"   Consider:")
                    print(f"   - Running more iterations")
                    print(f"   - Checking for model specification issues")
                    print(f"   - Reviewing divergences (if any)")
        
        except Exception as e:
            print(f"⚠️  Could not compute convergence diagnostics: {e}")
            traceback.print_exc()
        
        # Save inference data
        try:
            idata_file = os.path.join(NMIXTURE_DIR, 'nmixture_inference_data.nc')
            idata.to_netcdf(idata_file)
            print(f"\n✅ Inference data saved to {idata_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save inference data: {e}")
        
        return idata
    
    except Exception as e:
        print(f"❌ Error fitting n-mixture model: {e}")
        traceback.print_exc()
        raise

# =============================================================================
# Interpretation and Visualization Functions
# =============================================================================

def extract_protein_coefficients(idata, feature_names, target_protein='Wild meat', model_type='nmixture'):
    """Extract coefficients related to other protein types"""
    print(f"\n{'='*80}")
    print(f"EXTRACTING PROTEIN RELATIONSHIP COEFFICIENTS for {target_protein}")
    print(f"{'='*80}")
    
    # Get full posterior summary
    full_summary = az.summary(idata)
    
    # For occupancy models, check multiple beta parameters (beta_psi, beta_gamma, beta_epsilon)
    # For n-mixture models, use beta
    if model_type == 'occupancy':
        beta_params = ['beta_psi', 'beta_gamma', 'beta_epsilon']
        summary_parts = []
        for param in beta_params:
            try:
                param_summary = az.summary(idata, var_names=[param])
                if len(param_summary) > 0:
                    summary_parts.append(param_summary)
            except:
                pass
        if summary_parts:
            summary = pd.concat(summary_parts)
        else:
            summary = full_summary
    else:
        # n-mixture model uses 'beta'
        try:
            summary = az.summary(idata, var_names=['beta'])
        except:
            summary = full_summary
    
    # Identify protein-related features
    protein_features = {}
    other_proteins = ['fish', 'domestic', 'invertebrate']
    
    for protein in other_proteins:
        # Look for features containing this protein name
        matching_features = [f for f in feature_names if protein in f.lower()]
        protein_features[protein] = matching_features
    
    # Extract coefficients by matching feature names to their indices
    # First, create a mapping of feature names to their indices
    feature_to_idx = {feat: idx for idx, feat in enumerate(feature_names)}
    
    # Extract coefficients
    protein_coefs = {}
    
    for protein, features in protein_features.items():
        if features:
            protein_coefs[protein] = {}
            for feat in features:
                if feat not in feature_to_idx:
                    continue
                    
                feat_idx = feature_to_idx[feat]
                found = False
                row_idx = None
                
                # Try to find coefficient by matching index in parameter names
                # Parameters are named like "beta[0]", "beta_psi[5]", etc.
                for idx in summary.index:
                    idx_str = str(idx)
                    # Extract index from parameter name (e.g., "beta[5]" -> 5, "beta_psi[3]" -> 3)
                    match = re.search(r'\[(\d+)\]', idx_str)
                    if match:
                        param_idx = int(match.group(1))
                        # Match feature index to coefficient index
                        if param_idx == feat_idx:
                            row_idx = idx
                            found = True
                            break
                
                # If not found by index, try to match by feature name in parameter name
                if not found:
                    # Try exact feature name matching
                    feat_name_clean = feat.lower().replace('_', '').replace(' ', '')
                    for idx in summary.index:
                        idx_str = str(idx).lower().replace('_', '').replace(' ', '')
                        if feat_name_clean in idx_str or any(part in idx_str for part in feat_name_clean.split('_')):
                            row_idx = idx
                            found = True
                            break
                
                if found and row_idx is not None and row_idx in summary.index:
                    # Get HDI columns (handle different versions)
                    hdi_low_col = 'hdi_3%' if 'hdi_3%' in summary.columns else 'hdi_2.5%'
                    hdi_high_col = 'hdi_97%' if 'hdi_97%' in summary.columns else 'hdi_97.5%'
                    
                    try:
                        protein_coefs[protein][feat] = {
                            'mean': summary.loc[row_idx, 'mean'],
                            'sd': summary.loc[row_idx, 'sd'],
                            'hdi_3%': summary.loc[row_idx, hdi_low_col],
                            'hdi_97%': summary.loc[row_idx, hdi_high_col]
                        }
                        print(f"  ✅ Found coefficient for {feat}: beta[{feat_idx}] = {summary.loc[row_idx, 'mean']:.4f}")
                    except Exception as e:
                        print(f"  ⚠️  Could not extract coefficient for {feat}: {e}")
                else:
                    print(f"  ⚠️  Could not find coefficient for feature: {feat} (index {feat_idx})")
    
    return protein_coefs, full_summary

def convert_logit_to_prob(logit):
    """Convert logit (log-odds) to probability"""
    return np.exp(logit) / (1 + np.exp(logit))

def print_protein_relationships(protein_coefs, model_type='occupancy'):
    """Print interpretable summary of protein relationships"""
    print(f"\n{'='*80}")
    print(f"PROTEIN RELATIONSHIP SUMMARY ({model_type.upper()} MODEL)")
    print(f"{'='*80}")
    
    # For occupancy models, convert log-odds to probabilities
    # For n-mixture models, show log-scale effects (these are on log scale already)
    
    for protein, coefs in protein_coefs.items():
        if coefs:
            print(f"\n{protein.upper()} effects on Wild meat consumption:")
            print("-" * 80)
            for feat_name, coef_info in coefs.items():
                mean = coef_info['mean']
                sd = coef_info['sd']
                hdi_low = coef_info['hdi_3%']
                hdi_high = coef_info['hdi_97%']
                
                # Determine direction and significance
                if hdi_low > 0:
                    direction = "POSITIVE"
                    significance = "✅ Significant positive effect"
                elif hdi_high < 0:
                    direction = "NEGATIVE"
                    significance = "✅ Significant negative effect"
                else:
                    direction = "UNCERTAIN"
                    significance = "⚠️  Effect includes zero (not significant)"
                
                print(f"  Feature: {feat_name}")
                print(f"    Coefficient (mean): {mean:.4f} ± {sd:.4f}")
                print(f"    95% HDI: [{hdi_low:.4f}, {hdi_high:.4f}]")
                print(f"    Direction: {direction}")
                print(f"    {significance}")
                
                # For occupancy models, convert log-odds to probabilities
                if model_type == 'occupancy':
                    print(f"\n    📊 PROBABILITY INTERPRETATION:")
                    
                    # Baseline: no other protein consumption (all other covariates at 0)
                    # For interpretation, assume baseline logit = 0 (50% probability)
                    # In practice, this would depend on other covariates, but this gives a reference
                    baseline_logit = 0.0
                    baseline_prob = convert_logit_to_prob(baseline_logit)
                    
                    # With this protein: baseline_logit + coefficient
                    with_protein_logit = baseline_logit + mean
                    with_protein_prob = convert_logit_to_prob(with_protein_logit)
                    
                    # Probability change
                    prob_change = with_protein_prob - baseline_prob
                    prob_change_pct = (prob_change / baseline_prob) * 100
                    
                    print(f"      Baseline probability (no {protein}): {baseline_prob:.1%} (logit = {baseline_logit:.2f})")
                    print(f"      With {protein} consumption: {with_protein_prob:.1%} (logit = {with_protein_logit:.2f})")
                    print(f"      Change: {prob_change:+.1%} ({prob_change_pct:+.1f}% relative to baseline)")
                    
                    # Also show HDI bounds as probabilities
                    with_protein_prob_low = convert_logit_to_prob(baseline_logit + hdi_low)
                    with_protein_prob_high = convert_logit_to_prob(baseline_logit + hdi_high)
                    print(f"      95% HDI probability range: [{with_protein_prob_low:.1%}, {with_protein_prob_high:.1%}]")
                    
                    # Interpretation summary
                    if mean < 0:
                        print(f"      💡 Interpretation: Consuming {protein} REDUCES probability of Wild meat consumption")
                        print(f"         by approximately {abs(prob_change):.1%} (from {baseline_prob:.1%} to {with_protein_prob:.1%})")
                    elif mean > 0:
                        print(f"      💡 Interpretation: Consuming {protein} INCREASES probability of Wild meat consumption")
                        print(f"         by approximately {abs(prob_change):.1%} (from {baseline_prob:.1%} to {with_protein_prob:.1%})")
                
                elif model_type == 'n-mixture' or model_type == 'nmixture':
                    print(f"\n    📊 INTERPRETATION (Log-Scale Model):")
                    print(f"      Coefficient is on log-scale (log mass)")
                    print(f"      A 1-unit increase in log({protein} mass) changes log(Wild meat mass) by {mean:.4f}")
                    
                    # For log-scale models, interpret multiplicative effect
                    if mean < 0:
                        # Example: if log other protein increases by 0.693 (doubles), wild meat decreases by exp(mean * 0.693)
                        example_log_increase = 0.693  # log(2) ≈ 0.693 (doubling)
                        multiplicative_effect = np.exp(mean * example_log_increase)
                        print(f"      Example: If {protein} consumption doubles (log +0.693),")
                        print(f"                Wild meat consumption changes by factor of {multiplicative_effect:.4f}")
                        print(f"                (i.e., Wild meat ≈ {multiplicative_effect*100:.1f}% of original)")
                        print(f"      💡 Interpretation: Higher {protein} consumption is associated with LOWER Wild meat consumption")
                    elif mean > 0:
                        example_log_increase = 0.693
                        multiplicative_effect = np.exp(mean * example_log_increase)
                        print(f"      Example: If {protein} consumption doubles (log +0.693),")
                        print(f"                Wild meat consumption changes by factor of {multiplicative_effect:.4f}")
                        print(f"                (i.e., Wild meat ≈ {multiplicative_effect*100:.1f}% of original)")
                        print(f"      💡 Interpretation: Higher {protein} consumption is associated with HIGHER Wild meat consumption")
                
                print()

def plot_protein_relationships(idata, feature_names, target_protein, model_type, output_dir):
    """Create visualizations of protein relationships"""
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS for {target_protein} ({model_type.upper()} MODEL)")
    print(f"{'='*80}")
    
    # Get posterior samples
    posterior = idata.posterior
    
    # For occupancy models, we need to handle beta_psi, beta_gamma, beta_epsilon
    # For n-mixture models, we use beta
    if model_type == 'occupancy':
        beta_params = ['beta_psi', 'beta_gamma', 'beta_epsilon']
        # Use beta_psi for visualization (initial occupancy)
        beta_param_to_use = 'beta_psi'
    else:
        beta_params = ['beta']
        beta_param_to_use = 'beta'
    
    # Check if parameter exists
    if beta_param_to_use not in posterior:
        print(f"⚠️  Parameter '{beta_param_to_use}' not found in posterior. Available: {list(posterior.keys())}")
        return
    
    # Identify protein-related feature indices
    protein_indices = {}
    other_proteins = ['fish', 'domestic', 'invertebrate']
    
    for protein in other_proteins:
        indices = [i for i, f in enumerate(feature_names) if protein in f.lower()]
        if indices:
            protein_indices[protein] = indices
    
    if not protein_indices:
        print("⚠️  No protein-related features found for visualization")
        return
    
    # Create figure with subplots
    n_proteins = len(protein_indices)
    if n_proteins > 0:
        fig, axes = plt.subplots(n_proteins, 2, figsize=(16, 5*n_proteins))
        if n_proteins == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        for protein, indices in protein_indices.items():
            for idx in indices:
                feat_name = feature_names[idx]
                
                try:
                    # Extract posterior samples for this coefficient
                    # Dimension names differ: beta_psi_dim_0 for occupancy, beta_dim_0 for n-mixture
                    if model_type == 'occupancy':
                        # For occupancy, we'll show beta_psi (initial occupancy) as primary
                        # Dimension name is beta_psi_dim_0, not coord_dim_0
                        dim_name = f'{beta_param_to_use}_dim_0'
                        beta_samples = posterior[beta_param_to_use].sel({dim_name: idx}).values.flatten()
                        chain_samples = posterior[beta_param_to_use].isel(chain=0).sel({dim_name: idx}).values
                    else:
                        # For n-mixture, dimension name is beta_dim_0
                        dim_name = f'{beta_param_to_use}_dim_0'
                        beta_samples = posterior[beta_param_to_use].sel({dim_name: idx}).values.flatten()
                        chain_samples = posterior[beta_param_to_use].isel(chain=0).sel({dim_name: idx}).values
                    
                    # Plot 1: Posterior distribution
                    ax1 = axes[plot_idx, 0]
                    ax1.hist(beta_samples, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
                    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
                    ax1.axvline(np.mean(beta_samples), color='green', linestyle='-', linewidth=2, label='Mean')
                    ax1.set_xlabel('Coefficient Value')
                    ax1.set_ylabel('Density')
                    param_label = 'beta_psi' if model_type == 'occupancy' else 'beta'
                    ax1.set_title(f'{protein.upper()} Effect: {feat_name}\n{param_label} - Posterior Distribution')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot 2: Trace plot (for first chain)
                    ax2 = axes[plot_idx, 1]
                    ax2.plot(chain_samples, alpha=0.7, color='steelblue')
                    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
                    ax2.axhline(np.mean(beta_samples), color='green', linestyle='-', linewidth=1)
                    ax2.set_xlabel('Iteration')
                    ax2.set_ylabel('Coefficient Value')
                    ax2.set_title(f'{protein.upper()} Effect: {feat_name}\n{param_label} - Trace Plot (Chain 0)')
                    ax2.grid(True, alpha=0.3)
                    
                    plot_idx += 1
                    if plot_idx >= n_proteins * 2:
                        break
                except Exception as e:
                    print(f"  ⚠️  Could not plot {feat_name}: {e}")
                    continue
        
        if plot_idx > 0:
            plt.tight_layout()
            plot_file = os.path.join(output_dir, f'{model_type}_protein_relationships_{target_protein.replace(" ", "_")}.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"✅ Relationship plots saved to {plot_file}")
            plt.close()
        else:
            print("⚠️  No plots could be created")
            plt.close()

def create_coefficient_comparison_plot(protein_coefs, target_protein, model_type, output_dir):
    """Create a bar chart comparing effects of different proteins"""
    if not protein_coefs:
        return
    
    # Prepare data for plotting
    proteins = []
    means = []
    hdi_lows = []
    hdi_highs = []
    labels = []
    
    for protein, coefs in protein_coefs.items():
        if coefs:
            for feat_name, coef_info in coefs.items():
                proteins.append(protein.title())
                means.append(coef_info['mean'])
                hdi_lows.append(coef_info['hdi_3%'])
                hdi_highs.append(coef_info['hdi_97%'])
                # Clean up feature name for label
                clean_name = feat_name.replace('_', ' ').replace('log ', '').replace('binary', '').strip()
                labels.append(f"{protein.title()}\n({clean_name})")
    
    if not means:
        return
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(labels))
    colors = ['steelblue' if m > 0 else 'coral' for m in means]
    
    # Plot bars
    bars = ax.bar(x_pos, means, yerr=[np.array(means) - np.array(hdi_lows), 
                                     np.array(hdi_highs) - np.array(means)],
                 alpha=0.7, color=colors, edgecolor='black', capsize=5)
    
    # Add zero line
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
    
    # Customize
    ax.set_xlabel('Protein Type and Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Effect of Other Proteins on {target_protein} Consumption\n({model_type.upper()} Model)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.3f}', ha='center', va='bottom' if height > 0 else 'top',
               fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f'{model_type}_coefficient_comparison_{target_protein.replace(" ", "_")}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Coefficient comparison plot saved to {plot_file}")
    plt.close()

def save_detailed_coefficient_table(protein_coefs, summary, target_protein, model_type, output_dir):
    """Save detailed coefficient table to CSV"""
    rows = []
    for protein, coefs in protein_coefs.items():
        if coefs:
            for feat_name, coef_info in coefs.items():
                rows.append({
                    'model_type': model_type,
                    'target_protein': target_protein,
                    'other_protein': protein,
                    'feature_name': feat_name,
                    'coefficient_mean': coef_info['mean'],
                    'coefficient_sd': coef_info['sd'],
                    'hdi_lower': coef_info['hdi_3%'],
                    'hdi_upper': coef_info['hdi_97%'],
                    'effect_direction': 'positive' if coef_info['hdi_3%'] > 0 else ('negative' if coef_info['hdi_97%'] < 0 else 'uncertain'),
                    'is_significant': coef_info['hdi_3%'] > 0 or coef_info['hdi_97%'] < 0
                })
    
    if rows:
        df_coefs = pd.DataFrame(rows)
        csv_file = os.path.join(output_dir, f'{model_type}_protein_coefficients_{target_protein.replace(" ", "_")}.csv')
        df_coefs.to_csv(csv_file, index=False)
        print(f"✅ Detailed coefficient table saved to {csv_file}")
        return df_coefs
    return None

def save_full_coefficient_table(idata, feature_names, target_protein, model_type, output_dir):
    """Save full coefficient table with feature names mapped to beta indices.
    Includes all fixed effects (excluding random effects) with descriptive statistics.
    """
    print(f"\n{'='*80}")
    print(f"SAVING FULL COEFFICIENT TABLE ({model_type.upper()} MODEL)")
    print(f"{'='*80}")
    
    try:
        # Get summary for all beta parameters
        if model_type == 'occupancy':
            # For occupancy models, we have beta_psi, beta_gamma, beta_epsilon
            beta_params = ['beta_psi', 'beta_gamma', 'beta_epsilon']
            param_labels = ['Initial Occupancy (psi)', 'Colonization (gamma)', 'Extinction (epsilon)']
        else:
            # For n-mixture models (handles both 'nmixture' and 'n-mixture')
            beta_params = ['beta']
            param_labels = ['Fixed Effects']
        
        all_coef_rows = []
        
        for param, param_label in zip(beta_params, param_labels):
            try:
                param_summary = az.summary(idata, var_names=[param])
                
                if len(param_summary) > 0:
                    # Map indices to feature names
                    for idx, row in param_summary.iterrows():
                        # Extract index from parameter name (e.g., "beta_psi[0]" -> 0)
                        match = re.search(r'\[(\d+)\]', str(idx))
                        if match:
                            feat_idx = int(match.group(1))
                            # Get feature name
                            if feat_idx < len(feature_names):
                                feature_name = feature_names[feat_idx]
                            else:
                                feature_name = f"Feature_{feat_idx}"
                            
                            all_coef_rows.append({
                                'parameter_type': param,
                                'parameter_label': param_label,
                                'parameter_index': feat_idx,
                                'feature_name': feature_name,
                                'mean': row['mean'],
                                'sd': row['sd'],
                                'hdi_3%': row['hdi_3%'] if 'hdi_3%' in row.index else row['hdi_2.5%'],
                                'hdi_97%': row['hdi_97%'] if 'hdi_97%' in row.index else row['hdi_97.5%'],
                                'mcse_mean': row.get('mcse_mean', np.nan),
                                'mcse_sd': row.get('mcse_sd', np.nan),
                                'ess_bulk': row.get('ess_bulk', np.nan),
                                'ess_tail': row.get('ess_tail', np.nan),
                                'r_hat': row.get('r_hat', np.nan),
                                'target_protein': target_protein,
                                'model_type': model_type
                            })
            except Exception as e:
                print(f"  ⚠️  Could not extract {param}: {e}")
                continue
        
        if all_coef_rows:
            df_full_coefs = pd.DataFrame(all_coef_rows)
            
            # Sort by parameter type and index
            df_full_coefs = df_full_coefs.sort_values(['parameter_type', 'parameter_index'])
            
            # Save to CSV
            csv_file = os.path.join(output_dir, f'{model_type}_full_coefficients_{target_protein.replace(" ", "_")}.csv')
            df_full_coefs.to_csv(csv_file, index=False)
            print(f"✅ Full coefficient table saved to {csv_file}")
            print(f"   Total coefficients: {len(df_full_coefs)}")
            print(f"   Parameter types: {', '.join(df_full_coefs['parameter_type'].unique())}")
            
            # Print summary statistics
            print(f"\n📊 Summary Statistics:")
            print(f"   Mean |coefficient|: {df_full_coefs['mean'].abs().mean():.4f}")
            print(f"   Max |coefficient|: {df_full_coefs['mean'].abs().max():.4f}")
            print(f"   Significant coefficients (HDI excludes 0): {(df_full_coefs['hdi_3%'] > 0).sum() + (df_full_coefs['hdi_97%'] < 0).sum()} / {len(df_full_coefs)}")
            
            return df_full_coefs
        else:
            print(f"  ⚠️  No coefficients found to save")
            return None
    
    except Exception as e:
        print(f"  ❌ Error saving full coefficient table: {e}")
        traceback.print_exc()
        return None

# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("OCCUPANCY AND N-MIXTURE MODELS FOR PROTEIN CONSUMPTION")
    print("="*80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load and preprocess data
        outcome_data = load_and_preprocess_data()
        
        # Run models for each target protein
        # NOTE: Currently only analyzing Wild meat (commented out others to reduce complexity)
        target_proteins = ['Wild meat']  # Only analyze Wild meat
        # target_proteins = ['Wild meat', 'Fish', 'Domestic meat', 'Invertebrate']  # Full analysis
        
        for target_protein in target_proteins:
            print("\n" + "="*80)
            print(f"ANALYZING TARGET PROTEIN: {target_protein}")
            print("="*80)
            
            # 1. Dynamic Occupancy Model
            try:
                print("\n" + "-"*80)
                print("DYNAMIC OCCUPANCY MODEL")
                print("-"*80)
                occ_data = prepare_occupancy_data(outcome_data, target_protein)
                occ_model = build_occupancy_model_vectorized(occ_data)
                occ_idata = fit_occupancy_model(occ_model)
                
                # Save summary
                try:
                    summary = az.summary(occ_idata)
                    summary_file = os.path.join(OCCUPANCY_DIR, f'occupancy_summary_{target_protein.replace(" ", "_")}.csv')
                    summary.to_csv(summary_file)
                    print(f"✅ Summary saved to {summary_file}")
                except Exception as e:
                    print(f"⚠️  Could not save summary: {e}")
                
                # Extract and visualize protein relationships
                try:
                    print("\n" + "="*80)
                    print("INTERPRETING PROTEIN RELATIONSHIPS (OCCUPANCY MODEL)")
                    print("="*80)
                    
                    # Extract protein-related coefficients
                    protein_coefs, full_summary = extract_protein_coefficients(
                        occ_idata, occ_data['feature_names'], target_protein, model_type='occupancy'
                    )
                    
                    # Print interpretable summary
                    if protein_coefs:
                        print_protein_relationships(protein_coefs, 'occupancy')
                        
                        # Create visualizations
                        plot_protein_relationships(
                            occ_idata, occ_data['feature_names'], target_protein, 
                            'occupancy', OCCUPANCY_DIR
                        )
                        create_coefficient_comparison_plot(
                            protein_coefs, target_protein, 'occupancy', OCCUPANCY_DIR
                        )
                        save_detailed_coefficient_table(
                            protein_coefs, full_summary, target_protein, 'occupancy', OCCUPANCY_DIR
                        )
                    else:
                        print("⚠️  No protein-related features found in model")
                    
                    # Save full coefficient table with feature names
                    save_full_coefficient_table(
                        occ_idata, occ_data['feature_names'], target_protein, 'occupancy', OCCUPANCY_DIR
                    )
                
                except Exception as e:
                    print(f"⚠️  Could not extract protein relationships: {e}")
                    traceback.print_exc()
            
            except Exception as e:
                print(f"❌ Error in occupancy model for {target_protein}: {e}")
                traceback.print_exc()
            
            # 2. N-mixture Model
            try:
                print("\n" + "-"*80)
                print("N-MIXTURE MODEL")
                print("-"*80)
                nmixture_data = prepare_nmixture_data(outcome_data, target_protein)
                nmixture_model = build_nmixture_model(nmixture_data)
                nmixture_idata = fit_nmixture_model(nmixture_model)
                
                # Save summary
                try:
                    summary = az.summary(nmixture_idata)
                    summary_file = os.path.join(NMIXTURE_DIR, f'nmixture_summary_{target_protein.replace(" ", "_")}.csv')
                    summary.to_csv(summary_file)
                    print(f"✅ Summary saved to {summary_file}")
                except Exception as e:
                    print(f"⚠️  Could not save summary: {e}")
                
                # Extract and visualize protein relationships
                try:
                    print("\n" + "="*80)
                    print("INTERPRETING PROTEIN RELATIONSHIPS (N-MIXTURE MODEL)")
                    print("="*80)
                    
                    # Extract protein-related coefficients
                    protein_coefs, full_summary = extract_protein_coefficients(
                        nmixture_idata, nmixture_data['feature_names'], target_protein, model_type='nmixture'
                    )
                    
                    # Print interpretable summary
                    if protein_coefs:
                        print_protein_relationships(protein_coefs, 'n-mixture')
                        
                        # Create visualizations
                        plot_protein_relationships(
                            nmixture_idata, nmixture_data['feature_names'], target_protein, 
                            'n-mixture', NMIXTURE_DIR
                        )
                        create_coefficient_comparison_plot(
                            protein_coefs, target_protein, 'n-mixture', NMIXTURE_DIR
                        )
                        save_detailed_coefficient_table(
                            protein_coefs, full_summary, target_protein, 'n-mixture', NMIXTURE_DIR
                        )
                    else:
                        print("⚠️  No protein-related features found in model")
                    
                    # Save full coefficient table with feature names
                    save_full_coefficient_table(
                        nmixture_idata, nmixture_data['feature_names'], target_protein, 'nmixture', NMIXTURE_DIR
                    )
                
                except Exception as e:
                    print(f"⚠️  Could not extract protein relationships: {e}")
                    traceback.print_exc()
            
            except Exception as e:
                print(f"❌ Error in n-mixture model for {target_protein}: {e}")
                traceback.print_exc()
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOutputs saved to:")
        print(f"  - Occupancy models: {OCCUPANCY_DIR}")
        print(f"  - N-mixture models: {NMIXTURE_DIR}")
        
        print("\n" + "="*80)
        print("📊 INTERPRETATION SUMMARY")
        print("="*80)
        print(f"\nTo interpret the relationships between Wild meat and other proteins:")
        print(f"  1. Check coefficient CSV files:")
        print(f"     - {OCCUPANCY_DIR}/occupancy_protein_coefficients_Wild_meat.csv")
        print(f"     - {NMIXTURE_DIR}/nmixture_protein_coefficients_Wild_meat.csv")
        print(f"  2. View comparison plots:")
        print(f"     - {OCCUPANCY_DIR}/occupancy_coefficient_comparison_Wild_meat.png")
        print(f"     - {NMIXTURE_DIR}/nmixture_coefficient_comparison_Wild_meat.png")
        print(f"  3. View detailed relationship plots:")
        print(f"     - {OCCUPANCY_DIR}/occupancy_protein_relationships_Wild_meat.png")
        print(f"     - {NMIXTURE_DIR}/nmixture_protein_relationships_Wild_meat.png")
        print(f"\nKey interpretation:")
        print(f"  - Positive coefficients: Other protein consumption INCREASES Wild meat consumption")
        print(f"  - Negative coefficients: Other protein consumption DECREASES Wild meat consumption")
        print(f"  - Significant effect: 95% HDI does not include zero")
        print(f"  - Check the detailed coefficient tables for exact values and confidence intervals")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

