# =============================================================================
# Multivariate Protein Consumption Analysis in R
# =============================================================================
# Research Question: Modeling Protein Consumption with Temporal Autocorrelation
#
# This script implements the same analysis as the Python notebook using:
# - Dataset: protein_full_data.csv
# - Package: brms (Bayesian Regression Models using Stan)
# - Temporal autocorrelation: Lagged consumption variables
# - Updated model structure: Fixed effects + Random effects (household, week)
#
# Key Features:
# - Weekly aggregation using sums (not averages)
# - ISO calendar weeks for random effects
# - Festivity variable based on holidays
# - Multivariate model with correlated random effects across outcomes
# =============================================================================

# =============================================================================
# 1. Setup and Data Loading
# =============================================================================

# Set user library path (for packages installed to user directory)
user_lib <- file.path(Sys.getenv("HOME"), "R", "library")
if (!dir.exists(user_lib)) {
    dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(user_lib, .libPaths()))

# Load required libraries
# Use minimal packages instead of full tidyverse (much faster installation)
library(readr)      # for read_csv
library(dplyr)      # for filter, mutate, group_by, left_join, etc.
library(tidyr)      # for pivot operations if needed
library(lubridate)  # for date functions (year, month, etc.)
library(purrr)      # for map_dfr, map2, reduce functions
library(stringr)   # for str_split, str_detect, str_remove functions
library(brms)       # for Bayesian modeling
library(bayesplot)  # for diagnostics plots
library(posterior)  # for posterior samples
# Try to load cmdstanr (faster Stan backend), but fall back to rstan if not available
if (requireNamespace("cmdstanr", quietly = TRUE)) {
  library(cmdstanr)
  options(brms.backend = "cmdstanr")
  cat("Using cmdstanr backend (faster)\n")
} else {
  cat("cmdstanr not available, using rstan backend\n")
  options(brms.backend = "rstan")
}

# Set options for better performance
options(mc.cores = parallel::detectCores())

# Set random seed for reproducibility
set.seed(42)

cat("✅ Libraries loaded successfully!\n")
cat("Using brms version:", as.character(packageVersion("brms")), "\n")

# Set working directory to script directory
# This ensures the script finds protein_full_data.csv
# Get script directory - works in both RStudio and Rscript
get_script_dir <- function() {
  # Try to get script path from command line arguments
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg)
    return(dirname(normalizePath(script_path)))
  }
  # Fallback: use current working directory
  return(getwd())
}

script_dir <- get_script_dir()

# Try to find data file
data_file <- file.path(script_dir, "protein_full_data.csv")
if (!file.exists(data_file)) {
  # Try current working directory
  if (file.exists("protein_full_data.csv")) {
    script_dir <- getwd()
    data_file <- "protein_full_data.csv"
  } else {
    cat("⚠️  Error: protein_full_data.csv not found!\n")
    cat("Current working directory:", getwd(), "\n")
    cat("Script directory:", script_dir, "\n")
    stop("Cannot find protein_full_data.csv. Please check the file path.")
  }
}

# Set working directory to script directory
setwd(script_dir)
cat("Working directory:", getwd(), "\n")

# Create output directories
output_dir <- file.path(script_dir, "outputs")
models_dir <- file.path(output_dir, "models")
diagnostics_dir <- file.path(output_dir, "diagnostics")
plots_dir <- file.path(output_dir, "plots")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(models_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(diagnostics_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

cat("Output directories created:\n")
cat("  Models:", models_dir, "\n")
cat("  Diagnostics:", diagnostics_dir, "\n")
cat("  Plots:", plots_dir, "\n")

# Load the dataset
df <- read_csv(data_file, 
               col_types = cols(.default = "c",
                                hh.size2 = "i", ame = "d", wbi = "d", wbi.hh = "d",
                                age = "i", hh.size = "i", count.hunters = "i",
                                annual.income = "i", non.hunt.income = "i",
                                count.per.hh.per.date = "i", total.consumers = "i",
                                n.babies = "i", n.children = "i", n.youngmen = "i",
                                n.men = "i", n.oldmen = "i", n.nonpreg.women = "i",
                                n.oldwomen = "i", n.preg.women = "i",
                                n.breastfeeding.women = "i", protein.or.not.binary = "i",
                                X_id = "i", X_index_household = "i", X_index_food = "i",
                                income = "i", mass = "d", converted.mass = "d",
                                median.score = "d"))

cat("\nDataset loaded:\n")
cat("  Shape:", nrow(df), "observations,", ncol(df), "variables\n")
cat("  Columns:", paste(names(df), collapse = ", "), "\n")
cat("\nFirst few rows:\n")
print(head(df))

cat("\nMissing values:\n")
cat("  Total missing:", sum(is.na(df)), "\n")
cat("  Missing converted.mass:", sum(is.na(df$converted.mass)), "\n")

# =============================================================================
# 2. Data Preprocessing
# =============================================================================

cat("\n🔄 Starting data preprocessing...\n")

# Convert recall.date to date
df$recall.date <- as.Date(df$recall.date)

# Create year and month variables
df$year <- year(df$recall.date)
df$month <- month(df$recall.date)
df$month_num <- df$month

cat("✅ Date variables created\n")
cat("  Date range:", as.character(min(df$recall.date)), "to", as.character(max(df$recall.date)), "\n")
cat("  Years:", paste(sort(unique(df$year)), collapse = ", "), "\n")
cat("  Months:", paste(sort(unique(df$month)), collapse = ", "), "\n")

# Drop rows where converted.mass is NA
cat("\n🔄 Dropping rows with missing converted.mass...\n")
cat("  Original dataset shape:", nrow(df), "\n")
cat("  NaN values in converted.mass:", sum(is.na(df$converted.mass)), "\n")

df_clean <- df %>%
  filter(!is.na(converted.mass))

cat("  Dataset shape after dropping NaN:", nrow(df_clean), "\n")
cat("  Remaining NaN values:", sum(is.na(df_clean$converted.mass)), "\n")

# Organize data into weekly level (weekly sum per household per meat category)
cat("\n🔄 Organizing data into weekly level...\n")

# Create week identifier using ISO calendar week
df_clean <- df_clean %>%
  mutate(
    week = isoweek(recall.date),
    iso_year = isoyear(recall.date),
    week_num = isoweek(recall.date),
    year_week = paste0(year, "_", week)
  )

# First, count records per household per week (for record.days)
record_counts <- df_clean %>%
  group_by(household_ID, year_week) %>%
  summarise(record.days = n(), .groups = "drop")

# Group by household, year_week, and category to get weekly sums (not averages)
weekly_data <- df_clean %>%
  group_by(household_ID, year_week, category) %>%
  summarise(
    converted.mass = sum(converted.mass, na.rm = TRUE),  # Weekly sum consumption
    year = first(year),
    season = first(season),
    village = first(village),
    ame = first(ame),
    count.hunters = first(count.hunters),
    sex = first(sex),
    edu = first(edu),
    non.hunt.income = first(non.hunt.income),
    month = first(month),
    month_num = first(month_num),
    recall.date = first(recall.date),
    iso_year = first(iso_year),
    week_num = first(week_num),
    .groups = "drop"
  )

# Merge record.days (number of days on record per household per week)
weekly_data <- weekly_data %>%
  left_join(record_counts, by = c("household_ID", "year_week")) %>%
  mutate(record.days = ifelse(is.na(record.days), 0L, as.integer(record.days)))

cat("  Weekly data shape:", nrow(weekly_data), "observations\n")
cat("  Unique households:", n_distinct(weekly_data$household_ID), "\n")
cat("  Unique weeks:", n_distinct(weekly_data$year_week), "\n")

# Drop rows where weekly sum converted.mass is NaN (shouldn't happen after cleaning)
weekly_data <- weekly_data %>%
  filter(!is.na(converted.mass))

cat("  Weekly data shape after dropping NaN:", nrow(weekly_data), "\n")

# Now create lagged variables on the weekly data
cat("\n🔄 Creating lagged variables on weekly data...\n")

lagged_weekly_data <- list()

for (household in unique(weekly_data$household_ID)) {
  hh_data <- weekly_data %>%
    filter(household_ID == household)
  
  # For each category, create lagged mass using previous week's sum
  for (category in c("Fish", "Wild meat", "Domestic meat")) {
    cat_data <- hh_data %>%
      filter(category == !!category) %>%
      arrange(recall.date)
    
    if (nrow(cat_data) > 0) {
      # Create lagged variable using previous week's sum
      cat_data <- cat_data %>%
        mutate(lagged.converted.mass = lag(converted.mass, default = NA_real_))
      
      lagged_weekly_data[[length(lagged_weekly_data) + 1]] <- cat_data
    }
  }
}

# Combine all weekly data with lagged variables
df_with_lag <- bind_rows(lagged_weekly_data)

cat("✅ Weekly data with lagged variables created\n")
cat("  Final shape:", nrow(df_with_lag), "observations\n")
cat("  Missing lagged values:", sum(is.na(df_with_lag$lagged.converted.mass)), "\n")
cat("  First week entries with NaN lagged values:", 
    sum(is.na(df_with_lag$lagged.converted.mass)), "\n")

# =============================================================================
# 3. Calculate Festivity Variable
# =============================================================================

cat("\n🔄 Calculating festivity variable...\n")

# Define holiday dates for the period 2021-04-01 to 2022-03-31
# Note: Easter 2021 is hardcoded as April 4
holidays <- list(
  "Easter_2021" = as.Date("2021-04-04"),
  "Easter_2022" = as.Date("2022-04-17"),  # Calculate 2022 Easter
  "Christmas_2021" = as.Date("2021-12-25"),
  "Christmas_2022" = as.Date("2022-12-25"),  # Outside range but calculated anyway
  "NewYear_2022" = as.Date("2022-01-01"),
  "NewYear_2021" = as.Date("2021-01-01")  # Outside range but calculated anyway
)

cat("Holiday dates:\n")
for (name in names(holidays)) {
  cat("  ", name, ":", as.character(holidays[[name]]), "\n")
}

# Create date ranges for festive periods (±5 days around each holiday)
festive_periods <- list()
for (name in names(holidays)) {
  holiday_date <- holidays[[name]]
  start_date <- holiday_date - days(5)
  end_date <- holiday_date + days(5)
  festive_periods[[name]] <- tibble(start = start_date, end = end_date)
}

# Filter to only periods within our date range
data_start <- as.Date("2021-04-01")
data_end <- as.Date("2022-03-31")

festive_periods_in_range <- map_dfr(festive_periods, ~.x) %>%
  filter(start <= data_end & end >= data_start)

cat("\nFestive periods (10-day windows) within date range:\n")
for (i in 1:nrow(festive_periods_in_range)) {
  cat("  ", as.character(festive_periods_in_range$start[i]), 
      "to", as.character(festive_periods_in_range$end[i]), "\n")
}

# Create a vector of all dates in festive periods
festive_dates <- map2(festive_periods_in_range$start, 
                      festive_periods_in_range$end,
                      ~seq(.x, .y, by = "day")) %>%
  reduce(union)

# Mark dates as festive in the data
df_with_lag <- df_with_lag %>%
  mutate(
    date_only = as.Date(recall.date),
    is_festive_date = date_only %in% festive_dates
  )

# Now aggregate to weeks: a week is festive if it contains any festive dates
# Plus mark the week before and after (2 adjacent weeks)
festive_weeks <- df_with_lag %>%
  filter(is_festive_date) %>%
  distinct(year_week) %>%
  pull(year_week)

# Also add previous and next week for each festive week
for (week_key in festive_weeks) {
  parts <- str_split(week_key, "_", simplify = TRUE)
  year_val <- as.integer(parts[1])
  week_val <- as.integer(parts[2])
  
  # Previous week
  if (week_val > 1) {
    prev_week_key <- paste0(year_val, "_", week_val - 1)
  } else {
    prev_week_key <- paste0(year_val - 1, "_52")  # Approximate
  }
  festive_weeks <- c(festive_weeks, prev_week_key)
  
  # Next week
  next_week_key <- paste0(year_val, "_", week_val + 1)
  festive_weeks <- c(festive_weeks, next_week_key)
}

festive_weeks <- unique(festive_weeks)

# Create festivity variable (1 if week is festive, 0 otherwise)
df_with_lag <- df_with_lag %>%
  mutate(festivity = as.integer(year_week %in% festive_weeks))

cat("\n✅ Festivity variable created\n")
cat("  Number of festive weeks:", length(festive_weeks), "\n")
cat("  Festivity distribution:\n")
print(table(df_with_lag$festivity))

# =============================================================================
# 4. Create Outcome Matrix
# =============================================================================

cat("\n🔄 Creating outcome matrix...\n")

# Pivot to get one row per household-date with columns for each category
outcome_data <- df_with_lag %>%
  pivot_wider(
    id_cols = c(household_ID, recall.date, year, season, village, ame, count.hunters,
                record.days, sex, edu, non.hunt.income, month, month_num, week_num,
                iso_year, year_week, festivity),
    names_from = category,
    values_from = c(converted.mass, lagged.converted.mass),
    values_fill = 0,
    names_sep = "_"
  ) %>%
  rename(
    fish_mass = "converted.mass_Fish",
    wild_mass = "converted.mass_Wild meat",
    domestic_mass = "converted.mass_Domestic meat",
    lagged_fish_mass = "lagged.converted.mass_Fish",
    lagged_wild_mass = "lagged.converted.mass_Wild meat",
    lagged_domestic_mass = "lagged.converted.mass_Domestic meat"
  ) %>%
  # Replace NA with 0 for lagged variables
  mutate(across(starts_with("lagged_"), ~replace_na(., 0)))

cat("  Outcome data shape:", nrow(outcome_data), "observations\n")
cat("  Columns:", paste(names(outcome_data), collapse = ", "), "\n")
cat("\nFirst few rows:\n")
print(head(outcome_data))

# =============================================================================
# 5. Feature Engineering and Preparation
# =============================================================================

cat("\n🔄 Feature engineering...\n")

# Prepare data for modeling
model_data <- outcome_data %>%
  mutate(
    # Create factor variables
    household_ID = factor(household_ID),
    season = factor(season),
    village = factor(village),
    sex = factor(sex),
    edu = factor(edu),
    # Create week factor for random effects
    week_id = factor(paste0(iso_year, "_", week_num))
  )

# Check for missing values
cat("  Missing values check:\n")
missing_summary <- model_data %>%
  select(fish_mass, wild_mass, domestic_mass, ame, count.hunters, record.days,
         non.hunt.income, festivity, lagged_fish_mass, lagged_wild_mass,
         lagged_domestic_mass) %>%
  summarise_all(~sum(is.na(.)))
print(missing_summary)

# Fill any missing values
model_data <- model_data %>%
  mutate(
    # Fill continuous variables with median
    ame = ifelse(is.na(ame), median(ame, na.rm = TRUE), ame),
    count.hunters = ifelse(is.na(count.hunters), median(count.hunters, na.rm = TRUE), count.hunters),
    record.days = ifelse(is.na(record.days), 0L, record.days),
    non.hunt.income = ifelse(is.na(non.hunt.income), median(non.hunt.income, na.rm = TRUE), non.hunt.income),
    festivity = ifelse(is.na(festivity), 0L, festivity),
    lagged_fish_mass = ifelse(is.na(lagged_fish_mass), 0, lagged_fish_mass),
    lagged_wild_mass = ifelse(is.na(lagged_wild_mass), 0, lagged_wild_mass),
    lagged_domestic_mass = ifelse(is.na(lagged_domestic_mass), 0, lagged_domestic_mass)
  )

cat("\n✅ Feature engineering completed\n")
cat("  Final model data shape:", nrow(model_data), "observations\n")

# =============================================================================
# 6. Check Colinearity
# =============================================================================

cat("\n🔄 Checking for colinearity...\n")

# Create numeric predictors for correlation check
numeric_predictors <- model_data %>%
  select(ame, count.hunters, record.days, non.hunt.income, festivity,
         lagged_fish_mass, lagged_wild_mass, lagged_domestic_mass) %>%
  as.matrix()

cor_matrix <- cor(numeric_predictors, use = "complete.obs")

# Find high correlations (|r| > 0.8)
high_corr_pairs <- which(abs(cor_matrix) > 0.8 & abs(cor_matrix) < 1, arr.ind = TRUE)

if (nrow(high_corr_pairs) > 0) {
  cat("⚠️ High correlation pairs (|r| > 0.8):\n")
  for (i in 1:nrow(high_corr_pairs)) {
    row_idx <- high_corr_pairs[i, 1]
    col_idx <- high_corr_pairs[i, 2]
    cat("  ", rownames(cor_matrix)[row_idx], "-", 
        colnames(cor_matrix)[col_idx], ":",
        round(cor_matrix[row_idx, col_idx], 3), "\n")
  }
} else {
  cat("✅ No high correlations found. Proceeding with modeling.\n")
}

# =============================================================================
# 7. Model Building with brms
# =============================================================================

cat("\n🔄 Building multivariate model with brms...\n")

# Prepare outcome variables (log-transformed)
model_data <- model_data %>%
  mutate(
    log_fish = log(fish_mass + 1e-6),
    log_wild = log(wild_mass + 1e-6),
    log_domestic = log(domestic_mass + 1e-6)
  )

# Get indices for random effects
n_households <- n_distinct(model_data$household_ID)
n_weeks <- n_distinct(model_data$week_id)
n_obs <- nrow(model_data)

cat("  Number of observations:", n_obs, "\n")
cat("  Number of households:", n_households, "\n")
cat("  Number of weeks:", n_weeks, "\n")

# Scale continuous variables (to match Python StandardScaler)
# This is CRITICAL for convergence - unscaled variables cause poor R-hat and ESS
cat("\n🔄 Scaling continuous variables (matching Python StandardScaler)...\n")

# Store original values for later (if needed)
continuous_vars_orig <- model_data %>%
  select(ame, count.hunters, record.days, non.hunt.income, festivity,
         lagged_fish_mass, lagged_wild_mass, lagged_domestic_mass)

# Calculate mean and sd for each continuous variable
continuous_vars_stats <- model_data %>%
  summarise(
    ame_mean = mean(ame, na.rm = TRUE),
    ame_sd = sd(ame, na.rm = TRUE),
    count.hunters_mean = mean(count.hunters, na.rm = TRUE),
    count.hunters_sd = sd(count.hunters, na.rm = TRUE),
    record.days_mean = mean(record.days, na.rm = TRUE),
    record.days_sd = sd(record.days, na.rm = TRUE),
    non.hunt.income_mean = mean(non.hunt.income, na.rm = TRUE),
    non.hunt.income_sd = sd(non.hunt.income, na.rm = TRUE),
    festivity_mean = mean(festivity, na.rm = TRUE),
    festivity_sd = sd(festivity, na.rm = TRUE),
    lagged_fish_mass_mean = mean(lagged_fish_mass, na.rm = TRUE),
    lagged_fish_mass_sd = sd(lagged_fish_mass, na.rm = TRUE),
    lagged_wild_mass_mean = mean(lagged_wild_mass, na.rm = TRUE),
    lagged_wild_mass_sd = sd(lagged_wild_mass, na.rm = TRUE),
    lagged_domestic_mass_mean = mean(lagged_domestic_mass, na.rm = TRUE),
    lagged_domestic_mass_sd = sd(lagged_domestic_mass, na.rm = TRUE)
  )

# Scale continuous variables (z-score: (x - mean) / sd)
model_data_scaled <- model_data %>%
  mutate(
    # Scale continuous predictors (StandardScaler equivalent)
    ame = (ame - continuous_vars_stats$ame_mean[1]) / continuous_vars_stats$ame_sd[1],
    count.hunters = (count.hunters - continuous_vars_stats$count.hunters_mean[1]) / continuous_vars_stats$count.hunters_sd[1],
    record.days = (record.days - continuous_vars_stats$record.days_mean[1]) / continuous_vars_stats$record.days_sd[1],
    non.hunt.income = (non.hunt.income - continuous_vars_stats$non.hunt.income_mean[1]) / continuous_vars_stats$non.hunt.income_sd[1],
    festivity = (festivity - continuous_vars_stats$festivity_mean[1]) / ifelse(continuous_vars_stats$festivity_sd[1] == 0, 1, continuous_vars_stats$festivity_sd[1]),
    lagged_fish_mass = (lagged_fish_mass - continuous_vars_stats$lagged_fish_mass_mean[1]) / ifelse(continuous_vars_stats$lagged_fish_mass_sd[1] == 0, 1, continuous_vars_stats$lagged_fish_mass_sd[1]),
    lagged_wild_mass = (lagged_wild_mass - continuous_vars_stats$lagged_wild_mass_mean[1]) / ifelse(continuous_vars_stats$lagged_wild_mass_sd[1] == 0, 1, continuous_vars_stats$lagged_wild_mass_sd[1]),
    lagged_domestic_mass = (lagged_domestic_mass - continuous_vars_stats$lagged_domestic_mass_mean[1]) / ifelse(continuous_vars_stats$lagged_domestic_mass_sd[1] == 0, 1, continuous_vars_stats$lagged_domestic_mass_sd[1])
  )

# Save scaling statistics for later reference
write_csv(continuous_vars_stats, file.path(diagnostics_dir, "scaler_statistics.csv"))
cat("  ✅ Continuous variables scaled (StandardScaler equivalent)\n")
cat("  ✅ Scaling statistics saved to", file.path(diagnostics_dir, "scaler_statistics.csv"), "\n")

# Prepare data for brms
# brms multivariate models with mvbind() use wide format (not long format)
model_data_wide <- model_data_scaled %>%
  select(household_ID, week_id, recall.date, year, season, village, ame,
         count.hunters, record.days, sex, edu, non.hunt.income, festivity,
         lagged_fish_mass, lagged_wild_mass, lagged_domestic_mass,
         log_fish, log_wild, log_domestic) %>%
  # Ensure all numeric variables are numeric
  mutate(
    ame = as.numeric(ame),
    count.hunters = as.numeric(count.hunters),
    record.days = as.numeric(record.days),
    non.hunt.income = as.numeric(non.hunt.income),
    festivity = as.numeric(festivity),
    lagged_fish_mass = as.numeric(lagged_fish_mass),
    lagged_wild_mass = as.numeric(lagged_wild_mass),
    lagged_domestic_mass = as.numeric(lagged_domestic_mass),
    log_fish = as.numeric(log_fish),
    log_wild = as.numeric(log_wild),
    log_domestic = as.numeric(log_domestic)
  )

cat("\n✅ Data prepared for brms multivariate model\n")
cat("  Wide format shape:", nrow(model_data_wide), "observations\n")

# Check for problematic values before modeling
cat("\n🔄 Checking data quality...\n")
cat("  Checking for Inf values...\n")
if (any(sapply(model_data_wide[, c("log_fish", "log_wild", "log_domestic")], function(x) any(is.infinite(x))))) {
  cat("  ⚠️  Found Inf values in outcomes - removing rows\n")
  model_data_wide <- model_data_wide %>%
    filter(!is.infinite(log_fish), !is.infinite(log_wild), !is.infinite(log_domestic))
}

cat("  Checking for NaN values...\n")
if (any(sapply(model_data_wide[, c("log_fish", "log_wild", "log_domestic")], function(x) any(is.nan(x))))) {
  cat("  ⚠️  Found NaN values in outcomes - removing rows\n")
  model_data_wide <- model_data_wide %>%
    filter(!is.nan(log_fish), !is.nan(log_wild), !is.nan(log_domestic))
}

cat("  Final data shape:", nrow(model_data_wide), "observations\n")

# =============================================================================
# 8. Fit Multivariate Model with brms
# =============================================================================

cat("\n🔄 Fitting multivariate model with brms...\n")
cat("  This may take several minutes...\n")
cat("  Target convergence: RHAT ~1.0, ESS ~500\n")

# Define the multivariate model formula
# Using mvbind() for multiple outcomes with correlated random effects
# Note: In brms multivariate models (mvbind), random effects are automatically
# correlated across outcomes. Each grouping factor gets its own correlation matrix.
#
# IMPORTANT: Each grouping factor needs its own identifier if using 'p' syntax,
# OR use regular syntax and brms automatically handles correlations in mvbind models
model_formula <- bf(
  mvbind(log_fish, log_wild, log_domestic) ~ 
    # Fixed effects (same predictors for all outcomes, but separate coefficients per outcome)
    season + village + sex + edu + 
    ame + count.hunters + record.days + non.hunt.income + festivity +
    # Lagged variables - each outcome will have separate coefficients
    # Primary effect: log_fish uses lagged_fish_mass, log_wild uses lagged_wild_mass, etc.
    lagged_fish_mass + lagged_wild_mass + lagged_domestic_mass +
    # Random effects - automatically correlated across outcomes in mvbind models
    # Each grouping factor gets its own correlation matrix across the 3 outcomes
    (1 | household_ID) + (1 | week_id)
)

# Set priors (optional, brms will use defaults if not specified)
# Using default priors which are reasonable for most cases

# Fit the model
# Match Python settings exactly:
# - Python initial: chains=4, cores=4 (but adjusts for macOS)
# - Python fallback: chains=2, cores=1 (single-threaded)
# Use consistent settings across all platforms (matching Python exactly)
# Python uses: chains=2, cores=1 (conservative for stability)
n_cores_fit <- 1  # Single-threaded to avoid multiprocessing issues
n_chains_fit <- 2  # 2 chains for convergence checking (matches Python exactly)

cat("  Using", n_chains_fit, "chains with", n_cores_fit, "core(s) (consistent across platforms)\n")
cat("  Settings: 1000 warmup + 1000 draws per chain = 2000 total iterations per chain\n")
cat("  Acceptance rate: 0.95 (adapt_delta = 0.95)\n")

# Check if old model file exists and remove it if it has no posterior draws
# This ensures we always fit a fresh model
old_model_file <- file.path(models_dir, "multivariate_model_brms.rds")
if (file.exists(old_model_file)) {
  cat("  Checking existing model file...\n")
  tryCatch({
    old_model <- readRDS(old_model_file)
    # Check if old model has posterior draws
    test_draws <- tryCatch(as_draws_df(old_model), error = function(e) NULL)
    if (is.null(test_draws) || nrow(test_draws) == 0) {
      cat("  ⚠️  Old model file has no posterior draws - removing it\n")
      file.remove(old_model_file)
    } else {
      cat("  ℹ️  Old model file exists with posterior draws\n")
      cat("  Setting file_refit='always' to force re-fitting\n")
    }
  }, error = function(e) {
    cat("  ⚠️  Cannot check old model file - removing it to be safe\n")
    file.remove(old_model_file)
  })
}

start_time <- Sys.time()

# Match Python settings exactly:
# - target_accept=0.95 -> adapt_delta=0.95 in Stan/brms
# - draws=1000 + tune=1000 -> iter=2000 (total), warmup=1000
#   Note: In brms, iter = total iterations, warmup = adaptation, so iter = warmup + draws
# - chains=2 for macOS, 4 for others
# - cores=1 for macOS (single-threaded)
# - random_seed=42 -> seed=42
# - init='adapt_diag' -> default initialization in brms (brms doesn't have adapt_diag)

cat("  ⏱️  Model sampling started - this will take 30-60+ minutes for complex models\n")
cat("  📊 Progress will be shown every 10 iterations so you can see it's working\n")
cat("  Please be patient - first iteration can take several minutes for complex models\n\n")
cat("  💡 TIP: To check if Stan is running, open a terminal and run:\n")
cat("     ./check_stan_progress.sh\n")
cat("     Or check CPU usage: top -p $(pgrep -f '[R]script')\n\n")
cat("  ⚠️  NOTE: If you see 'Iteration: 1' for more than 10 minutes with no CPU activity,\n")
cat("     the model may be stuck. Stop it and check the data or model specification.\n\n")

# Flush output to ensure messages are printed immediately
flush.console()

tryCatch({
  model_fit <- brm(
    formula = model_formula,
    data = model_data_wide,
    family = gaussian(),
    chains = n_chains_fit,
    iter = 2000,  # total iterations: warmup (1000) + draws (1000) = 2000 (matches Python)
    warmup = 1000,  # tune=1000 (matches Python)
    cores = n_cores_fit,  # Use platform-appropriate cores
    seed = 42,  # random_seed=42 (matches Python)
    refresh = 10,  # Print progress every 10 iterations (so we can see early progress)
    control = list(
      adapt_delta = 0.95,  # target_accept=0.95 (matches Python, standard rate)
      max_treedepth = 12
    ),
    # Note: brms/rstan doesn't have 'adapt_diag' initialization like PyMC
    # Use default initialization (random) - brms handles this automatically
    save_pars = save_pars(all = TRUE),
    # Note: set_rescor is not a valid argument - residual correlation is automatic in mvbind
    # The warning says to use set_rescor() but that's a function, not an argument
    file = file.path(models_dir, "multivariate_model_brms"),  # Save model for later use
    file_refit = "always",  # Force re-fitting even if file exists
    backend = getOption("brms.backend", "rstan")  # Use rstan if cmdstanr fails
  )
}, error = function(e) {
  cat("\n⚠️  Error during model fitting:\n")
  cat("  ", conditionMessage(e), "\n")
  cat("\n🔄 Retrying with single-threaded sampling (matching Python fallback)...\n")
  
  # Retry with exact Python fallback settings
  cat("  ⏱️  Retrying model sampling - this will take 30-60+ minutes\n")
  cat("  📊 Progress will be shown every 10 iterations\n")
  cat("  First iteration can take several minutes for complex models\n\n")
  
  model_fit <<- brm(
    formula = model_formula,
    data = model_data_wide,
    family = gaussian(),
    chains = 2,  # chains=2 (matches Python fallback)
    iter = 2000,  # total iterations: warmup (1000) + draws (1000) = 2000 (matches Python)
    warmup = 1000,  # tune=1000 (matches Python)
    cores = 1,  # Single-threaded (matches Python fallback)
    seed = 42,  # random_seed=42 (matches Python)
    refresh = 10,  # Print progress every 10 iterations
    control = list(
      adapt_delta = 0.95,  # target_accept=0.95 (matches Python, standard rate)
      max_treedepth = 12
    ),
    # Note: brms/rstan doesn't have 'adapt_diag' initialization like PyMC
    # Use default initialization (random) - brms handles this automatically
    save_pars = save_pars(all = TRUE),
    file = file.path(models_dir, "multivariate_model_brms"),
    file_refit = "always",  # Force re-fitting even if file exists
    backend = "rstan"  # Force rstan backend
  )
})

elapsed_time <- Sys.time() - start_time

# Check if model fitting succeeded
if (!exists("model_fit") || inherits(model_fit, "try-error")) {
  cat("\n❌ Model fitting failed!\n")
  cat("  Trying with simpler initialization and fewer iterations...\n")
  
  # Try with exact Python settings as fallback
  cat("  ⏱️  Final retry with model sampling - this will take 30-60+ minutes\n")
  cat("  📊 Progress will be shown every 10 iterations\n")
  cat("  First iteration can take several minutes for complex models\n\n")
  
  model_fit <- brm(
    formula = model_formula,
    data = model_data_wide,
    family = gaussian(),
    chains = 2,  # matches Python fallback
    iter = 2000,  # total iterations: warmup (1000) + draws (1000) = 2000 (matches Python)
    warmup = 1000,  # tune=1000 (matches Python)
    cores = 1,  # Single-threaded (matches Python fallback)
    seed = 42,  # random_seed=42 (matches Python)
    refresh = 10,  # Print progress every 10 iterations
    control = list(
      adapt_delta = 0.95,  # target_accept=0.95 (matches Python, standard rate)
      max_treedepth = 12
    ),
    # Note: brms/rstan doesn't have 'adapt_diag' initialization like PyMC
    # Use default initialization (random) - brms handles this automatically
    save_pars = save_pars(all = TRUE),
    file = file.path(models_dir, "multivariate_model_brms"),
    file_refit = "always",  # Force re-fitting even if file exists
    backend = "rstan"
  )
}

cat("\n✅ Model fitting completed!\n")
cat("  Elapsed time:", as.numeric(elapsed_time, units = "mins"), "minutes\n")

# =============================================================================
# 9. Convergence Diagnostics
# =============================================================================

cat("\n📊 Checking convergence diagnostics...\n")

# Check if model_fit is valid
if (!exists("model_fit") || inherits(model_fit, "try-error")) {
  stop("Model fitting failed. Please check the error messages above and data quality.")
}

# Check if model has posterior draws
# In brms, check if the fit object has posterior samples
tryCatch({
  test_draws <- as_draws_df(model_fit)
  if (nrow(test_draws) == 0) {
    stop("No posterior draws found")
  }
}, error = function(e) {
  cat("⚠️  Model does not contain posterior draws!\n")
  cat("  This usually means sampling failed or didn't complete.\n")
  cat("  Error: ", conditionMessage(e), "\n")
  cat("  Checking model object...\n")
  
  # Try to get more info about what went wrong
  if (!is.null(model_fit$fit)) {
    if (inherits(model_fit$fit, "stanfit")) {
      cat("  Stan fit object exists but has no samples.\n")
      cat("  This may indicate sampling failed during execution.\n")
    }
  } else {
    cat("  Model fit object is NULL - sampling may not have run.\n")
  }
  
  stop("Model has no posterior draws. Sampling may have failed. Check Stan output above.")
})

cat("  Model has posterior draws. Checking convergence...\n")

# Get summary
summary_fit <- summary(model_fit)

# Check R-hat values - handle case where summary might not have Rhat
if (!"Rhat" %in% names(summary_fit$fixed)) {
  cat("⚠️  Warning: R-hat values not available in summary. Checking alternative method...\n")
  # Try alternative method to get R-hat
  tryCatch({
    rhat_vals <- rhat(model_fit)
    if (is.null(rhat_vals) || length(rhat_vals) == 0) {
      stop("Cannot extract R-hat values")
    }
  }, error = function(e) {
    cat("  Cannot extract R-hat values: ", conditionMessage(e), "\n")
    rhat_vals <- NULL
  })
} else {
  rhat_vals <- summary_fit$fixed$Rhat
}

if (is.null(rhat_vals) || length(rhat_vals) == 0 || all(is.na(rhat_vals))) {
  cat("⚠️  Warning: Unable to extract R-hat values from model\n")
  cat("  This may indicate the model didn't sample successfully\n")
  stop("Cannot extract convergence diagnostics. Model may not have sampled correctly.")
}
cat("\nRHAT statistics:\n")
cat("  Mean:", round(mean(rhat_vals, na.rm = TRUE), 4), "\n")
cat("  Min:", round(min(rhat_vals, na.rm = TRUE), 4), "\n")
cat("  Max:", round(max(rhat_vals, na.rm = TRUE), 4), "\n")
cat("  Parameters with RHAT > 1.01:", sum(rhat_vals > 1.01, na.rm = TRUE), 
    "/", length(rhat_vals), "\n")

# Check ESS - handle case where summary might not have ESS
if (!"Bulk_ESS" %in% names(summary_fit$fixed)) {
  cat("⚠️  Warning: ESS values not available in summary. Checking alternative method...\n")
  tryCatch({
    ess_bulk_vals <- ess_bulk(model_fit)
    ess_tail_vals <- ess_tail(model_fit)
    if (is.null(ess_bulk_vals) || length(ess_bulk_vals) == 0) {
      stop("Cannot extract ESS values")
    }
  }, error = function(e) {
    cat("  Cannot extract ESS values: ", conditionMessage(e), "\n")
    ess_bulk_vals <- NULL
    ess_tail_vals <- NULL
  })
} else {
  ess_bulk_vals <- summary_fit$fixed$Bulk_ESS
  ess_tail_vals <- summary_fit$fixed$Tail_ESS
}

if (is.null(ess_bulk_vals) || length(ess_bulk_vals) == 0 || all(is.na(ess_bulk_vals))) {
  cat("⚠️  Warning: Unable to extract ESS values from model\n")
} else {
  cat("\nESS statistics:\n")
  cat("  Bulk ESS - Mean:", round(mean(ess_bulk_vals, na.rm = TRUE), 0), "\n")
  cat("  Bulk ESS - Min:", round(min(ess_bulk_vals, na.rm = TRUE), 0), "\n")
  cat("  Bulk ESS - Max:", round(max(ess_bulk_vals, na.rm = TRUE), 0), "\n")
  cat("  Parameters with ESS < 400:", sum(ess_bulk_vals < 400, na.rm = TRUE),
      "/", length(ess_bulk_vals), "\n")
}

# Only check convergence if we have valid values
if (!is.null(rhat_vals) && !is.null(ess_bulk_vals) && 
    length(rhat_vals) > 0 && length(ess_bulk_vals) > 0 &&
    !all(is.na(rhat_vals)) && !all(is.na(ess_bulk_vals))) {
  max_rhat <- max(rhat_vals, na.rm = TRUE)
  min_ess <- min(ess_bulk_vals, na.rm = TRUE)
  
  if (is.finite(max_rhat) && is.finite(min_ess)) {
    if (max_rhat <= 1.01 && min_ess >= 400) {
      cat("\n✅ Convergence achieved! RHAT ~1 and ESS >= 400\n")
    } else {
      cat("\n⚠️ Convergence not fully achieved.\n")
      cat("  Max RHAT:", round(max_rhat, 4), " (target: <= 1.01)\n")
      cat("  Min ESS:", round(min_ess, 0), " (target: >= 400)\n")
      cat("  Consider increasing iterations.\n")
    }
  } else {
    cat("\n⚠️ Cannot assess convergence - invalid diagnostic values.\n")
  }
} else {
  cat("\n⚠️ Cannot assess convergence - missing diagnostic values.\n")
  cat("  R-hat available:", !is.null(rhat_vals) && length(rhat_vals) > 0, "\n")
  cat("  ESS available:", !is.null(ess_bulk_vals) && length(ess_bulk_vals) > 0, "\n")
}

# Print full summary
cat("\n📊 Full Model Summary:\n")
print(summary_fit)

# =============================================================================
# 10. Model Diagnostics and Visualization
# =============================================================================

cat("\n🔄 Creating model diagnostics...\n")

# Trace plots
pdf(file.path(plots_dir, "trace_plots.pdf"), width = 12, height = 8)
mcmc_trace(model_fit, pars = c("b_log_fish_Intercept", "b_log_wild_Intercept", 
                               "b_log_domestic_Intercept"))
dev.off()
cat("  Trace plots saved to", file.path(plots_dir, "trace_plots.pdf"), "\n")

# Posterior predictive checks (save to file)
pdf(file.path(plots_dir, "posterior_predictive_checks.pdf"), width = 12, height = 8)
pp_check(model_fit, resp = "logfish")
pp_check(model_fit, resp = "logwild")
pp_check(model_fit, resp = "logdomestic")
dev.off()
cat("  Posterior predictive checks saved to", file.path(plots_dir, "posterior_predictive_checks.pdf"), "\n")

# Extract posterior samples
posterior_samples <- as_draws_df(model_fit)

cat("\n✅ Model diagnostics completed\n")

# =============================================================================
# 11. Create Visualizations
# =============================================================================

cat("\n🔄 Creating visualizations...\n")

# Load ggplot2 if not already loaded
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
}

# Extract fixed effects for visualization
beta_summary <- summary_fit$fixed %>%
  rownames_to_column("parameter") %>%
  as_tibble()

# Create coefficient comparison plot (grouped bar chart)
cat("  Creating coefficient comparison plot...\n")
predictors <- coef_by_outcome %>%
  filter(!str_detect(predictor, "Intercept")) %>%
  distinct(predictor) %>%
  pull(predictor)

if (length(predictors) > 0 && nrow(coef_by_outcome) > 0) {
  # Create grouped bar chart
  pdf(file.path(plots_dir, "coefficient_comparison.pdf"), width = 16, height = 8)
  
  p <- coef_by_outcome %>%
    filter(!str_detect(predictor, "Intercept")) %>%
    ggplot(aes(x = predictor, y = Estimate, fill = outcome)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    geom_errorbar(aes(ymin = `l-95% CI`, ymax = `u-95% CI`),
                  position = position_dodge(0.9), width = 0.2) +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
    scale_fill_manual(values = c("Fish" = "#1f77b4", "Wild" = "#ff7f0e", "Domestic" = "#2ca02c")) +
    labs(title = "Coefficient Comparison Across Meat Types",
         x = "Predictors", y = "Coefficient Value") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.title = element_blank())
  
  print(p)
  dev.off()
  cat("    Coefficient comparison saved to", file.path(plots_dir, "coefficient_comparison.pdf"), "\n")
}

# Create effect plots by meat type (horizontal bar charts)
cat("  Creating effect plots by meat type...\n")
meat_types <- c("Fish", "Wild", "Domestic")
pdf(file.path(plots_dir, "effects_by_meat_type.pdf"), width = 20, height = 8)

# Prepare data for effect plots
effect_data <- coef_by_outcome %>%
  filter(!str_detect(predictor, "Intercept")) %>%
  mutate(
    ci_width = `u-95% CI` - `l-95% CI`,
    direction = case_when(
      `l-95% CI` > 0 ~ "Increase",
      `u-95% CI` < 0 ~ "Decrease",
      TRUE ~ "Uncertain"
    ),
    abs_effect = abs(Estimate)
  ) %>%
  arrange(outcome, abs_effect)

if (nrow(effect_data) > 0) {
  p <- effect_data %>%
    ggplot(aes(x = reorder(predictor, abs_effect), y = Estimate, fill = direction)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    geom_errorbar(aes(ymin = `l-95% CI`, ymax = `u-95% CI`), width = 0.2) +
    geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.6) +
    facet_wrap(~ outcome, nrow = 1, scales = "free") +
    coord_flip() +
    scale_fill_manual(values = c("Increase" = "#1f77b4", "Uncertain" = "#8c8c8c", "Decrease" = "#d62728")) +
    labs(title = "Effects by Meat Type",
         x = "Change in expected consumption (log scale)",
         y = "Predictor",
         fill = "Effect direction") +
    theme_minimal() +
    theme(legend.position = "right")
  
  print(p)
}
dev.off()
cat("    Effect plots saved to", file.path(plots_dir, "effects_by_meat_type.pdf"), "\n")

# Create uncertainty heatmap
cat("  Creating uncertainty heatmap...\n")
if (nrow(effect_data) > 0) {
  uncertainty_data <- effect_data %>%
    select(predictor, outcome, ci_width) %>%
    pivot_wider(names_from = outcome, values_from = ci_width, values_fill = 0)
  
  uncertainty_matrix <- as.matrix(uncertainty_data[, -1])
  rownames(uncertainty_matrix) <- uncertainty_data$predictor
  
  pdf(file.path(plots_dir, "uncertainty_heatmap.pdf"), width = 10, height = 12)
  heatmap(uncertainty_matrix, 
          Colv = NA, Rowv = NA,
          col = heat.colors(12),
          main = "Uncertainty Heatmap (CI Width)",
          margins = c(10, 10))
  dev.off()
  cat("    Uncertainty heatmap saved to", file.path(plots_dir, "uncertainty_heatmap.pdf"), "\n")
}

# Create coefficient sign heatmap
cat("  Creating coefficient sign heatmap...\n")
if (nrow(effect_data) > 0) {
  sign_data <- effect_data %>%
    select(predictor, outcome, Estimate) %>%
    pivot_wider(names_from = outcome, values_from = Estimate, values_fill = 0)
  
  sign_matrix <- as.matrix(sign_data[, -1])
  rownames(sign_matrix) <- sign_data$predictor
  
  pdf(file.path(plots_dir, "coefficient_sign_heatmap.pdf"), width = 10, height = 12)
  heatmap(sign_matrix,
          Colv = NA, Rowv = NA,
          col = colorRampPalette(c("blue", "white", "red"))(100),
          main = "Coefficient Sign by Predictor and Outcome",
          margins = c(10, 10))
  dev.off()
  cat("    Coefficient sign heatmap saved to", file.path(plots_dir, "coefficient_sign_heatmap.pdf"), "\n")
}

cat("\n✅ All visualizations created successfully!\n")

# =============================================================================
# 11. Extract Fixed Effects
# =============================================================================

cat("\n📊 Fixed Effects Summary:\n")

# Get fixed effects
fixed_effects <- summary_fit$fixed %>%
  rownames_to_column("parameter") %>%
  as_tibble()

print(fixed_effects)

# Extract coefficient estimates by outcome type
coef_by_outcome <- fixed_effects %>%
  filter(str_detect(parameter, "^b_")) %>%
  mutate(
    outcome = case_when(
      str_detect(parameter, "logfish") ~ "Fish",
      str_detect(parameter, "logwild") ~ "Wild",
      str_detect(parameter, "logdomestic") ~ "Domestic",
      TRUE ~ "Unknown"
    ),
    # Extract predictor name (remove b_logfish_, b_logwild_, b_logdomestic_ prefix)
    predictor = str_remove(parameter, "^b_(logfish|logwild|logdomestic)_") %>%
      str_remove("Intercept$") %>%
      replace_na("Intercept")
  )

cat("\n📊 Coefficients by Outcome:\n")
print(coef_by_outcome)

# =============================================================================
# 12. Save Results
# =============================================================================

cat("\n🔄 Saving results...\n")

# Save model
saveRDS(model_fit, file.path(models_dir, "multivariate_model_brms_fitted.rds"))
cat("  Model saved to", file.path(models_dir, "multivariate_model_brms_fitted.rds"), "\n")

# Save summary
write_csv(fixed_effects, file.path(diagnostics_dir, "fixed_effects_summary.csv"))
cat("  Fixed effects saved to", file.path(diagnostics_dir, "fixed_effects_summary.csv"), "\n")

write_csv(coef_by_outcome, file.path(diagnostics_dir, "coefficients_by_outcome.csv"))
cat("  Coefficients by outcome saved to", file.path(diagnostics_dir, "coefficients_by_outcome.csv"), "\n")

# Save posterior samples
write_csv(posterior_samples, file.path(diagnostics_dir, "posterior_samples.csv"))
cat("  Posterior samples saved to", file.path(diagnostics_dir, "posterior_samples.csv"), "\n")

# Save convergence diagnostics
convergence_diagnostics <- list(
  rhat = list(
    mean = mean(rhat_vals, na.rm = TRUE),
    min = min(rhat_vals, na.rm = TRUE),
    max = max(rhat_vals, na.rm = TRUE),
    n_high = sum(rhat_vals > 1.01, na.rm = TRUE),
    n_total = length(rhat_vals)
  ),
  ess_bulk = list(
    mean = mean(ess_bulk_vals, na.rm = TRUE),
    min = min(ess_bulk_vals, na.rm = TRUE),
    max = max(ess_bulk_vals, na.rm = TRUE),
    n_low = sum(ess_bulk_vals < 400, na.rm = TRUE),
    n_total = length(ess_bulk_vals)
  )
)

# Save as JSON (requires jsonlite package)
if (requireNamespace("jsonlite", quietly = TRUE)) {
  library(jsonlite)
  write_json(convergence_diagnostics, 
            file.path(diagnostics_dir, "convergence_diagnostics.json"),
            pretty = TRUE)
  cat("  Convergence diagnostics saved to", 
      file.path(diagnostics_dir, "convergence_diagnostics.json"), "\n")
} else {
  # Save as RDS if jsonlite not available
  saveRDS(convergence_diagnostics, 
         file.path(diagnostics_dir, "convergence_diagnostics.rds"))
  cat("  Convergence diagnostics saved to", 
      file.path(diagnostics_dir, "convergence_diagnostics.rds"), "\n")
}

cat("\n✅ All results saved successfully!\n")
cat("\n=== Analysis Complete ===\n")

