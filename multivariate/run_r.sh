#!/bin/bash
# Simple script to run R model
# Run this in your tmux session: ./run_r.sh

SCRIPT_DIR="/n/home00/msong300/conservation"
cd "$SCRIPT_DIR"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Create log file with timestamp
LOG_FILE="$SCRIPT_DIR/logs/r_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================================"
echo "R (BRMS) MODEL - STARTING"
echo "============================================================================"
echo "Start time: $(date)"
echo "Working directory: $SCRIPT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Load R module first (before checking anything else)
echo "Loading R module..."
if command -v module >/dev/null 2>&1; then
    module load R >/dev/null 2>&1 || module load r >/dev/null 2>&1 || module load R/4 >/dev/null 2>&1 || true
    if command -v Rscript >/dev/null 2>&1; then
        echo "✅ R module loaded successfully"
    else
        echo "⚠️  R module may not be loaded correctly"
    fi
else
    echo "⚠️  Module system not available, using system R"
fi

# Check if Rscript is available
if ! command -v Rscript >/dev/null 2>&1; then
    echo "❌ ERROR: Rscript not found!"
    echo "  Please load R module: module load R"
    exit 1
fi

R_VERSION=$(Rscript --version 2>&1 | head -n 1)
echo "R version: $R_VERSION"
echo ""

# Check and install R packages if missing
echo "Checking R packages..."
USER_R_LIB="$HOME/R/library"
mkdir -p "$USER_R_LIB"

# Clean up stale R package lock files from previous interrupted installations
echo "Checking for stale R package lock files..."
LOCK_DIRS=$(find "$USER_R_LIB" -maxdepth 1 -type d -name "00LOCK-*" 2>/dev/null)
if [ -n "$LOCK_DIRS" ]; then
    # Check if any R process is actively installing (should not remove active locks)
    ACTIVE_R=$(ps aux | grep -E "[R]script.*install.packages|[R] CMD INSTALL" | grep -v grep | wc -l)
    if [ "$ACTIVE_R" -eq 0 ]; then
        echo "⚠️  Found stale lock files from previous installation"
        echo "  Removing stale lock files..."
        find "$USER_R_LIB" -maxdepth 1 -type d -name "00LOCK-*" -exec rm -rf {} \; 2>/dev/null || true
        echo "✅ Lock files cleaned up"
    else
        echo "⚠️  R installation in progress, skipping lock cleanup"
    fi
fi

Rscript -e "
user_lib <- '$USER_R_LIB'
if (!dir.exists(user_lib)) {
    dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(user_lib, .libPaths()))
required <- c('readr', 'dplyr', 'tidyr', 'lubridate', 'purrr', 'stringr', 'brms', 'bayesplot', 'posterior')
missing <- required[!sapply(required, requireNamespace, quietly = TRUE)]
if (length(missing) == 0) {
    cat('✅ All R packages are already installed\n')
} else {
    cat('⚠️  Missing packages:', paste(missing, collapse = ', '), '\n')
    cat('Installing missing packages...\n')
    cat('This may take 10-20 minutes...\n')
    # Install all missing packages together for better dependency resolution
    cat('Installing packages:', paste(missing, collapse = ', '), '\n')
    tryCatch({
        install.packages(missing, 
                        lib = user_lib, 
                        repos = 'https://cloud.r-project.org', 
                        dependencies = TRUE, 
                        quiet = FALSE,
                        Ncpus = 1)  # Use single core to avoid race conditions
        # Verify each package was installed
        for (pkg in missing) {
            if (!requireNamespace(pkg, quietly = TRUE)) {
                cat('WARNING: ', pkg, 'may not have installed correctly\n')
            }
        }
    }, error = function(e) {
        cat('ERROR during installation:', conditionMessage(e), '\n')
        stop('Package installation failed')
    })
    .libPaths(c(user_lib, .libPaths()))
    # Verify installation
    missing <- required[!sapply(required, requireNamespace, quietly = TRUE)]
    if (length(missing) > 0) {
        stop('Failed to install: ', paste(missing, collapse = ', '))
    } else {
        cat('✅ All R packages installed successfully\n')
    }
}
" 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ ERROR: Failed to install R packages"
    echo "Check log: $LOG_FILE"
    exit 1
fi

echo ""
echo "Outputs will be saved to:"
echo "  Models: $SCRIPT_DIR/outputs/models/"
echo "  Diagnostics: $SCRIPT_DIR/outputs/diagnostics/"
echo "  Plots: $SCRIPT_DIR/outputs/plots/"
echo ""
echo "This may take 2-3 hours..."
echo "============================================================================"
echo ""

# Run R model and save output to log file
Rscript multivariate/Multivariate_Modeling_1101.R 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "R MODEL COMPLETED SUCCESSFULLY"
    echo "End time: $(date)"
    echo ""
    echo "Outputs saved to:"
    echo "  Models: $SCRIPT_DIR/outputs/models/"
    echo "  Diagnostics: $SCRIPT_DIR/outputs/diagnostics/"
    echo "  Plots: $SCRIPT_DIR/outputs/plots/"
    echo "  Log: $LOG_FILE"
else
    echo "R MODEL FAILED"
    echo "Exit code: $EXIT_CODE"
    echo "Check log: $LOG_FILE"
fi
echo "============================================================================"

exit $EXIT_CODE

