#!/bin/bash
# Robust R package installation script
# Handles memory issues, missing dependencies, and compilation errors

echo "============================================================================"
echo "R Package Installation - Robust Version"
echo "============================================================================"
echo "Start time: $(date)"
echo ""

# Load R module first
echo "Loading R module..."
if command -v module >/dev/null 2>&1; then
    module load R >/dev/null 2>&1 || module load r >/dev/null 2>&1 || module load R/4 >/dev/null 2>&1 || true
fi

# Check if Rscript is available
if ! command -v Rscript >/dev/null 2>&1; then
    echo "❌ ERROR: Rscript not found!"
    echo "  Please load R module first: module load R"
    exit 1
fi

R_VERSION=$(Rscript --version 2>&1 | head -n 1)
echo "R version: $R_VERSION"
echo ""

# Try to load CMake if available (needed for some packages)
echo "Checking for CMake..."
if command -v module >/dev/null 2>&1; then
    if module avail cmake 2>&1 | grep -q cmake; then
        module load cmake >/dev/null 2>&1 || true
        if command -v cmake >/dev/null 2>&1; then
            echo "✅ CMake module loaded"
        fi
    fi
fi

# Check if CMake is available (if not loaded via module, check system)
if ! command -v cmake >/dev/null 2>&1; then
    echo "⚠️  CMake not found - some packages may fail to install"
    echo "  Consider: module load cmake (if available)"
fi

# Set up user library
USER_R_LIB="$HOME/R/library"
mkdir -p "$USER_R_LIB"
echo "Using user R library: $USER_R_LIB"
echo ""

# Clean up stale lock files
echo "Cleaning up stale lock files..."
find "$USER_R_LIB" -maxdepth 1 -type d -name "00LOCK-*" 2>/dev/null | while read lockdir; do
    # Check if any R process is using it
    if ! lsof "$lockdir" >/dev/null 2>&1; then
        echo "  Removing stale lock: $lockdir"
        rm -rf "$lockdir" 2>/dev/null || true
    fi
done
echo ""

# Set up compilation environment to reduce memory usage
echo "Setting up compilation environment..."
export MAKEFLAGS="-j1"  # Single-threaded compilation to reduce memory usage
export R_MAKEVARS_USER="$HOME/.R/Makevars"

# Create Makevars file with memory-efficient settings
mkdir -p "$HOME/.R"
cat > "$R_MAKEVARS_USER" << 'EOF'
# Memory-efficient compilation settings
CXX = g++
CXX11 = g++
CXX14 = g++
CXX17 = g++
CXX20 = g++
CXXFLAGS = -O2 -g
CXX11FLAGS = -O2 -g
CXX14FLAGS = -O2 -g
CXX17FLAGS = -O2 -g
CXX20FLAGS = -O2 -g

# Limit parallel jobs to reduce memory usage
MAKEFLAGS = -j1

# Reduce optimization level to reduce memory during compilation
CXXFLAGS += -g -O2
EOF

echo "✅ Created Makevars file with memory-efficient settings"
echo ""

# Check for libiconv and set library paths if needed
echo "Checking for libiconv..."
LIBICONV_PATH=$(find /usr /opt /n/sw -name "libiconv.so*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
if [ -n "$LIBICONV_PATH" ]; then
    export LD_LIBRARY_PATH="$LIBICONV_PATH:$LD_LIBRARY_PATH"
    echo "✅ Found libiconv, added to LD_LIBRARY_PATH"
else
    echo "⚠️  libiconv not found in standard locations"
    echo "  Will try to use system libraries or install via R"
fi
echo ""

# Required packages
REQUIRED_PKGS=("readr" "dplyr" "tidyr" "lubridate" "brms" "bayesplot" "posterior")

# Check which packages are already installed
echo "Checking installed packages..."
Rscript -e "
user_lib <- '$USER_R_LIB'
.libPaths(c(user_lib, .libPaths()))
required <- c('readr', 'dplyr', 'tidyr', 'lubridate', 'brms', 'bayesplot', 'posterior')
installed <- sapply(required, requireNamespace, quietly = TRUE)
missing <- required[!installed]
if (length(missing) == 0) {
    cat('✅ All required packages are already installed\n')
    quit(save = 'no', status = 0)
} else {
    cat('Missing packages:', paste(missing, collapse = ', '), '\n')
    writeLines(missing, 'missing_packages.txt')
    quit(save = 'no', status = 1)
}
" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ All packages already installed!"
    exit 0
fi

MISSING_PKGS=$(cat missing_packages.txt 2>/dev/null || echo "")
rm -f missing_packages.txt

echo ""
echo "Installing missing packages: $MISSING_PKGS"
echo "This will install packages in smaller batches to avoid memory issues..."
echo ""

# Install packages in stages with error handling
Rscript -e "
# Set up library paths
user_lib <- '$USER_R_LIB'
if (!dir.exists(user_lib)) {
    dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(user_lib, .libPaths()))

# Set up environment variables for compilation
Sys.setenv(MAKEFLAGS = '-j1')
Sys.setenv(R_MAKEVARS_USER = Sys.getenv('R_MAKEVARS_USER'))

# Function to install package with retry and memory limits
install_with_retry <- function(pkg, max_retries = 2) {
    if (requireNamespace(pkg, quietly = TRUE)) {
        cat('✅', pkg, 'already installed\n')
        return(TRUE)
    }
    
    for (retry in 1:max_retries) {
        cat('Installing', pkg, '- attempt', retry, '...\n')
        tryCatch({
            # Use single-threaded compilation
            install.packages(pkg, 
                            lib = user_lib,
                            repos = 'https://cloud.r-project.org',
                            dependencies = TRUE,
                            type = 'source',  # Always use source for better control
                            INSTALL_opts = '--no-lock',
                            Ncpus = 1,  # Single CPU
                            quiet = FALSE)
            
            # Verify installation
            .libPaths(c(user_lib, .libPaths()))
            if (requireNamespace(pkg, quietly = TRUE)) {
                cat('✅', pkg, 'installed successfully\n')
                return(TRUE)
            } else {
                stop('Package installed but cannot be loaded')
            }
        }, error = function(e) {
            cat('⚠️  Error installing', pkg, ':', conditionMessage(e), '\n')
            if (retry < max_retries) {
                cat('  Retrying...\n')
                Sys.sleep(5)  # Wait before retry
            }
            return(FALSE)
        })
    }
    
    cat('❌ Failed to install', pkg, 'after', max_retries, 'attempts\n')
    return(FALSE)
}

# Install packages in dependency order
# Stage 1: Basic dependencies (most likely to succeed)
cat('\n=== Stage 1: Installing basic dependencies ===\n')
stage1 <- c('readr', 'dplyr', 'tidyr', 'lubridate')
stage1_missing <- stage1[!sapply(stage1, requireNamespace, quietly = TRUE)]
if (length(stage1_missing) > 0) {
    for (pkg in stage1_missing) {
        install_with_retry(pkg)
    }
}

# Stage 2: Try to install problematic packages individually
cat('\n=== Stage 2: Installing Stan-related packages ===\n')
cat('  Note: rstan and brms may take a long time and use lots of memory\n')
cat('  If they fail, you may need to request more memory/time\n\n')

# Try rstan first (brms depends on it)
if (!requireNamespace('rstan', quietly = TRUE)) {
    cat('Installing rstan (this may take 20-30 minutes and requires significant memory)...\n')
    install_with_retry('rstan', max_retries = 1)  # Only retry once for memory-intensive packages
}

# Then brms
if (!requireNamespace('brms', quietly = TRUE)) {
    cat('Installing brms...\n')
    install_with_retry('brms', max_retries = 1)
}

# Stage 3: Install remaining packages
cat('\n=== Stage 3: Installing remaining packages ===\n')
stage3 <- c('bayesplot', 'posterior')
stage3_missing <- stage3[!sapply(stage3, requireNamespace, quietly = TRUE)]
if (length(stage3_missing) > 0) {
    for (pkg in stage3_missing) {
        install_with_retry(pkg)
    }
}

# Final verification
.libPaths(c(user_lib, .libPaths()))
required <- c('readr', 'dplyr', 'tidyr', 'lubridate', 'brms', 'bayesplot', 'posterior')
missing <- required[!sapply(required, requireNamespace, quietly = TRUE)]

if (length(missing) == 0) {
    cat('\n✅ All required packages installed successfully!\n')
    quit(save = 'no', status = 0)
} else {
    cat('\n❌ Failed to install:', paste(missing, collapse = ', '), '\n')
    cat('\nTroubleshooting tips:\n')
    if ('rstan' %in% missing || 'brms' %in% missing) {
        cat('  - rstan/brms require significant memory (8GB+ recommended)\n')
        cat('  - Try requesting more memory: srun --mem=16G --time=2:00:00 bash\n')
        cat('  - Or install on a compute node with more memory\n')
    }
    cat('  - Check logs for specific error messages\n')
    cat('  - Some packages may need system dependencies (CMake, etc.)\n')
    quit(save = 'no', status = 1)
}
" 2>&1 | tee "$INSTALL_LOG_FILE"

INSTALL_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================================"
if [ $INSTALL_EXIT_CODE -eq 0 ]; then
    echo "✅ Package installation completed successfully!"
    echo ""
    echo "Installed packages are in: $USER_R_LIB"
    echo ""
    
    # Set up script directory and logging
    SCRIPT_DIR="/n/home00/msong300/conservation"
    cd "$SCRIPT_DIR" || exit 1
    
    # Create logs directory if it doesn't exist
    mkdir -p "$SCRIPT_DIR/logs"
    
    # Create log file for model run
    MODEL_LOG_FILE="$SCRIPT_DIR/logs/r_$(date +%Y%m%d_%H%M%S).log"
    
    echo "============================================================================"
    echo "AUTOMATICALLY STARTING R MODEL RUN"
    echo "============================================================================"
    echo "Start time: $(date)"
    echo "Working directory: $SCRIPT_DIR"
    echo "Model log file: $MODEL_LOG_FILE"
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
    echo "Running R model: Multivariate_Modeling_1101.R" | tee -a "$MODEL_LOG_FILE"
    echo "" | tee -a "$MODEL_LOG_FILE"
    
    Rscript Multivariate_Modeling_1101.R 2>&1 | tee -a "$MODEL_LOG_FILE"
    
    MODEL_EXIT_CODE=${PIPESTATUS[0]}
    
    echo ""
    echo "============================================================================"
    if [ $MODEL_EXIT_CODE -eq 0 ]; then
        echo "✅ R MODEL COMPLETED SUCCESSFULLY"
        echo "End time: $(date)"
        echo ""
        
        # Check and report what was saved
        echo "Checking saved outputs..."
        
        # Check models
        if [ -f "$SCRIPT_DIR/outputs/models/multivariate_model_brms.rds" ] || \
           [ -f "$SCRIPT_DIR/outputs/models/multivariate_model_brms_fitted.rds" ]; then
            echo "✅ Model file saved"
        else
            echo "⚠️  Model file not found"
        fi
        
        # Check diagnostics
        DIAG_FILES=$(find "$SCRIPT_DIR/outputs/diagnostics" -type f 2>/dev/null | wc -l)
        if [ "$DIAG_FILES" -gt 0 ]; then
            echo "✅ Diagnostics saved ($DIAG_FILES files)"
            find "$SCRIPT_DIR/outputs/diagnostics" -type f -name "*.csv" -o -name "*.json" | head -5 | while read f; do
                echo "  - $(basename "$f")"
            done
        else
            echo "⚠️  No diagnostic files found"
        fi
        
        # Check plots
        PLOT_FILES=$(find "$SCRIPT_DIR/outputs/plots" -type f 2>/dev/null | wc -l)
        if [ "$PLOT_FILES" -gt 0 ]; then
            echo "✅ Plots saved ($PLOT_FILES files)"
            find "$SCRIPT_DIR/outputs/plots" -type f -name "*.png" -o -name "*.pdf" | head -5 | while read f; do
                echo "  - $(basename "$f")"
            done
        else
            echo "⚠️  No plot files found"
        fi
        
        echo ""
        echo "Outputs saved to:"
        echo "  Models: $SCRIPT_DIR/outputs/models/"
        echo "  Diagnostics: $SCRIPT_DIR/outputs/diagnostics/"
        echo "  Plots: $SCRIPT_DIR/outputs/plots/"
        echo "  Log: $MODEL_LOG_FILE"
    else
        echo "❌ R MODEL FAILED"
        echo "Exit code: $MODEL_EXIT_CODE"
        echo "Check log: $MODEL_LOG_FILE"
    fi
    echo "============================================================================"
    
    exit $MODEL_EXIT_CODE
else
    echo "❌ Package installation had errors"
    echo ""
    echo "Check the log file: install_r_packages.log"
    echo ""
    echo "Common issues and solutions:"
    echo "  1. Memory errors (OOM):"
    echo "     - Request more memory: srun --mem=16G --time=4:00:00 bash"
    echo "     - Or run on a compute node with more RAM"
    echo ""
    echo "  2. Missing CMake:"
    echo "     - Try: module load cmake"
    echo ""
    echo "  3. Missing libiconv:"
    echo "     - May need to install system package: yum install libiconv-devel"
    echo "     - Or contact system administrator"
    echo ""
    echo "  4. rstan/brms compilation failures:"
    echo "     - These packages are very memory-intensive"
    echo "     - May require 16GB+ RAM and 1-2 hours to compile"
    echo "     - Consider using pre-compiled binaries if available"
    echo ""
    echo "⚠️  Cannot proceed to model run - packages must be installed first"
fi
echo "============================================================================"
echo "End time: $(date)"

exit $INSTALL_EXIT_CODE

