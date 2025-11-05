#!/bin/bash
# Fast R package installation
# Tries conda/mamba first (if writable), falls back to R install.packages

echo "Installing R packages..."

# Load R module if available
if command -v module >/dev/null 2>&1; then
    echo "🔄 Loading R module..."
    module load R >/dev/null 2>&1 || module load r >/dev/null 2>&1 || module load R/4 >/dev/null 2>&1 || true
fi

# Check if Rscript is available
if ! command -v Rscript >/dev/null 2>&1; then
    echo "❌ ERROR: Rscript not found!"
    echo "  Please load R module first: module load R"
    echo "  Or install R manually"
    exit 1
fi

# Create user library directory
USER_R_LIB="$HOME/R/library"
mkdir -p "$USER_R_LIB"
echo "Using user R library: $USER_R_LIB"

# Try conda/mamba if available and environment is writable
CONDA_WRITABLE=false
if command -v conda >/dev/null 2>&1; then
    # Check if we can write to conda environment
    CONDA_PREFIX=$(conda info --base 2>/dev/null || echo "")
    if [ -n "$CONDA_PREFIX" ] && [ -w "$CONDA_PREFIX" ]; then
        CONDA_WRITABLE=true
    fi
fi

if [ "$CONDA_WRITABLE" = true ]; then
    echo "Attempting to install via conda/mamba (faster)..."
    if command -v mamba >/dev/null 2>&1; then
        INSTALLER="mamba"
    else
        INSTALLER="conda"
    fi
    
    $INSTALLER install -y -c conda-forge \
        r-readr \
        r-dplyr \
        r-tidyr \
        r-lubridate \
        r-brms \
        r-bayesplot \
        r-posterior 2>&1 | grep -v "^$" || true
    
    # Check if installation succeeded
    Rscript -e "
    .libPaths(c('$USER_R_LIB', .libPaths()))
    if (dir.exists('$CONDA_PREFIX/lib/R/library')) {
        .libPaths(c('$CONDA_PREFIX/lib/R/library', .libPaths()))
    }
    required <- c('readr', 'dplyr', 'tidyr', 'lubridate', 'brms', 'bayesplot', 'posterior')
    missing <- required[!sapply(required, requireNamespace, quietly = TRUE)]
    if (length(missing) == 0) {
        cat('ALL_INSTALLED')
    } else {
        cat('MISSING:', paste(missing, collapse = ', '))
    }
    " 2>&1 | tail -1 | grep -q "ALL_INSTALLED"
    
    if [ $? -eq 0 ]; then
        echo "✅ R packages installed successfully via $INSTALLER"
        exit 0
    else
        echo "⚠️  Conda installation incomplete, falling back to R install.packages..."
    fi
else
    echo "⚠️  Conda environment not writable, using R install.packages..."
fi

# Install using R install.packages (to user library)
echo "Installing R packages using install.packages (this may take 10-20 minutes)..."
echo "  This is installing to: $USER_R_LIB"

Rscript -e "
# Set user library path
user_lib <- '$USER_R_LIB'
if (!dir.exists(user_lib)) {
    dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
}

# Add user library to .libPaths()
.libPaths(c(user_lib, .libPaths()))
cat('Using R library paths:', paste(.libPaths(), collapse = ', '), '\n')

# Function to install package if not available
install_if_missing <- function(pkg) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat('Installing', pkg, '...\n')
        tryCatch({
            install.packages(pkg, 
                            lib = user_lib,
                            repos = 'https://cloud.r-project.org', 
                            dependencies = TRUE, 
                            quiet = FALSE)
            .libPaths(c(user_lib, .libPaths()))
            return(TRUE)
        }, error = function(e) {
            cat('ERROR installing', pkg, ':', conditionMessage(e), '\n')
            return(FALSE)
        })
    } else {
        cat(pkg, 'already installed\n')
        return(TRUE)
    }
}

# Install required packages
required <- c('readr', 'dplyr', 'tidyr', 'lubridate', 'brms', 'bayesplot', 'posterior')
cat('Installing', length(required), 'required packages...\n')

for (pkg in required) {
    install_if_missing(pkg)
}

# Verify installation
.libPaths(c(user_lib, .libPaths()))
missing <- required[!sapply(required, requireNamespace, quietly = TRUE)]
if (length(missing) == 0) {
    cat('✅ All required packages installed successfully!\n')
} else {
    stop('Failed to install: ', paste(missing, collapse = ', '))
}
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Installation complete!"
    echo "  Packages installed to: $USER_R_LIB"
    echo ""
    echo "To verify installation, run:"
    echo "  Rscript -e \".libPaths(c('$USER_R_LIB', .libPaths())); sapply(c('readr', 'dplyr', 'tidyr', 'lubridate', 'brms', 'bayesplot', 'posterior'), requireNamespace, quietly=TRUE)\""
else
    echo ""
    echo "❌ Installation failed. Check error messages above."
    exit 1
fi

