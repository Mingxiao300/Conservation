#!/bin/bash
# Comprehensive Python (PyMC) model runner
# Sets up environment, runs model, logs everything, and reports outputs
# Usage: ./run_python_model.sh
# Or with srun: srun --mem=16G --time=6:00:00 --pty bash -c "./run_python_model.sh"

SCRIPT_DIR="/n/home00/msong300/conservation"
cd "$SCRIPT_DIR" || exit 1

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Create log file with timestamp
LOG_FILE="$SCRIPT_DIR/logs/python_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================================"
echo "PYTHON (PYMC) MODEL - STARTING"
echo "============================================================================"
echo "Start time: $(date)"
echo "Working directory: $SCRIPT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Check Python availability
echo "Checking Python environment..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ ERROR: python3 not found!"
    echo "  Please load Python module or activate conda environment"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python version: $PYTHON_VERSION"
echo ""

# Check required Python packages
echo "Checking required Python packages..."
python3 << 'PYTHON_CHECK'
import sys
required_packages = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'pymc': 'pymc',
    'arviz': 'arviz',
    'scipy': 'scipy',
    'sklearn': 'sklearn'
}

missing = []
for package_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"✅ {package_name} installed")
    except ImportError:
        print(f"❌ {package_name} NOT installed")
        missing.append(package_name)

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    print("  Please install missing packages:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("\n✅ All required packages are installed")
    sys.exit(0)
PYTHON_CHECK

PYTHON_CHECK_EXIT=$?
if [ $PYTHON_CHECK_EXIT -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Missing required Python packages"
    echo "  Please install missing packages before running the model"
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

# Run Python model and save output to log file
echo "Running Python model: multivariate/fit_model_pymc.py" | tee "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python3 multivariate/fit_model_pymc.py 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ PYTHON MODEL COMPLETED SUCCESSFULLY"
    echo ""
    
    # Check and report what was saved
    echo "Checking saved outputs..."
    
    # Check inference data
    if [ -f "$SCRIPT_DIR/outputs/models/inference_data.nc" ]; then
        echo "✅ Inference data saved: inference_data.nc"
        FILE_SIZE=$(du -h "$SCRIPT_DIR/outputs/models/inference_data.nc" | cut -f1)
        echo "  File size: $FILE_SIZE"
    else
        echo "⚠️  Inference data not found"
    fi
    
    # Check model pickle
    if [ -f "$SCRIPT_DIR/outputs/models/pymc_model.pkl" ]; then
        echo "✅ Model pickle saved: pymc_model.pkl"
    else
        echo "⚠️  Model pickle not found"
    fi
    
    # Check diagnostics
    DIAG_FILES=$(find "$SCRIPT_DIR/outputs/diagnostics" -type f 2>/dev/null | wc -l)
    if [ "$DIAG_FILES" -gt 0 ]; then
        echo "✅ Diagnostics saved ($DIAG_FILES files):"
        find "$SCRIPT_DIR/outputs/diagnostics" -type f -name "*.csv" -o -name "*.json" | head -10 | while read f; do
            FILE_SIZE=$(du -h "$f" 2>/dev/null | cut -f1 || echo "unknown")
            echo "  - $(basename "$f") ($FILE_SIZE)"
        done
    else
        echo "⚠️  No diagnostic files found"
    fi
    
    # Check plots
    PLOT_FILES=$(find "$SCRIPT_DIR/outputs/plots" -type f 2>/dev/null | wc -l)
    if [ "$PLOT_FILES" -gt 0 ]; then
        echo "✅ Plots saved ($PLOT_FILES files):"
        find "$SCRIPT_DIR/outputs/plots" -type f -name "*.png" -o -name "*.pdf" | head -10 | while read f; do
            FILE_SIZE=$(du -h "$f" 2>/dev/null | cut -f1 || echo "unknown")
            echo "  - $(basename "$f") ($FILE_SIZE)"
        done
    else
        echo "⚠️  No plot files found"
    fi
    
    # Check for preprocessing files
    if [ -f "$SCRIPT_DIR/outputs/models/encoder.pkl" ]; then
        echo "✅ Preprocessing files saved (encoder.pkl, scaler.pkl)"
    fi
    
    if [ -f "$SCRIPT_DIR/outputs/diagnostics/feature_correlation_matrix.csv" ]; then
        echo "✅ Feature correlation matrix saved"
    fi
    
    echo ""
    echo "============================================================================"
    echo "SUMMARY"
    echo "============================================================================"
    echo "Outputs saved to:"
    echo "  Models: $SCRIPT_DIR/outputs/models/"
    echo "  Diagnostics: $SCRIPT_DIR/outputs/diagnostics/"
    echo "  Plots: $SCRIPT_DIR/outputs/plots/"
    echo "  Log: $LOG_FILE"
    echo ""
    echo "To view results:"
    echo "  - Model outputs: ls -lh $SCRIPT_DIR/outputs/models/"
    echo "  - Diagnostics: ls -lh $SCRIPT_DIR/outputs/diagnostics/"
    echo "  - Plots: ls -lh $SCRIPT_DIR/outputs/plots/"
    echo "  - Full log: cat $LOG_FILE"
    echo ""
    
else
    echo "❌ PYTHON MODEL FAILED"
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "  1. Missing data file:"
    echo "     - Check if protein_full_data.csv exists in $SCRIPT_DIR"
    echo ""
    echo "  2. Memory errors:"
    echo "     - Request more memory: srun --mem=16G --time=6:00:00 --pty bash"
    echo "     - The model may need 8-16GB RAM depending on data size"
    echo ""
    echo "  3. Divergence warnings:"
    echo "     - Check the log for divergence warnings"
    echo "     - High divergences may indicate model geometry issues"
    echo "     - The script will continue but results may be unreliable"
    echo ""
    echo "  4. Missing Python packages:"
    echo "     - Check log for import errors"
    echo "     - Install missing packages: pip install <package>"
    echo ""
    echo "Check log file for details: $LOG_FILE"
    echo ""
    echo "Last 50 lines of log:"
    echo "============================================================================"
    tail -50 "$LOG_FILE"
fi

echo "============================================================================"
echo "End time: $(date)"
echo "============================================================================"

exit $EXIT_CODE


