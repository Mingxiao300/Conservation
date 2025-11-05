#!/bin/bash
# Occupancy and N-mixture Models runner
# Sets up environment, runs model, logs everything, and reports outputs
# Usage: ./run_occupancy_nmixture.sh
# Or with srun: srun --mem=16G --time=6:00:00 --pty bash -c "./run_occupancy_nmixture.sh"

SCRIPT_DIR="/n/home00/msong300/conservation"
cd "$SCRIPT_DIR" || exit 1

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Create log file with timestamp
LOG_FILE="$SCRIPT_DIR/logs/occupancy_nmixture_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================================"
echo "OCCUPANCY AND N-MIXTURE MODELS - STARTING"
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
echo "  Occupancy models: $SCRIPT_DIR/outputs/occupancy_models/"
echo "  N-mixture models: $SCRIPT_DIR/outputs/nmixture_models/"
echo ""
echo "This may take 2-4 hours (runs 4 proteins × 2 models = 8 models total)..."
echo "============================================================================"
echo ""

# Run Python model and save output to log file
echo "Running occupancy and n-mixture models: occupancy_nmixture_models.py" | tee "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python3 occupancy_nmixture/occupancy_nmixture_models.py 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "============================================================================" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ OCCUPANCY AND N-MIXTURE MODELS COMPLETED SUCCESSFULLY"
    echo ""
    
    # Check and report what was saved
    echo "Checking saved outputs..."
    
    # Check occupancy models
    if [ -d "$SCRIPT_DIR/outputs/occupancy_models" ]; then
        OCC_FILES=$(find "$SCRIPT_DIR/outputs/occupancy_models" -type f 2>/dev/null | wc -l)
        if [ "$OCC_FILES" -gt 0 ]; then
            echo "✅ Occupancy models saved ($OCC_FILES files):"
            find "$SCRIPT_DIR/outputs/occupancy_models" -type f -name "*.nc" -o -name "*.csv" | head -10 | while read f; do
                FILE_SIZE=$(du -h "$f" 2>/dev/null | cut -f1 || echo "unknown")
                echo "  - $(basename "$f") ($FILE_SIZE)"
            done
        else
            echo "⚠️  No occupancy model files found"
        fi
    else
        echo "⚠️  Occupancy models directory not found"
    fi
    
    # Check n-mixture models
    if [ -d "$SCRIPT_DIR/outputs/nmixture_models" ]; then
        NMIX_FILES=$(find "$SCRIPT_DIR/outputs/nmixture_models" -type f 2>/dev/null | wc -l)
        if [ "$NMIX_FILES" -gt 0 ]; then
            echo "✅ N-mixture models saved ($NMIX_FILES files):"
            find "$SCRIPT_DIR/outputs/nmixture_models" -type f -name "*.nc" -o -name "*.csv" | head -10 | while read f; do
                FILE_SIZE=$(du -h "$f" 2>/dev/null | cut -f1 || echo "unknown")
                echo "  - $(basename "$f") ($FILE_SIZE)"
            done
        else
            echo "⚠️  No n-mixture model files found"
        fi
    else
        echo "⚠️  N-mixture models directory not found"
    fi
    
    echo ""
    echo "============================================================================"
    echo "SUMMARY"
    echo "============================================================================"
    echo "Outputs saved to:"
    echo "  Occupancy models: $SCRIPT_DIR/outputs/occupancy_models/"
    echo "  N-mixture models: $SCRIPT_DIR/outputs/nmixture_models/"
    echo "  Log: $LOG_FILE"
    echo ""
    echo "To view results:"
    echo "  - Occupancy models: ls -lh $SCRIPT_DIR/outputs/occupancy_models/"
    echo "  - N-mixture models: ls -lh $SCRIPT_DIR/outputs/nmixture_models/"
    echo "  - Full log: cat $LOG_FILE"
    echo ""
    
else
    echo "❌ OCCUPANCY AND N-MIXTURE MODELS FAILED"
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
    echo "  3. Missing Python packages:"
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

