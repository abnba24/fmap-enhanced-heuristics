#!/bin/bash
# Setup script for FMAP experimental framework

echo "Setting up FMAP experimental framework..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install requirements
echo "Activating virtual environment and installing requirements..."
source venv/bin/activate
pip install -r requirements.txt

echo "✓ Virtual environment set up with all dependencies"

# Check if FMAP.jar exists
if [ -f "../FMAP.jar" ]; then
    echo "✓ FMAP.jar found"
else
    echo "⚠ Warning: FMAP.jar not found in project root"
    echo "  Please ensure FMAP.jar is in the parent directory"
fi

# Check if Domains directory exists
if [ -d "../Domains" ]; then
    domain_count=$(find ../Domains -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "✓ Domains directory found with $domain_count domains"
else
    echo "⚠ Warning: Domains directory not found"
    echo "  Please ensure domain files are in ../Domains/"
fi

# Test imports
echo "Testing Python imports..."
python3 -c "
import numpy, pandas, matplotlib, seaborn, scipy, psutil
print('✓ All Python packages imported successfully')
"

echo ""
echo "Setup complete! To run experiments:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run quick test: python run_experiments.py --quick"
echo "  3. Run full experiments: python run_experiments.py"
echo "" 