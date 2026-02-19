#!/bin/bash
# ============================================================
# Setup script for Stanford Sherlock
# Run ONCE before submitting jobs:
#   bash sherlock_setup.sh
# ============================================================

set -e

echo "=== CVCI-CF Sherlock Setup ==="

# --- Configuration (edit these) ---
ENV_NAME="cvci_cf"
PYTHON_VERSION="3.11"

# --- Load modules ---
module purge
module load python/${PYTHON_VERSION} 2>/dev/null || module load python/3.9
echo "Python: $(python3 --version)"

# --- Create virtual environment ---
VENV_DIR="${HOME}/envs/${ENV_NAME}"

if [ -d "$VENV_DIR" ]; then
    echo "Environment already exists at ${VENV_DIR}"
    echo "To recreate, run: rm -rf ${VENV_DIR} && bash sherlock_setup.sh"
else
    echo "Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
echo "Activated: $(which python)"

# --- Install dependencies ---
# Pin versions to avoid build-from-source issues with Sherlock's old GCC (4.8.5).
# These versions all have pre-built wheels for cp39-linux_x86_64.
pip install --upgrade pip
pip install \
    "numpy<2.1" \
    "scipy<1.14" \
    "scikit-learn<1.7" \
    "econml<0.17" \
    "matplotlib<3.10" \
    "pandas<2.2"

echo ""
echo "=== Verify installation ==="
python -c "
import numpy; print(f'  numpy:        {numpy.__version__}')
import sklearn; print(f'  scikit-learn: {sklearn.__version__}')
import econml; print(f'  econml:       {econml.__version__}')
import matplotlib; print(f'  matplotlib:   {matplotlib.__version__}')
print('  All imports OK')
"

echo ""
echo "=== Setup complete ==="
echo "Environment: ${VENV_DIR}"
echo ""
echo "To activate manually:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Next step: submit jobs with"
echo "  sbatch sherlock_submit.sh              # all experiments"
echo "  sbatch sherlock_submit.sh --prototype  # test run first"