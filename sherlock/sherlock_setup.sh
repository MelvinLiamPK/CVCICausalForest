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
pip install --upgrade pip
pip install \
    numpy \
    scipy \
    scikit-learn \
    econml \
    matplotlib \
    pandas

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
