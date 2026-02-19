#!/bin/bash
#
# CVCI-CF Simulations on Stanford Sherlock
#
# Uses SLURM job arrays: each axis×CATE combination runs as a separate job.
# 9 jobs total (3 axes × 3 CATEs), running in parallel.
#
# Usage:
#   sbatch sherlock_submit.sh                  # Full run (9 jobs, ~2-3h each)
#   sbatch sherlock_submit.sh --quick          # Quick run (9 jobs, ~30min each)
#   sbatch sherlock_submit.sh --prototype      # Test run (9 jobs, ~1min each)
#   sbatch sherlock_submit.sh --quick --cate constant   # 3 jobs, constant only
#   sbatch sherlock_submit.sh --quick --axis epsilon     # 3 jobs, epsilon only
#
# Monitor:
#   squeue -u $USER                            # Check job status
#   tail -f logs/cvci_cf_*.out                 # Watch output
#   sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS  # Resource usage
#
# ============================================================
# SLURM CONFIGURATION — edit these for your allocation
# ============================================================
#SBATCH --job-name=cvci_cf
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/cvci_cf_%A_%a.out
#SBATCH --error=logs/cvci_cf_%A_%a.err
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
# Uncomment and set your email:
# #SBATCH --mail-user=your_email@stanford.edu
# Uncomment if you have a PI allocation:
# #SBATCH --account=your_pi_group

# ============================================================
# CONFIGURATION — edit paths here
# ============================================================
ENV_NAME="cvci_cf"
VENV_DIR="${HOME}/envs/${ENV_NAME}"
# Where the code lives (clone your repo here):
CODE_DIR="${HOME}/CVCICF"
# Where results go (use $SCRATCH for large output):
RESULTS_DIR="${SCRATCH}/cvci_cf_results"

# ============================================================
# Parse arguments passed via: sbatch sherlock_submit.sh --quick
# SLURM passes extra args after the script name
# ============================================================
MODE=""
CATE_FILTER="all"
AXIS_FILTER="all"

for arg in "$@"; do
    case $arg in
        --prototype)  MODE="--prototype" ;;
        --ultra-quick) MODE="--ultra-quick" ;;
        --quick)      MODE="--quick" ;;
        --cate)       shift_next=cate ;;
        --axis)       shift_next=axis ;;
        constant|step|nonlinear)
            if [ "$shift_next" = "cate" ]; then
                CATE_FILTER="$arg"
                shift_next=""
            fi ;;
        epsilon|nobs|confounding)
            if [ "$shift_next" = "axis" ]; then
                AXIS_FILTER="$arg"
                shift_next=""
            fi ;;
    esac
done

# ============================================================
# Job array mapping: SLURM_ARRAY_TASK_ID -> (axis, cate)
# ============================================================
# ID  axis          cate
# 0   epsilon       constant
# 1   epsilon       step
# 2   epsilon       nonlinear
# 3   nobs          constant
# 4   nobs          step
# 5   nobs          nonlinear
# 6   confounding   constant
# 7   confounding   step
# 8   confounding   nonlinear

AXES=(epsilon epsilon epsilon nobs nobs nobs confounding confounding confounding)
CATES=(constant step nonlinear constant step nonlinear constant step nonlinear)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
AXIS=${AXES[$TASK_ID]}
CATE=${CATES[$TASK_ID]}

# Skip if filtered out
if [ "$CATE_FILTER" != "all" ] && [ "$CATE" != "$CATE_FILTER" ]; then
    echo "Skipping: axis=$AXIS cate=$CATE (filtered by --cate $CATE_FILTER)"
    exit 0
fi
if [ "$AXIS_FILTER" != "all" ] && [ "$AXIS" != "$AXIS_FILTER" ]; then
    echo "Skipping: axis=$AXIS cate=$CATE (filtered by --axis $AXIS_FILTER)"
    exit 0
fi

# ============================================================
# Setup environment
# ============================================================
echo "============================================================"
echo "CVCI-CF Simulation"
echo "  Job ID:    ${SLURM_JOB_ID} (array task ${TASK_ID})"
echo "  Node:      $(hostname)"
echo "  Axis:      ${AXIS}"
echo "  CATE:      ${CATE}"
echo "  Mode:      ${MODE:-full}"
echo "  Started:   $(date)"
echo "============================================================"

module purge
module load python/3.11 2>/dev/null || module load python/3.9

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    echo "Run 'bash sherlock_setup.sh' first."
    exit 1
fi

source "${VENV_DIR}/bin/activate"
echo "Python: $(which python) ($(python --version))"

# Create results and log directories
mkdir -p "${RESULTS_DIR}"
mkdir -p logs

# ============================================================
# Run simulation
# ============================================================
cd "${CODE_DIR}"

# Make src/ importable
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH}"

echo ""
echo "Running: python experiments/cf_simulations.py ${MODE} --axis ${AXIS} --cate ${CATE} --save-dir ${RESULTS_DIR}"
echo ""

python experiments/cf_simulations.py \
    ${MODE} \
    --axis "${AXIS}" \
    --cate "${CATE}" \
    --save-dir "${RESULTS_DIR}"

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  Finished: $(date)"
echo "  Exit code: ${EXIT_CODE}"
echo "  Results:   ${RESULTS_DIR}/"
echo "============================================================"

exit ${EXIT_CODE}
