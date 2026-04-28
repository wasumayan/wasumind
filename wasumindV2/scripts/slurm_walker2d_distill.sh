#!/bin/bash
#SBATCH --job-name=walker2d-distill
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/walker2d-distill-%j.out
#SBATCH --error=logs/walker2d-distill-%j.err

# ============================================================
# Phase 2 of Walker2d: Distill all architectures on Walker2d demos.
# Run on DELLA AFTER slurm_walker2d_teacher.sh completes.
# ============================================================

module purge
module load anaconda3/2025.6
set +u && conda activate spectramem && set -u
cd /home/$USER/wasumind
mkdir -p logs

pip install "gymnasium[mujoco]" mujoco -q 2>/dev/null

DEMO_DIR="experiments/v2/demos"
STUDENT_DIR="experiments/v2/students"
TOTAL=0
FAILED=0

echo "=== Walker2d distillation sweep — $(date) ==="

# Verify demos exist
if [ ! -d "${DEMO_DIR}/walker2d_pomdp" ]; then
    echo "FATAL: No Walker2d demos found. Run slurm_walker2d_teacher.sh first."
    exit 1
fi

for DEMO_SUBDIR in ${DEMO_DIR}/walker2d_pomdp/teacher_*; do
    [ -d "$DEMO_SUBDIR" ] || continue
    DEMO_PATH="${DEMO_SUBDIR}/demos.pt"
    [ -f "$DEMO_PATH" ] || continue
    DEMO_TAG=$(basename "$DEMO_SUBDIR")

    for ARCH in gru lstm stu transformer mlp framestack; do
        for SEED in 42 123 456; do
            OUTDIR="${STUDENT_DIR}/walker2d_pomdp/${DEMO_TAG}/${ARCH}/seed${SEED}"
            if [ -f "${OUTDIR}/result.json" ]; then
                echo "  SKIP (exists): $DEMO_TAG / $ARCH / seed=$SEED"
                continue
            fi
            echo "  RUN: walker2d / $DEMO_TAG / $ARCH / seed=$SEED"
            python wasumindV2/distillation/distill.py \
                --demo-path "$DEMO_PATH" \
                --arch "$ARCH" --seed "$SEED" \
                --output-dir "$STUDENT_DIR" \
                && TOTAL=$((TOTAL + 1)) \
                || { echo "  FAILED"; FAILED=$((FAILED + 1)); }
        done
    done
done

echo ""
echo "=== Compiling results ==="
python wasumindV2/evaluation/evaluate_sweep.py \
    --results-dir "${STUDENT_DIR}" \
    --output experiments/v2/walker2d_results.csv

echo "=== Done: $TOTAL succeeded, $FAILED failed — $(date) ==="
