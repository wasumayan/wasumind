#!/bin/bash
#SBATCH --job-name=metadrive-extras
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/metadrive-extras-%j.out
#SBATCH --error=logs/metadrive-extras-%j.err

# ============================================================
# After slurm_metadrive_full.sh completes:
#   1. Distill additional archs (stu, mlp, framestack) on MetaDrive demos
#   2. Evaluate ALL MetaDrive students (including gru/lstm/transformer from original run)
# Run on DELLA in the metadrive conda env.
# ============================================================

module purge
module load anaconda3/2025.6
set +u && conda activate metadrive && set -u
cd /home/$USER/wasumind
mkdir -p logs

DEMO_DIR="experiments/v2/demos_metadrive"
STUDENT_DIR="experiments/v2/students_metadrive"
TOTAL=0
FAILED=0

echo "=== MetaDrive extras — $(date) ==="

# --- Phase 0: Fix demo metadata (obs_dim was hardcoded to 259, actual is 91) ---
echo "--- Phase 0: Fix demo metadata ---"
python wasumindV2/evaluation/fix_metadrive_demos.py --demos-dir "$DEMO_DIR"

# --- Phase 1: Distill ALL archs on MetaDrive demos ---
echo ""
echo "--- Phase 1: Distill all architectures ---"

for DEMO_SUBDIR in ${DEMO_DIR}/teacher_*; do
    [ -d "$DEMO_SUBDIR" ] || continue
    DEMO_PATH="${DEMO_SUBDIR}/demos.pt"
    [ -f "$DEMO_PATH" ] || continue
    DEMO_TAG=$(basename "$DEMO_SUBDIR")

    for ARCH in gru lstm transformer stu mlp framestack; do
        for SEED in 42 123 456; do
            OUTDIR="${STUDENT_DIR}/metadrive/${DEMO_TAG}/${ARCH}/seed${SEED}"
            if [ -f "${OUTDIR}/result.json" ]; then
                echo "  SKIP (exists): $DEMO_TAG / $ARCH / seed=$SEED"
                continue
            fi
            echo "  RUN: metadrive / $DEMO_TAG / $ARCH / seed=$SEED"
            python wasumindV2/distillation/distill.py \
                --demo-path "$DEMO_PATH" \
                --arch "$ARCH" --seed "$SEED" \
                --output-dir "$STUDENT_DIR" \
                --skip-eval \
                && TOTAL=$((TOTAL + 1)) \
                || { echo "  FAILED"; FAILED=$((FAILED + 1)); }
        done
    done
done

# --- Phase 2: Evaluate ALL MetaDrive students ---
echo ""
echo "--- Phase 2: Evaluate all students in MetaDrive ---"
python wasumindV2/evaluation/evaluate_metadrive.py \
    --students-dir "$STUDENT_DIR" \
    --n-episodes 20

echo ""
echo "=== Compiling results ==="
python wasumindV2/evaluation/evaluate_sweep.py \
    --results-dir "$STUDENT_DIR" \
    --output experiments/v2/metadrive_results.csv

echo "=== Done: $TOTAL new distillations, $FAILED failed — $(date) ==="
