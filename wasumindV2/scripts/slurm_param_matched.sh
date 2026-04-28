#!/bin/bash
#SBATCH --job-name=param-match
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/param-match-%j.out
#SBATCH --error=logs/param-match-%j.err

# ============================================================
# Parameter-matched GRU: d_model=160 (~312K params) to match STU (~338K).
# If GRU still wins at matched params, the architecture claim holds.
# Run on ADROIT where demos already exist.
# ============================================================

module purge
module load anaconda3/2025.6
set +u && conda activate spectramem && set -u
cd /home/$USER/wasumind
mkdir -p logs

pip install "gymnasium[mujoco]" mujoco -q 2>/dev/null

DEMO_DIR="experiments/v2/demos"
STUDENT_DIR="experiments/v2/students_param_matched"
TOTAL=0
FAILED=0

echo "=== Parameter-matched GRU (d=160, ~312K params) — $(date) ==="

for ENV in halfcheetah_pomdp ant_pomdp; do
    for DEMO_SUBDIR in ${DEMO_DIR}/${ENV}/teacher_*; do
        [ -d "$DEMO_SUBDIR" ] || continue
        DEMO_PATH="${DEMO_SUBDIR}/demos.pt"
        [ -f "$DEMO_PATH" ] || continue
        DEMO_TAG=$(basename "$DEMO_SUBDIR")

        for SEED in 42 123 456; do
            OUTDIR="${STUDENT_DIR}/${ENV}/${DEMO_TAG}/gru/seed${SEED}"
            if [ -f "${OUTDIR}/result.json" ]; then
                echo "  SKIP (exists): $ENV / $DEMO_TAG / seed=$SEED"
                continue
            fi
            echo "  RUN: $ENV / $DEMO_TAG / gru_d160 / seed=$SEED"
            python wasumindV2/distillation/distill.py \
                --demo-path "$DEMO_PATH" \
                --arch gru --seed "$SEED" \
                --d-model 160 --n-layers 2 \
                --output-dir "$STUDENT_DIR" \
                && TOTAL=$((TOTAL + 1)) \
                || { echo "  FAILED"; FAILED=$((FAILED + 1)); }
        done
    done
done

echo ""
echo "=== Compiling results ==="
python wasumindV2/evaluation/evaluate_sweep.py \
    --results-dir "$STUDENT_DIR" \
    --output experiments/v2/param_matched_results.csv

echo "=== Done: $TOTAL succeeded, $FAILED failed — $(date) ==="
