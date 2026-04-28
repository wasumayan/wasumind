#!/bin/bash
#SBATCH --job-name=extra-seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/extra-seeds-%j.out
#SBATCH --error=logs/extra-seeds-%j.err

# ============================================================
# Seeds 789 + 1337 for GRU/LSTM/STU/Transformer on key conditions.
# Brings us from 3 → 5 seeds on the most important comparisons.
# Run on ADROIT where demos already exist.
# ============================================================

module purge
module load anaconda3/2025.6
set +u && conda activate spectramem && set -u
cd /home/$USER/wasumind
mkdir -p logs

pip install "gymnasium[mujoco]" mujoco -q 2>/dev/null

DEMO_DIR="experiments/v2/demos"
STUDENT_DIR="experiments/v2/students_extra_seeds"
TOTAL=0
FAILED=0

# Key conditions: strongest + weakest teachers, with and without noise
KEY_CONDITIONS=(
    "teacher_1000000_noise00"
    "teacher_2000000_noise00"
    "teacher_200000_noise03"
    "teacher_2000000_noise03"
)

echo "=== Extra seeds (789, 1337) — $(date) ==="

for ENV in halfcheetah_pomdp ant_pomdp; do
    for COND in "${KEY_CONDITIONS[@]}"; do
        DEMO_PATH="${DEMO_DIR}/${ENV}/${COND}/demos.pt"
        if [ ! -f "$DEMO_PATH" ]; then
            echo "  SKIP (no demos): $ENV / $COND"
            continue
        fi

        for ARCH in gru lstm stu transformer; do
            for SEED in 789 1337; do
                OUTDIR="${STUDENT_DIR}/${ENV}/${COND}/${ARCH}/seed${SEED}"
                if [ -f "${OUTDIR}/result.json" ]; then
                    echo "  SKIP (exists): $ENV / $COND / $ARCH / seed=$SEED"
                    continue
                fi
                echo "  RUN: $ENV / $COND / $ARCH / seed=$SEED"
                python wasumindV2/distillation/distill.py \
                    --demo-path "$DEMO_PATH" \
                    --arch "$ARCH" --seed "$SEED" \
                    --output-dir "$STUDENT_DIR" \
                    && TOTAL=$((TOTAL + 1)) \
                    || { echo "  FAILED"; FAILED=$((FAILED + 1)); }
            done
        done
    done
done

echo ""
echo "=== Compiling results ==="
python wasumindV2/evaluation/evaluate_sweep.py \
    --results-dir "$STUDENT_DIR" \
    --output experiments/v2/extra_seeds_results.csv

echo "=== Done: $TOTAL succeeded, $FAILED failed — $(date) ==="
