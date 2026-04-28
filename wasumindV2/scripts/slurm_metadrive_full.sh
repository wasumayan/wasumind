#!/bin/bash
#SBATCH --job-name=metadrive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/metadrive-%j.out
#SBATCH --error=logs/metadrive-%j.err

module purge
module load anaconda3/2025.6
set +u && conda activate metadrive && set -u
cd /home/$USER/wasumind
mkdir -p logs

# metadrive env has all deps pre-installed

TEACHER_DIR="experiments/v2/teachers_metadrive"
DEMO_DIR="experiments/v2/demos_metadrive"
STUDENT_DIR="experiments/v2/students_metadrive"

echo "=== MetaDrive Full Pipeline — $(date) ==="

# Train teachers (2 seeds)
for SEED in 42 123; do
    echo ""
    echo "=== Training teacher seed=$SEED ==="
    python wasumindV2/teachers/train_metadrive_teacher.py \
        --seed $SEED \
        --total-steps 1000000 \
        --output-dir "$TEACHER_DIR" \
        || echo "FAILED: teacher seed=$SEED"
done

# Use seed 42 teacher for distillation sweep
TEACHER_BASE="${TEACHER_DIR}/metadrive_seed42"
if [ ! -d "$TEACHER_BASE" ]; then
    echo "Teacher not found, exiting"
    exit 1
fi

# Collect demos and distill for each checkpoint
for CKPT_ZIP in ${TEACHER_BASE}/teacher_*.zip; do
    [ -f "$CKPT_ZIP" ] || continue
    CKPT="${CKPT_ZIP%.zip}"
    CKPT_NAME=$(basename "$CKPT")

    echo ""
    echo "=== $CKPT_NAME ==="

    for NOISE in 0.0 0.3; do
        echo "  Collecting demos (noise=$NOISE)..."
        python wasumindV2/teachers/collect_metadrive_demos.py \
            --teacher-path "$CKPT" \
            --n-episodes 500 \
            --noise-sigma "$NOISE" \
            --output-dir "$DEMO_DIR" \
            || { echo "  FAILED collecting demos"; continue; }

        NOISE_TAG="noise$(echo $NOISE | tr -d '.')"
        DEMO_PATH="${DEMO_DIR}/${CKPT_NAME}_${NOISE_TAG}/demos.pt"

        if [ ! -f "$DEMO_PATH" ]; then
            echo "  Demo file not found: $DEMO_PATH"
            continue
        fi

        for ARCH in gru lstm transformer; do
            for SEED in 42 123 456; do
                echo "  Distilling: $CKPT_NAME / noise=$NOISE / $ARCH / seed=$SEED..."
                python wasumindV2/distillation/distill.py \
                    --demo-path "$DEMO_PATH" \
                    --arch "$ARCH" --seed "$SEED" \
                    --output-dir "$STUDENT_DIR" \
                    --skip-eval \
                    || echo "  FAILED: $ARCH seed=$SEED"
            done
        done
    done
done

echo ""
echo "=== Compiling results ==="
python wasumindV2/evaluation/evaluate_sweep.py \
    --results-dir "$STUDENT_DIR" \
    --output experiments/v2/metadrive_sweep_results.csv

echo "=== Done: $(date) ==="
