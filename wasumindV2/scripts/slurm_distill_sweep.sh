#!/bin/bash
#SBATCH --job-name=distill-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/distill-sweep-%j.out
#SBATCH --error=logs/distill-sweep-%j.err

module purge
module load anaconda3/2025.6
set +u && conda activate spectramem && set -u
cd /home/$USER/wasumind
mkdir -p logs

pip install "gymnasium[mujoco]" mujoco -q 2>/dev/null

TEACHER_DIR="experiments/v2/teachers"
DEMO_DIR="experiments/v2/demos"
STUDENT_DIR="experiments/v2/students"

echo "=== Distillation sweep — $(date) ==="

# For each environment
for ENV in halfcheetah_pomdp ant_pomdp; do

    if [ "$ENV" = "halfcheetah_pomdp" ]; then
        TEACHER_PREFIX="halfcheetahv4"
    else
        TEACHER_PREFIX="antv4"
    fi

    # Find teacher checkpoints
    TEACHER_BASE="${TEACHER_DIR}/${TEACHER_PREFIX}_seed42"
    if [ ! -d "$TEACHER_BASE" ]; then
        echo "Teacher not found at $TEACHER_BASE, skipping $ENV"
        continue
    fi

    # For each checkpoint .zip file
    for CKPT_ZIP in ${TEACHER_BASE}/teacher_*.zip; do
        [ -f "$CKPT_ZIP" ] || continue

        # Remove .zip extension for sb3 load path
        CKPT="${CKPT_ZIP%.zip}"
        CKPT_NAME=$(basename "$CKPT")

        echo ""
        echo "=== $ENV / $CKPT_NAME ==="

        # Collect demos with different noise levels
        for NOISE in 0.0 0.3; do
            echo "  Collecting demos (noise=$NOISE)..."
            python wasumindV2/teachers/collect_demos.py \
                --teacher-path "$CKPT" \
                --env "$ENV" --n-episodes 500 \
                --noise-sigma "$NOISE" \
                --output-dir "$DEMO_DIR" \
                || { echo "  FAILED collecting demos for $CKPT_NAME noise=$NOISE"; continue; }

            NOISE_TAG="noise$(echo $NOISE | tr -d '.')"
            DEMO_PATH="${DEMO_DIR}/${ENV}/${CKPT_NAME}_${NOISE_TAG}/demos.pt"

            if [ ! -f "$DEMO_PATH" ]; then
                echo "  Demo file not found: $DEMO_PATH, skipping"
                continue
            fi

            # Distill each architecture
            for ARCH in gru lstm stu transformer; do
                for SEED in 42 123 456; do
                    echo "  Distilling: $ENV / $CKPT_NAME / noise=$NOISE / $ARCH / seed=$SEED..."
                    python wasumindV2/distillation/distill.py \
                        --demo-path "$DEMO_PATH" \
                        --arch "$ARCH" --seed "$SEED" \
                        --output-dir "$STUDENT_DIR" \
                        || echo "  FAILED: $ARCH seed=$SEED"
                done
            done
        done
    done
done

# Compile results
echo ""
echo "=== Compiling results ==="
python wasumindV2/evaluation/evaluate_sweep.py \
    --results-dir "$STUDENT_DIR" \
    --output experiments/v2/sweep_results.csv

echo "=== Done: $(date) ==="
