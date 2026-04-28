#!/bin/bash
#SBATCH --job-name=demo-ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/demo-ablation-%j.out
#SBATCH --error=logs/demo-ablation-%j.err

# ============================================================
# Demo count ablation: 50, 200, (500 exists), 2000 demos.
# Tests whether data efficiency patterns from CopyTask hold on MuJoCo.
# Uses 1M teacher (no noise) — the cleanest teacher quality level.
# Run on ADROIT where teachers + demos exist.
# ============================================================

module purge
module load anaconda3/2025.6
set +u && conda activate spectramem && set -u
cd /home/$USER/wasumind
mkdir -p logs

pip install "gymnasium[mujoco]" mujoco -q 2>/dev/null

TEACHER_DIR="experiments/v2/teachers"
DEMO_DIR="experiments/v2/demos"
ABLATION_DEMO_DIR="experiments/v2/demos_ablation"
STUDENT_DIR="experiments/v2/students_demo_ablation"
TOTAL=0
FAILED=0

echo "=== Demo count ablation — $(date) ==="

for ENV in halfcheetah_pomdp ant_pomdp; do
    if [ "$ENV" = "halfcheetah_pomdp" ]; then
        TEACHER_PREFIX="halfcheetahv4"
    else
        TEACHER_PREFIX="antv4"
    fi

    TEACHER_PATH="${TEACHER_DIR}/${TEACHER_PREFIX}_seed42/teacher_1000000"
    EXISTING_DEMO="${DEMO_DIR}/${ENV}/teacher_1000000_noise00/demos.pt"

    if [ ! -f "${TEACHER_PATH}.zip" ]; then
        echo "  SKIP: No 1M teacher for $ENV"
        continue
    fi

    # --- Subsample 50 and 200 from existing 500-episode demos ---
    for N_EP in 50 200; do
        DEST="${ABLATION_DEMO_DIR}/${ENV}/teacher_1000000_noise00_${N_EP}ep/demos.pt"
        if [ ! -f "$DEST" ]; then
            echo "  Subsampling $N_EP episodes for $ENV..."
            python wasumindV2/evaluation/subsample_demos.py \
                --input "$EXISTING_DEMO" \
                --n-episodes "$N_EP" --seed 42 \
                --output "$DEST" \
                || { echo "  FAILED subsampling"; continue; }
        fi
    done

    # --- Collect 2000 fresh demos ---
    DEST_2000="${ABLATION_DEMO_DIR}/${ENV}/teacher_1000000_noise00_2000ep"
    if [ ! -f "${DEST_2000}/demos.pt" ]; then
        echo "  Collecting 2000 demos for $ENV..."
        TEMP_DEMO_DIR="${ABLATION_DEMO_DIR}_tmp_$$"
        python wasumindV2/teachers/collect_demos.py \
            --teacher-path "$TEACHER_PATH" \
            --env "$ENV" --n-episodes 2000 \
            --noise-sigma 0.0 \
            --output-dir "$TEMP_DEMO_DIR" \
            || { echo "  FAILED collecting 2000 demos for $ENV"; rm -rf "$TEMP_DEMO_DIR"; continue; }
        # Move from auto-generated path to our naming scheme
        mkdir -p "$DEST_2000"
        mv "${TEMP_DEMO_DIR}/${ENV}/teacher_1000000_noise00/demos.pt" "$DEST_2000/"
        mv "${TEMP_DEMO_DIR}/${ENV}/teacher_1000000_noise00/demo_info.json" "$DEST_2000/" 2>/dev/null
        rm -rf "$TEMP_DEMO_DIR"
    fi

    # --- Distill on each demo size ---
    for N_EP in 50 200 2000; do
        if [ "$N_EP" = "2000" ]; then
            DEMO_PATH="${ABLATION_DEMO_DIR}/${ENV}/teacher_1000000_noise00_${N_EP}ep/demos.pt"
        else
            DEMO_PATH="${ABLATION_DEMO_DIR}/${ENV}/teacher_1000000_noise00_${N_EP}ep/demos.pt"
        fi

        if [ ! -f "$DEMO_PATH" ]; then
            echo "  SKIP: No demo file at $DEMO_PATH"
            continue
        fi

        for ARCH in gru lstm stu transformer; do
            for SEED in 42 123 456; do
                DEMO_TAG="teacher_1000000_noise00_${N_EP}ep"
                OUTDIR="${STUDENT_DIR}/${ENV}/${DEMO_TAG}/${ARCH}/seed${SEED}"
                if [ -f "${OUTDIR}/result.json" ]; then
                    echo "  SKIP (exists): $ENV / ${N_EP}ep / $ARCH / seed=$SEED"
                    continue
                fi
                echo "  RUN: $ENV / ${N_EP}ep / $ARCH / seed=$SEED"
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
    --output experiments/v2/demo_ablation_results.csv

echo "=== Done: $TOTAL succeeded, $FAILED failed — $(date) ==="
