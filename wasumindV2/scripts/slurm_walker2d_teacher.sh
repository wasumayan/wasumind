#!/bin/bash
#SBATCH --job-name=walker2d-teacher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/walker2d-teacher-%j.out
#SBATCH --error=logs/walker2d-teacher-%j.err

# ============================================================
# Phase 1 of Walker2d: Train teacher + collect ALL demos.
# Run on DELLA. After this completes, submit slurm_walker2d_distill.sh.
# ============================================================

module purge
module load anaconda3/2025.6
set +u && conda activate spectramem && set -u
cd /home/$USER/wasumind
mkdir -p logs

pip install "gymnasium[mujoco]" mujoco -q 2>/dev/null

TEACHER_DIR="experiments/v2/teachers"
DEMO_DIR="experiments/v2/demos"

echo "=== Walker2d teacher training — $(date) ==="

# Verify Walker2d obs layout before committing to a multi-hour train
python -c "
import gymnasium as gym
env = gym.make('Walker2d-v4')
obs, _ = env.reset()
assert obs.shape == (17,), f'Walker2d-v4 obs expected (17,), got {obs.shape}'
env.close()
print(f'Walker2d-v4 verified: obs={obs.shape}, positions=0:8, velocities=8:17')
" || { echo "FATAL: Walker2d-v4 obs layout verification failed"; exit 1; }

# Train teacher (seed 42 only — matching HalfCheetah/Ant sweep)
python wasumindV2/teachers/train_teacher.py \
    --env Walker2d-v4 --seed 42 \
    --total-steps 2000000 \
    --output-dir "$TEACHER_DIR" \
    || { echo "FATAL: teacher training failed"; exit 1; }

TEACHER_BASE="${TEACHER_DIR}/walker2dv4_seed42"
echo ""
echo "=== Collecting Walker2d demos — $(date) ==="

for CKPT_ZIP in ${TEACHER_BASE}/teacher_*.zip; do
    [ -f "$CKPT_ZIP" ] || continue
    CKPT="${CKPT_ZIP%.zip}"
    CKPT_NAME=$(basename "$CKPT")

    for NOISE in 0.0 0.3; do
        echo "  Collecting: $CKPT_NAME noise=$NOISE"
        python wasumindV2/teachers/collect_demos.py \
            --teacher-path "$CKPT" \
            --env walker2d_pomdp \
            --n-episodes 500 \
            --noise-sigma "$NOISE" \
            --output-dir "$DEMO_DIR" \
            || echo "  FAILED: $CKPT_NAME noise=$NOISE"
    done
done

echo ""
echo "=== Walker2d teacher + demos complete — $(date) ==="
echo "Now submit: sbatch wasumindV2/scripts/slurm_walker2d_distill.sh"
