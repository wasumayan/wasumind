#!/bin/bash
#SBATCH --job-name=teach-hc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/teach-hc-%j.out
#SBATCH --error=logs/teach-hc-%j.err

module purge
module load anaconda3/2025.6
set +u && conda activate spectramem && set -u
cd /home/$USER/wasumind
mkdir -p logs

# Install MuJoCo if not present
pip install "gymnasium[mujoco]" mujoco -q 2>/dev/null

echo "=== Teacher training: HalfCheetah-v4 seed 42 — $(date) ==="
python wasumindV2/teachers/train_teacher.py \
    --env HalfCheetah-v4 --seed 42 \
    --output-dir experiments/v2/teachers

echo "=== Teacher training: HalfCheetah-v4 seed 123 — $(date) ==="
python wasumindV2/teachers/train_teacher.py \
    --env HalfCheetah-v4 --seed 123 \
    --output-dir experiments/v2/teachers

echo "=== Done: $(date) ==="
