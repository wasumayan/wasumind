#!/bin/bash
#SBATCH --job-name=render-vis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/render-vis-%j.out
#SBATCH --error=logs/render-vis-%j.err

module purge
module load anaconda3/2025.6
set +u && conda activate spectramem && set -u
cd /home/$USER/wasumind
mkdir -p logs

pip install "gymnasium[mujoco]" mujoco matplotlib -q 2>/dev/null

echo "=== Rendering MuJoCo visuals — $(date) ==="
python wasumindV2/evaluation/render_mujoco_frames.py \
    --results-dir experiments/v2/students \
    --output-dir experiments/v2/visuals

echo "=== Done: $(date) ==="
