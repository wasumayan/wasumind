#!/bin/bash
#SBATCH --job-name=md-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/md-eval-%j.out
#SBATCH --error=logs/md-eval-%j.err

module purge
module load anaconda3/2025.6
set +u && conda activate metadrive && set -u
cd /home/$USER/wasumind
mkdir -p logs

echo "=== MetaDrive evaluation — $(date) ==="
python wasumindV2/evaluation/evaluate_metadrive.py \
    --students-dir experiments/v2/students_metadrive \
    --n-episodes 20
echo "=== Done: $(date) ==="
