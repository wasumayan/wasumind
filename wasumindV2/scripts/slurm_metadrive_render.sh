#!/bin/bash
#SBATCH --job-name=md-render
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/md-render-%j.out
#SBATCH --error=logs/md-render-%j.err

module purge
module load anaconda3/2025.6
set +u && conda activate metadrive && set -u
cd /home/$USER/wasumind
mkdir -p logs

pip install Pillow -q 2>/dev/null

echo "=== MetaDrive rendering — $(date) ==="
python wasumindV2/evaluation/render_metadrive.py \
    --students-dir experiments/v2/students_metadrive \
    --output-dir experiments/v2/metadrive_visuals \
    --teacher-tag teacher_500000_noise00 \
    --max-steps 300
echo "=== Done: $(date) ==="
