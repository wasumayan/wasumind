#!/bin/bash
#SBATCH --job-name=stu-copy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=wasumayan@princeton.edu
#SBATCH --output=logs/stu-copy-%j.out
#SBATCH --error=logs/stu-copy-%j.err

module purge
module load anaconda3/2025.6
set +u && conda activate spectramem && set -u
cd /home/$USER/wasumind
mkdir -p logs

echo "=== Full STU on CopyTask — $(date) ==="

# Override STUDENT_TYPES to include full STU
python -c "
import sys, os
sys.path.insert(0, '.')
os.environ['STUDENT_TYPES'] = 'stu,stu_t,gru,lstm,tiny_transformer'

# Monkey-patch the student types
import scripts.synthetic_experiment as se
se.STUDENT_TYPES = ['stu', 'stu_t', 'gru', 'lstm', 'tiny_transformer']

# Run with just CopyTask
import argparse
args = argparse.Namespace(
    tasks='copy',
    difficulties=None,
    output_dir='experiments/full_stu_synthetic',
    cpu=False,
    quick=False,
)
se.run_full_experiment(args)
" 2>&1 || echo "FAILED"

echo "=== Done: $(date) ==="
