# What Matters for Policy Distillation in POMDPs

**Teacher Quality, Memory Mechanism, and Architecture Choice**

Mayan Wasu · Princeton ECE · Spring 2026
Advisor: Professor Hossein Valavi

## Overview

This repo contains the code, experiments, and writeup for my Independent Work on policy distillation under partial observability. We compare GRU, LSTM, Transformer, and STU (spectral) architectures as student models, trained via behavioral cloning on demonstrations from RecurrentPPO teachers of varying quality.

**Key finding:** Teacher quality explains 49-72% of student return variance; architecture choice explains only 5-14%. But architecture matters more when teachers are bad (17-25% interaction effect), and the best architecture depends on the task: GRU for locomotion, Transformer for recall, STU/Transformer for spatially-structured driving.

## Repo structure

```
wasumindV2/
  distillation/     # Student training (BC) and model definitions
  teachers/          # RecurrentPPO teacher training and demo collection
  envs/              # POMDP wrappers (velocity masking) and MetaDrive setup
  evaluation/        # Evaluation, analysis, and figure generation
  scripts/           # SLURM job scripts for Princeton HPC (Della/Adroit)
writeup/             # LaTeX source and figures
poster/              # HTML poster and PDF export
```

## Quick start

```bash
pip install torch gymnasium[mujoco] mujoco sb3-contrib stable-baselines3

# Train a teacher
python wasumindV2/teachers/train_teacher.py --env HalfCheetah-v4 --seed 42

# Collect demos
python wasumindV2/teachers/collect_demos.py \
  --teacher-path experiments/v2/teachers/halfcheetahv4_seed42/teacher_1000000 \
  --env halfcheetah_pomdp --n-episodes 500

# Distill a student
python wasumindV2/distillation/distill.py \
  --demo-path experiments/v2/demos/halfcheetah_pomdp/teacher_1000000_noise00/demos.pt \
  --arch gru --seed 42
```

## Environments

- **HalfCheetah-POMDP** and **Ant-POMDP**: MuJoCo locomotion with velocity channels removed
- **MetaDrive**: Autonomous driving with 72-laser lidar (91-dim observations)
- **T-Maze** and **CopyTask**: Synthetic benchmarks for retention and recall

## Citation

```
Mayan Wasu. "What Matters for Policy Distillation in POMDPs: Teacher Quality,
Memory Mechanism, and Architecture Choice." Princeton University ECE Independent
Work, Spring 2026.
```
