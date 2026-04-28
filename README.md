# What Matters for Policy Distillation in POMDPs

**Teacher Quality, Memory Mechanism, and Architecture Choice**

Mayan Wasu · Princeton ECE · Spring 2026
Advisor: Professor Hossein Valavi

## Overview

This repo contains the code, experiments, and writeup for my Independent Work on policy distillation under partial observability. We compare GRU, LSTM, Transformer, and STU (spectral) architectures as student models, trained via behavioral cloning on demonstrations from RecurrentPPO teachers of varying quality.

**Key finding:** Teacher quality explains 49-72% of student return variance; architecture choice explains only 5-14%. But architecture matters more when teachers are bad (17-25% interaction effect), and the best architecture depends on the task: GRU for locomotion, Transformer for recall, STU/Transformer for spatially-structured driving.

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
