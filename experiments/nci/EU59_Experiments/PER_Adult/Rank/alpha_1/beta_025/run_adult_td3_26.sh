#!/bin/bash
#PBS -P eu59
#PBS -q gpuvolta
#PBS -l walltime=48:00:00
#PBS -l mem=24GB
#PBS -l jobfs=0
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -M david.timms@anu.edu.au
#PBS -l storage=scratch/sj53
#PBS -o out_TD326.txt
#PBS -e err_TD326.txt
#PBS -l software=python
module load python3/3.9.2
module load pytorch/1.9.0

python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id PER_Study/Per_Rank/TD3/alpha_1/beta_025/TD326_1 --patient_id 26 --replay_buffer_type per_rank --replay_buffer_alpha 1.0 --replay_buffer_beta 0.25  --seed 1 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id PER_Study/Per_Rank/TD3/alpha_1/beta_025/TD326_2 --patient_id 26 --replay_buffer_type per_rank --replay_buffer_alpha 1.0 --replay_buffer_beta 0.25  --seed 2 --debug 0 &
python3 /scratch/sj53/dt9478/G2P2C/experiments/run_RL_agent.py --agent td3 --folder_id PER_Study/Per_Rank/TD3/alpha_1/beta_025/TD326_3 --patient_id 26 --replay_buffer_type per_rank --replay_buffer_alpha 1.0 --replay_buffer_beta 0.25  --seed 3 --debug 0
wait
