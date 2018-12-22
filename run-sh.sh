#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=training
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:v100:1
# k80, p1080, p40, p100 and v100
#SBATCH --mail-type=end  # email me when the job ends

# Change the home directory
cd ~/ComputerVision/edison/
# module load pytorch/python3.6/0.3.0_4
wandb run python3 -u main.py --batch-size 128 
