#!/bin/bash

#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G  # NOTE DO NOT USE THE --mem= OPTION




#SBATCH --time=0-36:00:00
#SBATCH --account=ssd
#SBATCH --partition=ssd-gpu
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G  # NOTE DO NOT USE THE --mem= OPTION

