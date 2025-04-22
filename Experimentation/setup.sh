#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-48:00:00
#SBATCH -p gpu
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

CUDA_VISIBLE_DEVICES=0 python production.py
