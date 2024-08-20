#!/bin/bash
#SBATCH --job-name=treeTest
#SBATCH --output=/home/bwd29/self-join/results/Bee.out
#SBATCH --error=/home/bwd29/self-join/results/Bee.err
#SBATCH --time=30:00
#SBATCH --mem=0
#SBATCH -c 64 
#SBATCH -G 3
#SBATCH --partition=gowanlock
#SBATCH --account=gowanlock_condo
#SBATCH -w cn2


module load cuda


make clean
TPP=1 RP=6 BS=256 KB=1024 DIM=7490 ILP=8 make

./build/main bee_dataset_1D_feature_vectors.txt 7490 7 10000.0

