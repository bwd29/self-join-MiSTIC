#!/bin/bash
#SBATCH --job-name=nodeStatsProf
#SBATCH --output=/home/bwd29/self-join/results/results5prof.out
#SBATCH --error=/home/bwd29/self-join/results/error5prof.err
#SBATCH --time=1000:00
#SBATCH --mem=0
#SBATCH -c 64
#SBATCH -G 3
#SBATCH --partition=gowanlock
#SBATCH --account=gowanlock_condo
#SBATCH -w cn2


module load cuda

# make clean
# DIM=384 KT=1 make

# ncu --set full -o profileTINYK1.out -k nodeCalculationsKernel -c 1  ./build/main /scratch/bwd29/data/TINY_Normalized.bin 384 0.2

make clean
ORDP=32 KT=3 BS=1024 make
ncu --set full -o profileTINYK3-1.out -k nodeByPoint2 -c 1  ./build/main /scratch/bwd29/data/TINY_Normalized.bin 384 0.2


echo "Completed!"