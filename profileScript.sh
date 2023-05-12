#!/bin/bash
#SBATCH --job-name=nodeStatsProf
#SBATCH --output=/home/bwd29/self-join/results/prof.out
#SBATCH --error=/home/bwd29/self-join/results/prof.err
#SBATCH --time=1000:00
#SBATCH --mem=0
#SBATCH -c 64
#SBATCH -G 4
##SBATCH --partition=gowanlock
##SBATCH --account=gowanlock_condo
#SBATCH -w cn3


module load cuda

# make clean
# DIM=384 KT=1 make

# ncu --set full -o profileTINYK1.out -k nodeCalculationsKernel -c 1  ./build/main /scratch/bwd29/data/TINY_Normalized.bin 384 0.2

make clean
BS=256 KB=1024 make
cuda-memcheck ./build/main /scratch/bwd29/data/MSD.bin 90 0.091
ncu --set full -o profileMSDKT23.out -k nodeByPoint5 ./build/main /scratch/bwd29/data/MSD.bin 90 0.091


echo "Completed!"