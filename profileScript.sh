#!/bin/bash
#SBATCH --job-name=nodeStatsProf
#SBATCH --output=/home/bwd29/self-join/results/profMISTIC2.out
#SBATCH --error=/home/bwd29/self-join/results/profBINMISTIC2.err
#SBATCH --time=2000:00
#SBATCH --mem=0
#SBATCH -c 64
#SBATCH -G 4
# #SBATCH --partition=gowanlock
# #SBATCH --account=gowanlock_condo
#SBATCH -w cn3


module load cuda/11.2

# make clean
# DIM=384 KT=1 make

# ncu --set full -o profileTINYK1.out -k nodeCalculationsKernel -c 1  ./build/main /scratch/bwd29/data/TINY_Normalized.bin 384 0.2

make clean
BS=256 KB=1024 DIM=18 make

for j in 0.01703 
do
ncu --set full -c 1 -s 1 --clock-control none -f -o profileSUSYNODE.out -k nodeByPoint5 ./build/main /scratch/bwd29/data/SUSY_Normalized.bin 18 7 $j
done
# ncu --set full -o profileHiggs.out -k nodeByPoint5 ./build/main /scratch/bwd29/data/HIGSS_Normalized.bin 90 0.021375

echo "Completed!"