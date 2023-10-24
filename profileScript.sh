#!/bin/bash
#SBATCH --job-name=nodeStatsProf
#SBATCH --output=/home/bwd29/self-join/results/profBINCOTSS2.out
#SBATCH --error=/home/bwd29/self-join/results/profBINCOTSS2.err
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
TPP=1 RP=6 BS=256 KB=1024 DIM=57 ILP=4 make

for j in 0.0281 
do
ncu --set full -c 1 -s 1 --clock-control none -f -o profileBIGCROSS3.out -k searchKernelCOSStree ./build/main /scratch/bwd29/data/BIGCROSS_Normalized.bin 57 7 $j
done
# ncu --set full -o profileHiggs.out -k nodeByPoint5 ./build/main /scratch/bwd29/data/HIGSS_Normalized.bin 90 0.021375

echo "Completed!"