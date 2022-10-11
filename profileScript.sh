#!/bin/bash
#SBATCH --job-name=nodeStatsProf
#SBATCH --output=/home/bwd29/self-join/results/results5prof.out
#SBATCH --error=/home/bwd29/self-join/results/error5prof.err
#SBATCH --time=1000:00
#SBATCH --mem=0
#SBATCH -c 64
#SBATCH -G 4
##SBATCH --partition=gowanlock
##SBATCH --account=gowanlock_condo
#SBATCH -w cn3


module load cuda

make clean
MCPN=512 make

echo "1024x1024*2 launches, sqrt(N) x 0.01 sampling, 32 per layer, k rps, non-rand RP, dynamic calcs per thread max 250000, 30 registers"

       
        echo "SUSY ________________________________________________________________"
        echo "SUSY ________________________________________________________________"
        echo "SUSY ________________________________________________________________"

        ncu --set full -o profileMSD.out -k nodeCalculationsKernel -c 4  ./build/main /scratch/bwd29/data/MSD.bin 90 0.0091

done
echo "Completed!"