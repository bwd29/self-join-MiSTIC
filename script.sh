#!/bin/bash
#SBATCH --job-name=treeMSD 
#SBATCH --output=/home/bwd29/self-join/results/results5.out
#SBATCH --error=/home/bwd29/self-join/results/error5.err
#SBATCH --time=3000:00
#SBATCH --mem=0
#SBATCH -c 64
#SBATCH -G 3
#SBATCH --partition=gowanlock
#SBATCH --account=gowanlock_condo
#SBATCH -w cn2


module load cuda

make clean
make

echo "1024x1024 launches, 100 x 0.01 sampling, 32 per layer, 6 rps, non-rand rp, 200,000 calcs per thread"

echo "MSD ________________________________________________________________"
echo "MSD ________________________________________________________________"
echo "MSD ________________________________________________________________"

for j in 0.007  0.007525 0.00805 0.008575 0.0091  
do
     ./build/main /scratch/bwd29/data/MSD.bin 90 $j
done



echo "Census____________________________________________________________"
echo "Census____________________________________________________________"
echo "Census____________________________________________________________"

for j in 0.001 0.00325 0.0055 0.00775 0.01
do
        ./build/main /scratch/bwd29/data/USC_Normalized_No1Col.bin 68 $j
done

echo "Wave __________________________________________________"
echo "Wave __________________________________________________"
echo "Wave __________________________________________________"
for j in 0.002 0.00325 0.0045 0.00575 0.007
do
        ./build/main /scratch/bwd29/data/WAVE_Normalized.bin 49 $j
done

# echo "BIGCROSS____________________________________________________________"
# echo "BIGCROSS____________________________________________________________"
# echo "BIGCROSS____________________________________________________________"
# # for j in 0.001 0.00575 0.0105 0.01525 
# for j in 0.02
# do
#         ./build/main /scratch/bwd29/data/BIGCROSS_Normalized.bin 57 $j
# done

echo "SUSY ________________________________________________________________"
echo "SUSY ________________________________________________________________"
echo "SUSY ________________________________________________________________"

# for j in 0.01825 
for j in 0.01 0.01275 0.0155 0.01825 0.021  
do
    ./build/main /scratch/bwd29/data/SUSY_Normalized.bin 18 $j
done

echo "HIGGS ________________________________________________________________"
echo "HIGGS ________________________________________________________________"
echo "HIGGS ________________________________________________________________"
for j in 0.01 0.021375 0.03275 0.044125 0.0555
do
        ./build/main /scratch/bwd29/data/HIGSS_Normalized.bin 28 $j
done

echo "Tiny____________________________________________________"
echo "Tiny____________________________________________________"
echo "Tiny____________________________________________________"
for j in 0.2 0.26 0.32 0.38 0.44
do
        ./build/main /scratch/bwd29/data/TINY_Normalized.bin 384 $j
done

echo "Completed!"