#!/bin/bash
#SBATCH --job-name=graphStats
#SBATCH --output=/home/bwd29/self-join/results/results1.out
#SBATCH --error=/home/bwd29/self-join/results/error1.err
#SBATCH --time=1000:00
#SBATCH --mem=0
#SBATCH -c 64
#SBATCH -G 3
#SBATCH --partition=gowanlock
#SBATCH --account=gowanlock_condo
#SBATCH -w cn2


module load cuda

make clean

make

echo "1024x1024*2 launches, sqrt(N) x 0.01 sampling, 64 per layer, k rps, non-rand RP, dynamic calcs per thread max 250000, 30 registers"

for k in 6
do
        echo "MSD ________________________________________________________________"
        echo "MSD ________________________________________________________________"
        echo "MSD ________________________________________________________________"

        for j in 0.007 0.007525 0.00805 0.008575 0.0091  
        do
        ./build/main /scratch/bwd29/data/MSD.bin 90 $k $j
        done

        # echo "UNI ________________________________________________________________"
        # echo "UNI ________________________________________________________________"
        # echo "UNI ________________________________________________________________"

        # for j in 0.25 0.3 0.35 0.4 0.45
        # do
        # ./build/main /scratch/bwd29/data/ndim_10_2mil.bin 10 $k $j
        # done

        # echo "EXPO ________________________________________________________________"
        # echo "EXPO ________________________________________________________________"
        # echo "EXPO ________________________________________________________________"

        # for j in 0.03 0.035 0.04 0.045 0.05
        # do
        # ./build/main /scratch/bwd29/data/2mil16dexpo.bin 16 $k $j
        # done


        # echo "Census____________________________________________________________"
        # echo "Census____________________________________________________________"
        # echo "Census____________________________________________________________"

        # for j in 0.001 0.00325 0.0055 0.00775 0.01
        # do
        #         ./build/main /scratch/bwd29/data/USC_Normalized_No1Col.bin 68  $k $j
        # done

        # echo "Wave __________________________________________________"
        # echo "Wave __________________________________________________"
        # echo "Wave __________________________________________________"
        # for j in 0.002 0.00325 0.0045 0.00575 0.007
        # do
        #         ./build/main /scratch/bwd29/data/WAVE_Normalized.bin 49 $k $j
        # done

        # echo "BIGCROSS____________________________________________________________"
        # echo "BIGCROSS____________________________________________________________"
        # echo "BIGCROSS____________________________________________________________"
        # for j in 0.001 0.00575 0.0105 0.01525 
        # do
        #         ./build/main /scratch/bwd29/data/BIGCROSS_Normalized.bin 57 $k $j
        # done

        # echo "SUSY ________________________________________________________________"
        # echo "SUSY ________________________________________________________________"
        # echo "SUSY ________________________________________________________________"

        # # for j in 0.01825 
        # for j in 0.01 0.01275 0.0155 0.01825 0.021  
        # do
        # ./build/main /scratch/bwd29/data/SUSY_Normalized.bin 18 $k $j
        # done

        # echo "HIGGS ________________________________________________________________"
        # echo "HIGGS ________________________________________________________________"
        # echo "HIGGS ________________________________________________________________"
        # for j in 0.01 0.021375 0.03275 0.044125 0.0555
        # do
        #         ./build/main /scratch/bwd29/data/HIGSS_Normalized.bin 28 $k $j
        # done

        # echo "Tiny____________________________________________________"
        # echo "Tiny____________________________________________________"
        # echo "Tiny____________________________________________________"
        # for j in 0.2 0.26 0.32 0.38 0.44
        # do
        #         ./build/main /scratch/bwd29/data/TINY_Normalized.bin 384 $k $j
        # done
done
echo "Completed!"