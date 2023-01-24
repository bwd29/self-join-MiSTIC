#!/bin/bash
#SBATCH --job-name=nodeStats
#SBATCH --output=/home/bwd29/self-join/results/finalNEW_BS_CHECK2.out
#SBATCH --error=/home/bwd29/self-join/results/finalNEW2_BS_CHECK2.err
#SBATCH --time=3000:00
#SBATCH --mem=0
#SBATCH -c 64 
#SBATCH -G 3
#SBATCH --partition=gowanlock
#SBATCH --account=gowanlock_condo
#SBATCH -w cn2


module load cuda

# make clean

# CALC_MULTI=4 MIN_NODE_SIZE=1000 make

# echo -e "\nEPSILON | NUMPOINTS | MIN_NODE_SIZE | CALC_MULTI | NUMSUBS | LSUB | RP | NODES | CALCS | Construct T | Total T\n" >&2
echo -e "\nEPSILON | NUMPOINTS | RP | NODES | CALCS | Construct T | Total T\n" >&2


        # for i in 1
        # for i in 1024 512 256 128 64
        for i in 64 128 256 512 #1 2 4 8 16 32
        do

                make clean
                KT=3 BS=512 ORDP=$i make
                
                echo "MSD ________________________________________________________________"
                echo "MSD ________________________________________________________________"
                echo -e "\n\nMSD ________________________________________________________________\n" >&2

                for j in 0.007 0.007525 0.00805 0.008575 0.0091  
                do
                ./build/main /scratch/bwd29/data/MSD.bin 90 $j
                done

    
                
                echo "UNI ________________________________________________________________"
                echo "UNI ________________________________________________________________"
                echo -e "\n\nUNI ________________________________________________________________\n" >&2

                for j in 0.25 0.3 0.35 0.4 0.45
                do
                ./build/main /scratch/bwd29/data/ndim_10_2mil.bin 10 $j
                done

           

                echo "EXPO ________________________________________________________________"
                echo "EXPO ________________________________________________________________"
                echo -e "\n\nEXPO ________________________________________________________________\n" >&2 

                for j in 0.03 0.03375 0.0375 0.04125 0.045
                do
                ./build/main /scratch/bwd29/data/2mil16dexpo.bin 16 $j
                done



                echo "Census____________________________________________________________"
                echo "Census____________________________________________________________"
                echo -e "\n\nCensus____________________________________________________________\n" >&2

                for j in 0.001 0.00325 0.0055 0.00775 0.01
                do
                        ./build/main /scratch/bwd29/data/USC_Normalized_No1Col.bin 68 $j
                done

    
                echo "Wave __________________________________________________"
                echo "Wave __________________________________________________"
                echo -e "\n\nWave __________________________________________________\n" >&2
                for j in 0.002 0.00325 0.0045 0.00575 0.007
                do
                        ./build/main /scratch/bwd29/data/WAVE_Normalized.bin 49 $j
                done


                echo "BIGCROSS____________________________________________________________"
                echo "BIGCROSS____________________________________________________________"
                echo -e "\n\nBIGCROSS____________________________________________________________\n" >&2
                for j in 0.001 0.00575 0.0105 0.01525 0.02
                do
                        ./build/main /scratch/bwd29/data/BIGCROSS_Normalized.bin 57 $j
                done


                echo "SUSY ________________________________________________________________"
                echo "SUSY ________________________________________________________________"
                echo -e "\n\nSUSY ________________________________________________________________\n" >&2

                # for j in 0.01825 
                for j in 0.01 0.01275 0.0155 0.01825 0.021  
                do
                ./build/main /scratch/bwd29/data/SUSY_Normalized.bin 18 $j
                done


                echo "HIGGS ________________________________________________________________"
                echo "HIGGS ________________________________________________________________"
                echo -e "\n\nHIGGS ________________________________________________________________\n" >&2
                for j in 0.01 0.021375 0.03275 0.044125 0.0555
                do
                        ./build/main /scratch/bwd29/data/HIGSS_Normalized.bin 28 $j
                done


                echo "Tiny____________________________________________________"
                echo "Tiny____________________________________________________"
                echo -e "\nTiny____________________________________________________\n" >&2
                for j in 0.2 0.26 0.32 0.38 0.44
                do
                        ./build/main /scratch/bwd29/data/TINY_Normalized.bin 384 $j
                done
        done

echo "Completed!"