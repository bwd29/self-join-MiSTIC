#!/bin/bash
#SBATCH --job-name=treeTest
#SBATCH --output=/home/bwd29/self-join/results/NEWNODES-3.out
#SBATCH --error=/home/bwd29/self-join/results/NEWNODES-3.err
#SBATCH --time=300:00
#SBATCH --mem=0
#SBATCH -c 64 
#SBATCH -G 3
#SBATCH --partition=gowanlock
#SBATCH --account=gowanlock_condo
#SBATCH -w cn2


module load cuda

# make clean
# make

# CALC_MULTI=4 MIN_NODE_SIZE=1000 make

# echo -e "\nEPSILON | NUMPOINTS | MIN_NODE_SIZE | CALC_MULTI | NUMSUBS | LSUB | RP | NODES | CALCS | Construct T | Total T\n" >&2
# echo -e "\nEPSILON | NUMPOINTS | RP | NODES | CALCS | Construct T | Kernel T | Total T\n" >&2
echo -e "\nEPSILON | NUMPOINTS | MIN | MAX | SSE | STDEV | RP | Construct T | Kernel T | Total T\n" >&2


                for i in 256
                do
                        for t in 1024
                        do
                                for k in 1
                                do
                                        make clean
                                        BS=$i KB=$t DIM=90 make
                                        # make
                                        echo -e "\n\n _______________ $i _____ $t ______$k ______________________\n" >&2
                                        
                                        echo "MSD ________________________________________________________________"
                                        echo "MSD ________________________________________________________________"
                                        echo -e "\n\nMSD ________________________________________________________________\n" >&2

                                        for j in  0.0076 0.00913 0.011334 # 0.007 0.007525 0.00805 0.008575 0.0091 
                                        do
                                        ./build/main /scratch/bwd29/data/MSD.bin 90 7 $j
                                        done

                        
                                        # make clean
                                        # BS=$i KB=$t DIM=10 ILP=$l make

                                        # echo "UNI ________________________________________________________________"
                                        # echo "UNI ________________________________________________________________"
                                        # echo -e "\n\nUNI ________________________________________________________________\n" >&2

                                        # for j in 0.25 0.3 0.35 0.4 0.45
                                        # do
                                        # ./build/main /scratch/bwd29/data/ndim_10_2mil.bin 10 $j
                                        # done

                                
                                        # make clean
                                        # BS=$i KB=$t DIM=16 ILP=$l make

                                        # echo "EXPO ________________________________________________________________"
                                        # echo "EXPO ________________________________________________________________"
                                        # echo -e "\n\nEXPO ________________________________________________________________\n" >&2 

                                        # for j in 0.03 0.03375 0.0375 0.04125 0.045
                                        # do
                                        # ./build/main /scratch/bwd29/data/2mil16dexpo.bin 16 $j
                                        # done


                                        # make clean
                                        # RP=$l BS=$i KB=$t DIM=68 ILP=4 make

                                        # echo "Census____________________________________________________________"
                                        # echo "Census____________________________________________________________"
                                        # echo -e "\n\nCensus____________________________________________________________\n" >&2

                                        # for j in 0.02 0.023 0.026 #.001 0.00325 0.0055 0.00775 0.01 
                                        # do
                                        #         ./build/main /scratch/bwd29/data/USC_Normalized_No1Col.bin 68 6 $j
                                        # done

                                        make clean
                                        BS=$i KB=$t DIM=49 make

                                        echo "Wave __________________________________________________"
                                        echo "Wave __________________________________________________"
                                        echo -e "\n\nWave __________________________________________________\n" >&2
                                        for j in   0.0054 0.00702 0.008358 #0.002 0.00325 0.0045 0.00575 0.007
                                        do
                                                ./build/main /scratch/bwd29/data/WAVE_Normalized.bin 49 7 $j
                                        done

                                        make clean
                                        BS=$i KB=$t DIM=57 make
                                        echo "BIGCROSS____________________________________________________________"
                                        echo "BIGCROSS____________________________________________________________"
                                        echo -e "\n\nBIGCROSS____________________________________________________________\n" >&2
                                        for j in 0.0131 0.01994 0.0281 # 0.001 0.00575 0.0105 0.01525 0.02
                                        do
                                                ./build/main /scratch/bwd29/data/BIGCROSS_Normalized.bin 57 7 $j
                                        done

                                        make clean
                                        BS=$i KB=$t DIM=18 make
                                        # BS=256 KB=1024 DIM=18 ILP=4 make
                                        echo "SUSY ________________________________________________________________"
                                        echo "SUSY ________________________________________________________________"
                                        echo -e "\n\nSUSY ________________________________________________________________\n" >&2

                                        # for j in 0.01825 
                                        for j in 0.01703 0.02078 0.025555 #0.01 0.01275 0.0155 0.01825 0.021
                                        do
                                        ./build/main /scratch/bwd29/data/SUSY_Normalized.bin 18 7 $j
                                        done

                                        # make clean
                                        # BS=$i KB=$t DIM=28 make
                                        # # TPP=$k RP=$l BS=$i KB=$t DIM=28 ILP=8 make
                                        # echo "HIGGS ________________________________________________________________"
                                        # echo "HIGGS ________________________________________________________________"
                                        # echo -e "\n\nHIGGS ________________________________________________________________\n" >&2
                                        # for j in  0.049186 0.05558 0.063117 #0.01 0.021375 0.03275 0.044125 0.0555
                                        # do
                                        #         ./build/main /scratch/bwd29/data/HIGSS_Normalized.bin 28 7 $j
                                        # done

                                        # make clean
                                        # RP=$l BS=128 KB=1024 DIM=384 ILP=4 make
                                        # echo "Tiny____________________________________________________"
                                        # echo "Tiny____________________________________________________"
                                        # echo -e "\nTiny____________________________________________________\n" >&2
                                        # for j in 0.2 0.26 0.32 0.38 0.44
                                        # do
                                        #         ./build/main /scratch/bwd29/data/TINY_Normalized.bin 384 7 $j
                                        # done
                                done
                        done    
                done


echo "Completed!"