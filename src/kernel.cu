#include "include/kernel.cuh"


void launchKernel(int * addIndexes, int ** rangeIndexes, int ** rangeSizes, int * numValidRanges, unsigned long long *calcPerAdd, int nonEmptyBins, unsigned long long sumCalcs, unsigned long long sumAdds){

    int batchSizes = 1000000; //placeholdr value of 1 mil
    int numbatches = ceil(sumCalcs/batchSizes); //placeholder value 

    unsigned int calcsPerThread = 1000000; //placeholdr value of 1 mil

    int * numThreadsPerAddress = (int *)malloc(sizeof(int)*nonEmptyBins);
    for(int i = 0; i < nonEmptyBins; i++){
        
    }

    for(int i = 0; i < numbatches; i++){

    }





}


__device__ 
void distanceCalculations(int startAdd, int batchSize, int batchNum, int *addIndexes, int * numValidRanges, int ** rangeIndexes, int ** rangeSizes, int * numPointsInAdd){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    // num of threads assigned to address is batchSize / threads per batch
    int currentAdd = tid/tpa;
    int currentRange = 0;



    for(int i = 0; i < numValidRanges[currentAdd]; i++){
        for(int j = 0; j < rangeSizes[currentAdd][currentRange]; j++){
            for(int k = 0; k < rangeSizes[currentAdd][currentRange] * numPointsInAdd[currentAdd]; k++){
                
            }
        }
    }
}