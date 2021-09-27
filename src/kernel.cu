#include "include/kernel.cuh"


void launchKernel(int * addIndexes, int ** rangeIndexes, int ** rangeSizes, int * numValidRanges, unsigned long long *calcPerAdd, int nonEmptyBins, unsigned long long sumCalcs, unsigned long long sumAdds){
 
    unsigned int calcsPerThread = 1000000; //placeholdr value of 1 mil

    int * numThreadsPerAddress = (int *)malloc(sizeof(int)*nonEmptyBins);
    for(int i = 0; i < nonEmptyBins; i++){
        numThreadsPerAddress[i] = ceil(calcPerAdd[i] / calcsPerThread);
    }

    unsigned int threadsPerBatch = KERNEL_BLOCKS * BLOCK_SIZE;

    int currentAdd = 0;
    while(currentAdd < nonEmptyBins){

        unsigned long long sum = 0;
        unsigned int numAdds = 0;
        do{
            sum += calcPerAdd[currentAdd];
            currentAdd++;
            numAdds++;
        }while(sum < calcsPerThread*threadsPerBatch && currentAdd < nonEmptyBins);

        distanceCalculations(currentAdd, numAdds);

        

    }



}


__device__ 
void distanceCalculations(double * data, int startAdd, int numAdds, int batchSize, int batchNum, int *addIndexes, int * numValidRanges, int ** rangeIndexes, int ** rangeSizes, unsigned int * numPointsInAdd, int * addIndexRange){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;


    int currentAdd = tid/numThreadsPerAddress[currentAdd]; //check math on this
    int currentRange = 0;

    for(int i = 0; i < numValidRanges[currentAdd]; i++){
        for(int j = 0; j < rangeSizes[currentAdd][i] * numPointsInAdd[currentAdd]; j += numThreadsPerAddress[currentAdd]){
            unsigned int p1 = pointArray[addIndexRange[currentAdd] + j/rangeSizes[currentAdd][i]];
            unsigned int p2 = pointArray[rangeIndexes[currentAdd][i] + j % rangeSizes[currentAdd][i]];
            if (distanceCheck(epsilon2, dim, &data[p1], &data[p2])){
                //store point
            }
        }
    }
}

__device__ //may need to switch to inline
bool distanceCheck(double epsilon2, double dim, double * p1, double * p2){
    double sum = 0;
    for(int i = 0; i < dim; i++){
        sum += pow(p1[i]-p2[i],2);
        if(sum >= epsilon2) return false;
    }

    return true;
}