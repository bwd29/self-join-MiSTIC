#include "include/kernel.cuh"


void launchKernel(double epsilon, int * addIndexes, int ** rangeIndexes, int ** rangeSizes, int * numValidRanges, unsigned long long *calcPerAdd, int nonEmptyBins, unsigned long long sumCalcs, unsigned long long sumAdds){
 

    double epsilon2 = epsilon*epsilon;
    unsigned int calcsPerThread = 1000000; //placeholdr value of 1 mil

    int * numThreadsPerAddress = (int *)malloc(sizeof(int)*nonEmptyBins);


    int numBatches = 1;
    unsigned int threadsPerBatch = KERNEL_BLOCKS * BLOCK_SIZE;
    unsigned long long sum = 0;


    for(int i = 0; i < nonEmptyBins; i++){
        numThreadsPerAddress[i] = ceil(calcPerAdd[i] / calcsPerThread);
        if (sum + calcPerAdd[i] < calcsPerThread*threadsPerBatch || sum == 0){
            sum += calcPerAdd[i];
        }else{
            sum = calcPerAdd[i];
            numBatches++;
        }
    }

    unsigned long long * numCalcsPerBatch = (unsigned long long)malloc(sizeof(unsigned long long)*numBatches);
    unsigned int * numAddPerBatch = (unsigned int*)malloc(sizeof(unsigned int)*numBatches);
    unsigned int * numThreadsPerBatch = (unsigned int*)malloc(sizeof(unsigned int)*numBatches);
    sum = 0;
    int batchCount = 0;
    int addCount = 0;
    for(int i = 0; i < nonEmptyBins; i++){
        if (sum + calcPerAdd[i] < calcsPerThread*threadsPerBatch || sum == 0){
            sum += calcPerAdd[i];
            addCount++;
        }else{
            numCalcsPerBatch[count] = sum;
            numAddPerBatch[batchCount] = addCount;
            
            sum = calcPerAdd[i];

            addCount = 1;
            batchCount++;
        }

        numThreadsPerBatch[batchCount] += numThreadsPerAddress[i];

    }

    numCalcsPerBatch[numBatches-1] = sum; //for last
    numAddPerBatch[numBatches-1] = addCount;

    for(int i = 0; i < numBatches; i++){

        //launch distance kernel
        distanceCalculationsKernel<<<KERNEL_BLOCKS, BLOCK_SIZE>>>();

        //transfer back reuslts

    }



}


__device__ 
void distanceCalculationsKernel(int numThreadsPerBatch, double * data, int *addIndexes, int * numValidRanges, int ** rangeIndexes, int ** rangeSizes, unsigned int * numPointsInAdd, int * addIndexRange, unsigned long long *keyValueIndex, unsigned int * point_a, unsigned int * point_b){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if(tid > numThreadsPerBatch){
        return;
    }

    int currentAdd = tid/numThreadsPerAddress[currentAdd]; //check math on this

    for(int i = 0; i < numValidRanges[currentAdd]; i++){
        for(int j = 0; j < rangeSizes[currentAdd][i] * numPointsInAdd[currentAdd]; j += numThreadsPerAddress[currentAdd]){
            unsigned int p1 = pointArray[addIndexRange[currentAdd] + j/rangeSizes[currentAdd][i]];
            unsigned int p2 = pointArray[rangeIndexes[currentAdd][i] + j % rangeSizes[currentAdd][i]];
            if (distanceCheck(epsilon2, dim, &data[p1], &data[p2])){
                //store point
                unsigned int index = atomicAdd(key_value_index,(unsigned int)1);
                point_a[index] = p1; //stores the first point Number
                point_b[index] = p2; // this store the cooresponding point number to form a pair
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