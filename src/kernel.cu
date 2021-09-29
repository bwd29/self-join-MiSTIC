#include "include/kernel.cuh"


void launchKernel(double * data, int dim, int numPoints, double epsilon, int * addIndexes, int * pointArray, int ** rangeIndexes, unsigned int ** rangeSizes, int * numValidRanges, unsigned int * numPointsInAdd, unsigned long long *calcPerAdd, int nonEmptyBins, unsigned long long sumCalcs, unsigned long long sumAdds){
 
    double epsilon2 = epsilon*epsilon;
    unsigned long long calcsPerThread = 100000; //placeholder value of 1 mil

    unsigned long long * numThreadsPerAddress = (unsigned long long *)malloc(sizeof(unsigned long long)*nonEmptyBins);

    int numBatches = 0;
    unsigned long long threadsPerBatch = KERNEL_BLOCKS * BLOCK_SIZE;
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
    // if(numBatches == 0) numBatches =1;
    numBatches++;

    // printf("working with %d batches\n ",numBatches);

    
    unsigned long long * numCalcsPerBatch = (unsigned long long*)malloc(sizeof(unsigned long long)*numBatches);
    unsigned int * numAddPerBatch = (unsigned int*)malloc(sizeof(unsigned int)*numBatches);
    unsigned long long * numThreadsPerBatch = (unsigned long long*)calloc(numBatches,sizeof(unsigned long long));
    sum = 0;
    int batchCount = 0;
    int addCount = 0;
    for(int i = 0; i < nonEmptyBins; i++){
        // if(batchCount > numBatches-1) printf("batch count too high!\n");
        numThreadsPerBatch[batchCount] += numThreadsPerAddress[i];
        
        if (sum + calcPerAdd[i] < calcsPerThread*threadsPerBatch || sum == 0){
            sum += calcPerAdd[i];
            addCount++;
        }else{
            numCalcsPerBatch[batchCount] = sum;
            numAddPerBatch[batchCount] = addCount;

            sum = calcPerAdd[i];

            addCount = 1;
           
            batchCount++;
            // printf("current batch %d\n", batchCount);

        }

        

    }







    numCalcsPerBatch[numBatches-1] = sum; //for last
    numAddPerBatch[numBatches-1] = addCount;

    for(int i = 0; i < numBatches; i++){

        const double d_epsilon2 = epsilon2;
        const int d_dim = dim;
        const int d_numThreadsPerBatch = numThreadsPerBatch[i];

        //compute which thread does wich add
        int * addAssign = (int * )malloc(sizeof(int)*numThreadsPerBatch[i]);
        int * threadOffsets = (int*)malloc(sizeof(int)*numThreadsPerBatch[i]);
        unsigned int currentAdd = 0;
        unsigned int offsetCount = 0;

        for(unsigned int j = 0; j < numThreadsPerBatch[i]; j++){
            // if(currentAdd > nonEmptyBins) printf("current add is to large!");
            if ( offsetCount > numThreadsPerAddress[currentAdd]){
                currentAdd++;
                offsetCount = 0;
            }
            addAssign[j] = currentAdd;
            threadOffsets[j] = offsetCount;
            offsetCount++;
        }


        printf("BatchNumber: %d/%d, Calcs: %llu, Adds: %d, threads: %llu\n", i+1, numBatches, numCalcsPerBatch[i], numAddPerBatch[i], numThreadsPerBatch[i]);
        //launch distance kernel
        // distanceCalculationsKernel<<<KERNEL_BLOCKS, BLOCK_SIZE>>>();

        //transfer back reuslts

        free(addAssign);
        free(threadOffsets);
        
    }

    free(numCalcsPerBatch);
    free(numAddPerBatch);
    free(numThreadsPerBatch);
    free(numThreadsPerAddress);

}

__device__ 
void distanceCalculationsKernel(int * addAssign, int * threadOffsets, const double epsilon2, const int dim, const int numThreadsPerBatch, int * numThreadsPerAddress, double * data, int *addIndexes, int * numValidRanges, int ** rangeIndexes, unsigned int ** rangeSizes, unsigned int * numPointsInAdd, int * addIndexRange, int * pointArray, unsigned long long *keyValueIndex, unsigned int * point_a, unsigned int * point_b){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if(tid > numThreadsPerBatch){
        return;
    }

    int currentAdd = addAssign[tid];
    int threadOffset = threadOffsets[tid];

    for(int i = 0; i < numValidRanges[currentAdd]; i++){
        for(int j = 0; j < rangeSizes[currentAdd][i] * numPointsInAdd[currentAdd] + threadOffset; j += numThreadsPerAddress[currentAdd]){
            unsigned int p1 = pointArray[addIndexRange[currentAdd] + j/rangeSizes[currentAdd][i]];
            unsigned int p2 = pointArray[rangeIndexes[currentAdd][i] + j % rangeSizes[currentAdd][i]];
            if (distanceCheck(epsilon2, dim, &data[p1], &data[p2])){
                //store point
                unsigned int index = atomicAdd(keyValueIndex,(unsigned int)1);
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