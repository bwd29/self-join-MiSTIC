#include "include/kernel.cuh"


void launchKernel(int numLayers, double * data, int dim, int numPoints, double epsilon, int * addIndexes, int * addIndexRange, int * pointArray, int ** rangeIndexes, unsigned int ** rangeSizes, int * numValidRanges, unsigned int * numPointsInAdd, unsigned long long *calcPerAdd, int nonEmptyBins, unsigned long long sumCalcs, unsigned long long sumAdds){
 
    double epsilon2 = epsilon*epsilon;
    unsigned long long calcsPerThread = 1000; //placeholder value of 100k

    int numSearches = pow(3,numLayers);
    unsigned long long * numThreadsPerAddress = (unsigned long long *)malloc(sizeof(unsigned long long)*nonEmptyBins);

    int numBatches = 0;
    unsigned long long threadsPerBatch = KERNEL_BLOCKS * BLOCK_SIZE;
    unsigned long long sum = 0;

    printf("add 1 calcs: %llu\n", calcPerAdd[1]);
    for(int i = 0; i < nonEmptyBins; i++){
        numThreadsPerAddress[i] = ceil(calcPerAdd[i] / calcsPerThread);
        if (sum + calcPerAdd[i] < calcsPerThread*threadsPerBatch || sum == 0){
            sum += calcPerAdd[i];
        }else{
            sum = calcPerAdd[i];
            numBatches++;
        }
    }
    numBatches++;

    unsigned long long * numCalcsPerBatch = (unsigned long long*)malloc(sizeof(unsigned long long)*numBatches);
    unsigned int * numAddPerBatch = (unsigned int*)malloc(sizeof(unsigned int)*numBatches);
    unsigned long long * numThreadsPerBatch = (unsigned long long*)calloc(numBatches,sizeof(unsigned long long));
    sum = 0;
    int batchCount = 0;
    int addCount = 0;
    for(int i = 0; i < nonEmptyBins; i++){
        
        if (sum + calcPerAdd[i] < calcsPerThread*threadsPerBatch || sum == 0){
            numThreadsPerBatch[batchCount] += numThreadsPerAddress[i];
            sum += calcPerAdd[i];
            addCount++;
        }else{
            numCalcsPerBatch[batchCount] = sum;
            numAddPerBatch[batchCount] = addCount;

            sum = calcPerAdd[i];

            addCount = 1;
            batchCount++;

            numThreadsPerBatch[batchCount] += numThreadsPerAddress[i];

        }
    }

    numCalcsPerBatch[numBatches-1] = sum; //for last
    numAddPerBatch[numBatches-1] = addCount;

////////////////////////////////////////////////
double * d_data;
assert(cudaSuccess == cudaMalloc((void**)&d_data, sizeof(double)*numPoints*dim));
assert(cudaSuccess ==  cudaMemcpy(d_data, data, sizeof(double)*numPoints*dim, cudaMemcpyHostToDevice));

int * d_numThreadsPerAddress;
assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerAddress, sizeof(unsigned long long)*nonEmptyBins));
assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerAddress, numThreadsPerAddress, sizeof(unsigned long long)*nonEmptyBins, cudaMemcpyHostToDevice));

int * d_addIndexes;
assert(cudaSuccess == cudaMalloc((void**)&d_addIndexes, sizeof(int)*nonEmptyBins));
assert(cudaSuccess ==  cudaMemcpy(d_addIndexes, addIndexes, sizeof(int)*nonEmptyBins, cudaMemcpyHostToDevice));


int * d_numValidRanges;
assert(cudaSuccess == cudaMalloc((void**)&d_numValidRanges, sizeof(int)*nonEmptyBins));
assert(cudaSuccess ==  cudaMemcpy(d_numValidRanges, numValidRanges, sizeof(int)*nonEmptyBins, cudaMemcpyHostToDevice));

int * d_rangeIndexes; //double check this for errors
assert(cudaSuccess == cudaMalloc((void**)&d_rangeIndexes, sizeof(int)*nonEmptyBins*numSearches));
for(int i = 0; i < nonEmptyBins; i++){
    assert(cudaSuccess ==  cudaMemcpy(d_rangeIndexes+i*numSearches, rangeIndexes[i], sizeof(int)*numSearches, cudaMemcpyHostToDevice));
}

unsigned int * d_rangeSizes;
assert(cudaSuccess == cudaMalloc((void**)&d_rangeSizes, sizeof(unsigned int)*numSearches*nonEmptyBins));
for(int i = 0; i < nonEmptyBins; i++){
    assert(cudaSuccess ==  cudaMemcpy(d_rangeSizes+i*numSearches, rangeSizes[i], sizeof(unsigned int)*numSearches, cudaMemcpyHostToDevice));
}

unsigned int * d_numPointsInAdd;
assert(cudaSuccess == cudaMalloc((void**)&d_numPointsInAdd, sizeof(unsigned int)*nonEmptyBins));
assert(cudaSuccess ==  cudaMemcpy(d_numPointsInAdd, numPointsInAdd, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));


int * d_addIndexRange;
assert(cudaSuccess == cudaMalloc((void**)&d_addIndexRange, sizeof(int)*nonEmptyBins));
assert(cudaSuccess ==  cudaMemcpy(d_addIndexRange, addIndexRange, sizeof(int)*nonEmptyBins, cudaMemcpyHostToDevice));


int * d_pointArray;
assert(cudaSuccess == cudaMalloc((void**)&d_pointArray, sizeof(int)*numPoints));
assert(cudaSuccess ==  cudaMemcpy(d_pointArray, pointArray, sizeof(int)*numPoints, cudaMemcpyHostToDevice));


unsigned long long int * keyValueIndex;
assert(cudaSuccess == cudaMallocHost((void**)&keyValueIndex, sizeof(unsigned long long int)*numBatches));
for(int i = 0; i < numBatches; i++){
    keyValueIndex[i] = 0;
}
unsigned long long int * d_keyValueIndex;
assert(cudaSuccess == cudaMalloc((void**)&d_keyValueIndex, sizeof(unsigned long long int)*numBatches));
assert(cudaSuccess ==  cudaMemcpy(d_keyValueIndex, keyValueIndex, sizeof(unsigned long long int)*numBatches, cudaMemcpyHostToDevice));


unsigned int * d_pointA;
assert(cudaSuccess == cudaMalloc((void**)&d_pointA, sizeof(unsigned int)*resultsSize));

unsigned int * d_pointB;
assert(cudaSuccess == cudaMalloc((void**)&d_pointB, sizeof(unsigned int)*resultsSize));



///////////////////////////////////////////////
    

    for(int i = 0; i < numBatches; i++){

        const double d_epsilon2 = epsilon2;
        const int d_dim = dim;
        const int d_numThreadsPerBatch = numThreadsPerBatch[i];
        const int d_numSearches = pow(3,numLayers);

        //compute which thread does wich add
        int * addAssign = (int * )malloc(sizeof(int)*numThreadsPerBatch[i]);
        int * threadOffsets = (int*)malloc(sizeof(int)*numThreadsPerBatch[i]);
        unsigned int currentAdd = 0;
        unsigned int offsetCount = 0;

        printf("Threads in address 1: %llu\n", numThreadsPerAddress[0]);

        for(unsigned int j = 0; j < numThreadsPerBatch[i]; j++){
            if ( offsetCount > numThreadsPerAddress[currentAdd]){
                currentAdd++;
                offsetCount = 0;
            }
            addAssign[j] = currentAdd;
            threadOffsets[j] = offsetCount;
            offsetCount++;
        }

        /////////////////////////////////////////////////////////

        int * d_addAssign;
        assert(cudaSuccess == cudaMalloc((void**)&d_addAssign, sizeof(int)*numThreadsPerBatch[i]));
        assert(cudaSuccess ==  cudaMemcpy(d_addAssign, addAssign, sizeof(int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));


        int * d_threadOffsets;
        assert(cudaSuccess == cudaMalloc((void**)&d_threadOffsets, sizeof(int)*numThreadsPerBatch[i]));
        assert(cudaSuccess ==  cudaMemcpy(d_threadOffsets, threadOffsets, sizeof(int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));

        /////////////////////////////////////////////////////////

        cudaDeviceSynchronize();

        unsigned int totalBlocks = ceil(numThreadsPerBatch[i] / (unsigned long long)BLOCK_SIZE);


        printf("BatchNumber: %d/%d, Calcs: %llu, Adds: %d, threads: %llu, blocks:%d\n ", i+1, numBatches, numCalcsPerBatch[i], numAddPerBatch[i], numThreadsPerBatch[i], totalBlocks);
        
        
        
        
        
        //launch distance kernel
        distanceCalculationsKernel<<<totalBlocks, BLOCK_SIZE>>>(d_numSearches, d_addAssign, d_threadOffsets, d_epsilon2, d_dim, d_numThreadsPerBatch, d_numThreadsPerAddress, d_data, d_addIndexes, d_numValidRanges, d_rangeIndexes, d_rangeSizes, d_numPointsInAdd, d_addIndexRange, d_pointArray, &d_keyValueIndex[i], d_pointA, d_pointB);

        cudaDeviceSynchronize(); 

        assert(cudaSuccess ==  cudaMemcpy(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

        printf("Results: %llu\n", keyValueIndex[i]);
        //transfer back reuslts

        free(addAssign);
        free(threadOffsets);
        
    }

    free(numCalcsPerBatch);
    free(numAddPerBatch);
    free(numThreadsPerBatch);
    free(numThreadsPerAddress);

}

__global__ 
void distanceCalculationsKernel(const int numSearches, int * addAssign, int * threadOffsets, const double epsilon2, const int dim, const int numThreadsPerBatch, int * numThreadsPerAddress, double * data, int *addIndexes, int * numValidRanges, int * rangeIndexes, unsigned int * rangeSizes, unsigned int * numPointsInAdd, int * addIndexRange, int * pointArray, unsigned long long int *keyValueIndex, unsigned int * point_a, unsigned int * point_b){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if(tid > numThreadsPerBatch){
        return;
    }

    int currentAdd = addAssign[tid];
    int threadOffset = threadOffsets[tid];

    if(tid == 81) printf("current add: %d, offset: %d, threads in add: %d\n", currentAdd, threadOffset, numThreadsPerAddress[currentAdd]);


    for(int i = 0; i < numValidRanges[currentAdd]; i++){
        unsigned long long int numCalcs = rangeSizes[currentAdd*numSearches + i] * numPointsInAdd[currentAdd];
        if(tid == 81) printf("current range: %d, calcs: %llu\n", i, numCalcs);

        for(unsigned long long int j = threadOffset; j < numCalcs; j += numThreadsPerAddress[currentAdd]){
        //     unsigned int p1 = pointArray[addIndexRange[currentAdd] + j/rangeSizes[currentAdd*numSearches + i]];
        //     unsigned int p2 = pointArray[rangeIndexes[currentAdd*numSearches + i] + j % rangeSizes[currentAdd*numSearches + i]];
            // if (distanceCheck(epsilon2, dim, &data[p1], &data[p2])){
            //     //store point
                unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
            //     point_a[index] = p1; //stores the first point Number
            //     point_b[index] = p2; // this stores the cooresponding point number to form a pair
            // }
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