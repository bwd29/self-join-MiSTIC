#include "include/kernel.cuh"


void launchKernel(int numLayers, double * data, int dim, int numPoints, double epsilon, int * addIndexes, int * addIndexRange, int * pointArray, int ** rangeIndexes, unsigned int ** rangeSizes, int * numValidRanges, unsigned int * numPointsInAdd, unsigned long long *calcPerAdd, int nonEmptyBins, unsigned long long sumCalcs, unsigned long long sumAdds, int * linearRangeIndexes, unsigned int * linearRangeSizes){
 
    double epsilon2 = epsilon*epsilon;
    unsigned long long calcsPerThread = 1000; 

    unsigned int numSearches = pow(3,numLayers);
    unsigned int * numThreadsPerAddress = (unsigned int *)malloc(sizeof(unsigned int)*nonEmptyBins);

    int numBatches = 0;
    unsigned long long threadsPerBatch = KERNEL_BLOCKS * BLOCK_SIZE;
    unsigned long long sum = 0;

    // printf("add 0 calcs: %llu, num Points in that add: %u\n", calcPerAdd[0], numPointsInAdd[0]);
    for(int i = 0; i < nonEmptyBins; i++){
        numThreadsPerAddress[i] = ceil(calcPerAdd[i]*1.0 / calcsPerThread);
        if(numThreadsPerAddress[i] == 0) printf("\nERROR: Threads per address at %d: %u, cals per add: %llu\n",i, numThreadsPerAddress[i], calcPerAdd[i]);
        if (sum + calcPerAdd[i] < calcsPerThread*threadsPerBatch){
            sum += calcPerAdd[i];
        }else{
            sum = calcPerAdd[i];
            numBatches++;
        }
    }

    numBatches++;

    unsigned long long * numCalcsPerBatch = (unsigned long long*)calloc(numBatches,sizeof(unsigned long long));
    unsigned int * numAddPerBatch = (unsigned int*)calloc(numBatches, sizeof(unsigned int));
    unsigned int * numThreadsPerBatch = (unsigned int*)calloc(numBatches,sizeof(unsigned int));
    sum = 0;

    int currentBatch = 0;
    for(int i = 0; i < nonEmptyBins; i++){
        if(numThreadsPerBatch[currentBatch] == 0 || numThreadsPerBatch[currentBatch] + numThreadsPerAddress[i] < threadsPerBatch){
            numThreadsPerBatch[currentBatch] += numThreadsPerAddress[i];
            numAddPerBatch[currentBatch]++;
            numCalcsPerBatch[currentBatch] += calcPerAdd[i];
        } else {
            currentBatch++;
            i = i - 1;
        }

    }

    // for(int i = 0; i < numBatches; i++){
    //     printf("Batch: %d, numThreads: %u, numAdds: %u, numCalcs: %llu\n",i, numThreadsPerBatch[i],numAddPerBatch[i], numCalcsPerBatch[i]);
    // }



    ////////////////////////////////////////////////
    double * d_data;
    assert(cudaSuccess == cudaMalloc((void**)&d_data, sizeof(double)*numPoints*dim));
    assert(cudaSuccess ==  cudaMemcpy(d_data, data, sizeof(double)*numPoints*dim, cudaMemcpyHostToDevice));

    unsigned int * d_numThreadsPerAddress;
    assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerAddress, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerAddress, numThreadsPerAddress, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    int * d_addIndexes;
    assert(cudaSuccess == cudaMalloc((void**)&d_addIndexes, sizeof(int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_addIndexes, addIndexes, sizeof(int)*nonEmptyBins, cudaMemcpyHostToDevice));


    int * d_numValidRanges;
    assert(cudaSuccess == cudaMalloc((void**)&d_numValidRanges, sizeof(int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_numValidRanges, numValidRanges, sizeof(int)*nonEmptyBins, cudaMemcpyHostToDevice));

    int * d_rangeIndexes; //double check this for errors
    assert(cudaSuccess == cudaMalloc((void**)&d_rangeIndexes, sizeof(int)*nonEmptyBins*numSearches));
    assert(cudaSuccess ==  cudaMemcpy(d_rangeIndexes, linearRangeIndexes, sizeof(int)*numSearches*nonEmptyBins, cudaMemcpyHostToDevice));

    unsigned int * d_rangeSizes;
    assert(cudaSuccess == cudaMalloc((void**)&d_rangeSizes, sizeof(unsigned int)*numSearches*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_rangeSizes, linearRangeSizes, sizeof(unsigned int)*numSearches*nonEmptyBins, cudaMemcpyHostToDevice));


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

    double *d_epsilon2;
    assert(cudaSuccess == cudaMalloc((void**)&d_epsilon2, sizeof(double)));
    assert(cudaSuccess ==  cudaMemcpy(d_epsilon2, &epsilon2, sizeof(double), cudaMemcpyHostToDevice));

    int *d_dim;
    assert(cudaSuccess == cudaMalloc((void**)&d_dim, sizeof(int)));
    assert(cudaSuccess ==  cudaMemcpy(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice));

    unsigned int * d_numThreadsPerBatch;
    assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerBatch, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerBatch, numThreadsPerBatch, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));


    unsigned int * d_numSearches;
    assert(cudaSuccess == cudaMalloc((void**)&d_numSearches, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numSearches, &numSearches, sizeof(unsigned int), cudaMemcpyHostToDevice));

    unsigned int * d_numPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPoints, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numPoints, &numPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));





    ///////////////////////////////////////////////

    printf("numSearches: %d\n", numSearches);
    
    int batchFirstAdd = 0;
    for(int i = 0; i < numBatches; i++){

        //compute which thread does wich add
        int * addAssign = (int * )malloc(sizeof(int)*numThreadsPerBatch[i]);
        int * threadOffsets = (int*)malloc(sizeof(int)*numThreadsPerBatch[i]);
        unsigned int threadCount = 0;

        for(int j = 0; j < numAddPerBatch[i]; j++){
            if(numThreadsPerAddress[batchFirstAdd + j] == 0) {
                printf("ERROR: add %d has 0 threads\n", j + batchFirstAdd);
                // exit(0);
            }
            for(int k = 0; k < numThreadsPerAddress[batchFirstAdd + j]; k++){
                addAssign[threadCount] = j + batchFirstAdd;
                threadOffsets[threadCount] = k;
                threadCount++;
            }
        }

        batchFirstAdd += numAddPerBatch[i];

        // printf("\nBatch: %d, ThreadCount: %u, ThreadsPerBatch: %u\n",i,threadCount, numThreadsPerBatch[i]);

        /////////////////////////////////////////////////////////

        int * d_addAssign;
        assert(cudaSuccess == cudaMalloc((void**)&d_addAssign, sizeof(int)*numThreadsPerBatch[i]));
        assert(cudaSuccess ==  cudaMemcpy(d_addAssign, addAssign, sizeof(int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));


        int * d_threadOffsets;
        assert(cudaSuccess == cudaMalloc((void**)&d_threadOffsets, sizeof(int)*numThreadsPerBatch[i]));
        assert(cudaSuccess ==  cudaMemcpy(d_threadOffsets, threadOffsets, sizeof(int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));

        /////////////////////////////////////////////////////////

        cudaDeviceSynchronize();

        unsigned int totalBlocks = ceil(numThreadsPerBatch[i]*1.0 / BLOCK_SIZE);


        printf("BatchNumber: %d/%d, Calcs: %llu, Adds: %d, threads: %u, blocks:%d\n ", i+1, numBatches, numCalcsPerBatch[i], numAddPerBatch[i], numThreadsPerBatch[i], totalBlocks);
        
        
        
        
        
        //launch distance kernel
        distanceCalculationsKernel<<<totalBlocks, BLOCK_SIZE>>>(d_numPoints, d_numSearches, d_addAssign, d_threadOffsets, d_epsilon2, d_dim, &d_numThreadsPerBatch[i], d_numThreadsPerAddress, d_data, d_addIndexes, d_numValidRanges, d_rangeIndexes, d_rangeSizes, d_numPointsInAdd, d_addIndexRange, d_pointArray, &d_keyValueIndex[i], d_pointA, d_pointB);

        cudaDeviceSynchronize(); 

        assert(cudaSuccess ==  cudaMemcpy(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

        printf("Results: %llu\n", keyValueIndex[i]);
        //transfer back reuslts

        free(addAssign);
        free(threadOffsets);
        
    }

    unsigned long long totals = 0;
    for(int i = 0; i < numBatches; i++){
        totals += keyValueIndex[i];
    }

    printf("Total results Set Size: %llu\n", totals);

    free(numCalcsPerBatch);
    free(numAddPerBatch);
    free(numThreadsPerBatch);
    free(numThreadsPerAddress);

}

__global__ 
void distanceCalculationsKernel(unsigned int *numPoints, unsigned int *numSearches, int * addAssign, int * threadOffsets, double *epsilon2, int *dim, unsigned int *numThreadsPerBatch, unsigned int * numThreadsPerAddress, double * data, int *addIndexes, int * numValidRanges, int * rangeIndexes, unsigned int * rangeSizes, unsigned int * numPointsInAdd, int * addIndexRange, int * pointArray, unsigned long long int *keyValueIndex, unsigned int * point_a, unsigned int * point_b){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if(tid >= *numThreadsPerBatch){
        return;
    }

    int currentAdd = addAssign[tid]; 
    int threadOffset = threadOffsets[tid];

    for(int i = 0; i < numValidRanges[currentAdd]; i++){
        unsigned long long int numCalcs = rangeSizes[currentAdd*(*numSearches) + i] * numPointsInAdd[currentAdd];
        for(unsigned long long int j = threadOffset; j < numCalcs; j += numThreadsPerAddress[currentAdd]){

            unsigned int pointLocation1 = addIndexRange[currentAdd] + j / rangeSizes[currentAdd*(*numSearches) + i];
            unsigned int pointLocation2 = rangeIndexes[currentAdd*(*numSearches) + i] + j % rangeSizes[currentAdd*(*numSearches) + i];


            if(pointLocation1 > *numPoints) printf("ERROR 1: tid: %d, CurrentAdd: %d, Offset: %d, Point Locations: %u,%u, j: %llu, addVal: %d, size:%u, rangeIndexVal: %d, size:%u\n", tid, currentAdd, threadOffset, pointLocation1,pointLocation2,j,addIndexRange[currentAdd],numPointsInAdd[currentAdd],rangeIndexes[currentAdd*(*numSearches) + i],rangeSizes[currentAdd*(*numSearches) + i]);
            if(pointLocation2 > *numPoints) printf("ERROR 2: tid: %d, CurrentAdd: %d, Offset: %d, Point Locations: %u %u, j: %llu, rangeVal: %d, size: %u\n", tid, currentAdd, threadOffset, pointLocation1, pointLocation2,j,rangeIndexes[currentAdd*(*numSearches) + i],rangeSizes[currentAdd*(*numSearches) + i]);

            unsigned int p1 = pointArray[pointLocation1];
            unsigned int p2 = pointArray[pointLocation2];


            if (distanceCheck((*epsilon2), (*dim), data, p1, p2, (*numPoints))){
                //  store point
                unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
                point_a[index] = p1; //stores the first point Number
                point_b[index] = p2; // this stores the coresponding point number to form a pair
            }
        }
    }
}

__device__ //may need to switch to inline
bool distanceCheck(double epsilon2, int dim, double * data, unsigned int p1, unsigned int p2, unsigned int numPoints){
    double sum = 0;
    for(int i = 0; i < dim; i++){
        #if DATANORM
        sum+=pow(data[i*numPoints + p1] - data[i*numPoints + p2], 2);
        #else
        sum += pow(data[p1*dim+i]-data[p2*dim+i],2);
        #endif
        if(sum > epsilon2) return false;
    }

    return true;
}