#include "include/launcher.cuh"

//comparator function for sorting pairs and is used when checking results for duplicates
bool compPair(const std::pair<unsigned int, unsigned int> &x, const std::pair<unsigned int, unsigned int> &y){
    if(x.first < y.first){
        return true;
    }

    if(x.first == y.first && x.second < y.second){
        return true;
    }

    return false;

}


void constructNeighborTable(unsigned int * pointInDistValue, 
    unsigned int * pointersToNeighbors, 
    unsigned long long int * cnt, 
    unsigned int * uniqueKeys, 
    unsigned int * uniqueKeyPosition, 
    unsigned int numUniqueKeys,
    struct neighborTable * tables)
{

    #pragma omp parallel for
    for (unsigned int i=0; i < (*cnt); i++)
    {
        pointersToNeighbors[i] = pointInDistValue[i];
    }

    //////////////////////////////
    //NEW when Using unique on GPU
    //When neighbortable is initalized (memory reserved for vectors), we can write directly in the vector

    //if using unicomp we need to update different parts of the struct
    #pragma omp parallel for
    for (unsigned int i = 0; i < numUniqueKeys; i++) {

        unsigned int keyElem = uniqueKeys[i];
        //Update counter to write position in critical section
        omp_set_lock(&tables[keyElem].pointLock);

        unsigned int nextIdx = tables[keyElem].cntNDataArrays;
        tables[keyElem].cntNDataArrays++;
        omp_unset_lock(&tables[keyElem].pointLock);

        tables[keyElem].vectindexmin[nextIdx] = uniqueKeyPosition[i];
        tables[keyElem].vectdataPtr[nextIdx] = pointersToNeighbors;	

        //final value will be missing
        if (i == (numUniqueKeys - 1))
        {
            tables[keyElem].vectindexmax[nextIdx] = (*cnt)-1;
        }
        else
        {
           tables[keyElem].vectindexmax[nextIdx] = (uniqueKeyPosition[i+1]) - 1;
        }
    }

    return;
} 

//function to launch kcuda kernels and distance calculation kernels
struct neighborTable * launchKernel(unsigned int numLayers,// the number of layers in the tree
    double * data, //the dataset that has been ordered by dimensoins and possibly reorganized for colasced memory accsess
    unsigned int dim,//the dimensionality of the data
    unsigned int numPoints,//the number of points in the dataset
    double epsilon,//the distance threshold being searched
    unsigned int * addIndexes,//the non-empty index locations in the last layer of the tree
    unsigned int * addIndexRange,// the value of the non empty index locations  in the last layer of the tree, so the starting point number
    unsigned int * pointArray,// the array of point numbers ordered to match the sequence in the last array of the tree and the data
    unsigned int ** rangeIndexes,// the non-empty adjacent indexes for each non-empty index 
    unsigned int ** rangeSizes, // the size of the non-empty adjacent indexes for each non-empty index
    unsigned int * numValidRanges,// the number of adjacent non-empty indexes for each non-empty index
    unsigned int * numPointsInAdd,// the number of points in each non-empty index
    unsigned long long *calcPerAdd,// the number of calculations needed for each non-mepty index
    unsigned int nonEmptyBins,//the number of nonempty indexes
    unsigned long long sumCalcs,// the total number of calculations that will need to be made
    unsigned long long sumAdds,//the total number of addresses that will be compared to by other addresses for distance calcs
    unsigned int * linearRangeID,// an array for keeping trackj of starting points in the linear arrays
    unsigned int * linearRangeIndexes,// a linear version of rangeIndexes
    unsigned int * linearRangeSizes){ // a linear version of rangeSizes



    // store the squared value of epsilon because thats all that is needed for distance calcs
    double epsilon2 = epsilon*epsilon;

    //set a value for the number of calculations made by each thread per kernel invocation
    unsigned long long calcsPerThread = CALCS_PER_THREAD; 

    //the number of thrreads assigned to each non-empty address
    unsigned long long * numThreadsPerAddress = (unsigned long long *)malloc(sizeof(unsigned long long )*nonEmptyBins);

    //keeping track of the number of batches that will be needed
    unsigned int numBatches = 0;

    // the number of threads that will be avaliable for each kernel invocation
    unsigned long long threadsPerBatch = (unsigned long long)KERNEL_BLOCKS * BLOCK_SIZE;

    // keeping track of the number of threads in a kernel invocation
    unsigned long long sum = 0;

    //itterating through the non-empty bins to generate batch parameters
    for(unsigned int i = 0; i < nonEmptyBins; i++){

        // the number of threads for the address is the ceiling of the number of calcs for that address over calcs per thread
        numThreadsPerAddress[i] = calcPerAdd[i] / calcsPerThread;
        if(calcPerAdd[i] % calcsPerThread != 0) numThreadsPerAddress[i]++;

        if(calcPerAdd[i] == 0) printf("ERROR: add %d has 0 calcs\n", i);

        // check if the number of threads is higher than the number of threads for a batch
        if (sum + numThreadsPerAddress[i] < threadsPerBatch){
            // if the number of threads is less than keep adding addresses to the batch
            sum += numThreadsPerAddress[i];
        }else{
            // if we would exceed the number of threads for that batch, then dont add
            sum = numThreadsPerAddress[i];

            // check for an error
            // if(numThreadsPerAddress[i] > threadsPerBatch) printf("Warning: Address %d is too big. Needs: %d threads which is more than %d\n", i, numThreadsPerAddress[i], threadsPerBatch);

            //increment the number of batches needed
            numBatches++;
        }
    }

    //always need at least one batch
    numBatches++;

    //keeping track of the number of calculations for each batch
    unsigned long long * numCalcsPerBatch = (unsigned long long*)calloc(numBatches,sizeof(unsigned long long));

    //keeping track of the number of addresses that batch will compute
    unsigned int * numAddPerBatch = (unsigned int*)calloc(numBatches, sizeof(unsigned int));

    //keeping track of the number of threads that are in each batch
    unsigned long long * numThreadsPerBatch = (unsigned long long*)calloc(numBatches,sizeof(unsigned long long));

    // setting starting batch for loop
    unsigned int currentBatch = 0;

    // go through each non-empty index
    for(unsigned int i = 0; i < nonEmptyBins; i++){

        //error check
        if(currentBatch > numBatches) printf("ERROR 3: current batch %d is greater than num batches %d\n", currentBatch, numBatches);

        //check if the batch is new or if the number of threads per batch will exceed the max if added
        if(numThreadsPerBatch[currentBatch] == 0 || numThreadsPerBatch[currentBatch] + numThreadsPerAddress[i] < threadsPerBatch){
            //add the number of threads for index i to the number of threads for the batch
            numThreadsPerBatch[currentBatch] += numThreadsPerAddress[i];

            //increment the number of addresses in the current batch
            numAddPerBatch[currentBatch]++;

            // add the number of calculations for the address to the number of calcs for the batch
            numCalcsPerBatch[currentBatch] += calcPerAdd[i];
        } else { //if the number of threads for the batch will be too many, then need to add to the next batch instead
            currentBatch++;
            i = i - 1;
        }
    }


    // array to track which thread is assigned to witch address
    unsigned int ** addAssign = (unsigned int**)malloc(sizeof(unsigned int*)*numBatches);

    //the offset of the thread for calculations inside an address
    unsigned int ** threadOffsets = (unsigned int**)malloc(sizeof(unsigned int*)*numBatches);

    for(int i = 0; i < numBatches; i++){
        // array to track which thread is assigned to witch address
        addAssign[i] = (unsigned int * )malloc(sizeof(unsigned int)*numThreadsPerBatch[i]);

        //the offset of the thread for calculations inside an address
        threadOffsets[i] = (unsigned int*)malloc(sizeof(unsigned int)*numThreadsPerBatch[i]);
    }

    //setting the intital batch starting address
    unsigned int batchFirstAdd = 0;

    //keep track of the total numebr of threads
    unsigned long long totalThreads = 0;

    // array to keep track of where linear arrays start for threads based on the batch number
    unsigned int * batchThreadOffset = (unsigned int *)malloc(sizeof(unsigned int)*numBatches);

    // calculating the thread assignements
    for(unsigned int i = 0; i < numBatches; i++){

        unsigned int threadCount = 0;

        //compute which thread does wich add
        for(unsigned int j = 0; j < numAddPerBatch[i]; j++){

            //basic error check
            if(numThreadsPerAddress[batchFirstAdd + j] == 0) {
                printf("ERROR: add %d has 0 threads\n", j + batchFirstAdd);
            }

            //for each address in the batch, assigne threads to it
            for(unsigned int k = 0; k < numThreadsPerAddress[batchFirstAdd + j]; k++){

                //assign the thread to the current address
                addAssign[i][threadCount] = j + batchFirstAdd;

                //thread offset is set to the thread number for that address
                threadOffsets[i][threadCount] = k;

                //increment thread count for all threads in the batch
                threadCount++;
            }
        }

        //keep track of the thread numebr each batch starts at for use in linear arrays
        batchThreadOffset[i] = totalThreads;

        //keep total thread counts
        totalThreads += threadCount;


        //increment the first address in the batch for following batches
        batchFirstAdd += numAddPerBatch[i];
    }

    ////////////////////////////////////////////////
    //     Perfoming Data Transfers to Device     //
    ////////////////////////////////////////////////

    //device array which holds the dataset
    double * d_data;
    assert(cudaSuccess == cudaMalloc((void**)&d_data, sizeof(double)*numPoints*dim));
    assert(cudaSuccess ==  cudaMemcpy(d_data, data, sizeof(double)*numPoints*dim, cudaMemcpyHostToDevice));

    //device array to hold the number of threads in each address
    unsigned long long * d_numThreadsPerAddress;
    assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerAddress, sizeof(unsigned long long)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerAddress, numThreadsPerAddress, sizeof(unsigned long long )*nonEmptyBins, cudaMemcpyHostToDevice));

    // the device array to keep the values of the non-empty indexes in the final layer of the tree
    unsigned int * d_addIndexes;
    assert(cudaSuccess == cudaMalloc((void**)&d_addIndexes, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_addIndexes, addIndexes, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    //the number of adjacent non-empty indexes for each non-empty index
    unsigned int * d_numValidRanges;
    assert(cudaSuccess == cudaMalloc((void**)&d_numValidRanges, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_numValidRanges, numValidRanges, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    // copy over the linear rangeIDs for keeping track of loactions in the linear arrays
    unsigned int * d_linearRangeID;
    assert(cudaSuccess == cudaMalloc((void**)&d_linearRangeID, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_linearRangeID, linearRangeID, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));


    //copy over the linear range indexes wich kkeps track of the locations of adjacent non-empty indexes for each non-empty index
    unsigned int * d_rangeIndexes; //double check this for errors
    assert(cudaSuccess == cudaMalloc((void**)&d_rangeIndexes, sizeof(unsigned int)*sumAdds));
    assert(cudaSuccess ==  cudaMemcpy(d_rangeIndexes, linearRangeIndexes, sizeof(unsigned int)*sumAdds, cudaMemcpyHostToDevice));

    // copy over the size of the ranges in each adjacent non-empty index for each non-empty index
    unsigned int * d_rangeSizes;
    assert(cudaSuccess == cudaMalloc((void**)&d_rangeSizes, sizeof(unsigned int)*sumAdds));
    assert(cudaSuccess ==  cudaMemcpy(d_rangeSizes, linearRangeSizes, sizeof(unsigned int)*sumAdds, cudaMemcpyHostToDevice));

    // copy over array to keep track of number of points in each non-empty index
    unsigned int * d_numPointsInAdd;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPointsInAdd, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_numPointsInAdd, numPointsInAdd, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    //copy over the array that tracks the values of the non-empty indexes in the last layer of the tree
    unsigned int * d_addIndexRange;
    assert(cudaSuccess == cudaMalloc((void**)&d_addIndexRange, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_addIndexRange, addIndexRange, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    // copy over the ordered point array that corresponds with the point numbers and the dataset
    unsigned int * d_pointArray;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointArray, sizeof(unsigned int)*numPoints));
    assert(cudaSuccess ==  cudaMemcpy(d_pointArray, pointArray, sizeof(unsigned int)*numPoints, cudaMemcpyHostToDevice));

    // keep track of the number of pairs found in each batch
    unsigned long long * keyValueIndex;
    //use pinned memory for async copies back to the host
    assert(cudaSuccess == cudaMallocHost((void**)&keyValueIndex, sizeof(unsigned long long )*numBatches));
    for(int i = 0; i < numBatches; i++){
        keyValueIndex[i] = 0;
    }

    //copy over the array to keep track of the pairs found in each batch
    unsigned long long * d_keyValueIndex;
    assert(cudaSuccess == cudaMalloc((void**)&d_keyValueIndex, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_keyValueIndex, keyValueIndex, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));


    // copying over the squared epsilon value
    double *d_epsilon2;
    assert(cudaSuccess == cudaMalloc((void**)&d_epsilon2, sizeof(double)));
    assert(cudaSuccess ==  cudaMemcpy(d_epsilon2, &epsilon2, sizeof(double), cudaMemcpyHostToDevice));

    // copying over the dimensionality of the data
    unsigned int *d_dim;
    assert(cudaSuccess == cudaMalloc((void**)&d_dim, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_dim, &dim, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // copy over the number of threads for each batch
    unsigned long long * d_numThreadsPerBatch;
    assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerBatch, sizeof(unsigned long long)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerBatch, numThreadsPerBatch, sizeof(unsigned long long)*numBatches, cudaMemcpyHostToDevice));

    // copy over the number of points in the dataset
    unsigned int * d_numPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPoints, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numPoints, &numPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // the offsets into addAssign and threadOffsets based on batch number
    unsigned int * d_batchThreadOffset;
    assert(cudaSuccess == cudaMalloc((void**)&d_batchThreadOffset, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_batchThreadOffset, batchThreadOffset, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));


    // copy over the thread assignments for the current batch
    unsigned int * d_addAssign;
    assert(cudaSuccess == cudaMalloc((void**)&d_addAssign, sizeof(unsigned int)*totalThreads));
    for(unsigned int i = 0; i < numBatches; i++){
        assert(cudaSuccess ==  cudaMemcpy(&d_addAssign[batchThreadOffset[i]], addAssign[i], sizeof(unsigned int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));
    }

    // copy over the offsets for each thread in the batch
    unsigned int * d_threadOffsets;
    assert(cudaSuccess == cudaMalloc((void**)&d_threadOffsets, sizeof(unsigned int)*totalThreads));
    for(unsigned int i = 0; i < numBatches; i++){
        assert(cudaSuccess ==  cudaMemcpy(&d_threadOffsets[batchThreadOffset[i]], threadOffsets[i], sizeof(unsigned int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));
    }

    //array for keeping track of the paris found, this tyracks first value in pair
    unsigned int * d_pointA[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointA[i], sizeof(unsigned int)*resultsSize));
    }

    //array for keeping track of the paris found, this tyracks second value in pair
    unsigned int * d_pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_pointB[i], sizeof(unsigned int)*resultsSize));
    }

    unsigned int * pointB[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMallocHost((void**)&pointB[i], sizeof(unsigned int)*initalPinnedResultsSize));
    }

    unsigned int *uniqueCnt;
    assert(cudaSuccess == cudaMallocHost((void**)&uniqueCnt, numBatches*sizeof(unsigned int)));
    for (unsigned int i = 0; i < numBatches; i++) {
        uniqueCnt[i] = 0;
    }

    unsigned int *d_uniqueCnt;
    assert(cudaSuccess == cudaMalloc((void**)&d_uniqueCnt, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_uniqueCnt, uniqueCnt, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));

    unsigned int *d_uniqueKeyPosition[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeyPosition[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int *d_uniqueKeys[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        assert(cudaSuccess == cudaMalloc((void**)&d_uniqueKeys[i], sizeof(unsigned int)*numPoints));
    }

    unsigned int ** dataArray = (unsigned int **)malloc(sizeof(unsigned int*)*numBatches);

    //struct for sotring the results
    struct neighborTable * tables = (struct neighborTable*)malloc(sizeof(struct neighborTable)*numPoints);
    for (unsigned int i = 0; i < numPoints; i++)
    {	
    struct neighborTable temp;
    tables[i] = temp;
    //tables[i] = (struct neighborTable)malloc(sizeof(struct neighborTable));

    tables[i].cntNDataArrays = 1; 
    tables[i].vectindexmin.resize(numBatches+1);
    tables[i].vectindexmin[0] = i;
    tables[i].vectindexmax.resize(numBatches+1);
    tables[i].vectindexmax[0] = i;
    tables[i].vectdataPtr.resize(numBatches+1);
    tables[i].vectdataPtr[0] = pointArray;
    omp_init_lock(&tables[i].pointLock);
    }

    // vectors for tracking results in the CPU version of the Kernel
    std::vector<unsigned int> hostPointA;
    std::vector<unsigned int> hostPointB;

    cudaDeviceSynchronize(); 

    cudaStream_t stream[NUMSTREAMS];
    for (unsigned int i = 0; i < NUMSTREAMS; i++){
        cudaError_t stream_check = cudaStreamCreate(stream+i);
        assert(cudaSuccess == stream_check);
    }

    unsigned long long bufferSizes[NUMSTREAMS];
    for(unsigned int i = 0; i < NUMSTREAMS; i++){
        bufferSizes[i] = initalPinnedResultsSize;
    }

    #pragma omp parallel for num_threads(NUMSTREAMS) schedule(dynamic)
    for(unsigned int i = 0; i < numBatches; i++){

        unsigned int tid = omp_get_thread_num();
        unsigned int totalBlocks = ceil(numThreadsPerBatch[i]*1.0 / BLOCK_SIZE);

        printf("BatchNumber: %d/%d, Calcs: %llu, addresses: %d, threads: %u, blocks:%d \n", i+1, numBatches, numCalcsPerBatch[i], numAddPerBatch[i], numThreadsPerBatch[i], totalBlocks);

        //launch distance kernel
        #if HOST
            distanceCalculationsKernel_CPU(totalBlocks, 
                                    &numPoints,
                                    linearRangeID, 
                                    addAssign[i], 
                                    threadOffsets[i], 
                                    &epsilon2, 
                                    &dim, 
                                    &numThreadsPerBatch[i], 
                                    numThreadsPerAddress, 
                                    data, 
                                    addIndexes, 
                                    numValidRanges, 
                                    linearRangeIndexes, 
                                    linearRangeSizes, 
                                    numPointsInAdd, 
                                    addIndexRange, 
                                    pointArray, 
                                    &keyValueIndex[i],
                                    &hostPointA,
                                    &hostPointB);

        #else
            distanceCalculationsKernel<<<totalBlocks, BLOCK_SIZE, 0, stream[tid]>>>(d_numPoints,
                                                                                d_linearRangeID,
                                                                                &d_addAssign[batchThreadOffset[i]],
                                                                                &d_threadOffsets[batchThreadOffset[i]],
                                                                                d_epsilon2,
                                                                                d_dim,
                                                                                &d_numThreadsPerBatch[i],
                                                                                d_numThreadsPerAddress,
                                                                                d_data,
                                                                                d_numValidRanges,
                                                                                d_rangeIndexes,
                                                                                d_rangeSizes,
                                                                                d_numPointsInAdd,
                                                                                d_addIndexRange,
                                                                                &d_keyValueIndex[i],
                                                                                d_pointA[tid],
                                                                                d_pointB[tid]);

            cudaStreamSynchronize(stream[tid]);

            assert(cudaSuccess ==  cudaMemcpyAsync(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long ), cudaMemcpyDeviceToHost, stream[tid]));
            cudaStreamSynchronize(stream[tid]);

            printf("Batch %d Results: %llu\n", i,keyValueIndex[i]);


            if(keyValueIndex[i] > bufferSizes[tid]){
                //  printf("tid: %d first run\n", tid);
                cudaFreeHost(pointB[tid]);
                //printf("tid: %d freed memory\n", tid);
                assert(cudaSuccess == cudaMallocHost((void**) &pointB[tid], sizeof(unsigned int)*(keyValueIndex[i] + 1)));
                //printf("tid: %d pinned memory\n", tid);
                bufferSizes[tid] = keyValueIndex[i];
            }

            // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_pointA[tid], d_pointA[tid] + keyValueIndex[i], d_pointB[tid]);
            GPU_SortbyKey(stream[tid], d_pointA[tid], (unsigned int)keyValueIndex[i], d_pointB[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(pointB[tid], d_pointB[tid], sizeof(unsigned int)*keyValueIndex[i], cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            unsigned int totalBlocksUnique = ceil((1.0*keyValueIndex[i])/(1.0*BLOCK_SIZE));	
            kernelUniqueKeys<<<totalBlocksUnique, BLOCK_SIZE,0,stream[tid]>>>(d_pointA[tid],
                                                                &d_keyValueIndex[i], 
                                                                d_uniqueKeys[tid], 
                                                                d_uniqueKeyPosition[tid], 
                                                                &d_uniqueCnt[i]);

            cudaStreamSynchronize(stream[tid]);

            assert(cudaSuccess == cudaMemcpyAsync(&uniqueCnt[i], &d_uniqueCnt[i], sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[tid]));

            cudaStreamSynchronize(stream[tid]);

            // thrust::sort_by_key(thrust::cuda::par.on(stream[tid]), d_uniqueKeys[tid], d_uniqueKeys[tid]+uniqueCnt[i], d_uniqueKeyPosition[tid]);
            GPU_SortbyKey(stream[tid], d_uniqueKeys[tid], uniqueCnt[i], d_uniqueKeyPosition[tid]);


            unsigned int * uniqueKeys = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeys, d_uniqueKeys[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            unsigned int * uniqueKeyPosition = (unsigned int*)malloc(sizeof(unsigned int)*uniqueCnt[i]);
            assert(cudaSuccess == cudaMemcpyAsync(uniqueKeyPosition, d_uniqueKeyPosition[tid], sizeof(unsigned int)*uniqueCnt[i], cudaMemcpyDeviceToHost, stream[tid]));

            dataArray[i] = (unsigned int*)malloc(sizeof(unsigned int)*keyValueIndex[i]);

            cudaStreamSynchronize(stream[tid]);

            constructNeighborTable(pointB[tid], dataArray[i], &keyValueIndex[i], uniqueKeys,uniqueKeyPosition, uniqueCnt[i], tables);

            free(uniqueKeys);
            free(uniqueKeyPosition);


        #endif


    }

    #if HOST
        std::vector< std::pair<unsigned int,unsigned int>> pairs;

        for(unsigned int i = 0; i < hostPointA.size();i++){
            pairs.push_back(std::make_pair(hostPointA[i],hostPointB[i]));
        }

        std::sort(pairs.begin(), pairs.end(), compPair);

        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    #endif



    unsigned long long totals = 0;
    for(int i = 0; i < numBatches; i++){
        totals += keyValueIndex[i];
    }

    #if HOST
        printf("Total results Set Size: %llu , unique pairs: %lu\n", totals, pairs.size());
    #else
        printf("Total results Set Size: %llu \n", totals);
    #endif


    free(numCalcsPerBatch);
    free(numAddPerBatch);
    free(numThreadsPerBatch);
    free(numThreadsPerAddress);

    return tables;
}