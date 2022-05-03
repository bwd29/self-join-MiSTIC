#include "include/kernel.cuh"

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

//function to launch kcuda kernels and distance calculation kernels
void launchKernel(unsigned int numLayers,// the number of layers in the tree
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
    unsigned long long threadsPerBatch = KERNEL_BLOCKS * BLOCK_SIZE;

    // keeping track of the number of threads in a kernel invocation
    unsigned long long sum = 0;

    //itterating through the non-empty bins to generate batch parameters
    for(unsigned int i = 0; i < nonEmptyBins; i++){

        // the number of threads for the address is the ceiling of the number of calcs for that address over calcs per thread
        numThreadsPerAddress[i] = ceil(calcPerAdd[i]*1.0 / calcsPerThread);

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

    //array for keeping track of the paris found, this tyracks first value in pair
    unsigned int * d_pointA;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointA, sizeof(unsigned int)*resultsSize));

    //array for keeping track of the paris found, this tyracks second value in pair
    unsigned int * d_pointB;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointB, sizeof(unsigned int)*resultsSize));

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

    
    // vectors for trackking results in the CPU version of the Kernel
    std::vector<unsigned int> hostPointA;
    std::vector<unsigned int> hostPointB;

    // #pragma omp parallel for
    for(unsigned int i = 0; i < numBatches; i++){

        unsigned int totalBlocks = ceil(numThreadsPerBatch[i]*1.0 / BLOCK_SIZE);

        printf("BatchNumber: %d/%d, Calcs: %llu, Adds: %d, threads: %u, blocks:%d ", i+1, numBatches, numCalcsPerBatch[i], numAddPerBatch[i], numThreadsPerBatch[i], totalBlocks);

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
            distanceCalculationsKernel<<<totalBlocks, BLOCK_SIZE>>>(d_numPoints,
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
                                                                    d_pointA,
                                                                    d_pointB);

        cudaDeviceSynchronize(); 

        assert(cudaSuccess ==  cudaMemcpy(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long ), cudaMemcpyDeviceToHost));
        
        #endif
        


        printf("Results: %llu\n", keyValueIndex[i]);
        ///////////////////////
        //transfer back reuslts
        ///////////////////////

        // free(addAssign);
        // free(threadOffsets);
        
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

}

__global__ 
void distanceCalculationsKernel(unsigned int *numPoints,
                                unsigned int * linearRangeID,
                                unsigned int * addAssign,
                                unsigned int * threadOffsets,
                                double *epsilon2,
                                unsigned int *dim,
                                unsigned long long  *numThreadsPerBatch,
                                unsigned long long  * numThreadsPerAddress,
                                double * data, 
                                unsigned int * numValidRanges,
                                unsigned int * rangeIndexes,
                                unsigned int * rangeSizes,
                                unsigned int * numPointsInAdd,
                                unsigned int * addIndexRange,
                                unsigned long long  *keyValueIndex,
                                unsigned int * point_a,
                                unsigned int * point_b){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if(tid >= *numThreadsPerBatch){
        return;
    }

    unsigned int currentAdd = addAssign[tid]; 
    unsigned int threadOffset = threadOffsets[tid];
    unsigned int startingRangeID = linearRangeID[currentAdd];

    for(unsigned int i = 0; i < numValidRanges[currentAdd]; i++){
        unsigned long long int numCalcs = rangeSizes[startingRangeID + i] * numPointsInAdd[currentAdd];
        for(unsigned long long int j = threadOffset; j < numCalcs; j += numThreadsPerAddress[currentAdd]){

            unsigned int p1 = addIndexRange[currentAdd] + j / rangeSizes[startingRangeID + i];
            unsigned int p2 = rangeIndexes[startingRangeID + i] + j % rangeSizes[startingRangeID+ i];

            if (distanceCheck((*epsilon2), (*dim), data, p1, p2, (*numPoints))){
                //  store point
                unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
                point_a[index] = p1; //stores the first point Number
                point_b[index] = p2; // this stores the coresponding point number to form a pair
            }
        }
    }
}



void distanceCalculationsKernel_CPU(unsigned int totalBlocks,
                                    unsigned int *numPoints,
                                    unsigned int * linearRangeID,
                                    unsigned int * addAssign,
                                    unsigned int * threadOffsets,
                                    double *epsilon2,
                                    unsigned int *dim,
                                    unsigned long long  *numThreadsPerBatch,
                                    unsigned long long  * numThreadsPerAddress,
                                    double * data,
                                    unsigned int *addIndexes,
                                    unsigned int * numValidRanges,
                                    unsigned int * rangeIndexes,
                                    unsigned int * rangeSizes,
                                    unsigned int * numPointsInAdd,
                                    unsigned int * addIndexRange,
                                    unsigned int * pointArray,
                                    unsigned long long  *keyValueIndex,
                                    std::vector<unsigned int> * hostPointA,
                                    std::vector<unsigned int> * hostPointB){


    for(unsigned int h = 0; h < BLOCK_SIZE*totalBlocks; h++)
    {

        unsigned int tid = h;

        if(tid < *numThreadsPerBatch ){
            
            //the current index/address that we are searching for
            unsigned int currentAdd = addAssign[tid]; 

            //the offset of the thread based on the address we are currently in
            unsigned int threadOffset = threadOffsets[tid];

            // the strating location for this index into the linear arrays
            unsigned int startingRangeID = linearRangeID[currentAdd];
        
            //go through each adjacent index and calcualte
            for(unsigned int i = 0; i < numValidRanges[currentAdd]; i++){

                //the number of calcs for this address is the nuymber of points in it * number of points in current address
                unsigned long long int numCalcs = rangeSizes[startingRangeID + i] * numPointsInAdd[currentAdd];

                //each thread starts at its offset then increments by the number of threads for the address untill past the numebr of calcs
                for(unsigned long long int j = threadOffset; j < numCalcs; j += numThreadsPerAddress[currentAdd]){
        
                    // the first point will be from the home address
                    // the starting point from addIndexRange plus the floor of the itteration over the number of points in the adjacent address
                    // once all of the threads have made the calcs for a point they all move to the next point in the home address
                    unsigned int p1 = addIndexRange[currentAdd] + j / rangeSizes[startingRangeID + i];
                    unsigned int p2 = rangeIndexes[startingRangeID + i] + j % rangeSizes[startingRangeID+ i];

                    if (distanceCheck((*epsilon2), (*dim), data, p1, p2, (*numPoints))){
                        //  store point
                        *keyValueIndex += 1;
                        
                        // unsigned long long int index = *keyValueIndex;
                        (*hostPointA).push_back(p1); //stores the first point Number
                        (*hostPointB).push_back(p2); // this stores the coresponding point number to form a pair
                    }
                }
            }
        }
    }
}



__host__ __device__ //may need to switch to inline (i did)
inline bool distanceCheck(double epsilon2, unsigned int dim, double * data, unsigned int p1, unsigned int p2, unsigned int numPoints){
    double sum = 0;
    for(unsigned int i = 0; i < dim; i++){
        #if DATANORM
        sum += pow(data[i*numPoints + p1] - data[i*numPoints + p2], 2);
        #else
        sum += pow(data[p1*dim+i]-data[p2*dim+i],2);
        #endif
        if(sum > epsilon2) return false;
    }

    return true;
}