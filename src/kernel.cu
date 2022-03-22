#include "include/kernel.cuh"

//comparator function for sorting pairs and is used when checking results for duplicates
bool compPair(const std::pair<int, int> &x, const std::pair<int, int> &y){
    if(x.first < y.first){
        return true;
    }

    if(x.first == y.first && x.second < y.second){
        return true;
    }

    return false;

}

//function to launch kcuda kernels and distance calculation kernels
void launchKernel(int numLayers,// the number of layers in the tree
                  double * data, //the dataset that has been ordered by dimensoins and possibly reorganized for colasced memory accsess
                  int dim,//the dimensionality of the data
                  unsigned int numPoints,//the number of points in the dataset
                  double epsilon,//the distance threshold being searched
                  int * addIndexes,//the non-empty index locations in the last layer of the tree
                  int * addIndexRange,// the value of the non empty index locations  in the last layer of the tree, so the starting point number
                  int * pointArray,// the array of point numbers ordered to match the sequence in the last array of the tree and the data
                  int ** rangeIndexes,// the non-empty adjacent indexes for each non-empty index 
                  unsigned int ** rangeSizes, // the size of the non-empty adjacent indexes for each non-empty index
                  int * numValidRanges,// the number of adjacent non-empty indexes for each non-empty index
                  unsigned int * numPointsInAdd,// the number of points in each non-empty index
                  unsigned long long *calcPerAdd,// the number of calculations needed for each non-mepty index
                  int nonEmptyBins,//the number of nonempty indexes
                  unsigned long long sumCalcs,// the total number of calculations that will need to be made
                  unsigned long long sumAdds,//the total number of addresses that will be compared to by other addresses for distance calcs
                  int * linearRangeID,// an array for keeping trackj of starting points in the linear arrays
                  int * linearRangeIndexes,// a linear version of rangeIndexes
                  unsigned int * linearRangeSizes){ // a linear version of rangeSizes


    // store the squared value of epsilon because thats all that is needed for distance calcs
    double epsilon2 = epsilon*epsilon;

    //set a value for the number of calculations made by each thread per kernel invocation
    unsigned long long calcsPerThread = CALCS_PER_THREAD; 

    //the number of thrreads assigned to each non-empty address
    unsigned int * numThreadsPerAddress = (unsigned int *)malloc(sizeof(unsigned int)*nonEmptyBins);

    //keeping track of the number of batches that will be needed
    int numBatches = 0;

    // the number of threads that will be avaliable for each kernel invocation
    unsigned long long threadsPerBatch = KERNEL_BLOCKS * BLOCK_SIZE;

    // keeping track of the number of threads in a kernel invocation
    unsigned long long sum = 0;

    //itterating through the non-empty bins to generate batch parameters
    for(int i = 0; i < nonEmptyBins; i++){

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
    unsigned int * numThreadsPerBatch = (unsigned int*)calloc(numBatches,sizeof(unsigned int));

    // setting starting batch for loop
    int currentBatch = 0;

    // go through each non-empty index
    for(int i = 0; i < nonEmptyBins; i++){

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
    int ** addAssign = (int**)malloc(sizeof(int*)*numBatches);

    //the offset of the thread for calculations inside an address
    int ** threadOffsets = (int**)malloc(sizeof(int*)*numBatches);

    for(int i = 0; i < numBatches; i++){
        // array to track which thread is assigned to witch address
        addAssign[i] = (int * )malloc(sizeof(int)*numThreadsPerBatch[i]);

        //the offset of the thread for calculations inside an address
        threadOffsets[i] = (int*)malloc(sizeof(int)*numThreadsPerBatch[i]);
    }

    //setting the intital batch starting address
    int batchFirstAdd = 0;

    // calculating the thread assignements
    for(int i = 0; i < numBatches; i++){

        unsigned int threadCount = 0;

        //compute which thread does wich add
        for(int j = 0; j < numAddPerBatch[i]; j++){

            //basic error check
            if(numThreadsPerAddress[batchFirstAdd + j] == 0) {
                printf("ERROR: add %d has 0 threads\n", j + batchFirstAdd);
            }

            //for each address in the batch, assigne threads to it
            for(int k = 0; k < numThreadsPerAddress[batchFirstAdd + j]; k++){

                //assign the thread to the current address
                addAssign[i][threadCount] = j + batchFirstAdd;

                //thread offset is set to the thread number for that address
                threadOffsets[i][threadCount] = k;

                //increment thread count for all threads in the batch
                threadCount++;
            }
        }

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
    unsigned int * d_numThreadsPerAddress;
    assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerAddress, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerAddress, numThreadsPerAddress, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    // the device array to keep the values of the non-empty indexes in the final layer of the tree
    int * d_addIndexes;
    assert(cudaSuccess == cudaMalloc((void**)&d_addIndexes, sizeof(int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_addIndexes, addIndexes, sizeof(int)*nonEmptyBins, cudaMemcpyHostToDevice));

    //the number of adjacent non-empty indexes for each non-empty index
    int * d_numValidRanges;
    assert(cudaSuccess == cudaMalloc((void**)&d_numValidRanges, sizeof(int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_numValidRanges, numValidRanges, sizeof(int)*nonEmptyBins, cudaMemcpyHostToDevice));

    // copy over the linear rangeIDs for keeping track of loactions in the linear arrays
    int * d_linearRangeID;
    assert(cudaSuccess == cudaMalloc((void**)&d_linearRangeID, sizeof(int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_linearRangeID, linearRangeID, sizeof(int)*nonEmptyBins, cudaMemcpyHostToDevice));


    //copy over the linear range indexes wich kkeps track of the locations of adjacent non-empty indexes for each non-empty index
    int * d_rangeIndexes; //double check this for errors
    assert(cudaSuccess == cudaMalloc((void**)&d_rangeIndexes, sizeof(int)*sumAdds));
    assert(cudaSuccess ==  cudaMemcpy(d_rangeIndexes, linearRangeIndexes, sizeof(int)*sumAdds, cudaMemcpyHostToDevice));

    // copy over the size of the ranges in each adjacent non-empty index for each non-empty index
    unsigned int * d_rangeSizes;
    assert(cudaSuccess == cudaMalloc((void**)&d_rangeSizes, sizeof(unsigned int)*sumAdds));
    assert(cudaSuccess ==  cudaMemcpy(d_rangeSizes, linearRangeSizes, sizeof(unsigned int)*sumAdds, cudaMemcpyHostToDevice));

    // copy over array to keep track of number of points in each non-empty index
    unsigned int * d_numPointsInAdd;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPointsInAdd, sizeof(unsigned int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_numPointsInAdd, numPointsInAdd, sizeof(unsigned int)*nonEmptyBins, cudaMemcpyHostToDevice));

    //copy over the array that tracks the values of the non-empty indexes in the last layer of the tree
    int * d_addIndexRange;
    assert(cudaSuccess == cudaMalloc((void**)&d_addIndexRange, sizeof(int)*nonEmptyBins));
    assert(cudaSuccess ==  cudaMemcpy(d_addIndexRange, addIndexRange, sizeof(int)*nonEmptyBins, cudaMemcpyHostToDevice));

    // copy over the ordered point array that corresponds with the point numbers and the dataset
    int * d_pointArray;
    assert(cudaSuccess == cudaMalloc((void**)&d_pointArray, sizeof(int)*numPoints));
    assert(cudaSuccess ==  cudaMemcpy(d_pointArray, pointArray, sizeof(int)*numPoints, cudaMemcpyHostToDevice));

    // keep track of the number of pairs found in each batch
    unsigned long long int * keyValueIndex;
    //use pinned memory for async copies back to the host
    assert(cudaSuccess == cudaMallocHost((void**)&keyValueIndex, sizeof(unsigned long long int)*numBatches));
    for(int i = 0; i < numBatches; i++){
        keyValueIndex[i] = 0;
    }

    //copy over the array to keep track of the pairs found in each batch
    unsigned long long int * d_keyValueIndex;
    assert(cudaSuccess == cudaMalloc((void**)&d_keyValueIndex, sizeof(unsigned long long int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_keyValueIndex, keyValueIndex, sizeof(unsigned long long int)*numBatches, cudaMemcpyHostToDevice));

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
    int *d_dim;
    assert(cudaSuccess == cudaMalloc((void**)&d_dim, sizeof(int)));
    assert(cudaSuccess ==  cudaMemcpy(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice));

    // copy over the number of threads for each batch
    unsigned int * d_numThreadsPerBatch;
    assert(cudaSuccess == cudaMalloc((void**)&d_numThreadsPerBatch, sizeof(unsigned int)*numBatches));
    assert(cudaSuccess ==  cudaMemcpy(d_numThreadsPerBatch, numThreadsPerBatch, sizeof(unsigned int)*numBatches, cudaMemcpyHostToDevice));

    // copy over the number of points in the dataset
    unsigned int * d_numPoints;
    assert(cudaSuccess == cudaMalloc((void**)&d_numPoints, sizeof(unsigned int)));
    assert(cudaSuccess ==  cudaMemcpy(d_numPoints, &numPoints, sizeof(unsigned int), cudaMemcpyHostToDevice));


    
    // vectors for trackking results in the CPU version of the Kernel
    std::vector<int> hostPointA;
    std::vector<int> hostPointB;

    // #pragma omp parallel for
    for(int i = 0; i < numBatches; i++){
       
        //need to move this and use linear arrays
        /////////////////////////////////////////////////////////

        // copy over the thread assignments for the current batch
        int * d_addAssign;
        assert(cudaSuccess == cudaMalloc((void**)&d_addAssign, sizeof(int)*numThreadsPerBatch[i]));
        assert(cudaSuccess ==  cudaMemcpy(d_addAssign, addAssign[i], sizeof(int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));

        // copy over the offsets for each thread in the batch
        int * d_threadOffsets;
        assert(cudaSuccess == cudaMalloc((void**)&d_threadOffsets, sizeof(int)*numThreadsPerBatch[i]));
        assert(cudaSuccess ==  cudaMemcpy(d_threadOffsets, threadOffsets[i], sizeof(int)*numThreadsPerBatch[i], cudaMemcpyHostToDevice));

        /////////////////////////////////////////////////////////

        cudaDeviceSynchronize();

        unsigned int totalBlocks = ceil(numThreadsPerBatch[i]*1.0 / BLOCK_SIZE);


        printf("BatchNumber: %d/%d, Calcs: %llu, Adds: %d, threads: %u, blocks:%d\n ", i+1, numBatches, numCalcsPerBatch[i], numAddPerBatch[i], numThreadsPerBatch[i], totalBlocks);

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
                                                                    d_addAssign,
                                                                    d_threadOffsets,
                                                                    d_epsilon2,
                                                                    d_dim,
                                                                    &d_numThreadsPerBatch[i],
                                                                    d_numThreadsPerAddress,
                                                                    d_data,
                                                                    d_addIndexes,
                                                                    d_numValidRanges,
                                                                    d_rangeIndexes,
                                                                    d_rangeSizes,
                                                                    d_numPointsInAdd,
                                                                    d_addIndexRange,
                                                                    d_pointArray,
                                                                    &d_keyValueIndex[i],
                                                                    d_pointA,
                                                                    d_pointB);

        cudaDeviceSynchronize(); 

        assert(cudaSuccess ==  cudaMemcpy(&keyValueIndex[i], &d_keyValueIndex[i], sizeof(unsigned long long int), cudaMemcpyDeviceToHost));
        
        #endif
        


        printf("Results: %llu\n", keyValueIndex[i]);
        ///////////////////////
        //transfer back reuslts
        ///////////////////////

        // free(addAssign);
        // free(threadOffsets);
        
    }

    #if HOST
    std::vector< std::pair<int,int>> pairs;

    for(int i = 0; i < hostPointA.size();i++){
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
                                int * linearRangeID,
                                int * addAssign,
                                int * threadOffsets,
                                double *epsilon2,
                                int *dim,
                                unsigned int *numThreadsPerBatch,
                                unsigned int * numThreadsPerAddress,
                                double * data, int *addIndexes,
                                int * numValidRanges,
                                int * rangeIndexes,
                                unsigned int * rangeSizes,
                                unsigned int * numPointsInAdd,
                                int * addIndexRange,
                                int * pointArray,
                                unsigned long long int *keyValueIndex,
                                unsigned int * point_a,
                                unsigned int * point_b){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if(tid >= *numThreadsPerBatch){
        return;
    }

    int currentAdd = addAssign[tid]; 
    int threadOffset = threadOffsets[tid];
    int startingRangeID = linearRangeID[currentAdd];

    for(int i = 0; i < numValidRanges[currentAdd]; i++){
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
                                    int * linearRangeID,
                                    int * addAssign,
                                    int * threadOffsets,
                                    double *epsilon2,
                                    int *dim,
                                    unsigned int *numThreadsPerBatch,
                                    unsigned int * numThreadsPerAddress,
                                    double * data,
                                    int *addIndexes,
                                    int * numValidRanges,
                                    int * rangeIndexes,
                                    unsigned int * rangeSizes,
                                    unsigned int * numPointsInAdd,
                                    int * addIndexRange,
                                    int * pointArray,
                                    unsigned long long int *keyValueIndex,
                                    std::vector<int> * hostPointA,
                                    std::vector<int> * hostPointB){


    for(int h = 0; h < BLOCK_SIZE*totalBlocks; h++)
    {

        unsigned int tid = h;

        if(tid < *numThreadsPerBatch ){
            
            //the current index/address that we are searching for
            int currentAdd = addAssign[tid]; 

            //the offset of the thread based on the address we are currently in
            int threadOffset = threadOffsets[tid];

            // the strating location for this index into the linear arrays
            int startingRangeID = linearRangeID[currentAdd];
        
            //go through each adjacent index and calcualte
            for(int i = 0; i < numValidRanges[currentAdd]; i++){

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
inline bool distanceCheck(double epsilon2, int dim, double * data, unsigned int p1, unsigned int p2, unsigned int numPoints){
    double sum = 0;
    for(int i = 0; i < dim; i++){
        #if DATANORM
        sum += pow(data[i*numPoints + p1] - data[i*numPoints + p2], 2);
        #else
        sum += pow(data[p1*dim+i]-data[p2*dim+i],2);
        #endif
        if(sum > epsilon2) return false;
    }

    return true;
}