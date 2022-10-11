#include "include/kernel.cuh"

__global__
void GPUGenerateRanges(unsigned int * tree, //points to the tree constructed with buildTree()
                        unsigned int * lastTreeLayer,
                        unsigned int * numPoints, // the number of points in the dataset
                        unsigned int * pointBinNumbers, // the bin numbers of the points relative to the reference points
                        unsigned int * numLayers, // the number of layer the tree has
                        unsigned int * binSizes,	// the number of bins for each layer, or rather the width of the tree in bins for that layer
                        unsigned int * binAmounts, // the number of bins for each reference point, ranhge/epsilon
                        unsigned int * addIndexes, // where generateRanges will return the non-empty index locations in the tree's final layer
                        unsigned int * rangeIndexes, // the index locations that are adjacent to each non-empty index
                        unsigned int * rangeSizes, // the number of points in adjacent non-empty indexes for each non-empty index
                        unsigned int * numValidRanges, // the numnber of adjacent non-empty indexes for each non-empty index
                        unsigned long long * calcPerAdd, // the number of calculations that will be needed for each non-empty index
                        unsigned int * numPointsInAdd, // the number of points in each non-empty index
                        unsigned int * nonEmptyBins,
                        unsigned int * binNumbers,
                        unsigned int * tempAdd,
                        unsigned int * numSearches){ 
    
    // guiding function for dynamic programing of either the binary searches or tree traversals

    //assign one thread to each address
    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    // if the tid is greawter than the number of non empty bins then return
    if(tid >= *nonEmptyBins){
        return;
    }

    //searches can be perfomed with either dynamic programming or witht he single thread sequentialy

    #if BINARYSEARCH
        // call the GPU binary sesarch with dynamic programing

        //call the GPU binary search for a single thread
        GPUBinarySearch( tid, // the bin to search in bin numebrs
                        tempAdd, //temporary address for searching
                        binNumbers, //array of bin numbrs 
                        *nonEmptyBins, //size of binNumebrs
                        *numLayers, //number of reference points
                        lastTreeLayer,
                        binAmounts, // the range of bins from a reference points, i.e. range / epsilon
                        addIndexes, // location of nonempty bin in tree
                        calcPerAdd, // for keeping track of the number of distance calculations to be performed
                        numValidRanges, // the number of adjacent non-empty addresses/indexes
                        rangeIndexes, // this addresses/index's array to keep track of adjacent indexes
                        rangeSizes, // the number of points in the indexes in localRangeIndexes
                        numPointsInAdd, // the number of points in this nonempty address
                        *numSearches); // the number of searches that need to be performed, 3^r

    #else // do a tree traversal

        // call the GPU tree traversal with dynamic programing

        //call the GPU tree traversal with a single thread
        GPUTreeTraversal(tid,
                        tempAdd, // array of int for temp storage for searching
                        tree, // pointer to the tree made with buildTree()
                        lastTreeLayer,
                        binSizes, // the widths of each layer of the tree measured in bins
                        binAmounts, // the range of bins from a reference points, i.e. range / epsilon
                        binNumbers, // the address/bin numbers of the current index/address
                        *numLayers, // the number of reference points or layers in the tree, same thing
                        calcPerAdd, // for keeping track of the number of distance calculations to be performed
                        numValidRanges, // the number of adjacent non-empty addresses/indexes
                        rangeIndexes, // this addresses/index's array to keep track of adjacent indexes
                        rangeSizes, // the number of points in the indexes in localRangeIndexes
                        numPointsInAdd, // the number of points in this nonempty address
                        *numSearches); // the number of searches that need to be performed, 3^r

    #endif


}











__device__
void GPUBinarySearch(	unsigned int searchIndex, // the bin to search in bin numbers
                    unsigned int  * tempAdd, //temporary address for searching
                    unsigned int * binNumbers, //array of bin numbrs 
                    unsigned int nonEmptyBins, //size of binNumebrs
                    unsigned int numLayers, //number of reference points
                    unsigned int * lastTreeLayer, //the last layer of the tree structure
                    unsigned int * binAmounts, // the range of bins from a reference points, i.e. range / epsilon
                    unsigned int * addIndexs, //location of nonempty bins in the tree
                    unsigned long long * numCalcs, // the place to retrun the number of calcs that will be needed
                    unsigned int * numberRanges, // the return location for the number of adjacent non-empty indexes
                    unsigned int * rangeIndexes, // the array of non-empty adjacent index locations
                    unsigned int * rangeSizes, // the number of points in each of the adjacent non-empty indexes
                    unsigned int * numPointsInAdd, //the number of points in the home address/iondex
                    unsigned int numSearches){ //the number of searches that are being perfomred for each addresss


    //keep track of the number of calcs that will be needed
    unsigned long long localNumCalcs = 0;

    // keep track of the number of non-empty adjacent indexes
    unsigned int localNumRanges = 0;

    //permute through bin variations (3^r) and run depth searches
    for(unsigned int i = 0; i < numSearches; i++){

        //modify temp add for the search based on our itteration i
        for(unsigned int j = 0; j < numLayers; j++){
            tempAdd[searchIndex*numLayers + j] = binNumbers[searchIndex*numLayers + j] + (i / (int)pow((double)3, (double)j) % 3)-1;
        }
        //perform the search and get the index location of the return 
        long int index = GPUBSearch(searchIndex, &tempAdd[searchIndex*numLayers], binNumbers, nonEmptyBins, numLayers);

        //check if the index location was non empty
        if(index >= 0){
            unsigned int newIndex = addIndexs[index];
            //store the non empty index location
            rangeIndexes[searchIndex*numSearches + localNumRanges] = newIndex;

            //calcualte the size of the index, i.e. the number of points in the index
            unsigned long long size = lastTreeLayer[newIndex+1] - lastTreeLayer[newIndex]; //may need to +- to index here!!!!!!!!!!!!!!!

            //store that in the sizes array
            rangeSizes[searchIndex*numSearches + localNumRanges] = size;

            // keep running total of the sizes for getting the number of calculations latter
            localNumCalcs += size;

            //keep track of the number of non-empty adjacent indexes
            localNumRanges++;
        }
    }

    // get the index of the home address
    long int homeIndex = addIndexs[GPUBSearch(searchIndex, &binNumbers[searchIndex*numLayers], binNumbers, nonEmptyBins, numLayers)];
    // find the number of points in the home address
    *numPointsInAdd = lastTreeLayer[homeIndex+1] - lastTreeLayer[homeIndex]; //may need to +- one to index here !!!!!!!!!
    // use the running total of points in adjacent addresses and multiply it by the number of points in the home address for number of total calcs
    *numCalcs = localNumCalcs*(*numPointsInAdd);
    
    *numberRanges = localNumRanges;
}


__device__
void GPUTreeTraversal(unsigned int tid,
                    unsigned int * tempAdd, //twmp array for the address being searched
                    unsigned int * tree, // the pointer to the tree
                    unsigned int * lastTreeLayer, // pointer to the last layer of the tree
                    unsigned int * binSizes, // the width of the tree for each layer mesuared in number of bins
                    unsigned int * binAmounts, // the number of bins for each reference point
                    unsigned int * binNumbers, // the bin number for the home address
                    unsigned int numLayers, // the number of reference points/layers in the tree
                    unsigned long long * numCalcs, // the place to retrun the number of calcs that will be needed
                    unsigned int * numberRanges, // the return location for the number of adjacent non-empty indexes
                    unsigned int * rangeIndexes, // the array of non-empty adjacent index locations
                    unsigned int * rangeSizes, // the number of points in each of the adjacent non-empty indexes
                    unsigned int  * numPointsInAdd, //the number of points in the home address/iondex
                    unsigned int numSearches){ //the number of searches that are being perfomred for each addresss

    //keep track of the number of calcs that will be needed
    unsigned long long localNumCalcs = 0;

    // keep track of the number of non-empty adjacent indexes
    unsigned int localNumRanges = 0;

    //permute through bin variations (3^r) and run depth searches
    for(unsigned int i = 0; i < numSearches; i++){

        //modify temp add for the search based on our itteration i
        for(unsigned int j = 0; j < numLayers; j++){
            tempAdd[tid*numLayers+j] = binNumbers[tid*numLayers+j] + (i / (int)pow((double)3, (double)j) % 3)-1;
        }

        //perform the search and get the index location of the return 

        int index = GPUDepthSearch(tid, tree, binSizes, binAmounts, numLayers, &tempAdd[tid*numLayers]);

        //check if the index location was non empty
        if(index >= 0){
            unsigned int newIndex = index;
            //store the non empty index location
            rangeIndexes[tid*numSearches + localNumRanges] = newIndex;

            //calcualte the size of the index, i.e. the number of points in the index
            unsigned long long size = lastTreeLayer[newIndex+1] - lastTreeLayer[newIndex]; //may need to +- to index here!!!!!!!!!!!!!!!

            //store that in the sizes array
            rangeSizes[tid*numSearches+localNumRanges] = size;

            // keep running total of the sizes for getting the number of calculations latter
            localNumCalcs += size;

            //keep track of the number of non-empty adjacent indexes
            localNumRanges++;
        }
    }

    // get the index of the home address
    int homeIndex = GPUDepthSearch(tid, tree, binSizes, binAmounts, numLayers, &binNumbers[tid*numLayers]);

    // find the number of points in the home address
    *numPointsInAdd = lastTreeLayer[homeIndex+1] - lastTreeLayer[homeIndex]; //may need to +- one to index here !!!!!!!!!

    // use the running total of points in adjacent addresses and multiply it by the number of points in the home address for number of total calcs
    *numCalcs = localNumCalcs*(*numPointsInAdd);

    *numberRanges = localNumRanges;

}






__device__
int GPUDepthSearch(unsigned int tid,
                    unsigned int * tree, //pointer to the tree built with buildTree()
                    unsigned int * binSizes,
                    unsigned int * binAmounts, // the number of bins for each reference point, i.e. range/epsilon
                    unsigned int numLayers, //the number of layers in the tree
                    unsigned int * searchBins){ // the bin number that we are searching for

    // the offset is used for keeping track of the offset from the begining of each layer to the index
    unsigned int offset = 0;
    unsigned int layerOffset = 0;
    //go through each layer up to the last to determine if the index is non-empty and if it is then find the offset into the next layer
    for(unsigned int i = 0; i < numLayers-1; i++){

        //check the current layer at the bin number + offset may or may not need -1 here
        if (tree[layerOffset +offset + searchBins[i]] == 0){
            return -2;
        }

        // the next offset will be the previous layer index number * the number of bins for the reference point in the next layer
        offset = (tree[layerOffset + searchBins[i]+offset]-1)*binAmounts[i+1];

        layerOffset += binSizes[i];
    }

    //the index will be the last layers bin number plus the offset for the last layer
    long int index = searchBins[ numLayers-1]+offset;

    //if last layer has poionts then return the index value
    if(tree[layerOffset + index] < tree[layerOffset+index+1]){
        return index;
    }else{
        return -1;
    }

}


__device__
int GPUBSearch(unsigned int tid,
            unsigned int * tempAdd, //address to search for
            unsigned int * binNumbers, //array of addresses
            unsigned int nonEmptyBins, //number of bins
            unsigned int numLayers) //number of layers or size of addresses
            {

    // initial conditions of the search
    unsigned int left = 0;
    unsigned int right = nonEmptyBins-1;

    //while loop for halving search each itterations
    while(left <= right){
        //calculate the middle
        unsigned int mid = (left + right)/2;
        // -1 for smaller, 1 for larger, 0 for equal
        int loc = 0; //compareBins( binNumbers[mid], tempAdd, numLayers);

        for(unsigned int i = 0; i < numLayers; i++){
            if(binNumbers[mid*numLayers + i] < tempAdd[ i]){
                left = mid + 1;
                break;
            }
            if(binNumbers[mid*numLayers+i] > tempAdd[ i]){
                right = mid - 1;
                break;
            }
        }

        //if we found the index
        if( loc == 0) return mid;

    }

    return -1;

}



//unique key array on the GPU
__global__ 
void kernelUniqueKeys(unsigned int * pointIDKey,
                      unsigned long long int * N, 
                      unsigned int * uniqueKey, 
                      unsigned int * uniqueKeyPosition, 
                      unsigned int * cnt)
{
	unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

	if (tid >= *N){
		return;
	}

	if (tid == 0)
	{
		unsigned int idx = atomicAdd(cnt,(unsigned int)1);
		uniqueKey[idx] = pointIDKey[0];
		uniqueKeyPosition[idx] = 0;
		return;
	
	}
	
	//All other threads, compare to previous value to the array and add
	
	if (pointIDKey[tid-1] != pointIDKey[tid])
	{
		unsigned int idx = atomicAdd(cnt,(unsigned int)1);
		uniqueKey[idx] = pointIDKey[tid];
		uniqueKeyPosition[idx] = tid;
	}
	
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

    // unsigned int currentAdd = addAssign[tid]; 
    // unsigned int threadOffset = threadOffsets[tid];
    // unsigned int startingRangeID = linearRangeID[currentAdd];

    for(unsigned int i = 0; i < numValidRanges[addAssign[tid]]; i++){
        // unsigned long long int numCalcs = (unsigned long long int)rangeSizes[startingRangeID + i] * numPointsInAdd[currentAdd];
        for(unsigned long long int j = threadOffsets[tid]; j < (unsigned long long int)rangeSizes[linearRangeID[addAssign[tid]] + i] * numPointsInAdd[addAssign[tid]]; j += numThreadsPerAddress[addAssign[tid]]){

            unsigned int p1 = addIndexRange[addAssign[tid]] + j / rangeSizes[linearRangeID[addAssign[tid]] + i];
            unsigned int p2 = rangeIndexes[linearRangeID[addAssign[tid]] + i] + j % rangeSizes[linearRangeID[addAssign[tid]]+ i];

            // double sum = 0;
            // for(unsigned int i = 0; i < *dim; i++){
            //     #if DATANORM
            //     sum += pow(data[i*(*numPoints) + p1] - data[i*(*numPoints) + p2], 2);
            //     #else
            //     sum += pow(data[p1*(*dim)+i]-data[p2*(*dim)+i],2);
            //     #endif
            //     if(sum > *epsilon2) break;
            // }

            if (distanceCheck((*epsilon2), (*dim), data, p1, p2, (*numPoints))){
            // if (sum <= *epsilon2){
                //  store point
                unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
                point_a[index] = p1; //stores the first point Number
                point_b[index] = p2; // this stores the coresponding point number to form a pair
            }
        }
    }
}

__global__ 
void nodeCalculationsKernel(unsigned int *numPoints,
                                unsigned int * pointOffsets,
                                unsigned int * nodeAssign,
                                unsigned int * threadOffsets,
                                double *epsilon2,
                                unsigned int *dim,
                                unsigned long long  *numThreadsPerBatch,
                                unsigned long long  * numThreadsPerNode,
                                double * data, 
                                unsigned int * numNeighbors,
                                unsigned int * nodePoints,
                                unsigned int * neighbors,
                                unsigned int * neighborOffset,
                                unsigned long long  *keyValueIndex,
                                unsigned int * point_a,
                                unsigned int * point_b){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if(tid >= *numThreadsPerBatch){
        return;
    }
    // unsigned int assignedNode = nodeAssign[tid];

    for(unsigned int i = 0; i < numNeighbors[ nodeAssign[tid]]; i++){
        unsigned int neighborIndex = neighbors[neighborOffset[nodeAssign[tid]] + i];
        // unsigned int neighborPoints = nodePoints[neighborIndex];
        // unsigned long long numCals = (unsigned long long int)nodePoints[ nodeAssign[tid]]* nodePoints[ neighbors[neighborOffset[ nodeAssign[tid]] + i]];
        for(unsigned long long int j = threadOffsets[tid]; j < (unsigned long long int)nodePoints[ nodeAssign[tid]]* nodePoints[ neighborIndex]; j += numThreadsPerNode[ nodeAssign[tid]]){

            unsigned int p1 = pointOffsets[ nodeAssign[tid]] + j / nodePoints[ neighborIndex];
            unsigned int p2 = pointOffsets[ neighborIndex] + j % nodePoints[ neighborIndex];

            if (distanceCheck((*epsilon2), (*dim), data, p1, p2, (*numPoints))){
            // if (sum <= *epsilon2){
                //  store point
                unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
                point_a[index] = p1; //stores the first point Number
                point_b[index] = p2; // this stores the coresponding point number to form a pair
            }
        }
    }
}

void nodeCalculationsKernel_CPU( unsigned int numNodes,
                            unsigned int totalBlocks,
                            unsigned int *numPoints,
                            unsigned int * pointOffsets,
                            unsigned int * nodeAssign,
                            unsigned int * threadOffsets,
                            double *epsilon2,
                            unsigned int *dim,
                            unsigned long long  *numThreadsPerBatch,
                            unsigned long long  * numThreadsPerNode,
                            double * data, 
                            unsigned int * numNeighbors,
                            unsigned int * nodePoints,
                            unsigned int * neighbors,
                            unsigned int * neighborOffset,
                            unsigned long long  *keyValueIndex){

    #pragma omp parallel for
    for(unsigned int h = 0; h < BLOCK_SIZE*totalBlocks; h++)
    {
        unsigned int tid = h;
        if(tid < *numThreadsPerBatch){
            // if(nodeAssign[tid]>numNodes) printf("ERROR0: %u / %u", nodeAssign[tid],numNodes);
            for(unsigned int i = 0; i < numNeighbors[nodeAssign[tid]]; i++){
                unsigned int neighborIndex = neighbors[neighborOffset[nodeAssign[tid]]+i];
                for(unsigned long long int j = threadOffsets[tid]; j < (unsigned long long int)nodePoints[nodeAssign[tid]]* nodePoints[neighborIndex]; j += numThreadsPerNode[nodeAssign[tid]]){
    
                    unsigned int p1 = pointOffsets[nodeAssign[tid]] + j / nodePoints[neighborIndex];
                    unsigned int p2 = pointOffsets[neighbors[neighborOffset[nodeAssign[tid]] + i]] + j % nodePoints[neighborIndex];
    
                    // if(p1 > *numPoints) printf("ERROR1: %u, %u, %u, %u\n", p1, nodeAssign[tid], neighborIndex, pointOffsets[nodeAssign[tid]]);
                    // if(p2 > *numPoints) printf("ERROR2: %u, %u, %u, %u\n", p2, nodeAssign[tid], neighborIndex, pointOffsets[nodeAssign[tid]]);
    
                    if (distanceCheck((*epsilon2), (*dim), data, p1, p2, (*numPoints))){
    
                        #pragma omp critical
                        {
                            keyValueIndex++;
                        }
                    }
                }
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


    #pragma omp parallel for
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
                unsigned long long int numCalcs = (unsigned long long int)rangeSizes[startingRangeID + i] * numPointsInAdd[currentAdd];

                //each thread starts at its offset then increments by the number of threads for the address untill past the numebr of calcs
                for(unsigned long long int j = threadOffset; j < numCalcs; j += numThreadsPerAddress[currentAdd]){
        
                    // the first point will be from the home address
                    // the starting point from addIndexRange plus the floor of the itteration over the number of points in the adjacent address
                    // once all of the threads have made the calcs for a point they all move to the next point in the home address
                    unsigned int p1 = addIndexRange[currentAdd] + j / rangeSizes[startingRangeID + i];
                    unsigned int p2 = rangeIndexes[startingRangeID + i] + j % rangeSizes[startingRangeID+ i];

                    if (distanceCheck((*epsilon2), (*dim), data, p1, p2, (*numPoints))){
                        //  store point

                        #pragma omp critical
                        {
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
}

__global__ 
void nodeByPoint( double * data, //
                  double * epsilon2,//
                  unsigned int * numPoints, //
                  unsigned int * batchPoints, //
                  unsigned int * nodeID, //
                  unsigned int * numNeighbors, //
                  unsigned int * numPointsNode, //
                  unsigned int * neighborNodes, //
                  unsigned int * neighborOffset, //
                  unsigned int * pointOffset, //
                  unsigned int * point_a, //
                  unsigned int * point_b, //
                  unsigned long long * keyValueIndex){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if(tid >= KERNEL_BLOCKS*BLOCK_SIZE / TPP || tid/TPP + (*batchPoints) >= (*numPoints) ){
        return;
    }

    //assign thread to a point
    unsigned int point = (*batchPoints) + tid/TPP;
    unsigned int node = nodeID[point];

    //for every neighboring node
    for(unsigned int i = 0; i < numNeighbors[node]; i++){
        
        unsigned int neighborNodeIndex = neighborNodes[neighborOffset[node]+i];

        for(unsigned int j = 0; j < numPointsNode[neighborNodeIndex]; j++){
            unsigned int comparePoint = pointOffset[neighborNodeIndex]+j;
            
            if (distanceCheck((*epsilon2), DIM, data, point, comparePoint, (*numPoints))){
                // if (sum <= *epsilon2){
                    //  store point
                    unsigned long long int index = atomicAdd(keyValueIndex,(unsigned long long int)1);
                    point_a[index] = point; //stores the first point Number
                    point_b[index] = comparePoint; // this stores the coresponding point number to form a pair
                }
        }
    }

}


__host__ __device__ //may need to switch to inline (i did)
inline bool distanceCheck(double epsilon2, unsigned int dim, double * data, unsigned int p1, unsigned int p2, unsigned int numPoints){
    
    double sum[8];
    
    #pragma unroll
    for(unsigned int i = 0; i < 8; i++){
        sum[i] = 0;
    }

    for(unsigned int i = 0; i < DIM; i+=8){
        
        #pragma unroll
        for(unsigned int j = 0; j < 8 && (i + j) < DIM; j++){
            sum[j] += (data[(i+j)*numPoints + p1] - data[(i+j)*numPoints + p2])*(data[(i+j)*numPoints + p1] - data[(i+j)*numPoints + p2]);
        }

        #pragma unroll
        for(unsigned int j = 1; j < 8; j++){
            sum[0] += sum[j];
            sum[j] = 0;
        }

        if(sum[0] > epsilon2) return false;
        
    }

    // #pragma unroll
    // for(unsigned int i = 0; i < dim%8; i++){
    //     sum[i] += (data[(dim/8*8+i)*numPoints + p1] - data[(dim/8*8+i)*numPoints + p2])*(data[(dim/8*8+i)*numPoints + p1] - data[(dim/8*8+i)*numPoints + p2]);
    // }

    // #pragma unroll
    // for(unsigned int j = 1; j < dim%8; j++){
    //     sum[0] += sum[j];
    // }

    // if(sum[0] > epsilon2) return false;

    return true;
}

__global__
void binningKernel(unsigned int * binNumbers, //array numPoints long
                    unsigned int * numPoints,
                    unsigned int * dim,
                    double * data, //all data
                    double * RP, //single rp
                    double * epsilon){

    unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;

    if(tid >= *numPoints){
        return;
    }

    for(unsigned int i = 0; i < RPPERLAYER; i++){

        double distance = 0;
        for(unsigned int j = 0; j < *dim; j++){
            // distance += pow(data[tid*(*dim) + j]-RP[j + (*dim)*i],2);
            distance += (data[tid*(*dim) + j]-RP[j + (*dim)*i]) * (data[tid*(*dim) + j]-RP[j + (*dim)*i]);
        }
    
        
        binNumbers[tid+(*numPoints)*i] = floor(sqrt(distance) / (*epsilon));
    
    }

    return;
}