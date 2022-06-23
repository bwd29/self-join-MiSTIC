#include "include/kernel.cuh"


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

    unsigned int currentAdd = addAssign[tid]; 
    unsigned int threadOffset = threadOffsets[tid];
    unsigned int startingRangeID = linearRangeID[currentAdd];

    for(unsigned int i = 0; i < numValidRanges[currentAdd]; i++){
        unsigned long long int numCalcs = (unsigned long long int)rangeSizes[startingRangeID + i] * numPointsInAdd[currentAdd];
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