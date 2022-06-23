#pragma once

#include "include/params.cuh"

__global__ 
void kernelUniqueKeys(unsigned int * pointIDKey,
                      unsigned long long int * N, 
                      unsigned int * uniqueKey, 
                      unsigned int * uniqueKeyPosition, 
                      unsigned int * cnt);



__host__ __device__ //may need to switch to inline
bool distanceCheck(double epsilon2, 
                   unsigned int dim,
                   double * data,
                   unsigned int p1, 
                   unsigned int p2, 
                   unsigned int numPoints);


__global__ 
void distanceCalculationsKernel(unsigned int *numPoints,
                                unsigned int * linearRangeID,
                                unsigned int * addAssign,
                                unsigned int * threadOffsets,
                                double *epsilon2,
                                unsigned int *dim,
                                unsigned long long *numThreadsPerBatch,
                                unsigned long long * numThreadsPerAddress,
                                double * data, 
                                unsigned int * numValidRanges,
                                unsigned int * rangeIndexes,
                                unsigned int * rangeSizes,
                                unsigned int * numPointsInAdd,
                                unsigned int * addIndexRange,
                                unsigned long long *keyValueIndex,
                                unsigned int * point_a,
                                unsigned int * point_b);

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
    std::vector<unsigned int> * hostPointB);
