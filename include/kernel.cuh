#pragma once

#include "include/params.cuh"

// #include "include/utils.cuh"




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
    unsigned int * linearRangeSizes); // a linear version of rangeSizes


__host__ __device__ //may need to switch to inline
bool distanceCheck(double epsilon2, int dim, double * data, unsigned int p1, unsigned int p2, unsigned int numPoints);


__global__ 
void distanceCalculationsKernel(unsigned int *numPoints,
                                int * linearRangeID,
                                unsigned int * addAssign,
                                unsigned int * threadOffsets,
                                double *epsilon2,
                                int *dim,
                                unsigned long long *numThreadsPerBatch,
                                unsigned long long * numThreadsPerAddress,
                                double * data, 
                                int * numValidRanges,
                                int * rangeIndexes,
                                unsigned int * rangeSizes,
                                unsigned int * numPointsInAdd,
                                int * addIndexRange,
                                unsigned long long *keyValueIndex,
                                unsigned int * point_a,
                                unsigned int * point_b);

void distanceCalculationsKernel_CPU(unsigned int totalBlocks,
    unsigned int *numPoints,
    int * linearRangeID,
    unsigned int * addAssign,
    unsigned int * threadOffsets,
    double *epsilon2,
    int *dim,
    unsigned long long  *numThreadsPerBatch,
    unsigned long long  * numThreadsPerAddress,
    double * data,
    int *addIndexes,
    int * numValidRanges,
    int * rangeIndexes,
    unsigned int * rangeSizes,
    unsigned int * numPointsInAdd,
    int * addIndexRange,
    int * pointArray,
    unsigned long long  *keyValueIndex,
    std::vector<int> * hostPointA,
    std::vector<int> * hostPointB);
