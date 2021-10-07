#pragma once

#include "include/params.cuh"
// #include "include/utils.cuh"

__device__ //may need to switch to inline
bool distanceCheck(double epsilon2, int dim, double * data, unsigned int p1, unsigned int p2, unsigned int numPoints);


__global__
void distanceCalculationsKernel(unsigned int *numPoints, unsigned int *numSearches, int * addAssign, int * threadOffsets, double *epsilon2, int *dim, unsigned int *numThreadsPerBatch, unsigned int * numThreadsPerAddress, double * data, int *addIndexes, int * numValidRanges, int * rangeIndexes, unsigned int * rangeSizes, unsigned int * numPointsInAdd, int * addIndexRange, int * pointArray, unsigned long long int *keyValueIndex, unsigned int * point_a, unsigned int * point_b);

void launchKernel(int numLayers, double * data, int dim, int numPoints, double epsilon, int * addIndexes, int * addIndexRange, int * pointArray, int ** rangeIndexes, unsigned int ** rangeSizes, int * numValidRanges, unsigned int * numPointsInAdd, unsigned long long *calcPerAdd, int nonEmptyBins, unsigned long long sumCalcs, unsigned long long sumAdds, int * linearRangeIndexes, unsigned int * linearRangeSizes);
