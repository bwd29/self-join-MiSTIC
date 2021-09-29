#pragma once

#include "include/params.cuh"
#include "include/utils.cuh"

__device__ //may need to switch to inline
bool distanceCheck(double epsilon2, double dim, double * p1, double * p2);


__device__ 
void distanceCalculationsKernel(int * addAssign, int * threadOffsets, const double epsilon2, const int dim, const int numThreadsPerBatch, int * numThreadsPerAddress, double * data, int *addIndexes, int * numValidRanges, int ** rangeIndexes, unsigned int ** rangeSizes, unsigned int * numPointsInAdd, int * addIndexRange, int * pointArray, unsigned long long *keyValueIndex, unsigned int * point_a, unsigned int * point_b);

void launchKernel(double * data, int dim, int numPoints, double epsilon, int * addIndexes, int * addIndexRange, int * pointArray, int ** rangeIndexes, unsigned int ** rangeSizes, int * numValidRanges, unsigned int * numPointsInAdd, unsigned long long *calcPerAdd, int nonEmptyBins, unsigned long long sumCalcs, unsigned long long sumAdds);
