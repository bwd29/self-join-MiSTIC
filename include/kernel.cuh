#pragma once

#include "include/params.cuh"

__device__ //may need to switch to inline
bool distanceCheck(double epsilon2, double dim, double * p1, double * p2);


__device__ 
void distanceCalculationsKernel(double * data, int *addIndexes, int * numValidRanges, int ** rangeIndexes, int ** rangeSizes, unsigned int * numPointsInAdd, int * addIndexRange, unsigned long long *keyValueIndex, unsigned int * point_a, unsigned int * point_b);
