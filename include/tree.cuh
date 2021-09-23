#pragma once

#include "include/utils.cuh"
#include "include/params.cuh"

int buildTree(int *** rbins, double * data, int dim, unsigned long long numPoints, double epsilon, int maxBinAmount,  int * pointArray, int *** rpointBinNumbers, unsigned int * binSizes, unsigned int * binAmounts);

__device__ 
int searchTree_linear(int * tree, int numRP, int * searchAddress, int * binSizes, int * binAmounts,  int numPoints, int * range);