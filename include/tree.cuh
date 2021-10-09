#pragma once

#include "include/utils.cuh"
#include "include/params.cuh"

int buildTree(int *** rbins, double * data, int dim, unsigned long long numPoints, double epsilon, int maxBinAmount,  int * pointArray, int *** rpointBinNumbers, unsigned int * binSizes, unsigned int * binAmounts);

__host__ __device__
void treeTraversal(int * tempAdd, int ** tree, unsigned int * binSizes, unsigned int * binAmounts, int * binNumbers, int numLayers, unsigned long long * numCalcs, int * numberRanges, int ** rangeIndexes, unsigned int ** rangeSizes, unsigned int * numPointsInAdd, unsigned int numSearches);

__host__ __device__
int depthSearch(int ** tree, unsigned int * binAmounts, int numLayers, int * searchBins);


int generateRanges(int ** tree, int numPoints, int ** pointBinNumbers, int numLayers, unsigned int * binSizes, unsigned int * binAmounts, int ** addIndexes, int *** rangeIndexes, unsigned int *** rangeSizes, int ** numValidRanges, unsigned long long ** calcPerAdd, unsigned int ** numPointsInAdd );
