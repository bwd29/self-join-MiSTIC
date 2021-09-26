#pragma once

#include "include/utils.cuh"
#include "include/params.cuh"

int buildTree(int *** rbins, double * data, int dim, unsigned long long numPoints, double epsilon, int maxBinAmount,  int * pointArray, int *** rpointBinNumbers, unsigned int * binSizes, unsigned int * binAmounts);

void treeTraversal(int ** tree, unsigned int * binSizes, unsigned int * binAmounts, int * binNumbers, int numLayers, int numPoints, unsigned long long * numCalcs, int * numberRanges, int ** rangeIndexes, unsigned int ** rangeSizes);

int depthSearch(int ** tree, unsigned int * binSizes, unsigned int * binAmounts, int numLayers, int currentLayer, int initalOffset, int numPoints, int * searchBins, int * rangeIndexResult);


int generateRanges(int ** tree, int numPoints, int ** pointBinNumbers, int numLayers, unsigned int * binSizes, unsigned int * binAmounts, int ** addIndexes, int *** rangeIndexes, unsigned int *** rangeSizes, int ** numValidRanges, unsigned long long ** calcPerAdd );