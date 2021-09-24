#pragma once

#include "include/utils.cuh"
#include "include/params.cuh"

int buildTree(int *** rbins, double * data, int dim, unsigned long long numPoints, double epsilon, int maxBinAmount,  int * pointArray, int *** rpointBinNumbers, unsigned int * binSizes, unsigned int * binAmounts);

void generateRanges(int ** tree,int numPoints, int* pointArray, int ** pointBinNumbers, int numLayers, int * binSizes, int * binAmounts, int * addIndexes, int ** rangeIndexes, int ** rangeSizes, int * numValidRanges, int * calcPerAdd );

void treeTraversal(int ** tree, int * binSizes, int * binAmounts, int * binNumbers, int numLayers, int numPoints, int * numCalcs, int * numberRanges, int ** rangeIndexes, int ** rangeSizes);

int depthSearch(int ** tree, int * binSizes, int * binAmounts, int numLayers, int currentLayer, int initalOffset, int numPoints, int * searchBins, int * rangeIndexResult);

void generateRanges(int ** tree, int numPoints, int* pointArray, int ** pointBinNumbers, int numLayers, int * binSizes, int * binAmounts, int * addIndexes, int *** rangeIndexes, int *** rangeSizes, int * numValidRanges, int * calcPerAdd );
