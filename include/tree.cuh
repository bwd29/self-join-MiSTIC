#pragma once

#include "include/utils.cuh"
#include "include/params.cuh"

int buildTree(int *** rbins, double * data, int dim, unsigned long long numPoints, double epsilon, int maxBinAmount,  int * pointArray, int *** rpointBinNumbers, unsigned int * binSizes, unsigned int * binAmounts);

// __host__ __device__
void treeTraversal(int * tempAdd, int ** tree, unsigned int * binSizes, unsigned int * binAmounts, int * binNumbers, int numLayers, unsigned long long * numCalcs, int * numberRanges, int ** rangeIndexes, unsigned int ** rangeSizes, unsigned int * numPointsInAdd, unsigned int numSearches);

// __host__ __device__
int depthSearch(int ** tree, unsigned int * binAmounts, int numLayers, int * searchBins);


int generateRanges(int ** tree, int numPoints, int ** pointBinNumbers, int numLayers, unsigned int * binSizes, unsigned int * binAmounts, int ** addIndexes, int *** rangeIndexes, unsigned int *** rangeSizes, int ** numValidRanges, unsigned long long ** calcPerAdd, unsigned int ** numPointsInAdd );

// __host__ __device__
int bSearch(int * tempAdd, //address to search for
			int ** binNumbers, //array of addresses
			int nonEmptyBins, //number of bins
			int numLayers); //numebr of layers or size of addresses

// __host__ __device__
inline int compareBins(int * bin1, int * bin2, int binSize);

// __host__ __device__
void binarySearch(	int searchIndex, // the bin to search in bin numebrs
				 	int  * tempAdd, //temporary address for searching
				  	int ** binNumbers, //array of bin numbrs 
					int nonEmptyBins, //size of binNumebrs
					int numLayers, //number of reference points
					int ** tree, //the tree structure
					unsigned int * binAmounts, // the range of bins from a reference points, i.e. range / epsilon
					int * addIndexes, //location of nonempty bins in the tree
					unsigned long long * numCalcs, // the place to retrun the number of calcs that will be needed
					int * numberRanges, // the return location for the number of adjacent non-empty indexes
					int ** rangeIndexes, // the array of non-empty adjacent index locations
					unsigned int ** rangeSizes, // the number of points in each of the adjacent non-empty indexes
					unsigned int * numPointsInAdd, //the number of points in the home address/iondex
					unsigned int numSearches);