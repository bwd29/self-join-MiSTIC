#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>



#include "include/tree.cuh"
#include "include/utils.cuh"
#include "include/kernel.cuh"


int main(int argc, char*argv[]){
    


    //reading in command line arguments
	char *filename = argv[1];
	int dim = atoi(argv[2]);
	int concurent_streams = 2;
	double epsilon;
	sscanf(argv[3], "%lf", &epsilon);

	double time0 = omp_get_wtime();

	std::ifstream file(	filename, std::ios::in | std::ios::binary);
	file.seekg(0, std::ios::end); 
	size_t size = file.tellg();  
	file.seekg(0, std::ios::beg); 
	char * read_buffer = (char*)malloc(sizeof(char)*size);
	file.read(read_buffer, size*sizeof(double));
	file.close();


	double time00 = omp_get_wtime();
	printf("\nTime to read in file: %f\n", time00-time0);

	double* A = (double*)read_buffer;//reinterpret as doubles

	int numPoints = size/sizeof(double)/dim;

	//////////////
	// numPoints =10000;
	////////////

	printf("\nNumber points: %d ", numPoints);
	printf("\nNumber Dimensions: %d ", dim);
	printf("\nNumber Concurent Streams: %d", concurent_streams);
	printf("\nDistance Threshold: %f \n*********************************\n\n", epsilon);


	int *dimensionOrder = (int*)malloc(sizeof(int)*dim);
	double * dimOrderedData = (double*)malloc(sizeof(double)*numPoints*dim);

    dimensionOrder = stddev(A, dim, numPoints);
    #pragma omp parallel for
    for(int i = 0; i < numPoints; i++){
        for(int j = 0; j < dim; j++){
            dimOrderedData[i*dim + j] = A[i*dim + dimensionOrder[j]];
        }
    }

	double time1 = omp_get_wtime();


	//build tree
	int * pointArray = (int*)malloc(sizeof(int)*numPoints);
	for (int i = 0; i < numPoints; i++){
		pointArray[i] = i;
	}


	int ** pointBinNumbers;

	unsigned int * binSizes = (unsigned int*)malloc(sizeof(unsigned int)*MAXRP);
	unsigned int * binAmounts = (unsigned int*)malloc(sizeof(unsigned int)*MAXRP);
	int maxBinAmount = MAX_BIN;
	int ** tree;
	int numLayers = buildTree(
					&tree,
					dimOrderedData,
					dim,
					numPoints,
					epsilon,
					maxBinAmount,
					pointArray,
					&pointBinNumbers,
					binSizes,
					binAmounts);


    double * data = (double *)malloc(sizeof(double)*numPoints*dim);
    #pragma omp parallel for
	for(int i = 0; i < numPoints; i++){
		for(int j = 0; j < dim; j++){
			data[i*dim+j] = dimOrderedData[pointArray[i]*dim+j];
		}
	}


	printf("Last Bin Size: %d\nTree Check: %d\n",binSizes[numLayers-1], tree[numLayers-1][binSizes[numLayers-1]-1]);

	double time2 = omp_get_wtime();

    printf("Time to build tree: %f\n", time2-time1);

    int * addIndexes;
    int ** rangeIndexes;
    unsigned int ** rangeSizes;
    int * numValidRanges;
    unsigned long long * calcPerAdd;
	unsigned int *numPointsInAdd;
    int nonEmptyBins = generateRanges(tree, numPoints, pointBinNumbers, numLayers, binSizes, binAmounts, &addIndexes, &rangeIndexes, &rangeSizes, &numValidRanges, &calcPerAdd, &numPointsInAdd);

    unsigned long long sumCalcs = 0;
    unsigned long long sumAdds = 0;
    for(int i = 0; i < nonEmptyBins; i++){
        sumCalcs += calcPerAdd[i];
        sumAdds += numValidRanges[i];
    }

	int * addIndexRange = (int*)malloc(sizeof(int)*nonEmptyBins);
	for(int i = 0; i < nonEmptyBins; i++){
		addIndexRange[i] = tree[numLayers-1][addIndexes[i]];
	}

    printf("Number non-empty bins: %d\nNumber of calcs: %llu\nNumber Address for calcs: %llu\n", nonEmptyBins, sumCalcs, sumAdds);


	double time3 = omp_get_wtime();

	printf("Tree search time: %f\n", time3-time2);


	launchKernel(numLayers, data, dim ,numPoints, epsilon, addIndexes, addIndexRange, pointArray, rangeIndexes, rangeSizes, numValidRanges, numPointsInAdd, calcPerAdd, nonEmptyBins, sumCalcs, sumAdds);














//////////////////////////////////////////////////////////////////////

	for(int i = 0; i < nonEmptyBins; i++){
		free(rangeIndexes[i]);
		free(rangeSizes[i]);
	}
	for(int i = 0; i < numLayers; i++){
		free(tree[i]);
	}
	free(tree);
	for(int i = 0; i < numPoints; i++){
		free(pointBinNumbers[i]);
	}

	free(pointBinNumbers);
	free(numValidRanges);
	free(calcPerAdd);
	free(addIndexes);
	free(rangeIndexes);
	free(rangeSizes);
	free(A);
	free(binSizes);
	free(binAmounts);
	free(pointArray);
	free(data);
	free(dimensionOrder);
	// free(read_buffer);
	free(dimOrderedData);
	free(addIndexRange);

    return 1;

}