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
	int tpp = 8;
	int concurent_streams = 2;
	double epsilon;
	sscanf(argv[3], "%lf", &epsilon);

	double time0 = omp_get_wtime();

	std::ifstream file(	filename, std::ios::in | std::ios::binary);
	file.seekg(0, std::ios::end); 
	size_t size = file.tellg();  
	file.seekg(0, std::ios::beg); 
	char * read_buffer = new char[size];
	file.read(read_buffer, size*sizeof(double));
	file.close();

	double time00 = omp_get_wtime();
	printf("\nTime to read in file: %f\n", time00-time0);

	double* A = (double*)read_buffer;//reinterpret as doubles

	int numPoints = size/sizeof(double)/dim;


	printf("\nNumber points: %d ", numPoints);
	printf("\nNumber Dimensions: %d ", dim);
	printf("\nNumber Threads Per Point: %d ", tpp);
	printf("\nNumber Concurent Streams: %d", concurent_streams);
	printf("\nDistance Threshold: %f \n*********************************\n\n", epsilon);




	int *dimension_order = (int*)malloc(sizeof(int)*dim);
	double * dim_ordered_data = (double*)malloc(sizeof(double)*numPoints*dim);

    dimension_order = stddev(A, dim, numPoints);
    #pragma omp parallel for
    for(int i = 0; i < numPoints; i++){
        for(int j = 0; j < dim; j++){
            dim_ordered_data[i*dim + j] = A[i*dim + dimension_order[j]];
        }
    }
    A = dim_ordered_data;;



	double time1 = omp_get_wtime();


	//build tree
	int * pointArray = (int*)malloc(sizeof(int)*numPoints);
	for (int i = 0; i < numPoints; i++){
		point_array[i] = i;
	}


	int ** pointBinNumbers;

	unsigned int binSizes[MAXRP];
	unsigned int binAmounts[MAXRP];
	int maxBinAmount = MAX_BIN;
	int ** tree;
	int numLayers = buildTree(
					&tree,
					A,
					dim,
					numPoints,
					epsilon,
					maxBinAmount,
					pointArray,
					&pointBinNumbers,
					binSizes,
					binAmounts);


    double * point_ordered_data = (double *)malloc(sizeof(double)*numPoints*dim);
    #pragma omp parallel for
	for(int i = 0; i < numPoints; i++){
		for(int j = 0; j < dim; j++){
			point_ordered_data[i*dim+j] = A[pointArray[i]*dim+j];
		}
	}
	A = point_ordered_data;


    int * addIndexes;
    int ** rangeIndexes;
    int ** rangeSizes;
    int * numValidRanges;
    int * calcPerAdd;
    int nonEmptyBins = generateRanges(tree, numPoints, pointArray, pointBinNumbers, numLayers, binSizes, binAmounts, &addIndexes, &rangeIndexes, &rangeSizes, &numValidRanges, &calcPerAdd);

    long long sumCalcs = 0;
    long long sumAdds = 0;
    for(int i = 0; i < nonEmptyBins; i++){
        sumCalcs += calcPerAdd[i];
        sumAdds += numValidRanges[i];
    }

    printf("Number non-empty bins: %d\nNumber of calcs: %ld\nNumber Address for calcs: %ld\n", nonEmptyBins, sumCalcs, sumAdds);

    double time2 = omp_get_wtime();

    printf("Time to build tree: %f\n", time2-time1);

    return 1;

}