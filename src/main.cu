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
	char *filename = argv[1]; // first argument is the file with the dataset as a .bin
	int dim = atoi(argv[2]); // second argument is the dimensionality of the data, i.e. number of columns
	int concurent_streams = 2; // number of cuda streams, should only ever need to be 2 but can be set to a parameter
	double epsilon;
	sscanf(argv[3], "%lf", &epsilon); // third argumernt is the distance threshold being searched

	double time0 = omp_get_wtime(); //start initial timer

	//read in file from binary, only works with doubles if file saved as doubles
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

	int numPoints = size/sizeof(double)/dim; // calculate number of points based on the siez of the input

	// can set a subset of the data for easier debugging
	//////////////
	// numPoints = 10000;
	////////////

	printf("\nNumber points: %d ", numPoints);
	printf("\nNumber Dimensions: %d ", dim);
	printf("\nNumber Concurent Streams: %d", concurent_streams);
	printf("\nDistance Threshold: %f \n*********************************\n\n", epsilon);

	//if using a small datset for debugging, also run brute force so we can double check results
	if(numPoints <= 10000) 	brute_force( numPoints, dim, epsilon, A);

	double time1 = omp_get_wtime();

	//dimensionOrder holds the order of dimensions sorted by thier varience
	int *dimensionOrder = (int*)malloc(sizeof(int)*dim);

	//dimOrderedData holds the dataset after it has been reordered based on dimensional varience
	double * dimOrderedData = (double*)malloc(sizeof(double)*numPoints*dim);

    dimensionOrder = stddev(A, dim, numPoints); //find the order of dimensions by varience

	// reorder the origional data into the dimesionaly ordered data
	// this makes earlier columns of the data have higher varience than later columns
	// the points maintain thier order relative to other points
	// this can increase short circuiting because higher variences are calculated earlier
    #pragma omp parallel for
    for(int i = 0; i < numPoints; i++){
        for(int j = 0; j < dim; j++){
            dimOrderedData[i*dim + j] = A[i*dim + dimensionOrder[j]];
        }
    }

	// allocate and set an array to keep the order of the points
	// this allows us to refer to the row of the intial data when returning pairs
	int * pointArray = (int*)malloc(sizeof(int)*numPoints);
	for (int i = 0; i < numPoints; i++){
		pointArray[i] = i;
	}


	// poinmt bin numbers holds the bins relative to reference points that each point is in
	int ** pointBinNumbers;

	// binSizes is the number of bins for each layer of the tree , which includes the spread from the previous layer
	unsigned int * binSizes = (unsigned int*)malloc(sizeof(unsigned int)*MAXRP);

	//bin amounts is the number of bins for that reference point, i.e. the range of points / epsilon
	unsigned int * binAmounts = (unsigned int*)malloc(sizeof(unsigned int)*MAXRP);

	//maxBinAmount limits the number of bins that can be in a layer to reduce space complexity, not usually an issue
	int maxBinAmount = MAX_BIN;

	//this will be the tree structure of pointers to layers
	int ** tree;

	//build the tree into tree and returns the number of layers that was selected for the tree
	int numLayers = buildTree(
					&tree, // outputs into this pointer
					dimOrderedData, //uses the data after ordered for dimensions
					dim, // the dimensionality of the data
					numPoints, // the number of points in the dataset
					epsilon, // the distance threshold being searched
					maxBinAmount, // the limmiter on tree width in number of bins
					pointArray, // the ordered points, this will be rearanged when building the tree
					&pointBinNumbers, // this will hold of the bin number for each point relative to each reference point
					binSizes, // the width for each layer of the tree which is built in the fuinction
					binAmounts); //the number of bins for each reference point as in the range / epsilon


	// allocate a data array for used with distance calcs
	// the data is moved around so that point in bin are near eachother in the array
	// the location is based  on the point array that was altered during tree construction
	// data can be organized 2 ways:
	// 1. if DATANORM = true
	//    the data is organized so that the the dimensions of each point are next to eachother
	//	  this allows for coalesced memory accsess on the gpu to increase perfomance
	//
	// 2. if DATANORM is false
	//	  this is the standard stride that was used after dimensional ordering
    double * data = (double *)malloc(sizeof(double)*numPoints*dim);
    #pragma omp parallel for
	for(int i = 0; i < numPoints; i++){
		for(int j = 0; j < dim; j++){
			#if DATANORM
			data[i+numPoints*j] = dimOrderedData[pointArray[i]*dim+j];
			#else
			data[i*dim+j] = dimOrderedData[pointArray[i]*dim+j];
			#endif
		}
	}





	// checking that the last bin size is not negative or zero and that the tree has every data point in it
	printf("Last Layer Bin Count: %d\nTree Check: %d\n",binSizes[numLayers-1], tree[numLayers-1][binSizes[numLayers-1]-1]);

	double time2 = omp_get_wtime();

    printf("Time to build tree: %f\n", time2-time1);


	// addIndexes holds the return from generating ranges which contains the non-empty index locations in the last layer of tree
    int * addIndexes;

	// rangeIndexes holds the return from generating ranges that correspond to the non-empty indexes 
	// and has the adjacent non-empty index locations
    int ** rangeIndexes;

	// range sizes has the number of points in the adjacent ranges in rangeIndexes
    unsigned int ** rangeSizes;

	// the number of adjacent indexes for each noin-empty index
    int * numValidRanges;

	//the number of distance calculations that will be needed for each non-empty index
    unsigned long long * calcPerAdd;

	// the number of points in each non-empty index
	unsigned int *numPointsInAdd;

	//generate the ranges and perform the searches
    int nonEmptyBins = generateRanges(tree, //the tree arrays pointer
									  numPoints, //the number of points in the dataset
									  pointBinNumbers, // the bin numbers for each point
									  numLayers, // the number of layers of the tree
									  binSizes, // the width of each layer of the tree
									  binAmounts, // the number of bins for each reference point
									  &addIndexes, // for returning the addIndex info
									  &rangeIndexes, //for returning the range index info
									  &rangeSizes,// for returning the range size info
									  &numValidRanges, //for returning the valid range count
									  &calcPerAdd, // for returning the number of clacs for each address
									  &numPointsInAdd); //for returning the number of points in each non-empty index

	// keep track of the number of total calcs needed
    unsigned long long sumCalcs = 0;

	//keep track of the number of address that were found in searching for each address
    unsigned long long sumAdds = 0;

	//itterating through just to find sum values
    for(int i = 0; i < nonEmptyBins; i++){
        sumCalcs += calcPerAdd[i];
        sumAdds += numValidRanges[i];
    }

	// add index range has the values at each nonempty index of the last layer of the tree
	int * addIndexRange = (int*)malloc(sizeof(int)*nonEmptyBins);
	for(int i = 0; i < nonEmptyBins; i++){
		addIndexRange[i] = tree[numLayers-1][addIndexes[i]];
	}

	// an array for keeping track of where the linear indexs start for each elemetn in the 2d one
	int * linearRangeID = (int*)malloc(sizeof(int) * nonEmptyBins);

	// a linear range index made from the 2d range index array to keep track of adjacent non-empty indexes for each non-empty index
	int * linearRangeIndexes = (int*)malloc(sizeof(int)*sumAdds);

	//a linear range sizes made from the range siezes array to keep track of the number of points in adjacent indexes
	unsigned int * linearRangeSizes = (unsigned int*)malloc(sizeof(unsigned int)*sumAdds);

	//running total for keeping track of starts for each index in the linear arrays
	int runningTotal = 0;
	//linearizeing 2d arrays to 1d arrays for GPU work
	for(int i = 0; i < nonEmptyBins; i++){

		//keeping track of start locations in the linear arrays
		linearRangeID[i] = runningTotal;
		

		//populating the linear arrays from the 2d ones
		for(int j = 0; j < numValidRanges[i]; j++){
			linearRangeIndexes[runningTotal + j] = tree[numLayers-1][rangeIndexes[i][j]];
			linearRangeSizes[runningTotal + j] = rangeSizes[i][j];
		}

		//increment the running total by the number of ranges for the current index
		runningTotal += numValidRanges[i];
	}

    printf("Number non-empty bins: %d\nNumber of calcs: %llu\nNumber Address for calcs: %llu\n", nonEmptyBins, sumCalcs, sumAdds);


	double time3 = omp_get_wtime();

	printf("Tree search time: %f\n", time3-time2);


	launchKernel(numLayers, // the number of layers in the tree
				data, //the dataset that has been ordered by dimensoins and possibly reorganized for colasced memory accsess
				dim, //the dimensionality of the data
				numPoints, //the number of points in the dataset
				epsilon, //the distance threshold being searched
				addIndexes, //the non-empty index locations in the last layer of the tree
			    addIndexRange, // the value of the non empty index locations  in the last layer of the tree, so the starting point number
				pointArray, // the array of point numbers ordered to match the sequence in the last array of the tree and the data
				rangeIndexes, // the non-empty adjacent indexes for each non-empty index 
				rangeSizes, // the size of the non-empty adjacent indexes for each non-empty index
				numValidRanges, // the number of adjacent non-empty indexes for each non-empty index
				numPointsInAdd, // the number of points in each non-empty index
				calcPerAdd, // the number of calculations needed for each non-mepty index
				nonEmptyBins, //the number of nonempty indexes
				sumCalcs, // the total number of calculations that will need to be made
				sumAdds, //the total number of addresses that will be compared to by other addresses for distance calcs
				linearRangeID, // an array for keeping trackj of starting points in the linear arrays
				linearRangeIndexes, // a linear version of rangeIndexes
				linearRangeSizes); // a linear version of rangeSizes



	double time4 = omp_get_wtime();

	printf("Kernel time: %f\n", time4-time3);

	printf("Total Time: %f\n",time4-time1); //note that this does not include time to read in the data from disk to main memory









// just freeing memory here
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
	free(dimOrderedData);
	free(addIndexRange);

    return 1;

}