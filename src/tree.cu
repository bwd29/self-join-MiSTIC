#include "include/tree.cuh"

int buildTree(int *** rbins, //this will be where the tree itself is returned
			  double * data, //this is the input data
			  int dim, // the dimensionality of the data
			  unsigned long long numPoints, // the number of points in the dataset
			  double epsilon, // the distance threshold being searched
			  int maxBinAmount, // the maximum nnumber of bins for one layer of the tree
			  int * pointArray, // where the ordered point number will be returned
			  int *** rpointBinNumbers, // the bin numbers for each point
			  unsigned int * binSizes, // the number bins for that layer of the tree
			  unsigned int * binAmounts){ // the range / epsilon for that rp of that layer of the tree

	int maxRP = MAXRP; // setting the max number of reference points
	int numRPperLayer = RPPERLAYER; // setting how many reference points are checked for each layer

	//an array to hold the bins of the tree that will be passed back to the calling function in rbins
	int ** bins = (int **)malloc(sizeof(int*)*maxRP);

	//an array to hold the number of bins per layer of the tree
	int * binCounts = (int*)malloc(sizeof(int)*maxRP);

	//an array to keep track of the number of non-empty bins in each layer
	int * binNonEmpty = (int*)malloc(sizeof(int)*maxRP);

	//a 2d array that keeps track of the offset of each point for placement in the next layer
	int ** pointBinOffsets = (int**)malloc(sizeof(int*)*maxRP);
	for(int i = 0; i < maxRP; i++){
		pointBinOffsets[i] = (int*)malloc(sizeof(int)*numPoints);
	}

	//a 2d array that keeps track of the bin numbers for each point relative to the reference points
	int ** pointBinNumbers = (int**)malloc(sizeof(int*)*numPoints);
	for(int i = 0; i < numPoints; i++){ 
		pointBinNumbers[i] = (int*)malloc(sizeof(int)*maxRP);
	}

	//an array to hold the sum of squares error for calculating which reference point to use for each layer
	double * sumSqrsTemp = (double*)malloc(sizeof(double)*numRPperLayer);

	//an array to hold the final sum sqrs for each layer, helps determine the number of reference points to use
	double * sumSqrsLayers = (double*)malloc(sizeof(double)*maxRP);

	// variable for exiting the while loop, evaluates to false when sumsqrs is beyond a threshold
	bool check = true;

	// keeping track of which layer of the tree is being built
	int currentLayer = 0;

	while(check && currentLayer < maxRP){
		// printf("\nCurrent layer: %d\n",currentLayer);
		
		sumSqrsLayers[currentLayer] = 0;

		for(int i = 0; i < numRPperLayer; i++){
			sumSqrsTemp[i] = 0;
		}

		if(RAND) srand(omp_get_wtime());
		double * RPArray = createRPArray(data, numRPperLayer, dim, numPoints);
		// double * RPArray = (double *)malloc(sizeof(double)*dim*numRPperLayer); 

		// for(int i = 0; i < numRPperLayer*dim; i++){
		// 	RPArray[i] = (double)rand()/(double)RAND_MAX;
		// }
		
		double * distMat = (double * )malloc(sizeof(double)*numPoints*numRPperLayer);
		int ** layerBins = (int**)malloc(sizeof(int*)*numRPperLayer);
		int * layerBinCount = (int*)malloc(sizeof(int)*numRPperLayer);
		int * layerBinNonEmpty = (int*)malloc(numRPperLayer*sizeof(int));
		int ** layerBinOffsets = (int**)malloc(sizeof(int*)*numRPperLayer);
		int * layerNumBins = (int*)malloc(sizeof(int)*numRPperLayer);
		int * skipBins = (int*)malloc(sizeof(int)*numRPperLayer);


		for(int i = 0; i < numRPperLayer; i ++){
			layerBinOffsets[i] = (int*)malloc(sizeof(int)*numPoints);
		}

		// printf("refChecking: \n");
		#pragma omp parallel for
		for(int i = 0; i < numRPperLayer; i++){
			layerBinNonEmpty[i] = 0;

			double maxDistance = 0; 
			double minDistance = euclideanDistance(&data[0*dim], dim , &RPArray[i*dim]);
			for(int j = 0; j < numPoints; j++){
				distMat[i*numPoints + j] = euclideanDistance(&data[j*dim], dim , &RPArray[i*dim]);
				if(distMat[i*numPoints + j] > maxDistance){
					maxDistance = distMat[i*numPoints + j];
				}
				if(distMat[i*numPoints + j] < minDistance){
					minDistance = distMat[i*numPoints + j];
				}
			}



			// printf("Max dist = %f, ",maxDistance);
			skipBins[i] = floor(minDistance/epsilon) - 2;
			layerNumBins[i] = ceil(maxDistance / epsilon) + 2 - skipBins[i];
			// layerNumBins[i] = ceil(maxDistance / epsilon) + 1;
			if(currentLayer == 0){
				layerBinCount[i] = layerNumBins[i];
			} else {
				layerBinCount[i] = binNonEmpty[currentLayer - 1] * layerNumBins[i];
				// printf("nonEmpties prev = %d, max bins = %d, layerBinCount = %d\n", binNonEmpty[currentLayer-1],layerNumBins[i], layerBinCount[i]);

			}
			

			layerBins[i] = (int*)malloc(layerBinCount[i] * sizeof(int));
			for(int j = 0; j < layerBinCount[i]; j++){
				layerBins[i][j] = 0;
			}

			if(currentLayer == 0){
				for(int j = 0; j < numPoints; j++){
					int binNumber = floor(distMat[i*numPoints + j] / epsilon) - skipBins[i];
					// int binNumber = floor(distMat[i*numPoints + j] / epsilon);
					if(layerBins[i][binNumber] == 0){
						layerBinNonEmpty[i]++;
					}
					layerBins[i][binNumber]++; 
					layerBinOffsets[i][j] = binNumber; 
				}
			} else {
				for(int j = 0; j < numPoints; j++){
					int part1 = pointBinOffsets[currentLayer-1][j];
					int part2 = bins[ currentLayer - 1 ][part1] - 1;
					int part3 = layerNumBins[i];//binNonEmpty[currentLayer-1];
					// int offset = (bins[ currentLayer - 1 ][ pointBinOffsets[currentLayer-1][j] ] - 1)*binNonEmpty[currentLayer-1];
					int offset = part2*part3;
					int binNumber = floor(distMat[i*numPoints + j] / epsilon) - skipBins[i];
					// int binNumber = floor(distMat[i*numPoints + j] / epsilon);

					// if(offset+binNumber > layerBinCount[i] && checkers == true) {
					// 	checkers = false;
					// 	printf("offset+binNumber is the problem with offset = %d, binnumber = %d, and layerbincount = %d, pointBinOff = %d, bins = %d\n", offset,binNumber,layerBinCount[i], part1, part2);
					// }

					if(layerBins[i][offset+binNumber] == 0){
						layerBinNonEmpty[i]++;
					}
					layerBins[i][offset+binNumber]++;

					layerBinOffsets[i][j] = offset + binNumber;

				}

			}

			// printf("Non Empties = %d", layerBinNonEmpty[i]);

			for(int j = 0; j < layerBinCount[i]; j++){
				sumSqrsTemp[i] += layerBins[i][j] * layerBins[i][j];
			}
			// printf(", %d\n ",i);

		}

		
		// printf("\n");
 
		//pick the one with lowest sum sqrs? sure why not
		int minSumIdx = 0;
		for(int i = 1; i < numRPperLayer; i++){
			if(sumSqrsTemp[minSumIdx] > sumSqrsTemp[i] ){
				minSumIdx = i; 
			}
		}

		// printf("Found Layer min at %d\n", minSumIdx);

		sumSqrsLayers[currentLayer] = sumSqrsTemp[minSumIdx];

		for(int i = 0; i < numPoints; i++){
			pointBinOffsets[currentLayer][i] = layerBinOffsets[minSumIdx][i];
		}
		
		int part4 = layerBinCount[minSumIdx];
		bins[currentLayer] = (int*)malloc(part4*sizeof(int));
		int tmpCount = 0;
		// printf("\n");
		for(int i = 0; i < layerBinCount[minSumIdx]; i++){
			if(layerBins[minSumIdx][i] != 0){
				tmpCount++;
				bins[currentLayer][i] = tmpCount;
			} else {
				bins[currentLayer][i] = 0;
			}

			// printf("%d, ", bins[currentLayer][i]);
 
		} 
		// printf("\n");



		// printf("tmpCount: %d, non Empty: %d\n", tmpCount, layerBinNonEmpty[minSumIdx] );

		// printf("fff: %d\n", layerBinCount[minSumIdx]);
		binCounts[currentLayer] = layerBinCount[minSumIdx];
		if(currentLayer != 0){
			binSizes[currentLayer] = layerNumBins[minSumIdx]*binNonEmpty[currentLayer-1];
		} else {
			binSizes[currentLayer] = layerNumBins[minSumIdx];
		}
		
		// printf("layer %d bincount: %d\n", currentLayer, binCounts[currentLayer]);

		binNonEmpty[currentLayer] = layerBinNonEmpty[minSumIdx];
		binAmounts[currentLayer] = layerNumBins[minSumIdx];

		for(int i = 0; i < numPoints; i++){
			pointBinNumbers[i][currentLayer] = floor(distMat[minSumIdx*numPoints+i] / epsilon)-skipBins[minSumIdx];
			// pointBinNumbers[i][currentLayer] = floor(distMat[minSumIdx*numPoints+i] / epsilon);
		}

		// printf("Finishing up layer\n");
		if(currentLayer >= MINRP){
			if(sumSqrsLayers[currentLayer-1]/sumSqrsLayers[currentLayer] < LAYER_DIFF || currentLayer == maxRP - 1){
				// printf("Found final layer!\n");
				check = false;

				int runningTotal = 0;
				for(int i = 0; i < binCounts[currentLayer]; i++){
					bins[currentLayer][i] = runningTotal;
					runningTotal += layerBins[minSumIdx][i];
					
				}
				// printf("Made it here!\n"); 
			}
			
		} 

		currentLayer++;

		free(distMat);
		for(int i = 0; i < numRPperLayer; i++){
			free(layerBins[i]);
			free(layerBinOffsets[i]);
		}
		free(layerBins);
		free(layerBinOffsets);
		free(layerBinCount);
		free(layerBinNonEmpty);
		free(layerNumBins);
		free(skipBins);
		free(RPArray);
	}

	int numRP = currentLayer;
	// selectedRP = &numRP;
	printf("Selected %d reference points\n", numRP);

    // sort the point arrays from bottom to top using a stable sort
	thrust::host_vector<int*> pointVector(numPoints);
	for(int i = 0; i < numPoints; i++)
	{
		pointVector[i] = (int*)malloc(sizeof(int)*(numRP+1));
		for(int j = 0; j < numRP; j++){
			pointVector[i][j+1] = pointBinNumbers[i][j];
		}
	}

	for( unsigned long long i = 0; i < numPoints; i++){
		pointVector[i][0] = pointArray[i];
	}

	// thrust::device_vector<int*> d_pointVector(pointVector);
	double startSortTime = omp_get_wtime();

    for(int i = numRP-1; i >= 0; i--){

		int * oneBin = (int*)malloc(sizeof(int)*numPoints);
		// thrust::host_vector<int> oneBin(numPoints);

		// #pragma omp parallel for
		for(int j = 0; j < numPoints; j++ ){
			oneBin[j] = pointVector[j][i+1];
		}

		// thrust::device_vector<int> d_oneBin(oneBin);

		thrust::stable_sort_by_key(thrust::omp::par, oneBin, oneBin+numPoints, pointVector.begin());
		// thrust::stable_sort_by_key(thrust::device, oneBin.begin(), oneBin.end(), pointVector.begin());

		free(oneBin);
	}

	double endSortTime = omp_get_wtime();

	// printf("\nTree sorting time: %f\n", endSortTime-startSortTime);

	#pragma omp parallel for
	for(int i = 0; i < numPoints; i++){
		pointArray[i] = pointVector[i][0];
		for(int j = 0; j < numRP; j++){
			pointBinNumbers[i][j] = pointVector[i][j+1];
		}
		free(pointVector[i]);
	}

	for(int i = 0; i < maxRP; i++){
		free(pointBinOffsets[i]);
	}
	free(pointBinOffsets);
	free(binCounts);
	free(binNonEmpty);
	free(sumSqrsLayers);
	free(sumSqrsTemp);

	*rbins = bins;
	*rpointBinNumbers = pointBinNumbers;

	return(numRP);
}


int generateRanges(int ** tree, //points to the tree constructed with buildTree()
				   int numPoints, // the number of points in the dataset
				   int ** pointBinNumbers, // the bin numbers of the points relative to the reference points
				   int numLayers, // the number of layer the tree has
				   unsigned int * binSizes,	// the number of bins for each layer, or rather the width of the tree in bins for that layer
				   unsigned int * binAmounts, // the number of bins for each reference point, ranhge/epsilon
				   int ** addIndexes, // where generateRanges will return the non-empty index locations in the tree's final layer
				   int *** rangeIndexes, // the index locations that are adjacent to each non-empty index
				   unsigned int *** rangeSizes, // the number of points in adjacent non-empty indexes for each non-empty index
				   int ** numValidRanges, // the numnber of adjacent non-empty indexes for each non-empty index
				   unsigned long long ** calcPerAdd, // the number of calculations that will be needed for each non-empty index
				   unsigned int ** numPointsInAdd ){ // the number of points in each non-empty index

    // an array to temproarily hold the indexes of only non empty bins, over allocates to the maximum possible
    int*tempIndexes = (int*)malloc(sizeof(int)*binSizes[numLayers-1]);

	// the number of non empty bins in the final layer or number of addresses with points
    int nonEmptyBins = 0;

	// counting the number of non empty bins and keeping track of the indexes of those nonempty bins
    for(int i = 0; i < binSizes[numLayers-1]-1; i++){ 
        
		// if the tree value on the last layer is less than the next then it has points in it
		if(tree[numLayers-1][i] < tree[numLayers-1][i+1]){
			//keep track of that non empty index
            tempIndexes[nonEmptyBins] = i;
            nonEmptyBins++;
        }
    }

	// local ranges keeps track of the indexes that are searched for each address, and were distancs calcs are needed
	// i.e. for each nonempty last layer index, the indexes that are adjacent
    int ** localRangeIndexes = (int**)malloc(sizeof(int*)*nonEmptyBins);

	// local range sizes keeps track of how large each localRangeIndexes is, 
	// so the number of points in an adjacent nonempty indexe
	unsigned int ** localRangeSizes = (unsigned int **)malloc(sizeof(unsigned int*)*nonEmptyBins);

	// a temp variable to keep track of the num valid ranges for a single index 
    int * tempNumValidRanges = (int *)malloc(sizeof(int)*nonEmptyBins);

	// this keeps track of the number of distance calculations that will be needed for a non empty index/address
	unsigned long long * tempCalcPerAdd = (unsigned long long*)malloc(sizeof(unsigned long long)*nonEmptyBins);

	// an array for holding the non empty index locations in the final tree layer
    int * tempAddIndexes = (int*)malloc(sizeof(int)*nonEmptyBins);

	// as array for keeping track of the number of points int the non empty  indexes, correlated with tempAddIndexes
	unsigned int * tempNumPointsInAdd = (unsigned int*)malloc(sizeof(unsigned int)*nonEmptyBins);

	// copy the temp indexes which keeps track of the nonepty indexes into an array that is the correct size
    for(int i = 0; i < nonEmptyBins; i++){
        tempAddIndexes[i] = tempIndexes[i];
    }

	// free the overallocated array now that we have the data stored in tempAddIndexes
    free(tempIndexes);

	// the number of searches that are needed for a full search is the number of referecnes point cubed. This is always true.
	const unsigned int numSearches = pow(3,numLayers);

	//go through each non empty bin and do all the needed searching and generate the arrasy that are needed for the calculations kernels
	#pragma omp parallel for
    for(int i = 0; i < nonEmptyBins; i++){

		// the bin numbers of the current nonempty bin is found from the first point in that bin
		int * binNumbers = pointBinNumbers[ tree[ numLayers-1 ][ tempAddIndexes[i] ] ];

		// this will record the number of calculations that need to be made by this index/address
		unsigned long long numCalcs;

		// the number of ranges, so the number of adjacent nonempty indexes 
		int numRanges;

		// a temporary address that will be modifed to perfom the searches
		int tempAdd[numLayers];

		// the number of points in this non empty index
		unsigned int localNumPointsInAdd;

		treeTraversal(tempAdd, // array of int for temp storage for searching
					  tree, // pointer to the tree made with buildTree()
					  binSizes, // the widths of each layer of the tree measured in bins
					  binAmounts, // the range of bins from a reference points, i.e. range / epsilon
					  binNumbers, // the address/bin numbers of the current index/address
					  numLayers, // the number of reference points or layers in the tree, same thing
					  &numCalcs, // for keeping track of the number of distance calculations to be performed
					  &numRanges, // the number of adjacent non-empty addresses/indexes
					  &localRangeIndexes[i], // this addresses/index's array to keep track of adjacent indexes
					  &localRangeSizes[i], // the number of points in the indexes in localRangeIndexes
					  &localNumPointsInAdd, // the number of points in this nonempty address
					  numSearches); // the number of searches that need to be performed, 3^r

		// storing variables into arrays that coorespond to the non-empty indexes/addresses
		tempCalcPerAdd[i] = numCalcs;
		tempNumValidRanges[i] = numRanges;
		tempNumPointsInAdd[i] = localNumPointsInAdd;


    }

	// assiging pointers for accsess outside of this functions scope
	*addIndexes = tempAddIndexes;
	*calcPerAdd = tempCalcPerAdd;
	*numValidRanges = tempNumValidRanges;
	*rangeSizes = localRangeSizes;
	*rangeIndexes = localRangeIndexes;
	*numPointsInAdd = tempNumPointsInAdd;

	//return the number of non-empty bins
	return nonEmptyBins;

}

__host__ __device__ 
int depthSearch(int ** tree, //pointer to the tree built with buildTree()
				unsigned int * binAmounts, // the number of bins for each reference point, i.e. range/epsilon
				int numLayers, //the number of layers in the tree
				int * searchBins){ // the bin number that we are searching for
	
	// the offset is used for keeping track of the offset from the begining of each layer to the index
	int offset = 0;
	
	//go through each layer up to the last to determine if the index is non-empty and if it is then find the offset into the next layer
	for(int i = 0; i < numLayers-1; i++){
		
		//check the current layer at the bin number + offset may or may not need -1 here
		if (tree[i][offset + searchBins[i]] == 0){
			return -2;
		}

		// the next offset will be the previous layer index number * the number of bins for the reference point in the next layer
		offset = (tree[i][searchBins[i]+offset]-1)*binAmounts[i+1];
	}

	//the index will be the last layers bin number plus the offset for the last layer
	int index = searchBins[numLayers-1]+offset;

	//if last layer has poionts then return the index value
	if(tree[numLayers-1][index] < tree[numLayers-1][index+1]){
		return index;
	}else{
		return -1;
	}

}

__host__ __device__
void treeTraversal(int * tempAdd, //twmp array for the address being searched
				   int ** tree, // the pointer to the tree
				   unsigned int * binSizes, // the width of the tree for each layer mesuared in number of bins
				   unsigned int * binAmounts, // the number of bins for each reference point
				   int * binNumbers, // the bin number for the home address
				   int numLayers, // the number of reference points/layers in the tree
				   unsigned long long * numCalcs, // the place to retrun the number of calcs that will be needed
				   int * numberRanges, // the return location for the number of adjacent non-empty indexes
				   int ** rangeIndexes, // the array of non-empty adjacent index locations
				   unsigned int ** rangeSizes, // the number of points in each of the adjacent non-empty indexes
				   unsigned int * numPointsInAdd, //the number of points in the home address/iondex
				   unsigned int numSearches){ //the number of searches that are being perfomred for each addresss

	//keep track of the number of calcs that will be needed
    unsigned long long localNumCalcs = 0;

	// keep track of the number of non-empty adjacent indexes
    unsigned int localNumRanges = 0;

	//keep track of the locations of adjacent indexes that are not empty
	int * localRangeIndexes = (int*)malloc(sizeof(int)*numSearches);

	// the number of points in the adjacent non-empty indexes
	unsigned int * localRangeSizes = (unsigned int*)malloc(sizeof(unsigned int)*numSearches);

	//permute through bin variations (3^r) and run depth searches
	for(unsigned int i = 0; i < numSearches; i++){
		
		//modify temp add for the search based on our itteration i
		for(unsigned int j = 0; j < numLayers; j++){
			tempAdd[j] = binNumbers[j] + (i / (int)pow(3, j) % 3)-1;
		}
		
		//perform the search and get the index location of the return 

		int index = depthSearch(tree, binAmounts, numLayers, tempAdd);

		//check if the index location was non empty
		if(index >= 0){
			//store the non empty index location
			localRangeIndexes[localNumRanges] = index;

			//calcualte the size of the index, i.e. the number of points in the index
			unsigned long long size = tree[numLayers-1][index+1] - tree[numLayers-1][index]; //may need to +- to index here!!!!!!!!!!!!!!!

			//store that in the sizes array
			localRangeSizes[localNumRanges] = size;

			// keep running total of the sizes for getting the number of calculations latter
			localNumCalcs += size;

			//keep track of the number of non-empty adjacent indexes
			localNumRanges++;
		}
	}

	// get the index of the home address
	int homeIndex = depthSearch(tree, binAmounts, numLayers, binNumbers);

	// find the number of points in the home address
	unsigned long long numHomePoints = tree[numLayers-1][homeIndex+1] - tree[numLayers-1][homeIndex]; //may need to +- one to index here !!!!!!!!!

	if(numHomePoints == 0)printf("ERROR: no points found in address at index: %d\n", homeIndex);
	// use the running total of points in adjacent addresses and multiply it by the number of points in the home address for number of total calcs
	*numCalcs = localNumCalcs*numHomePoints;

	// return the arrays with pointer magic
	*rangeIndexes = localRangeIndexes;
	*rangeSizes = localRangeSizes;

	*numberRanges = localNumRanges;

	*numPointsInAdd = numHomePoints;


}