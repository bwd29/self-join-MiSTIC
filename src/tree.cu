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

	// cinstruct the layers in this loop, terminates when we reach the max layer count or when loss function calls for return
	while(check && currentLayer < maxRP){
		
		// initial sumsqrs for layer is 0
		sumSqrsLayers[currentLayer] = 0;

		// set the temp sum sqrs for possible reference points to 0
		for(int i = 0; i < numRPperLayer; i++){
			sumSqrsTemp[i] = 0;
		}

		// if we want full random, then set seed based on time
		if(RAND){
			srand(omp_get_wtime());
		} 

		// make the array of reference points
		double * RPArray = createRPArray(data, numRPperLayer, dim, numPoints);
		
		// this is the distance matrix for the points to the reference points in RPArray
		double * distMat = (double * )malloc(sizeof(double)*numPoints*numRPperLayer);

		// a 2d array that keeps track of the bins for the current layer for each possible reference point
		int ** layerBins = (int**)malloc(sizeof(int*)*numRPperLayer);

		// an array for the number of bins in the current layer of the tree
		int * layerBinCount = (int*)malloc(sizeof(int)*numRPperLayer);

		// the number of non empty bins in the current layer of the tree
		int * layerBinNonEmpty = (int*)malloc(numRPperLayer*sizeof(int));

		// the offset of each point into the current layer
		int ** layerBinOffsets = (int**)malloc(sizeof(int*)*numRPperLayer);

		// allocateing the 2d array structure for layer bin offsets
		for(int i = 0; i < numRPperLayer; i ++){
			layerBinOffsets[i] = (int*)malloc(sizeof(int)*numPoints);
		}

		// the nuber of bins from the reference point to points
		int * layerNumBins = (int*)malloc(sizeof(int)*numRPperLayer);

		//the number of bins to skip form the start of the reference point
		int * skipBins = (int*)malloc(sizeof(int)*numRPperLayer);

		// printf("refChecking: \n");

		// entering into a loop to create a bunch of different possible layers for the current layer. then the best one is chosen
		#pragma omp parallel for
		for(int i = 0; i < numRPperLayer; i++){

			//set the current number of non empty bins to 0
			layerBinNonEmpty[i] = 0;

			//fill out the distance matrix
			double maxDistance = 0;  // for saving the max distance and used for determining layer size
			double minDistance = euclideanDistance(&data[0*dim], dim , &RPArray[i*dim]); // the min distance starts as just a point

			// going through each point to build the distance matrix
			for(int j = 0; j < numPoints; j++){
				distMat[i*numPoints + j] = euclideanDistance(&data[j*dim], dim , &RPArray[i*dim]);

				// checking fot the max distance
				if(distMat[i*numPoints + j] > maxDistance){
					maxDistance = distMat[i*numPoints + j];
				}

				//checkign for the min distance
				if(distMat[i*numPoints + j] < minDistance){
					minDistance = distMat[i*numPoints + j];
				}
			}



			// printf("Max dist = %f, ",maxDistance);

			// a modifier for padding the bin sizes
			int modifier = 0;

			//the number of bins to skip from the beigning to save on space
			skipBins[i] = floor(minDistance/epsilon) - 1 ;

			//the number of bins for a given reference point, i.e. the farthest bin number minus the closest bin number
			layerNumBins[i] = ceil(maxDistance / epsilon) + modifier - skipBins[i];
			// layerNumBins[i] = ceil(maxDistance / epsilon) + 1;

			//if the current layer is the first layer, then the number of total bins for that layer will be just the number of bins for the reference point
			if(currentLayer == 0){ 
				layerBinCount[i] = layerNumBins[i];
			} else {
				// the number of bins in the layer is the number of bins for a reference point times the number of non empty bins in the previous layer
				layerBinCount[i] = binNonEmpty[currentLayer - 1] * layerNumBins[i];
			}
			
			// now that we know the size of the layer we can allocate the array to hold the bin values
			layerBins[i] = (int*)malloc(layerBinCount[i] * sizeof(int));

			// the bin values will all start at 0
			for(int j = 0; j < layerBinCount[i]; j++){
				layerBins[i][j] = 0;
			}

			// the first layer is unique because there are no offsets from the previous layer
			if(currentLayer == 0){
				// itterate through each point in the data set to assign point to bins
				for(int j = 0; j < numPoints; j++){
					// the bin number will be the the floor of this distance from the point to the reference point - the numebr of bins we skipped on the start
					int binNumber = floor(distMat[i*numPoints + j] / epsilon) - skipBins[i];

					// if the bin is empty, then increase the non empty bin count by 1
					if(layerBins[i][binNumber] == 0){
						layerBinNonEmpty[i]++;
					}

					//increment the value of the bin to count the number of points in that bin
					layerBins[i][binNumber]++; 

					// the offset from the first layer will be the bin number
					layerBinOffsets[i][j] = binNumber; 
				}
			} else {
				// assign every point to a bin for layers past the first one
				for(int j = 0; j < numPoints; j++){

					// the offset will be the bin number of the previous layer, which will be the number of non empty bins before it, times the number of bins for the reference point in the current layer
					int offset = (bins[currentLayer -1][pointBinOffsets[currentLayer-1][j]]-1)*layerNumBins[i];

					// the bin number will be the the floor of this distance from the point to the reference point - the numebr of bins we skipped on the start
					int binNumber = floor(distMat[i*numPoints + j] / epsilon) - skipBins[i];

					// if(offset+binNumber > layerBinCount[i] && checkers == true) {
					// 	checkers = false;
					// 	printf("offset+binNumber is the problem with offset = %d, binnumber = %d, and layerbincount = %d, pointBinOff = %d, bins = %d\n", offset,binNumber,layerBinCount[i], part1, part2);
					// }
					
					// if the bin was empty, then increment the number of non empty bins
					if(layerBins[i][offset+binNumber] == 0){
						layerBinNonEmpty[i]++;
					}

					// increment the value of the bin to count the number of points in that bin
					layerBins[i][offset+binNumber]++;

					// the offset for the point will be the previous offset + the bin number
					layerBinOffsets[i][j] = offset + binNumber;

				}

			}

			// calculate the sum of squares based on the number of points in each bin. 
			// the lower the sum of squares, the more even the distribution of points
			for(int j = 0; j < layerBinCount[i]; j++){
				sumSqrsTemp[i] += layerBins[i][j] * layerBins[i][j];
			}

		}

		//pick the one with lowest sum sqrs? sure why not
		int minSumIdx = 0; // this will be the reference point form all of the possible one that will be used for final layer construction
		for(int i = 1; i < numRPperLayer; i++){
			if(sumSqrsTemp[minSumIdx] > sumSqrsTemp[i] ){
				minSumIdx = i; 
			}
		}

		// assign the sum sqrs for the layer as the one chosen
		sumSqrsLayers[currentLayer] = sumSqrsTemp[minSumIdx];

		// copy over the bin offsets based on the reference point that was chosen
		for(int i = 0; i < numPoints; i++){
			pointBinOffsets[currentLayer][i] = layerBinOffsets[minSumIdx][i];
		}
		
		// the size of the bins will be the size of the layer form the chosen one
		bins[currentLayer] = (int*)malloc(layerBinCount[minSumIdx]*sizeof(int));
		
		// tmnpcount will keep track of the number of previous non empty bins
		int tmpCount = 0;

		//go through each bin inn the layer
		for(int i = 0; i < layerBinCount[minSumIdx]; i++){

			//if the bin is not empty then increment our counter and assign that value to the bin
			if(layerBins[minSumIdx][i] != 0){
				tmpCount++;
				bins[currentLayer][i] = tmpCount;

			// if the bin is empty, then  leave the value as zero
			} else {
				bins[currentLayer][i] = 0;
			} 
		} 

		// the bin counts will be the number of bins in that layer
		binSizes[currentLayer] = layerBinCount[minSumIdx];

		// copy over the number of non empty bins for the layer
		binNonEmpty[currentLayer] = layerBinNonEmpty[minSumIdx];

		// copy the number of bins for the reference point
		binAmounts[currentLayer] = layerNumBins[minSumIdx];

		// assign the point bin numbers and save them
		for(int i = 0; i < numPoints; i++){
			pointBinNumbers[i][currentLayer] = floor(distMat[minSumIdx*numPoints+i] / epsilon)-skipBins[minSumIdx];
		}

		// check if the current layer is at least the min number of layers for the tree
		if(currentLayer >= MINRP){
			// check if the sum of sqrs is still decreasing by adding layers or if we are at the max number of layers
			if(sumSqrsLayers[currentLayer-1]/sumSqrsLayers[currentLayer] < LAYER_DIFF || currentLayer == maxRP - 1){
				
				//set check to false to exit the while loop
				check = false;

				// keep track of the running total of points in bins
				int runningTotal = 0;

				// go through each bin of the final layer
				for(int i = 0; i < binSizes[currentLayer]; i++){
					// the final layer of the tree will have the running total of points in the bins as values
					bins[currentLayer][i] = runningTotal;
					// update the running total number of points
					runningTotal += layerBins[minSumIdx][i];
					
				}
			}
		} 

		//move on to the next layer
		currentLayer++;

		//free all the memory used for construction of this layer
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

	// the number of layers/ reference points is the current layer
	int numRP = currentLayer;

	printf("Selected %d reference points\n", numRP);

    // sort the point arrays from bottom to top using a stable sort
	
	// use a thrust vector that will keep track of the point numbers
	thrust::host_vector<int*> pointVector(numPoints);

	//go through each point an load the point bin numbers
	for(int i = 0; i < numPoints; i++){
		// the point vector will contain the point number i.e. the row of the point in the original data, and then each bin of the point
		pointVector[i] = (int*)malloc(sizeof(int)*(numRP+1));

		//copy over the bin numbers
		for(int j = 0; j < numRP; j++){
			pointVector[i][j+1] = pointBinNumbers[i][j];
		}
	}

	// assign the point numbers which start as sequential
	for( int i = 0; i < numPoints; i++){
		pointVector[i][0] = pointArray[i];
	}

	// go through each layer of the tree backwards to sort the points
    for(int i = numRP-1; i >= 0; i--){

		// array to keep track of the bin values that are being used for this sort at the layer i
		int * oneBin = (int*)malloc(sizeof(int)*numPoints);

		// copy over the bin numebrs for sorting
		#pragma omp parallel for
		for(int j = 0; j < numPoints; j++ ){
			oneBin[j] = pointVector[j][i+1];
		}

		// run the stabel sort with the bin numbers at i as the key
		thrust::stable_sort_by_key(thrust::omp::par, oneBin, oneBin+numPoints, pointVector.begin());

		free(oneBin);
	}

	// copy over results after sorting
	#pragma omp parallel for
	for(int i = 0; i < numPoints; i++){
		// point array will have all of the point numbers is order
		pointArray[i] = pointVector[i][0];
		for(int j = 0; j < numRP; j++){
			// the point bin numbers will have all of the bin numbers
			pointBinNumbers[i][j] = pointVector[i][j+1];
		}
		free(pointVector[i]);
	}

	// free data used only in tree cointsruction
	for(int i = 0; i < maxRP; i++){
		free(pointBinOffsets[i]);
	}
	free(pointBinOffsets);
	free(binNonEmpty);
	free(sumSqrsLayers);
	free(sumSqrsTemp);

	// return the tree
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