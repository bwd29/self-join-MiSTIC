#include "include/tree.cuh"

int buildTree(int *** rbins, double * data, int dim, unsigned long long numPoints, double epsilon, int maxBinAmount,  int * pointArray, int *** rpointBinNumbers, unsigned int * binSizes, unsigned int * binAmounts){

	int maxRP = MAXRP;
	int numRPperLayer = RPPERLAYER;
	double layerDifReq = 1;
	// srand(omp_get_wtime());
	int ** bins = (int **)malloc(sizeof(int*)*maxRP);
	int * binCounts = (int*)malloc(sizeof(int)*maxRP);
	int * binNonEmpty = (int*)malloc(sizeof(int)*maxRP);

	int ** pointBinOffsets = (int**)malloc(sizeof(int*)*maxRP);
	for(int i = 0; i < maxRP; i++){
		pointBinOffsets[i] = (int*)malloc(sizeof(int)*numPoints);
	}

	int ** pointBinNumbers = (int**)malloc(sizeof(int*)*numPoints);
	for(int i = 0; i < numPoints; i++){ 
		pointBinNumbers[i] = (int*)malloc(sizeof(int)*maxRP);
	}

	double * sumSqrsTemp = (double*)malloc(sizeof(double)*numRPperLayer);
	double * sumSqrsLayers = (double*)malloc(sizeof(double)*maxRP);

	bool check = true;
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
			skipBins[i] = floor(minDistance/epsilon) - 1;
			layerNumBins[i] = ceil(maxDistance / epsilon) + 1 - skipBins[i];
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
		binSizes[currentLayer] = layerBinCount[minSumIdx];//layerNumBins[minSumIdx]*layerBinNonEmpty[minSumIdx];
		printf("layer %d bincount: %d\n", currentLayer, binCounts[currentLayer]);

		binNonEmpty[currentLayer] = layerBinNonEmpty[minSumIdx];
		binAmounts[currentLayer] = layerNumBins[minSumIdx];

		for(int i = 0; i < numPoints; i++){
			pointBinNumbers[i][currentLayer] = floor(distMat[minSumIdx*numPoints+i] / epsilon)-skipBins[minSumIdx];
		}

		// printf("Finishing up layer\n");
		if(currentLayer != 0 && currentLayer >= MINRP){
			if(sumSqrsLayers[currentLayer]*layerDifReq <= sumSqrsLayers[currentLayer-1] || currentLayer == maxRP - 1){
				// printf("Found final layer!\n");
				check = false;

				int runningTotal = 0;
				for(int i = 0; i < binCounts[currentLayer]; i++){
					runningTotal += layerBins[minSumIdx][i];
					bins[currentLayer][i] = runningTotal;
					
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


	*rbins = bins;
	*rpointBinNumbers = pointBinNumbers;
	// printf("\n!!%d\n", pointBinNumbers[0][0]);

	return(numRP);
}


int generateRanges(int ** tree, int numPoints, int ** pointBinNumbers, int numLayers, unsigned int * binSizes, unsigned int * binAmounts, int ** addIndexes, int *** rangeIndexes, unsigned int *** rangeSizes, int ** numValidRanges, unsigned long long ** calcPerAdd ){
    
    int*tempIndexes = (int*)malloc(sizeof(int)*binSizes[numLayers-1]);

    int nonEmptyBins = 0;
    for(int i = 1; i < binSizes[numLayers-1]-1; i++){
        if(tree[numLayers-1][i-1] < tree[numLayers-1][i]){
            tempIndexes[nonEmptyBins] = i;
            nonEmptyBins++;
        }
    }

    if(tree[numLayers-1][binSizes[numLayers-1]-1] != numPoints){
        tempIndexes[nonEmptyBins] = binSizes[numLayers-1]-1;
        nonEmptyBins++;
    }



    int ** localRangeIndexes = (int**)malloc(sizeof(int*)*nonEmptyBins);
	unsigned int ** localRangeSizes = (unsigned int **)malloc(sizeof(unsigned int*)*nonEmptyBins);

    int * tempNumValidRanges = (int *)malloc(sizeof(int)*nonEmptyBins);
	unsigned long long * tempCalcPerAdd = (unsigned long long*)malloc(sizeof(unsigned long long)*nonEmptyBins);
    int * tempAddIndexes = (int*)malloc(sizeof(int)*nonEmptyBins);

    for(int i = 0; i < nonEmptyBins; i++){
        tempAddIndexes[i] = tempIndexes[i];
    }

    free(tempIndexes);


	#pragma omp parallel for
    for(int i = 0; i < nonEmptyBins; i++){

		int * binNumbers = pointBinNumbers[ tree[ numLayers-1 ][ tempAddIndexes[i] ] -1 ];

		int * tempRangeIndexes;
		unsigned int * tempRangeSizes;
		unsigned long long numCalcs;
		int numRanges;

		treeTraversal(tree, binSizes, binAmounts, binNumbers, numLayers, numPoints, &numCalcs, &numRanges, &tempRangeIndexes, &tempRangeSizes);

		tempCalcPerAdd[i] = numCalcs;
		tempNumValidRanges[i] = numRanges;
		localRangeIndexes[i] = tempRangeIndexes;
		localRangeSizes[i] = tempRangeSizes;


    }


	*addIndexes = tempAddIndexes;
	*calcPerAdd = tempCalcPerAdd;
	*numValidRanges = tempNumValidRanges;
	*rangeSizes = localRangeSizes;
	*rangeIndexes = localRangeIndexes;

	return nonEmptyBins;

}

int depthSearch(int ** tree, unsigned int * binSizes, unsigned int * binAmounts, int numLayers, int currentLayer, int initalOffset, int numPoints, int * searchBins, int * rangeIndexResult){
	
	
	int offset = initalOffset;
	//starting at current layer
	for(int i = currentLayer; i < numLayers-1; i++){
		//get child
		int child = tree[i][offset + searchBins[i]];

		if (child == 0){
			return -1;
		}

		offset = (tree[i][searchBins[i]+offset]-1)*binAmounts[i+1];
	}

	//for final layer need ranges
	int index = searchBins[numLayers-1]+offset;
	if(tree[numLayers-1][index] < tree[numLayers-1][index+1]){
		*rangeIndexResult = index;
	}else{
		return -1;
	}

	return 1;

}

void treeTraversal(int ** tree, unsigned int * binSizes, unsigned int * binAmounts, int * binNumbers, int numLayers, int numPoints,unsigned long long * numCalcs, int * numberRanges, int ** rangeIndexes, unsigned int ** rangeSizes){

    int numSearches = pow(3,numLayers);
    unsigned long long localNumCalcs = 0;
    int localNumRanges = 0;
    int* tempRangeIndexes = (int*)malloc(sizeof(int)*numSearches);
	int offset = 0;

	for(int i = 0; i < numLayers-1; i++){

			//search left
			if(tree[i][binNumbers[i] + offset-1] != 0){
				//create bins for full depth search
				int searchBins[numLayers-1-i];
				for(int j = i; j < numLayers-i; j++){
					searchBins[j] = binNumbers[j];
				}

				searchBins[0] = searchBins[0] - 1; //moving to the left

				//launch search
				int tempRangeIndex;
				int searchResults = depthSearch(tree, binSizes, binAmounts, numLayers, i, offset, numPoints, searchBins, &tempRangeIndex);
				if(searchResults > 0){
					tempRangeIndexes[localNumRanges] = tempRangeIndex;
					localNumRanges++;
				}

			}

			//search right
			if(tree[i][binNumbers[i]+offset+1] != 0){
				//create bins for full depth search
				int searchBins[numLayers-1-i];
				for(int j = i; j < numLayers-i; j++){
					searchBins[j] = binNumbers[j];
				}

				searchBins[0] = searchBins[0] + 1; //moving to the right

				//launch search
				int tempRangeIndex;
				int searchResults = depthSearch(tree, binSizes, binAmounts, numLayers, i, offset, numPoints, searchBins, &tempRangeIndex);
				if(searchResults > 0){
					tempRangeIndexes[localNumRanges] = tempRangeIndex;
					localNumRanges++;
				}
			}


			// if(i < numLayers - 1){
				offset = (tree[i][binNumbers[i]+offset]-1) * binAmounts[i+1];
				// if(offset < 0){
				// 	printf("!!! tree value at [%d,%d]: %d binAmounts: %d\n", i,binNumbers[i], (tree[i][binNumbers[i]]-1),binAmounts[i+1]);
				// }
				// printf("Offset for layer %d: %d\n", i, offset);
			// }
		
	}


	//get home bin values and adjacent in last layer
	//check left
	if(tree[numLayers-1][offset+binNumbers[numLayers-1] -1] < tree[numLayers-1][offset+binNumbers[numLayers-1]]){
		tempRangeIndexes[localNumRanges] = offset+binNumbers[numLayers-1]-1;
		localNumRanges++;
	}

	//check right
	if(tree[numLayers-1][offset+binNumbers[numLayers-1]] < tree[numLayers-1][offset+binNumbers[numLayers-1]+1]){
		tempRangeIndexes[localNumRanges] = offset+binNumbers[numLayers-1]+1;
		localNumRanges++;
	}

	tempRangeIndexes[localNumRanges] = offset+binNumbers[numLayers-1];
	localNumRanges++;

	//get number of points in home address
	// if(tempRangeIndexes[localNumRanges - 1]+1 > binSizes[numLayers-1] -1){
	// 	printf("!!! temprange is too large:%d offset: %d, binNumber %d, binAmounts:%d\n", tempRangeIndexes[localNumRanges - 1]+1, offset, binNumbers[numLayers-1],binAmounts[numLayers-1]);
	// }
	int numHomePoints = tree[numLayers-1][tempRangeIndexes[localNumRanges - 1]+1] - tree[numLayers-1][tempRangeIndexes[localNumRanges - 1]];
	
	//allocate for "returns"
	int * localRangeIndexes = (int*)malloc(sizeof(int)*localNumRanges);
	unsigned int * localRangeSizes = (unsigned int*)malloc(sizeof(unsigned int)*localNumRanges);

	//get number of calcs / load in array values
	for(int i = 0; i < localNumRanges; i++){
		localRangeIndexes[i] = tempRangeIndexes[i];
		unsigned int size = tree[numLayers-1][tempRangeIndexes[i]+1] - tree[numLayers-1][tempRangeIndexes[i]];
		localRangeSizes[i] = size;
		localNumCalcs += (unsigned long long)size;
	}

	*numberRanges = localNumRanges;
	*numCalcs = localNumCalcs*numHomePoints;
	*rangeIndexes = localRangeIndexes;
	*rangeSizes = localRangeSizes;


    free(tempRangeIndexes);

}