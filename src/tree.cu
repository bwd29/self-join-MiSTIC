#include "include/tree.cuh"

int buildTree(int *** rbins, double * data, int dim, unsigned long long numPoints, double epsilon, int maxBinAmount,  int * pointArray, int *** rpointBinNumbers, unsigned int * binSizes, unsigned int * binAmounts){

	int maxRP = MAXRP;
	int numRPperLayer = 64;
	double layerDifReq = 1;
	srand(omp_get_wtime());
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
		// srand(omp_get_wtime());
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


		for(int i = 0; i < numRPperLayer; i ++){
			layerBinOffsets[i] = (int*)malloc(sizeof(int)*numPoints);
		}

		// printf("refChecking: \n");
		#pragma omp parallel for
		for(int i = 0; i < numRPperLayer; i++){
			layerBinNonEmpty[i] = 0;

			double maxDistance = 0; 
			for(int j = 0; j < numPoints; j++){
				distMat[i*numPoints + j] = euclideanDistance(&data[j*dim], dim , &RPArray[i*dim]);
				if(distMat[i*numPoints + j] > maxDistance){
					maxDistance = distMat[i*numPoints + j];
				}
			}



			// printf("Max dist = %f, ",maxDistance);
			layerNumBins[i] = ceil(maxDistance / epsilon);
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
					int binNumber = floor(distMat[i*numPoints + j] / epsilon);
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
					int binNumber = floor(distMat[i*numPoints + j] / epsilon);

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
		binSizes[currentLayer] = layerNumBins[minSumIdx]*layerBinNonEmpty[minSumIdx];
		printf("layer %d bincount: %d\n", currentLayer, binCounts[currentLayer]);

		binNonEmpty[currentLayer] = layerBinNonEmpty[minSumIdx];
		binAmounts[currentLayer] = layerNumBins[minSumIdx];

		for(int i = 0; i < numPoints; i++){
			pointBinNumbers[i][currentLayer] = floor(distMat[minSumIdx*numPoints+i] / epsilon);
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
	}

	int numRP = currentLayer;
	// selectedRP = &numRP;
	printf("Selected %d reference points\n", numRP);

	// unsigned int * binSizes = (unsigned int*)malloc(sizeof(unsigned int)*numRP);
	// unsigned int * binAmounts = (unsigned int*)malloc(sizeof(unsigned int)*numRP);





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
	}

	// for(int i = 0; i < binSizes[numRP-1]; i++){
	// 	printf("%d,",bins[numRP-1][i]);
	// }
	// printf("\n");


    // delete(bins);
    // free(RPArray);

	// return(binArrays);
	// return(bins); 
	// *rbinAmounts = binAmounts;

	// *rbinSizes = binSizes;

	*rbins = bins;
	*rpointBinNumbers = pointBinNumbers;
	// printf("\n!!%d\n", pointBinNumbers[0][0]);

	return(numRP);
}


void generateRanges(int ** tree, int numPoints, int* pointArray, int ** pointBinNumbers, int numLayers, int * binSizes, int * binAmounts, int * addIndexes, int *** rangeIndexes, int *** rangeSizes, int * numValidRanges, int * calcPerAdd ){
    
    int*tempIndexes = (int*)malloc(sizeof(int)*binSizes[numLayers-1]);

	//linearize tree
	int * tree_bins = (int*)calloc(total_bins,sizeof(int));
	unsigned int tree_count = 0;
	for(int i = 0; i < rps; i++){
]		for(int j = 0; j < binSizes[i]; j++){
			tree_bins[tree_count] = binArrays[i][j];
			tree_count++;
		}
	}

    int nonEmptyBins = 0;
    for(int i = 0; i < binSizes[numLayers-1]-1; i++){
        if(tree[numLayers-1][i] < tree[numLayers-1][i+1]){
            addIndexes[nonEmptyBins] = i;
            nonEmptyBins++;
        }
    }

    if(tree[numLayers-1][binSizes[numLayers-1]-1] != numPoints){
        addIndexes[nonEmptyBins] = binSizes[numLayers-1]-1;
        nonEmptyBins++;
    }



    *rangeIndexes = (int**)malloc(sizeof(int*)*nonEmptyBins);
    *rangeSizes = (int **)malloc(sizeof(int*)*nonEmptyBins);

    numValidRanges = (int *)malloc(sizeof(int)*nonEmptyBins);
    calcPerAdd = (int*)malloc(sizeof(int)*nonEmptyBins);
    addIndexes = (int*)malloc(sizeof(int)*nonEmptyBins);

    for(int i = 0; i < nonEmptyBins; i++){
        addIndexes[i] = tempIndexes[i];
    }

    free(tempIndexes);


	// #pragma omp parallel for
    for(int i = 0; i < nonEmptyBins; i++){

        int * binNumbers = pointBinNumbers[tree[numLayers-1][addIndexes[i]]]; //may need to add 1 for inclusive

		int numSearches = pow(3,numLayers);
		int * tempRangeIndexes;
		int * tempRangeSizes;

		treeTraversal(tree, binSizes, binAmounts, binNumbers, numLayers, numPoints, &calcPerAdd[i], &numValidRanges[i], &tempRangeIndexes, &tempRangeSizes);

		*rangeIndexes[i] = tempRangeIndexes;
		*rangeSizes[i] = tempRangeSizes;

    }

}

int depthSearch(int ** tree, int * binSizes, int * binAmounts, int numLayers, int currentLayer, int initalOffset, int numPoints, int * searchBins, int * rangeIndexResult){
	
	
	int offset = initalOffset;
	//starting at current layer
	for(int i = currentLayer; i < numLayers-1; i++){
		//get child
		int child = tree[i][offset + searchBins[i]];

		if (child == 0){
			return -1;
		}

		offset = (tree[i][searchBins[i]]-1)*binAmounts[i+1];
	}

	//for final layer need ranges
	int index = searchBins[numLayers-1]+offset;
	if((index < binsSizes[numlayer-1]-1 && tree[numLayers-1][index] < tree[numLayers-1][index-1]) || (index == binSizes[numLayers-1]-1 && tree[numLayers-1][index] < numPoints)){
		*rangeIndexResult = index;
	}else{
		return -1;
	}

	return 1;

}

void treeTraversal(int ** tree, int * binSizes, int * binAmounts, int * binNumbers, int numLayers, int numPoints, int * numCalcs, int * numberRanges, int ** rangeIndexes, int ** rangeSizes){

    int numSearches = pow(3,numLayers);
    int localNumCalcs = 0;
    int localNumRanges = 0;
    int* tempRangeIndexes = (int*)malloc(sizeof(int)*numSearches);
	int offset = 0;

	for(int i = 0; i < numlayers; i++){

			//search left
			if(binNumbers[i] > 0 && tree[i][binNumbers[i]-1] != 0){
				//create bins for full depth search
				int searchBins[numLayers-1-i];
				for(int j = i; j < numLayers-i; j++){
					searchBins[j] = binNumbers[j];
				}

				searchBins[0] = searchBins[0] - 1; //moving to the left

				//launch search

				int searchResults = depthSearch(tree, binSizes, binAmounts, numLayers, i, offset, numPoints, searchBins, &tempRangeIndexes[localNumRanges]);
				if(searchResults > 0){
					localNumRanges++;
				}

			}

			//search right
			if(binNumbers[i] < binSizes -1){
				//create bins for full depth search
				int searchBins[numLayers-1-i];
				for(int j = i; j < numLayers-i; j++){
					searchBins[j] = binNumbers[j];
				}

				searchBins[0] = searchBins[0] + 1; //moving to the right

				//launch search

				int searchResults = depthSearch(tree, binSizes, binAmounts, numLayers, i, offset, numPoints, searchBins, &tempRangeIndexes[localNumRanges]);
				if(searchResults > 0){
					localNumRanges++;
				}
			}


			if(i < numlayers - 1){
				offset = (tree[i][binNumbers[i]]-1) * binAmounts[i+1];
			}
		
	}

	//get home bin values
	tempRangeIndexes[localNumRanges] = offset+binNumbers[numLayers-1];
	localNumRanges++;

	//allocate for "returns"
	*rangeIndexes = (int*)malloc(sizeof(int)*numLocalRanges);
	*rangeSizes = (int*)malloc(sizeof(int)*numLocalRanges);

	//get number of calcs / load in array values
	for(int i = 0; i < numLocalRanges; i++){
		rangeIndexes[i] = tempRangeIndexes[i];
		int size;
		if(tempRangeIndexes[i] == binSizes[numLayers-1]-1){
			size = numPoints - tree[numLayers-1][tempRangeIndexes[i]];
		}else{
			size = tree[numLayers-1][tempRangeIndexes[i+1]] - tree[numLayers-1][tempRangeIndexes[i]];
		}

		rangeSizes[i] = size;
		localNumCalcs += size;
	}

	*numberRanges = localNumRanges;
	*numCalcs = localNumCalcs;


    free(tempRangeIndexes);

}