#include "include/utils.cuh"


int * stddev( double * A, int dim, int num_points) {
	double mean, devmean;
	double *deviation = (double*)malloc(sizeof(double) * dim);
	int *dimension = (int*)malloc(sizeof(int) * dim);
	for(int i = 0; i < dim; i++) {
		dimension[i] = i;
	}
	for(int i = 0; i < dim; i++){
		mean = 0.0;
		for(int j = 0; j < num_points; j++){
			mean += A[dim*j+i];
		}
		mean /= num_points;
		devmean = 0.0;
		for(int j = 0; j < num_points; j++){
			devmean += pow(A[dim*j + i] - mean,2);
		}
		devmean /= num_points;
		deviation[i] = sqrt(devmean);
	}
	thrust::sort_by_key(deviation, &deviation[dim-1], dimension);
	double *deviationret = (double*)malloc(sizeof(double) * dim);
	int *dimensionret = (int*)malloc(sizeof(int) * dim);
	for(int i = 0; i < dim; i++){
		deviationret[i] = deviation[dim-1-i];
		dimensionret[i] = dimension[dim-1-i];
	}
	free(deviationret);
	free(deviation);
	free(dimension);
	return dimensionret;
}

double euclideanDistance(double * dataPoint, int dim, double * RP){
    // get the euclidean distance
    double distance = 0;
    for(int i = 0; i < dim; i++){
        double diff = (RP[i] - dataPoint[i]);
        distance += diff * diff;
    }
    distance = sqrt(distance);
    return distance;
}

double * createRPArray(double * data, int numRP, int dim, unsigned long long numPoints){

	int sample_size = numPoints*SAMPLE_PER;

	double * testRPArray = new double[TEST_RP*dim];

	//randomly place the rps
	// #pragma omp parallel for
	for(int i = 0; i < TEST_RP*dim; i++){
		testRPArray[i] = (double)rand()/(double)RAND_MAX;
	}

	//get the distances
	double *distmat = new double[TEST_RP*sample_size];

	// #pragma omp parallel for
	for(int i = 0; i < sample_size; i++){
		for(int j = 0; j < TEST_RP; j++){
			distmat[i*TEST_RP+j] = euclideanDistance(&data[i*dim], dim, &testRPArray[j*dim]);
		}
	}

	//get std dev of dist mat
	int * order = stddev(distmat, TEST_RP, sample_size);

	//get first numRP rps
	double * RPArray = (double *)malloc(sizeof(double)*numRP*dim);

	// #pragma omp parallel for
	for(int i = 0; i < numRP; i++){
		for(int j = 0; j < dim; j++){
			RPArray[i*dim+j] = testRPArray[ order[i]*dim + j ];
		}
	}

	// delete(testRPArray);
	// delete(distmat);

    return RPArray;
}
