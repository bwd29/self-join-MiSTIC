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

	int num_points = size/sizeof(double)/dim;


	printf("\nNumber points: %d ", num_points);
	printf("\nNumber Dimensions: %d ", dim);
	printf("\nNumber Threads Per Point: %d ", tpp);
	printf("\nNumber Concurent Streams: %d", concurent_streams);
	printf("\nDistance Threshold: %f \n*********************************\n\n", epsilon);




	int *dimension_order = (int*)malloc(sizeof(int)*dim);
	double * dim_ordered_data = (double*)malloc(sizeof(double)*num_points*dim);

    dimension_order = stddev(A, dim, num_points);
    #pragma omp parallel for
    for(int i = 0; i < num_points; i++){
        for(int j = 0; j < dim; j++){
            dim_ordered_data[i*dim + j] = A[i*dim + dimension_order[j]];
        }
    }
    A = dim_ordered_data;;



	double time1 = omp_get_wtime();


	//build tree
	int * point_array = (int*)malloc(sizeof(int)*num_points);
	for (int i = 0; i < num_points; i++){
		point_array[i] = i;
	}


	int ** pointBinNumbers;

	unsigned int binSizes[MAXRP];
	unsigned int binAmounts[MAXRP];
	int maxBinAmount = MAX_BIN;
	int ** binArrays;
	int rps = buildTree(
					&binArrays,
					A,
					dim,
					num_points,
					epsilon,
					maxBinAmount,
					point_array,
					&pointBinNumbers,
					binSizes,
					binAmounts);


    double * point_ordered_data = (double *)malloc(sizeof(double)*num_points*dim);
    #pragma omp parallel for
	for(int i = 0; i < num_points; i++){
		for(int j = 0; j < dim; j++){
			point_ordered_data[i*dim+j] = A[point_array[i]*dim+j];
		}
	}
	A = point_ordered_data;





    double time2 = omp_get_wtime();

    printf("Time to build tree: %f\n", time2-time1);

    return 1;

}