#pragma once
#include "include/params.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "omp.h"
#include <unistd.h>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h> 
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>   

struct result{
    int pid;
    int numNeighbors;
    int *neighbors;
};

double euclideanDistance(double * dataPoint, int dim, double * RP);

double * createRPArray(double * data, int numRP, int dim, unsigned long long numPoints);

int * stddev( double * A, int dim, int num_points);