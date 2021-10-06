#pragma once

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
 
#include "include/params.cuh"

struct result{
    int pid;
    int numNeighbors;
    int *neighbors;
};

double euclideanDistance(double * dataPoint, int dim, double * RP);

double * createRPArray(double * data, int numRP, int dim, unsigned long long numPoints);

int * stddev( double * A, int dim, int num_points);
int brute_force( int num_points, int dim, double epsilon, double *A);