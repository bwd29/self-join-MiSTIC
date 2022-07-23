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
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>  
#include <vector>

#define DATANORM true
#define HOST false
#define BINARYSEARCH 1
#define GPUSEARCH false
#define TESTING_SEARCH false
#define MINSQRS true
#define NODES true

#define CUDA_DEVICE 2

const unsigned int MAXRP = 6;

const unsigned int MINRP = 5;


const unsigned int NUMSTREAMS = 2;
const unsigned int BLOCK_SIZE = 1024; 
const unsigned int KERNEL_BLOCKS = 2*1024*1024/BLOCK_SIZE;
const int BRUTE = false;
const int RANDOM = false;
const int BOXED_RP = false;
const unsigned int TEST_RP = 100; 
const double SAMPLE_PER = 0.01;


const unsigned int CALCS_PER_THREAD = 100000;
const unsigned int MAX_CALCS_PER_THREAD = 250000;
const unsigned int MIN_CALCS_PER_THREAD = 10000;

const unsigned int MAX_BIN = 1000000; 
 

const double LAYER_DIFF = 1.0;

const unsigned int RPPERLAYER = 16;

const int RAND = false;
const unsigned long long initalPinnedResultsSize = 10000;
const unsigned long long int resultsSize = 1000000000; // 400MB