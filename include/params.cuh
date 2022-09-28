#pragma once

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
#include <stack>

#define DATANORM true
#define HOST false
#define BINARYSEARCH 1
#define GPUSEARCH false
#define TESTING_SEARCH false
#define MINSQRS true
#define NODES true
#define ERRORPRINT true
#define NODETEST false
#define RANDRP false

#define SUBG false

#define DEVICE_BUILD true

#define CUDA_DEVICE 1
// #define CALC_MULTI 4
// #define MIN_NODE_SIZE 1000

const unsigned int RPPERLAYER = 10;

const unsigned int MAXRP = 20;

const unsigned int MINRP = 2;


const unsigned int NUMSTREAMS = 2;
const unsigned int BLOCK_SIZE = 1024; 
const unsigned int KERNEL_BLOCKS = 2*1024*1024/BLOCK_SIZE;
const int BRUTE = false;
const int RANDOM = false;
const int BOXED_RP = false;
const unsigned int TEST_RP = 100; 
const double SAMPLE_PER = 0.01;
// const unsigned int MIN_NODE_SIZE = 1000; //value of 1 shuts this off

const unsigned int CALCS_PER_THREAD = 100000;
const unsigned int MAX_CALCS_PER_THREAD = 250000;
const unsigned int MIN_CALCS_PER_THREAD = 10000;

const unsigned int MAX_BIN = 100000; 

// const unsigned long long int calcsPerSecond = 50000000000;
// const unsigned int nodesPerSecond = 2000;

const double LAYER_DIFF = 1.25;



const int RAND = false;
const unsigned long long initalPinnedResultsSize = 100000;
const unsigned long long int resultsSize = 1000000000; // 400MB