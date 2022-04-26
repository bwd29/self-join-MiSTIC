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

const int BLOCK_SIZE = 1024; 
const int KERNEL_BLOCKS = 1024;
const int BRUTE = false;
const int RANDOM = false;
const int TEST_RP = 100; 
const double SAMPLE_PER = 0.02;

const int CALCS_PER_THREAD = 100000;

const int MAX_BIN = 1000000; 
 
const int MAXRP = 6;

const int MINRP = 5;

const double LAYER_DIFF = 1.0;

const int RPPERLAYER = 10;

const int RAND = false;

const int resultsSize = 1000000000; // 400MB