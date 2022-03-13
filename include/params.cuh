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
#define HOST true

const int BLOCK_SIZE = 1024; 
const int KERNEL_BLOCKS = 32;
const int BRUTE = false;
const int RANDOM = false;
const int TEST_RP = 100; 
const double SAMPLE_PER = 0.01;

const int CALCS_PER_THREAD = 10000;

const int MAX_BIN = 100000; 
 
const int MAXRP = 1;

const int MINRP = 0;

const int RPPERLAYER = 10;

const int RAND = false;

const int resultsSize = 10000000; // 40MB