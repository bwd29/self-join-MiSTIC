#include "include/kernel.cuh"



__global__
void distanceClalcs(){

    int tid = blockIDx.x*blockDim.x+threadIdx.x; //thread id that only works inside the invocation

    //number of calcs to make
    int modifier = 0;
    long long numCalcs = totalNumCalcs / threads;
    if(tid < totalNumCalcs % threads){
        numCalcs = numCalcs + 1;
        modifier = totalNumCalcs % threads - tid;
    }

    long long start = tid * (totalNumCalcs / threads) + (totalNumCalcs % threads) - modifier;

    //scan to start
    int startAddress = 0;
    int startLocationInAdd = 0;
    long long sum = 0;
    for(int i = 0; i < nonEmptyBins; i++){
        sum += calcPerAdd[i];
        if(sum > start){
            startAddress = i;
            startLocationInAdd = sum - start
            break;
        }
    }

    int startRange = 0;
    int startLocationInRange = 0;
    sum = 0;
    for(int i = 0; i < numValidRanges[startAddress]; i++){

        if(sum < startLocationInAdd){
            startRange = i;
            startLocationInRange = sum - startLocationInAdd;
            break;
        }

    }

    int currentAdd = startAddress;
    int currentRange = startRange;
    int currentRangeEnd = rangeIndexes[currentAdd][currentRange + 1]; //need to account for end here somehow
    while(true){
        for(int i = 0; i < currentRangeEnd - startLocationInRange; i++){
            
        }
    }


}