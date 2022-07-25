
#include "include/nodes.cuh"

//function to build up net of nodes
unsigned int buildNodeNet(double * data,
                 unsigned int dim,
                 unsigned int numPoints,
                 unsigned int numRP,
                 unsigned int * pointArray,
                 double epsilon,
                 std::vector<struct Node> * nodes){


    std::vector<struct Node> newNodes;
    unsigned int numNodes;
    
    // need to go through each reference point
    for(unsigned int i = 0; i < numRP; i++){ 

        // generate some reference points
        double * RPArray = createRPArray(data, numRP, dim, numPoints);
    
        std::vector<struct Node> layerNodes;
        unsigned long long int lowestDistCalcs = ULLONG_MAX;
        unsigned int bestRP = 0;

        // #pragma omp parallel for num_threads(RPPERLAYER)
        for(unsigned int j = 0; j < RPPERLAYER; j++){
            // need to compare num dist calcs for different potental RP
            std::vector<struct Node> tempNodes;
            unsigned int tempNumNodes = 0;
            if(i == 0){tempNumNodes = initNodes(data, dim, numPoints, epsilon, &RPArray[j], pointArray, &tempNodes);}
            else{tempNumNodes = splitNodes(&RPArray[j], newNodes, newNodes.size(), epsilon, data, dim, numPoints, &tempNodes);}
            
            // printf("check: %llu\n", tempNodes[0].numCalcs);
            
            unsigned long long numCalcs = totalNodeCalcs(tempNodes, tempNumNodes);
            unsigned long long sumSqrs = nodeSumSqrs(tempNodes, tempNumNodes);
            // printf("Layer %d for RP %d has Nodes: %u with calcs: %llu , and sumSQRs: %llu\n", i, j, tempNumNodes, numCalcs, sumSqrs);

            // #pragma omp critical
            {
                if(numCalcs < lowestDistCalcs){
                    lowestDistCalcs = numCalcs;
                    bestRP = j;
                    layerNodes = tempNodes;
                    numNodes = tempNumNodes;
                }
            }
            
        }

        printf("Layer %d Selecting RP %d with Nodes: %u and calcs: %llu\n", i, bestRP, numNodes, lowestDistCalcs);

        newNodes = layerNodes;
    }

    // printf("check: %llu\n", newNodes[0].numCalcs);
    unsigned long long numCalcs = totalNodeCalcs(newNodes, numNodes);
    unsigned long long sumSqrs = nodeSumSqrs(newNodes, numNodes);

    printf("Final graph has %u nodes with: %llu calcs and sumSqrs: %llu\n", numNodes, numCalcs, sumSqrs);


    //rearange the pointArray
    unsigned int counter = 0;
    for(unsigned int i = 0; i < newNodes.size(); i++){
        //append own index to neighbors
        newNodes[i].pointOffset = counter;
        for(unsigned int j = 0; j < newNodes[i].numNodePoints; j++){
            pointArray[counter] = newNodes[i].nodePoints[j];
            counter++;
        }
    }

    *nodes = newNodes;


    return numNodes;

}

unsigned int initNodes(double * data,
                        unsigned int dim,
                        unsigned int numPoints,
                        double epsilon,
                        double * RP,
                        unsigned int * pointArray,
                        std::vector<struct Node> * nodes){



    std::vector<struct Node> newNodes;

    //make the first set of nodes
    unsigned int * binNumber = (unsigned int * )malloc(sizeof(unsigned int)*numPoints);

    #pragma omp parallel for
    for(unsigned int i = 0; i < numPoints; i++){
        //get distance of each point in the node to the reference point
        binNumber[i] = floor( euclideanDistance(&data[i*dim],dim,RP) / epsilon);
    }

    // sort the node points based on their bin numbers
    thrust::sort_by_key(thrust::host, &binNumber[0], &binNumber[numPoints], &pointArray[0]);


    //if all the points are in the same bin
    if(binNumber[0] == binNumber[numPoints-1]){

        newNodes.push_back(newNode(numPoints, pointArray, binNumber[0], 0));

        updateNodeCalcs(&newNodes, newNodes.size());
        //free temp memory
        free(binNumber);

        *nodes = newNodes;
        //go to the next node
        return 1;
    }

    //go through and make nodes

    //variable to keep track of last bin end
    unsigned int tempBinPointer = 0;

    //variable to count new nodes
    unsigned int numNewNodes = 0;

    unsigned int bcounter = 0;
    //scan through and create a new node for each non-empty bin
    for(unsigned int i = 0; i < numPoints; i++){
        bcounter++;
        //check if need to make a new node
        if(i == numPoints-1 || binNumber[i] != binNumber[i+1]){
            // printf("making new node, j: %d, tempBinPointer: %d, numPoints in the new node:%d\n", i, tempBinPointer, i - tempBinPointer+1 );
            // if(i== numPoints - 1) {
            //     printf("BinNumber#%u: %u->%u: p=%u->%u\n", numNewNodes, binNumber[i], 0 ,bcounter,i-tempBinPointer+1);
            // }else{ 
            //     printf("BinNumber#%u: %u->%u: p=%u->%u\n", numNewNodes, binNumber[i], binNumber[i+1], bcounter,i-tempBinPointer+1);
            // }
            //push back the new node onto the temporary vector of nodes
            newNodes.push_back( newNode(i-tempBinPointer+1, pointArray+tempBinPointer, binNumber[i], numNewNodes ) );
            tempBinPointer = i+1;
            numNewNodes++;
            bcounter = 0;
        }
    }

    //create the connections
            //now that the split nodes exist, modify neighbor values
    //special case for the first
    if(newNodes[0].binNumbers.back() == newNodes[1].binNumbers.back() - 1 ){ //already know at least 2 nodes in list
        newNodes[0].neighborIndex.push_back(1);
    } 

    // handle all middle nodes
    for(unsigned int i = 1; i < numNewNodes-1; i++){
        //check if lower bin is one away
        if(newNodes[i].binNumbers.back() == newNodes[i-1].binNumbers.back() + 1){
            newNodes[i].neighborIndex.push_back(i-1);
        }
        //check if upper bin is one away
        if(newNodes[i].binNumbers.back() == newNodes[i+1].binNumbers.back() - 1){
            newNodes[i].neighborIndex.push_back(i+1);
        }
    }

    //special case for last node
    if(newNodes[numNewNodes-1].binNumbers.back() == newNodes[numNewNodes-2].binNumbers.back() + 1 ){ //already know at least 2 nodes in list
        newNodes[numNewNodes-1].neighborIndex.push_back(numNewNodes-2);
    }


    updateNodeCalcs(&newNodes, newNodes.size());
    //assign the vector to the return
 
    // printf("check: %llu\n", newNodes[0].numCalcs);

    *nodes = newNodes;
    
    free(binNumber);

    return numNewNodes;

}

//splits a node based on a reference point and return the number of new nodes
unsigned int splitNodes(double * RP, //the reference point used for the split
                    std::vector<struct Node> nodes,// the array of nodes
                    unsigned int numNodes,//the number of nodes
                    double epsilon, //the distance threshold of the search
                    double * data, //the dataset
                    unsigned int dim,//the number of dimensions of the data
                    unsigned int numPoints,// number of points in the dataset
                    std::vector<struct Node> * newNodes){  // pointer for returning the new nodes
    

    // printf("Start split\n");
    //need to keep track of all of the new split nodes
    std::vector<std::vector<struct Node>> tempNewNodes;
    tempNewNodes.resize(numNodes);

    // printf("allocated vec for %d nodes\n", numNodes);
    // go through each node and split
    for(unsigned int i = 0; i < numNodes; i++){

        // printf("For node %d, starting binning\n", i);
        //temp array to hold each points new bin number
        unsigned int * binNumber = (unsigned int * )malloc(sizeof(unsigned int)*nodes[i].numNodePoints);


        // break nodes into new nodes
        #pragma omp parallel for
        for(unsigned int j = 0; j < nodes[i].numNodePoints; j++){
            //get distance of each point in the node to the reference point
            binNumber[j] = floor( euclideanDistance(&data[j*dim],dim,RP) / epsilon);
        }

        // printf("finished binning\n");


        // sort the node points based on their bin numbers
        thrust::sort_by_key(thrust::host, &binNumber[0], &binNumber[nodes[i].numNodePoints], &nodes[i].nodePoints[0]);
        
        // printf("finished sorting\n");
        
        //temp vector to hold new nodes
        std::vector<struct Node> tempNodes;


        //if all the points are in the same bin
        if(binNumber[0] == binNumber[nodes[i].numNodePoints-1]){
            // printf("no splits\n");
            //add the bin number
            // nodes[i].binNumbers.push_back(binNumber[0]);
            tempNodes.push_back(newNode(nodes[i].numNodePoints, &(nodes[i].nodePoints[0]), nodes[i], binNumber[0], 0 ) );
            tempNewNodes[i] = tempNodes;
            //free temp memory
            free(binNumber);

            //go to the next node
            continue;
        }

        // printf("finished same bin Check\n");




        //variable to keep track of last bin end
        unsigned int tempBinPointer = 0;

        //variable to count new nodes
        unsigned int numNewNodes = 0;

        unsigned int bcounter = 0;

        //scan through and create a new node for each non-empty bin
        // printf("NumNodePoints for node %u: %u\n", i,  nodes[i].numNodePoints);
        for(unsigned int j = 0; j < nodes[i].numNodePoints; j++){
            bcounter++;
            
            //check if need to make a new node
            if(j == nodes[i].numNodePoints-1 || binNumber[j] != binNumber[j+1]){
                
                // printf("making new node, j: %d, numNodePoints: %d, tempBinPointer: %d, numPoints in the new node:%d\n", j,nodes[i].numNodePoints, tempBinPointer, j - tempBinPointer+1 );
                //push back the new node onto the temporary vector of nodes
                if(j== nodes[i].numNodePoints - 1) {
                    printf("BinNumber#%u: %u->%u: p=%u->%u; j: %u\n", numNewNodes, binNumber[j], 0 ,bcounter,j-tempBinPointer+1, j);
                }else{ 
                    printf("BinNumber#%u: %u->%u: p=%u->%u; j: %u\n", numNewNodes, binNumber[j], binNumber[j+1], bcounter,j-tempBinPointer+1, j);
                }
                tempNodes.push_back( newNode(j-tempBinPointer+1, &(nodes[i].nodePoints[0]) + tempBinPointer, nodes[i], binNumber[j], numNewNodes ) );
                tempBinPointer = j+1;
                numNewNodes++;
                bcounter = 0;
            }
        }

        // printf("finished creating new nodes for non empty bins: new nodes : %u\n", numNewNodes);

        //now that the split nodes exist, modify neighbor values
        //special case for the first
        if(tempNodes[0].binNumbers.back() == tempNodes[1].binNumbers.back() - 1 ){ //already know at least 2 nodes in list
            tempNodes[0].neighborIndex.push_back(1);
        } 

        // handle all middle nodes
        for(unsigned int j = 1; j < numNewNodes-1; j++){
            //check if lower bin is one away
            if(tempNodes[j].binNumbers.back() == tempNodes[j-1].binNumbers.back() + 1){
                tempNodes[j].neighborIndex.push_back(j-1);
            }
            //check if upper bin is one away
            if(tempNodes[j].binNumbers.back() == tempNodes[j+1].binNumbers.back() - 1){
                tempNodes[j].neighborIndex.push_back(j+1);
            }
        }

        //special case for last node
        if(tempNodes[numNewNodes-1].binNumbers.back() == tempNodes[numNewNodes-2].binNumbers.back() + 1 ){ //already know at least 2 nodes in list
            tempNodes[numNewNodes-1].neighborIndex.push_back(numNewNodes-2);
        } 

        
        //copy the temp nodes back into the larger vector of nodes
        tempNewNodes[i]  = tempNodes;
        free(binNumber);
    }
    // printf("finished intital splitting\n");

    ////////////////////////////////////////////////////////////////
    // All nodes in the temp nodes now point to their neighbors within the smaller array
    // the neighbor index values will have to be updated when merging with the other vectors
    /////////////////////////////////////////////////////////////

    //need to make linear index ids now
    //a variable for the index offsets
    unsigned int nodeIndexOffset = 0;
    for(unsigned int i = 0; i < numNodes; i++){
        for(unsigned int j = 0; j < tempNewNodes[i].size(); j++){
            tempNewNodes[i][j].nodeIndex += nodeIndexOffset;
            for(unsigned int k = 0; k < tempNewNodes[i][j].neighborIndex.size(); k++){
                tempNewNodes[i][j].neighborIndex[k] += nodeIndexOffset;
            }
        }
        nodeIndexOffset += tempNewNodes[i].size();
    }

    // go through the old node list and compare bins to adjacent splits of old nodes

    for(unsigned int i = 0; i < numNodes; i++){ //for each old node
        for(unsigned int j = 0; j < tempNewNodes[i].size(); j++){ //go through each split off node
            // bin number that we are looking for adjacents to
            unsigned int nodeBinNumber = tempNewNodes[i][j].binNumbers.back();
            for(unsigned int k = 1; k < nodes[i].neighborIndex.size(); k++){ //and check the neighbors
                //neighbor to check
                unsigned int neighborNodesIndex = nodes[i].neighborIndex[k]; // this will also give the index of the vector of split nodes
                
                //go through each neighbors split nodes
                for(unsigned int l = 0; l < tempNewNodes[neighborNodesIndex].size(); l++){
                    unsigned int checkBin = tempNewNodes[neighborNodesIndex][l].binNumbers.back();
                    if(checkBin + 1 == nodeBinNumber || 
                        checkBin - 1 == nodeBinNumber ||
                        checkBin == nodeBinNumber){
                        tempNewNodes[i][j].neighborIndex.push_back(tempNewNodes[neighborNodesIndex][l].nodeIndex);
                    }
                }
            }
        }
    }

    // printf("finsihed updating neighbors\n");


    //make a new linear array of nodes
    std::vector<struct Node> nodeVec;
    for(unsigned int i = 0; i < tempNewNodes.size(); i++){
        nodeVec.insert(nodeVec.end(), tempNewNodes[i].begin(), tempNewNodes[i].end());
        // for(unsigned int j = 0; j < tempNewNodes[i].size(); j++){
        //     nodeVec.push_back(tempNewNodes[i][j]);
        // }
    }

    updateNodeCalcs(&nodeVec, nodeVec.size());

    // printf("NumNodes: %u, TotalCalcs: %llu\n", (unsigned int)nodeVec.size(), totalNodeCalcs(nodeVec, nodeVec.size()));

    *newNodes = nodeVec;

    return (unsigned int)nodeVec.size();

}


struct Node newNode(unsigned int numNodePoints, //number of points to go into the node
                    unsigned int * nodePoints, // the start of the points that will go into the node
                    struct Node parent, //the parent node
                    unsigned int binNumber,//the bin number of the node
                    unsigned int nodeNumber){ //the index number of the node

    struct Node newNode;
    newNode.nodeIndex = nodeNumber;
    newNode.numNodePoints = numNodePoints;
    // newNode.binNumbers = parent.binNumbers;
    for(unsigned int i = 0; i < parent.binNumbers.size();i++){
        newNode.binNumbers.push_back(parent.binNumbers[i]);
    }
    newNode.binNumbers.push_back(binNumber);
    newNode.neighborIndex.push_back(nodeNumber);
    // newNode.nodePoints.insert(newNode.nodePoints.begin(), &nodePoints[0], &nodePoints[numNodePoints-1] ); //double check this
    newNode.nodePoints.resize(numNodePoints);
    for(unsigned int i = 0; i < numNodePoints; i++){
        newNode.nodePoints[i] = nodePoints[i];
    }
    return newNode;
};

struct Node newNode(unsigned int numNodePoints, //number of points to go into the node
                    unsigned int * nodePoints, // the start of the points that will go into the node
                    unsigned int binNumber,//the bin number of the node
                    unsigned int nodeNumber){ //the index number of the node

    struct Node newNode;
    newNode.nodeIndex = nodeNumber;
    newNode.numNodePoints = numNodePoints;
    newNode.binNumbers.push_back(binNumber);
    newNode.neighborIndex.push_back(nodeNumber);
    // newNode.nodePoints.insert(newNode.nodePoints.begin(), &nodePoints[0], &nodePoints[numNodePoints-1]); //double check this
    newNode.nodePoints.resize(numNodePoints);
    for(unsigned int i = 0; i < numNodePoints; i++){
        newNode.nodePoints[i] = nodePoints[i];
    }
    return newNode;
};

void updateNodeCalcs(std::vector<struct Node> * nodes,
                     unsigned int numNodes){

    bool verboseNodeInfo = true;
    for(unsigned int i = 0; i < numNodes; i++){
        if(verboseNodeInfo) printf("Node %d has:\n",i);
        unsigned long long int numNeighboringPoints = 0;
        if(verboseNodeInfo) printf("    %u points\n", (*nodes)[i].numNodePoints);
        if(verboseNodeInfo) printf("    %lu neighbors\n", (*nodes)[i].neighborIndex.size());
        for(unsigned int j = 0; j < (*nodes)[i].neighborIndex.size(); j++){
            if(verboseNodeInfo) printf("    neighbors bin: %d with numPoints: %u\n", (*nodes)[i].neighborIndex[j],(*nodes)[(*nodes)[i].neighborIndex[j]].numNodePoints);
            numNeighboringPoints += (unsigned long long int)(*nodes)[(*nodes)[i].neighborIndex[j]].numNodePoints;
        }
        (*nodes)[i].numCalcs = (unsigned long long int)numNeighboringPoints*(*nodes)[i].numNodePoints;
        if(verboseNodeInfo) printf("    %llu total calcs to make\n", (*nodes)[i].numCalcs);
    }

}

unsigned long long totalNodeCalcs(std::vector<struct Node> nodes, unsigned int numNodes){
    unsigned long long totalCalcs = 0;
    
    for(unsigned int i = 0; i < nodes.size(); i++){
        
        totalCalcs += nodes[i].numCalcs;
    }

    return totalCalcs;
}

unsigned long long nodeSumSqrs(std::vector<struct Node> nodes, unsigned int numNodes){
    unsigned long long sumSqrs = 0;
    
    for(unsigned int i = 0; i < nodes.size(); i++){
        sumSqrs += nodes[i].numNodePoints*nodes[i].numNodePoints;
    }

    return sumSqrs;
}

unsigned long long nodeForce(std::vector<struct Node> * nodes, double epsilon, double * data, unsigned int dim, unsigned int numPoints){

    std::vector<unsigned int> PA;
    std::vector<unsigned int> PB;
    
    bool check1 = true;
    bool check2 = true;
    bool verboseNodeInfo = true;
    // #pragma omp parallel for
    if(verboseNodeInfo)printf("\n************************************************\n");
    if(verboseNodeInfo)printf("Number of Nodes: %lu\n", (*nodes).size());
    for(unsigned int i = 0; i < (*nodes).size();i++){
        if(verboseNodeInfo) printf("Node: %u\n  numNeighbors: %lu\n", i, (*nodes)[i].neighborIndex.size());
        unsigned long long sum = 0;
        for(unsigned int j = 0; j < (*nodes)[i].neighborIndex.size();j++){
            if(verboseNodeInfo) printf("    neighbor at index: %u\n", (*nodes)[i].neighborIndex[j]);
            if(verboseNodeInfo) printf("        numPoints: %u\n", (*nodes)[(*nodes)[i].neighborIndex[j]].numNodePoints);
            if(verboseNodeInfo){
                printf("        Bin Numbers: |");
                for(unsigned int h = 0; h <  (*nodes)[(*nodes)[i].neighborIndex[j]].binNumbers.size(); h++){
                    printf(" %u |", (*nodes)[(*nodes)[i].neighborIndex[j]].binNumbers[h]);
                }
                printf("\n");
            }
            for(unsigned int k = 0; k < (*nodes)[i].numNodePoints; k++){
                for(unsigned int l = 0; l <(*nodes)[(*nodes)[i].neighborIndex[j]].numNodePoints; l++){
                    unsigned int a = (*nodes)[i].nodePoints[k];
                    unsigned int b = (*nodes)[(*nodes)[i].neighborIndex[j]].nodePoints[l];
                    if(a == 50 && check1) {
                        printf("                 point 50 Node Number: %u\n", i);
                        check1 = false;
                    }
                    if(a == 53 && check2) {
                        printf("                 point 53 Node Number: %u\n", i);
                        check2 = false;
                    }
                    if(verboseNodeInfo) if(b == a && j != 0) printf("ERROR:%u\n",a);
                    double running = 0;
                    for(unsigned int d = 0; d < dim; d++){
                        running += pow(data[a*dim + d] - data[b*dim + d], 2);
                    }
                    if(running <= epsilon*epsilon){
                        sum++;
                        PA.push_back(a);
                        PB.push_back(b);
                    }
                }

            }
        }
        (*nodes)[i].numResults = sum;
    }

    unsigned long long total = 0;
    for(unsigned int i = 0; i < (*nodes).size(); i++){
        total += (*nodes)[i].numResults;
    }
    // if(verboseNodeInfo)printf("TOTAL PAIRS: %llu\n", total);

    std::vector<unsigned int> BA;
    std::vector<unsigned int> BB;

    unsigned int brute_count = 0;
	omp_lock_t brute;
	omp_init_lock(&brute);

	#pragma omp parallel for
	for(unsigned int i = 0; i < numPoints; i++)
	{
		for (unsigned int j = 0; j < numPoints; j++)
		{
		double distance = 0;
			for (unsigned int k = 0; k < dim; k++)
			{
				if(distance > epsilon*epsilon)
				{
					break;
				} else {
					double a1 = data[i*dim + k];
					double a2 = data[j*dim + k];
					distance += (a1-a2)*(a1-a2);
				}
				}
				if(distance <= epsilon*epsilon){
					omp_set_lock(&brute);
					brute_count++;
                    BA.push_back(i);
                    BB.push_back(j);
					omp_unset_lock(&brute);
				}
		}
	}

    // printf("BrUTe count: %u", brute_count);

    std::vector< std::pair<unsigned int,unsigned int>> pairsB;

    for(unsigned int i = 0; i < BA.size();i++){
        pairsB.push_back(std::make_pair(BA[i],BB[i]));
    }

    std::sort(pairsB.begin(), pairsB.end(), compPair);

    std::vector< std::pair<unsigned int,unsigned int>> pairs;

    for(unsigned int i = 0; i < PA.size();i++){
        pairs.push_back(std::make_pair(PA[i],PB[i]));
    }

    std::sort(pairs.begin(), pairs.end(), compPair);

    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    
    std::vector< std::pair<unsigned int,unsigned int>> missing;

    // std::set_difference(pairsB.begin(), pairsB.end(), pairs.begin(), pairs.end(), std::inserter(missing, missing.begin()));
    unsigned int counter = 0;
    for(unsigned int i = 0; i < pairsB.size();i++){
        if(pairsB[i].first == pairs[counter].first && pairsB[i].second == pairs[counter].second){
            counter++;
        }else{
            missing.push_back(pairsB[i]);
        }
    }

    printf("Missing %lu pairs:\n", missing.size());
    for(unsigned int i = 0; i < missing.size(); i++){
        printf("(%u,%u),",missing[i].first, missing[i].second);
    }

    printf("\nTotal NodeFORCE results Set Size: %llu , unique pairs: %lu\n", total, pairs.size());

    if(verboseNodeInfo)printf("\n************************************************\n");


    return total;
}