
#include "include/nodes.cuh"





//function to build up net of nodes
unsigned int buildNodeNet(double * data,
                 unsigned int dim,
                 unsigned int numPoints,
                 unsigned int numRP,
                 unsigned int * pointArray,
                 double epsilon,
                 struct Node ** nodes){

    // generate some reference points
    
    struct Node * newNodes;
    unsigned int numNodes;
    
    // need to go through each reference point
    for(unsigned int i = 0; i < numRP; i++){ 
        double * RPArray = createRPArray(data, numRP, dim, numPoints);
        struct Node * layerNodes;
        unsigned long long int lowestDistCalcs = ULLONG_MAX;
        unsigned int bestRP = 0;
        for(unsigned int j = 0; j < RPPERLAYER; j++){
            // need to compare num dist calcs for different potental RP
            struct Node * tempNodes;
            unsigned int tempNumNodes;
            if(i == 0){tempNumNodes = initNodes(data, dim, numPoints, epsilon, &RPArray[j*dim], pointArray, &tempNodes);}
            else{tempNumNodes = splitNodes(&RPArray[j*dim], newNodes, numNodes, epsilon, data, dim, numPoints, &tempNodes);}
            unsigned long long numCalcs = totalNodeCalcs(tempNodes, tempNumNodes);
            unsigned long long sumSqrs = nodeSumSqrs(tempNodes, tempNumNodes);
            printf("Layer %d for RP %d has Nodes: %u with calcs: %llu , and sumSQRs: %llu\n", i, j, tempNumNodes, numCalcs, sumSqrs);

            if(numCalcs < lowestDistCalcs){
                lowestDistCalcs = numCalcs;
                bestRP = j;
                layerNodes = tempNodes;
                numNodes = tempNumNodes;
            }
            
        }

        printf("Layer %d Selecting RP %d with Nodes: %u and calcs: %llu\n", i, bestRP, numNodes, lowestDistCalcs);

        newNodes = layerNodes;
    }

    unsigned long long numCalcs = totalNodeCalcs(&newNodes[0], numNodes);
    unsigned long long sumSqrs = nodeSumSqrs(&newNodes[0], numNodes);

    printf("Final graph has %u nodes with: %llu calcs and sumSqrs: %llu", numNodes, numCalcs, sumSqrs);

    *nodes = &newNodes[0];

    //rearange the pointArray
    unsigned int counter = 0;
    for(unsigned int i = 0; i < numNodes; i++){
        newNodes[i].pointOffset = counter;
        for(unsigned int j = 0; j < newNodes[i].numNodePoints; j++){
            pointArray[counter] = newNodes[i].nodePoints[j];
            counter++;
        }
    }


    return numNodes;

}

unsigned int initNodes(double * data,
                        unsigned int dim,
                        unsigned int numPoints,
                        double epsilon,
                        double * RP,
                        unsigned int * pointArray,
                        struct Node ** nodes){



    std::vector<struct Node> newNodes;

    //make the first set of nodes
    unsigned int * binNumber = (unsigned int * )malloc(sizeof(unsigned int)*numPoints);
    for(unsigned int i = 0; i < numPoints; i++){
        //get distance of each point in the node to the reference point
        binNumber[i] = floor( euclideanDistance(&data[i*dim],dim,RP) / epsilon);
    }

    // sort the node points based on their bin numbers
    thrust::sort_by_key(thrust::host, &binNumber[0], &binNumber[numPoints-1], &pointArray[0]);


    //if all the points are in the same bin
    if(binNumber[0] == binNumber[numPoints-1]){

        newNodes.push_back(newNode(numPoints, pointArray, binNumber[0], 0));
        *nodes = &newNodes[0];
        //free temp memory
        free(binNumber);

        //go to the next node
        return 1;
    }

    //go through and make nodes

    //variable to keep track of last bin end
    unsigned int tempBinPointer = 0;

    //variable to count new nodes
    unsigned int numNewNodes = 0;

    //scan through and create a new node for each non-empty bin
    for(unsigned int i = 0; i < numPoints; i++){

        //check if need to make a new node
        if(i == numPoints-1 || binNumber[i] != binNumber[i+1]){
            //push back the new node onto the temporary vector of nodes
            newNodes.push_back( newNode(i-tempBinPointer, pointArray, binNumber[i], numNewNodes ) );
            tempBinPointer = i;
            numNewNodes++;
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


    updateNodeCalcs(&newNodes[0], newNodes.size());
    //assign the vector to the return
    *nodes = &newNodes[0];

    free(binNumber);

    return numNewNodes;

}

//splits a node based on a reference point and return the number of new nodes
unsigned int splitNodes(double * RP, //the reference point used for the split
                    struct Node* nodes,// the array of nodes
                    unsigned int numNodes,//the number of nodes
                    double epsilon, //the distance threshold of the search
                    double * data, //the dataset
                    unsigned int dim,//the number of dimensions of the data
                    unsigned int numPoints,// number of points in the dataset
                    struct Node ** newNodes){  // pointer for returning the new nodes
    

    printf("Start split node\n");
    //need to keep track of all of the new split nodes
    std::vector<std::vector<struct Node>> tempNewNodes;
    tempNewNodes.resize(numNodes);
   
    printf("allocate vector node\n");


    // go through each node and split
    for(unsigned int i = 0; i < numNodes; i++){


        //temp array to hold each points new bin number
        unsigned int * binNumber = (unsigned int * )malloc(sizeof(unsigned int)*nodes[i].numNodePoints);


        // break nodes into new nodes
        for(unsigned int j = 0; j < nodes[i].numNodePoints; j++){
            //get distance of each point in the node to the reference point
            binNumber[j] = floor( euclideanDistance(&data[j*dim],dim,RP) / epsilon);
        }


        // sort the node points based on their bin numbers
        thrust::sort_by_key(thrust::host, &binNumber[0], &binNumber[nodes[i].numNodePoints-1], &nodes[i].nodePoints[0]);

        //if all the points are in the same bin
        if(binNumber[0] == binNumber[nodes[i].numNodePoints-1]){

            //add the bin number
            nodes[i].binNumbers.push_back(binNumber[0]);

            //free temp memory
            free(binNumber);

            //go to the next node
            continue;
        }



        //temp vector to hold new nodes
        std::vector<struct Node> tempNodes;


        //variable to keep track of last bin end
        unsigned int tempBinPointer = 0;

        //variable to count new nodes
        unsigned int numNewNodes = 0;

        //scan through and create a new node for each non-empty bin
        for(unsigned int j = 0; j < nodes[i].numNodePoints; j++){
            
            //check if need to make a new node
            if(j == nodes[i].numNodePoints-1 || binNumber[j] != binNumber[j+1]){
                //push back the new node onto the temporary vector of nodes
                tempNodes.push_back( newNode(j-tempBinPointer, &(nodes[j].nodePoints[tempBinPointer]), nodes[i], binNumber[j], numNewNodes ) );
                tempBinPointer = j;
                numNewNodes++;
            }
        }

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

    // go through the old node list and compare bins to adgacent splits of old nodes

    for(unsigned int i = 0; i < numNodes; i++){ //for eacch old node
        for(unsigned int j = 0; j < tempNewNodes[i].size(); j++){ //go through each split off node
            // bin number that we are lloking for adjacents to
            unsigned int nodeBinNumber = tempNewNodes[i][j].binNumbers.back();
            for(unsigned int k = 0; k < nodes[i].neighborIndex.size(); k++){ //and check the neighbors
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


    //make a new linear array of nodes
    std::vector<struct Node> nodeVec;
    for(unsigned int i = 0; i < numNodes; i++){
        nodeVec.insert(nodeVec.end(), tempNewNodes[i].begin(), tempNewNodes[i].end());
    }

    updateNodeCalcs(&nodeVec[0], nodeVec.size());

    printf("NumNodes: %u, TotalCalcs: %llu", (unsigned int)nodeVec.size(), totalNodeCalcs(&nodeVec[0], nodeVec.size()));

    *newNodes = &nodeVec[0];

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
    newNode.binNumbers = parent.binNumbers;
    newNode.binNumbers.push_back(binNumber);
    newNode.nodePoints.insert(newNode.nodePoints.begin(), &nodePoints[0], &nodePoints[numNodePoints-1] ); //double check this

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
    newNode.nodePoints.insert(newNode.nodePoints.begin(), &nodePoints[0], &nodePoints[numNodePoints-1]); //double check this

    return newNode;
};

void updateNodeCalcs(struct Node * nodes,
                     unsigned int numNodes){

    for(unsigned int i = 0; i < numNodes; i++){
        unsigned long long int numNeighboringPoints = 0;
        for(unsigned int j = 0; j < nodes[i].neighborIndex.size(); j++){
            numNeighboringPoints += nodes[nodes[i].neighborIndex[j]].numNodePoints;
        }
        nodes[i].numCalcs = numNeighboringPoints*nodes[i].numNodePoints;
    }

}

unsigned long long totalNodeCalcs(struct Node * nodes, unsigned int numNodes){
    unsigned long long totalCalcs = 0;
    
    for(unsigned int i = 0; i < numNodes; i++){
        totalCalcs += nodes[i].numCalcs;
    }

    return totalCalcs;
}

unsigned long long nodeSumSqrs(struct Node * nodes, unsigned int numNodes){
    unsigned long long totalCalcs = 0;
    
    for(unsigned int i = 0; i < numNodes; i++){
        totalCalcs += nodes[i].numNodePoints*nodes[i].numNodePoints;
    }

    return totalCalcs;
}