#pragma once

#include "include/utils.cuh"
#include "include/params.cuh"


//struct to contain the node information
typedef struct Node{

    //the index of this node
    unsigned int nodeIndex;

    //the index of neighbor nodes in the array
    std::vector<unsigned int> neighborIndex;

    //the point numbres in this node
    std::vector<unsigned int> nodePoints;

    //the bin numbers of this node
    std::vector<unsigned int> binNumbers;

    //the number of points in this node
    unsigned int numNodePoints;

    //the number of calculations for this node
    //form numNodePoints * neighbors' numNodePoints
    unsigned long long int numCalcs;

    //start of points in pointArray
    unsigned int pointOffset;


}Node;





unsigned int buildNodeNet(double * data,
    unsigned int dim,
    unsigned int numPoints,
    unsigned int numRP,
    unsigned int * pointArray,
    double epsilon,
    std::vector<struct Node> * nodes);

//splits a node based on a reference point and return the number of new nodes
unsigned int splitNodes(double * RP, //the reference point used for the split
    std::vector<struct Node> nodes,// the array of nodes
    unsigned int numNodes,//the number of nodes
    double epsilon, //the distance threshold of the search
    double * data, //the dataset
    unsigned int dim,//the number of dimensions of the data
    unsigned int numPoints,// number of points in the dataset
    std::vector<struct Node> * newNodes);

struct Node newNode(unsigned int numNodePoints, //number of points to go into the node
    unsigned int * nodePoints, // the start of the points that will go into the node
    struct Node parent, //the parent node
    unsigned int binNumber,//the bin number of the node
    unsigned int nodeNumber);

struct Node newNode(unsigned int numNodePoints, //number of points to go into the node
    unsigned int * nodePoints, // the start of the points that will go into the node
    unsigned int binNumber,//the bin number of the node
    unsigned int nodeNumber);

void updateNodeCalcs(std::vector<struct Node> * nodes,
    unsigned int numNodes);

unsigned long long totalNodeCalcs(std::vector<struct Node> nodes, unsigned int numNodes);
unsigned long long nodeSumSqrs(std::vector<struct Node> nodes, unsigned int numNodes);

unsigned int initNodes(double * data,
    unsigned int dim,
    unsigned int numPoints,
    double epsilon,
    double * RP,
    unsigned int * pointArray,
    std::vector<struct Node> * nodes);