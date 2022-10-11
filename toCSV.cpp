
#include <sys/stat.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define VERBOSE false


int main(int argc, char*argv[])
{
	//reading in command line arguments
	char *filename = argv[1];
    char *outfile = argv[2];
    int dim = atoi(argv[3]);

    std::ifstream file(	filename, std::ios::in | std::ios::binary);
    file.seekg(0, std::ios::end); 
    size_t size = file.tellg();  
    file.seekg(0, std::ios::beg); 
    char * read_buffer = new char[size];
    file.read(read_buffer, size*sizeof(double));
    file.close();

    double* A = (double*)read_buffer;//reinterpret as doubles

    // for (int i = 0; i < 10; i++){
    //     printf("%f, ", A[i]);
    // }

    std::ofstream myFile;
    myFile.open(outfile, std::ios::out);

    for(int i = 0; i < size/sizeof(double); i+=dim){
        for(int j = 0; j < dim; j++){
            myFile << A[i+j] << ',';
        }
        myFile << '\n';
    }

    // myFile.write(pointer, data.size()*sizeof(double));

    myFile.close();



    // //reading in file
	// FILE *fptr;
	// fptr = fopen(filename, "r");
	// if (fptr == NULL)
	// {
	// 	printf("No such File\n");
	// 	exit(0);
	// }
	// double check = 0;
    
    // std::vector<double> data;

	// while( fscanf(fptr, "%lf, ", &check) == 1 || fscanf(fptr, "%lf ", &check) == 1)
	// {
	// 	data.push_back(check);
	// }
	// fclose(fptr);

    // if(VERBOSE){

    //     for (int i = 0; i < data.size(); i++){
    //         printf("%f, ", data[i]);
    //     }
    //     printf("\n**************************************************************\n");
    // }

    // //writing a binary file
    // const char* pointer = reinterpret_cast<const char*>(&data[0]);

    // std::ofstream myFile;
    // myFile.open(outfile, std::ios::out | std::ios::binary);

    // myFile.write(pointer, data.size()*sizeof(double));

    // myFile.close();

    // if(VERBOSE){

    //     std::ifstream file(	outfile, std::ios::in | std::ios::binary);
    //     file.seekg(0, std::ios::end); 
    //     size_t size = file.tellg();  
    //     file.seekg(0, std::ios::beg); 
    //     char * read_buffer = new char[size];
    //     file.read(read_buffer, size*sizeof(double));
    //     file.close();

    //     double* A = (double*)read_buffer;//reinterpret as doubles

    //     for (int i = 0; i < size/sizeof(double); i++){
    //         printf("%f, ", A[i]);
    //     }
    // }
}