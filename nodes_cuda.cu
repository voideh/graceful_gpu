#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include "time.h"
using namespace std;
int factorial(int n)
{
  return (n == 0 || n == 1) ? 1 : factorial(n-1) * n;
}

__global__ void calculate_edges(int *perms, int *children, int *stops, int *edges, int NUMNODES, int NUMPERMS)
{
    /*
       Since the permutation array is a flattened 2D array which was NUMNODES wide and NUMPERMS long, then we
       must start at the begining of every row which would be offset by NUMNODES.
   */
    int element = (blockIdx.x * blockDim.x + threadIdx.x) * NUMNODES; 
    int total = NUMNODES * NUMPERMS; // make sure we do not exceed the size of the permutation array
    int edge_counter = 0; // keep track of where in the edge array we are putting the next edge label
    int last_index = 0; // keep track of the last index from the stop array.
    int edge_start = (blockIdx.x * blockDim.x + threadIdx.x) * (NUMNODES-1); // calculate where in the edge array we should begin placing labels
    if(element < total)
    {
        // Only go thorugh each NUMNODE group of labels
        for(int i = element; i < element + NUMNODES; i++)
        {
            // check for sentinel value of -1
            if(stops[i % NUMNODES] != -1)
            {
                // If this is our first time we start at 0, otherwise we continue from the the last index
                for(int j = (last_index == 0) ? 0 : last_index+1; j<=stops[i % NUMNODES]; j++)
                    // place the absolute difference of each end point into the edge array
                    edges[edge_start + edge_counter++] = abs(perms[i] - perms[children[j] + element]);
                last_index = stops[i % NUMNODES];
            }
        } }
}

__global__ void check_gracefulness(int *edges, int *graceful_labels, int NUMNODES, int NUMPERMS)
{
    /*
       Go through edge array and check for any duplicates. If there are duplicates found, exit the loop and mark this label
       as being nongraceful , which is designated by a -1 in the label array. If no duplicates are found, the labeling is graceful and
       the index of the permutation is stored.
   */
    int element = (blockIdx.x * blockDim.x + threadIdx.x) * (NUMNODES-1); 
    int total = NUMNODES * NUMPERMS;
    bool graceful = true;
    if(element < total)
    {
        for(int i = element; i < element + NUMNODES-1; i++)
        {
            int current = edges[i];
            for(int j = i + 1; j < element + NUMNODES-1; j++)
            {
                if(current == edges[j])
                {
                    graceful = false;
                    break;
                }
            }
            if(!graceful) break;
        }
        if(graceful)
            graceful_labels[element / (NUMNODES-1)] = element/(NUMNODES-1)*NUMNODES;
        if(!graceful)
            graceful_labels[element / (NUMNODES-1)] = -1;
    }
}

void execute_gpu(int perms[], int children[], int stops[], int graceful_labels[], int edges[], int NUMNODES, int NUMPERMS)
{
    int *d_perms, *d_children, *d_graceful_labels, *d_stops, *d_edges;

    // define sizes for convenience
    const size_t perm_size = NUMNODES*NUMPERMS*sizeof(int);
    const size_t edge_size = (NUMNODES-1)*NUMPERMS*sizeof(int);
    const size_t child_size = (NUMNODES-1)*sizeof(int);
    const size_t stop_size = NUMNODES*sizeof(int);
    const size_t label_size = NUMPERMS*sizeof(int);

    // 768 cores available on my home computer
    // 1024 cores available on starship
    int numCores = (NUMNODES * NUMPERMS)/ 1024 + 1;
    int numThreads = 1024;

    // Allocate memory on GPU
    cudaMalloc(&d_perms, perm_size);
    cudaMalloc(&d_edges, edge_size);
    cudaMalloc(&d_children, child_size);
    cudaMalloc(&d_stops, stop_size);
    cudaMalloc(&d_graceful_labels, label_size);

    // Copy over necessary arrays to GPU
    cudaMemcpy(d_perms, perms, perm_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_children, children, child_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stops, stops, stop_size, cudaMemcpyHostToDevice);

    // Calculate edge labelings for each permutation
    calculate_edges<<<numCores, numThreads>>>(d_perms,
            d_children,
            d_stops,
            d_edges,
            NUMNODES,
            NUMPERMS);

    // Don't need these for the next step, so just free the memory up.
    cudaFree(&d_perms);
    cudaFree(&d_stops);
    cudaFree(&d_children);

    // For debugging  purposes only
//    cudaMemcpy(edges, d_edges, edge_size, cudaMemcpyDeviceToHost);

    // Now check the gracefulness of the given edge labelings.
    check_gracefulness<<<numCores, numThreads>>>(d_edges, d_graceful_labels, NUMNODES, NUMPERMS);

    // Copy back the evaluated labelings
    cudaMemcpy(graceful_labels, d_graceful_labels, label_size, cudaMemcpyDeviceToHost);

    // Free up the rest of the memory
    cudaFree(&d_graceful_labels);
    cudaFree(&d_edges);
}

int main()
{
   //const int NUMNODES = 3;
   //const int NUMPERMS = factorial(NUMNODES);
   //int stops [] = {1, -1, -1};
  // const int NUMNODES = 9;
  // const int NUMPERMS = 100500;
  // int stops [] = {2,5,7,-1,-1,-1,-1,-1,-1};
   //const int NUMNODES = 8;
   //const int NUMPERMS = 10000*NUMNODES;
   //int stops [] = {1,3,5,-1,-1,-1,6,-1};
   //const int NUMNODES = 11;
   //const int NUMPERMS = 1000*NUMNODES;
   //int stops [] = {2,-1,5,-1, -1,6,-1,7,9,-1,-1};
   const int NUMNODES = 12;
   const int NUMPERMS = 1000*NUMNODES;
   int stops [] = {1, 4, 6, -1, -1, -1, 9, -1, -1, 10, -1};
   int found = 0;
   bool has_next = false;
   bool has_started = false;

    int children[NUMNODES-1], labels[NUMNODES];
    float iter = 0;
    // generate both children and label array
    for(int i = 0; i < NUMNODES; i++)
    {
        labels[i] = i;
        if(i < NUMNODES - 1) children[i] = i+1;
    }
do{
    int edges[NUMPERMS*(NUMNODES-1)], perms[NUMPERMS*NUMNODES], graceful_labels[NUMPERMS];
    // create all permutations of given nodes
    for(int i = 0; i < NUMPERMS; i++)
    {
        for(int j = 0; j < NUMNODES; j++)
        {
            perms[i*NUMNODES+j] = labels[j];
            //edges[i*NUMNODES+j] = 0;
        }
        graceful_labels[i] = -1;
        has_next = next_permutation(labels, labels+NUMNODES);
if(!has_next) break;
    }
    if(!has_started)
    {
    	has_started = true;
	init(NUMNODES);
    }
    execute_gpu(perms, children, stops, graceful_labels, edges, NUMNODES, NUMPERMS);
    for(int i = 0; i < NUMPERMS; i++)
    {
        if(graceful_labels[i] != -1)
	{
		for(int j = 0; j < NUMNODES; j++)
		cout << perms[graceful_labels[i] + j] << " ";
		cout << endl;
		    found=1;
		break;
	}
	
    }

iter++;
}while(has_next && found != 1);
 finish(NUMNODES);
    cout << "Found " << found << " graceful labelings." << endl;
    cout << "Took " << iter << " iterations" << endl;
    return 0;
}
