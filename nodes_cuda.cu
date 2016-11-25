#include <stdlib.h>
#include <iostream>
#include <algorithm>
using namespace std;

int factorial(int n)
{
  return (n == 0 || n == 1) ? 1 : factorial(n-1) * n;
}

__global__ void calculate_edges(int *perms, int *children, int *stops, int *graceful, int *temp, int NUMNODES, int NUMPERMS)
{
    // will start at begining of node permutation
    int element = (blockIdx.x * blockDim.x + threadIdx.x) * NUMNODES; 
    int total = NUMNODES * NUMPERMS;
    int temp_counter = 0;
    int last_index = 0;
    if(element < total)
    {
        for(int i = element; i < element + NUMNODES; i++)
        {
            if(stops[i % NUMNODES] != -1)
            {
                for(int j = (last_index == 0) ? 0 : last_index+1; j<=stops[i % NUMNODES]; j++)
                    temp[element + temp_counter++] = abs(perms[i] - perms[children[j] + element]);
                last_index = stops[i % NUMNODES];
            }
        }
    }
}

void execute_gpu(int perms[], int children[], int stops[], int graceful_labels[], int temp[], int NUMNODES, int NUMPERMS)
{
    int *d_perms, *d_children, *d_graceful_labels, *d_stops, *temp_calc;

    const size_t perm_size = NUMNODES*NUMPERMS*sizeof(int);
    const size_t child_size = (NUMNODES-1)*sizeof(int);
    const size_t stop_size = NUMNODES*sizeof(int);
    const size_t label_size = NUMPERMS*sizeof(int);

    int numCores = (NUMNODES * NUMPERMS)/ 768 + 1;
    int numThreads = 1024;

    cudaMalloc(&d_perms, perm_size);
    cudaMalloc(&temp_calc, perm_size);
    cudaMalloc(&d_children, child_size);
    cudaMalloc(&d_stops, stop_size);
    cudaMalloc(&d_graceful_labels, label_size);

    cudaMemcpy(d_perms, perms, perm_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_children, children, child_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_stops, stops, stop_size, cudaMemcpyHostToDevice);

    calculate_edges<<<numCores, numThreads>>>(d_perms,
            d_children,
            d_stops,
            d_graceful_labels,
            temp_calc,
            NUMNODES,
            NUMPERMS);
    cudaMemcpy(temp, temp_calc, perm_size, cudaMemcpyDeviceToHost);

    cudaFree(&d_perms);
    cudaFree(&d_stops);
    cudaFree(&d_children);
    cudaFree(&d_graceful_labels);
    cudaFree(&temp_calc);
}

int main()
{
    const int NUMNODES = 5;
    const int NUMPERMS = factorial(NUMNODES);
    int children[NUMNODES-1], perms[NUMPERMS*NUMNODES], graceful_labels[NUMPERMS], labels[NUMNODES];
    int temp[NUMPERMS*NUMNODES-1];
    int stops[] = {1, 3, -1, -1, -1};
    // generate both children and label array
    for(int i = 0; i < NUMNODES; i++)
    {
        labels[i] = i;
        if(i < NUMNODES - 1) children[i] = i+1;
    }

    // create all permutations of given nodes
    for(int i = 0; i < NUMPERMS; i++)
    {
        for(int j = 0; j < NUMNODES; j++)
        {
            perms[i*NUMNODES+j] = labels[j];
            temp[i*NUMNODES+j] = 0;
        }
        graceful_labels[i] = -1;
        next_permutation(labels, labels+NUMNODES);
    }
    execute_gpu(perms, children, stops, graceful_labels, temp, NUMNODES, NUMPERMS);

    for(int i = 0; i < NUMPERMS; i++)
    {
        for(int j = 0; j < NUMNODES-1; j++)
            cout << temp[i*NUMNODES+j] << " ";
        cout << endl;
    }
}
