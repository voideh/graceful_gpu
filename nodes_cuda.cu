#include<iostream>
#include <algorithm>
using namespace std;
const int NUMNODES = 8;

int factorial(int n)
{
  return (n == 0 || n == 1) ? 1 : factorial(n-1) * n;
}
const int NUMPERMS = factorial(NUMNODES);

__global__ void gpu_check_edges(int *c, int *l, int * stop, int* store)
{
}

void prepare_gpu(int NUMPERMS, int NUMNODES,  int c[], int p[NUMPERMS][NUMNODES] , int results[], int s[])
{
    int* graceful_labels; 
    int* children;
    int* permutations;
    int* stop;
    size_t pitch;
    int numCores = NUMPERMS / 768 + 1;
    int numThreads = 1024;
    // allocate mem
    cudaMalloc(&graceful_labels, sizeof(int) * NUMPERMS);
    cudaMalloc(&children, sizeof(int) * NUMNODES-1);
    cudaMallocPitch((void**)&permutations, &pitch, NUMNODES * sizeof(int), NUMPERMS);
    cudaMalloc(&stop, sizeof(int) * NUMNODES);
    // copy over stuff
    cudaMemcpy(children, c, sizeof(int) * NUMNODES-1, cudaMemcpyHostToDevice);
    cudaMemcpy2D(permutations, pitch, p, pitch, NUMNODES * sizeof(int), NUMPERMS, cudaMemcpyHostToDevice);
    cudaMemcpy(stop, s, sizeof(int) * NUMNODES, cudaMemcpyHostToDevice);
    // run code
    gpu_check_edges<<numCores, numThreads>>(children, permutations, stop, graceful_labels);
    cudaMemcpy(results, graceful_labels, sizeof(int)*NUMPERMS, cudaMemcpyDeviceToHost);
    cudaFree(&graceful_labels);
    cudaFree(permutations);
    cudaFree(&stop);
    cudaFree(&children);
}

int main()
{
    int NUMNODES = 8;
    int NUMPERMS = factorial(NUMNODES);
    int children[NUMNODES-1];
    int labels[NUMNODES];
    int results[NUMPERMS];
    int stop[] = {1, 3, 4, -1, 6, -1 ,-1 ,-1};
    int permutations[NUMPERMS][NUMNODES];
    for(int i = 0; i < NUMNODES; i++)
        labels[i] = i;
    for(int i = 0; i < NUMNODES; i++)
        children[i] = i+1;
    for(int i = 0; i < NUMPERMS; i++)
    {
        for(int j = 0; j < NUMNODES; j++)
            permutations[i][j] = labels[j];
        next_permutation(labels, labels+NUMNODES);
    }
    return 0;
}
