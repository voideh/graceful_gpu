#include<iostream>
#include <algorithm>
using namespace std;

int factorial(int n)
{
  return (n == 0 || n == 1) ? 1 : factorial(n-1) * n;
}

__global__ void gpu_check_edges()
{
    /*
       do some real fancy stuff here 
     */
}


void prepare_gpu(int NUMPERMS, int NUMNODES)
{
    int* results;
    int numCores = NUMPERMS / 768 + 1;
    int numThreads = 1024;
    cudaMalloc(&results, (sizeof(int)*NUMPERMS*2 + sizeof(int)*NUMNODES*2));
    cudaFree(&results);
    //gpu_check_edges<<numCores, numThreads>>();
}

int main()
{
    int NUMNODES = 4;
    int NUMPERMS = factorial(NUMNODES);
    int stop[] = {2, -1, 4, -1, 5, -1, -1};
    int children[NUMNODES-1];
    int labels[NUMNODES];
    for(int i = 0; i < NUMNODES; i++)
        labels[i] = i;
    for(int i = 0; i < NUMNODES-1; i++)
        children[i] = i+1;
    // create all permutations for the given labels
    int permutations[NUMPERMS*NUMNODES];
    for(int i = 0; i < NUMPERMS*NUMNODES; i++)
        permutations[i] = 0;
    for(int i = 0; i < NUMPERMS; i++)
    {
        for(int j = 0; j < NUMNODES; j++)
        {
            permutations[NUMNODES*i+j] = labels[j];
        }
        next_permutation(labels, labels+NUMNODES);
    }
    // check that all permutations are available
    for(int i = 0; i < NUMPERMS*NUMNODES; i+=4)
    {
        cout << permutations[i] << permutations[i+1] << permutations[i+2] << endl;
    }
    prepare_gpu(NUMPERMS, NUMNODES);
}
