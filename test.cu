#include <iostream>


__global__ void sqrt(int n)
{
    int test[n];
    for(int i = 0; i < n; i++)
        test[i] = i * i;
}

void gpu(int N)
{
    int numThreads = 1024;
    int numCores = N/768 + 1;
    int* gpu;
   cudaMalloc(&gpu, N*sizeof(float)); // Allocate enough memory on the GPU
   sqrt<<numCores, numThreads>>(N);
   cudaFree(&gpu);
}

int main()
{
    gpu(50);
    return 0;
}
