#include<iostream>
#include <algorithm>
using namespace std;
int NUMNODES = 12;
unsigned long int NUMPERMS = factorial(NUMNODES);

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


void prepare_gpu()
{
}

int main()
{
    int stop[] = {1, -1, -1};
    int children[NUMNODES-1];
    int labels[NUMNODES];
    for(int i = 0; i < NUMNODES-1; i++)
        children[i] = i+1;
    // create all permutations for the given labels
    int permutations[NUMPERMS][NUMNODES];
    for(int i = 0; i < NUMPERMS; i++)
    {
        for(int j = 0; j < NUMNODES; j++)
            permutations[i][j] = labels[j];
        next_permutation(labels, labels+NUMNODES);
    }
    // check that all permutations are available
    for(int i = 0; i < NUMPERMS; i++)
    {
        for(int j = 0; j < NUMNODES; j++)
        {
            cout << permutations[i][j] << " ";
        }
        cout << endl;
    }
}
