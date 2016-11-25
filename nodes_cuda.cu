#include<iostream>
#include <algorithm>
using namespace std;
int NUMNODES = 3;
int NUMPERMS = 6;

__global__ void gpu_check_edges(bool isgraceful, int labels[], int children[], int stop[], int NUMNODES)
{
    /*
       do some real fancy stuff here 
       */
}

int main()
{
    int stop[] = {1, -1, -1};
    int children[NUMNODES-1];
    int labels[NUMNODES];
    for(int i = 0; i < NUMNODES; i++)
        labels[i] = i;
    for(int i = 0; i < NUMNODES-1; i++)
        children[i] = i+1;
    int permutations[NUMPERMS][NUMNODES];
    for(int i = 0; i < NUMPERMS; i++)
    {
        for(int j = 0; j < NUMNODES; j++)
            permutations[i][j] = labels[j];
        next_permutation(labels, labels+NUMNODES);
    }
    for(int i = 0; i < NUMPERMS; i++)
    {
        for(int j = 0; j < NUMNODES; j++)
        {
            cout << permutations[i][j] << " ";
        }
        cout << endl;
    }
}
