#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <array>

const int NUMNODES = 5;
const int EDGES = NUMNODES - 1;

bool check_edges(int children[], int labels[], int stop[])
{
    std::array<int, EDGES> edges;
    edges.fill(0);
    int edgecounter = 0;
    for(int i = 0; i < NUMNODES; i++)
    {
        if(stop[i] == -1) {}
        else
        {
            for(int j = stop[i] - 1; j <= stop[i]; j++)
            {
                int diff = abs(labels[i] - labels[children[j]]);
                bool contains = std::find(edges.begin(), edges.end(), diff) != edges.end();
                if(contains)
                {
                    return false;
                }
                else
                {
                    edges[edgecounter++] = diff;
                }
            }
        }
    }
    return true;
}

int main()
{
    bool isgraceful = false;
    int children [] = {1, 2, 3, 4};
    int stop [] = {1, -1, 3, -1, -1};
    int labels[NUMNODES];
    for(int i = 0; i < NUMNODES; i++)
        labels[i] = i;
    do
    {
        isgraceful = check_edges(children, labels, stop);
        if(!isgraceful)
            std::next_permutation(labels, labels+NUMNODES);
    }while(!isgraceful);
    if(isgraceful)
    {
        std::cout << "Graceful labeling found. " << std::endl;
        for(int e : labels)
            std::cout << e << " ";
        std::cout << std::endl;
    }
    return 0;
}
