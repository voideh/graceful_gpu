#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <array>

const int NUMNODES = 8;
const int EDGES = NUMNODES - 1;

bool check_edges(int children[], int labels[], int stop[])
{
    std::array<int, EDGES> edges;
    edges.fill(0);
    int last_index = 0;
    int edgecounter = 0;
    for(int i = 0; i < NUMNODES; i++)
    {
        if(stop[i] == -1) {}
        else
        {
            for(int j = ((last_index == 0) ? 0 : last_index+1); j <= stop[i]; j++)
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
            last_index = stop[i];
        }
    }
    return true;
}

int main()
{
    bool isgraceful = false;

    int children[NUMNODES-1];
    int stop [] = {1, 3, 4, -1, 6, -1, -1, -1};
    int labels[NUMNODES];
    for(int i = 0; i < NUMNODES; i++)
        labels[i] = i;
    for(int i = 0; i < NUMNODES-1; i++)
        children[i] = i+1;
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
