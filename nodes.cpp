#include <iostream>
#include <algorithm> #include <stdlib.h>
#include <array>
const int NUMNODES = 8;
const int EDGES = NUMNODES - 1;

bool check_edges(int children[], int labels[], int stop[])
{
    // make temp array to check whether the current permutation of labels is graceful or not
    std::array<int, EDGES> edges;
    // just to make sure that the garbage values in the array dont mess up our check. there should never be a difference between nodes that 
    // equals 0
    edges.fill(0);
    // to keep track of where we left off in the stop array
    int last_index = 0;
    // to keep track of where to place the next edge label
    int edgecounter = 0;
    for(int i = 0; i < NUMNODES; i++)
    {
        // check for sentinel value 
        if(stop[i] == -1) {}
        else
        {
            // if the last index is 0 we start at 0, otherwise we will start at the index after the previous index
            for(int j = ((last_index == 0) ? 0 : last_index+1); j <= stop[i]; j++)
            {
                int diff = abs(labels[i] - labels[children[j]]);
                // simply try to find the calculated difference of labells in the temp edge array
                bool contains = std::find(edges.begin(), edges.end(), diff) != edges.end();
                if(contains)
                {
                    // theres a duplicate therefore it is not graceful
                    return false;
                }
                else
                {
                    edges[edgecounter++] = diff;
                }
            }
            // update where we stopped at 
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
    // this will check all possible permutations of an array of numbers for a graceful configuration.
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
