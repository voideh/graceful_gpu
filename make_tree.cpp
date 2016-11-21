#include <iostream>
#include <fstream>
#include <string>
#include "nodes.h"
#include <sstream>
#include <algorithm>
#include <iterator>
using namespace std;
__global__()
{
}
int main()
{
    string treeline;
    ifstream trees;
    trees.open("trees");
    if(trees.is_open())
    {
        while(getline(trees, treeline) )
        {
            int numnodes;
            int next; 
            // make the node pairs to iterate through;
            vector<string> tokens;
            istringstream iss(treeline);
            copy(istream_iterator<string>(iss),
                    istream_iterator<string>(),
                    back_inserter(tokens));
            for()



        }
        trees.close();
        return 0;
    }
    else
    {
        cout << "no file" << endl;
        return 1;
    }
}
