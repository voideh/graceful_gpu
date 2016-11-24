#include <iostream>
#include <fstream>
#include <string>
#include "nodes.h"
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
using namespace std;
Node** make_tree(string treeline)
{
    Node** tree;
    int numnodes;
    int next; 
    // make the node pairs to iterate through;
    vector<string> pairs;
    istringstream iss(treeline);
    copy(istream_iterator<string>(iss),
            istream_iterator<string>(),
            back_inserter(pairs));
    // iterating through the node pairs and creating nodes
    numnodes = pairs.size() + 1;
    tree = new Node*[numnodes];
    for(int i = 0; i < numnodes; i++)
        tree[i] = new Node(i, numnodes-1);

    // create the realtionships between each node
    for(string pair : pairs)
    {
        string::size_type sz;
        // only pull out the numbers from string
        int parent = stoi(pair.substr(0,1), &sz);
        int child = stoi(pair.substr(2,2), &sz);

        cout << "Parent Node # : " << parent << endl;
        cout << "Child Node # : " << child << endl;
        tree[parent]->add_child(tree[child]);
        cout << "--------------------" << endl;
    }
        return tree;
    }

int main()
{
    vector<Node**> trees;
    string treeline;
    ifstream trees_file;
    trees_file.open("trees");
    int count = 0;
    if(trees_file.is_open())
    {
        while(getline(trees_file, treeline))
        {
            trees.push_back(make_tree(treeline));
        }
        for(Node** tree : trees)
        {

            cout << " Begin tree " << ++count << endl;
            for(int i = 0; i < tree[0]->tree_size; i++)
                cout << tree[i]->arb_label << endl;
        }
        trees_file.close();
        return 0;
    }
    else
    {
        cout << "No file foud.. " << endl;
        return 1;
    }
}
