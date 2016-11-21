#include <iostream>
#include "nodes.h"
using namespace std;
int NODES = 7;
const int EDGES = NODES-1;
int DIFF_COUNT = 0;

void calculate_difference(Node *root, int edges[])
{
  if(root->num_children > 0)
  {
    for(int i = 0; i < root->num_children; i++)
    {
      edges[DIFF_COUNT++] = abs(root->label - root->children[i]->label);
      calculate_difference(root->children[i], edges);
    }
  }
}

int main()
{
  int edges[EDGES];
  return 0xc0ff3;
}
