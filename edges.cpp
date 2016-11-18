#include <iostream>
#include "nodes.h"
using namespace std;
int NODES = 7;
int EDGES = NODES -1 ;
int DIFF_COUNT = 0;

void calculate_difference(Node *root, int edges[EDGES])
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
  Node *root;
  Node *child;
  root = new Node(2);
  child = new Node(2);
  root->label = 0;
  child->label = 1;
  root->add_child(child);

  Node *child2 = new Node(4);
  child2->label = 2;
  Node *child3 = new Node(1);
  child3->label = 3;
  Node *child4 = new Node(1);
  child4->label = 4;
  Node *child5 = new Node(1);
  child5->label = 5;
  Node *child6 = new Node(1);
  child6->label = 6;

  child2->add_child(child4);
  child2->add_child(child5);
  child2->add_child(child6);

  child->add_child(child2);
  root->add_child(child3);

  calculate_difference(root, edges);
  for(int i = 0; i < EDGES; i++)
    cout << edges[i] << endl;

  return 0xc0ff3;
}
