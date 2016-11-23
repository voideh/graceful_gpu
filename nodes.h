#include <stdlib.h>
#include <iostream>
struct Node{
    int num_children;
    int tree_size;
    int graceful_label;
    int arb_label;
    Node **children;

    Node(int d, int max_children)
    {
      arb_label = d;
      children = (Node**)malloc(sizeof(Node) * max_children);
      tree_size = max_children + 1;
      num_children = 0;
    }

    void add_child(Node *c)
    {
      std::cout << "child node " << c->arb_label << " to parent node " << this->arb_label << std::endl;
      children[num_children++] = c;
    }
};
