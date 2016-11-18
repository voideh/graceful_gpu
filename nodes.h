#include <stdlib.h>
class Node{
  public:
    int num_children;
    int degree;
    int label;
    Node **children;

    Node(int d)
    {
      degree = d;
      children = (Node**)malloc(sizeof(Node) * degree - 1);
      num_children = 0;
    }

    void add_child(Node *c)
    {
      children[num_children++] = c;
      degree++;
    }
};
