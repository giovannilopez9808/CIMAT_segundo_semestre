#ifndef nodes_H
#define nodes_H
#define n_nodes 272
#include <iostream>
#include <cstdlib>
#include <memory>
class node
{
private:
    int channel;

public:
    void set_channel(int);
    int get_channel();
    ~node();
};
class nodes_class
{
private:
    node nodes[n_nodes];

public:
    nodes_class(unsigned,
                int);
    ~nodes_class();
};
#endif