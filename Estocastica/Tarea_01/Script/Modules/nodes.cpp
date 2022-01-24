#include "nodes.h"

void node::set_channel(int n_channels)
{
    channel = 1 + (rand() % n_channels);
}
int node::get_channel()
{
    return channel;
}
node::~node()
{
}

nodes_class::nodes_class(int channels)
{
    for (int i = 0; i < n_nodes; i++)
    {
        nodes[i].set_channel(channels);
    }
}
node nodes_class::get_node_i(int i)
{
    return nodes[i];
}
nodes_class::~nodes_class()
{
}