#include "nodes.h"
// Ingresa la informacion del canal
void node::set_channel(int n_channels)
{
    // Obtiene numeros entre 1 y n_channels
    channel = 1 + (rand() % n_channels);
}
// Regresa el canal del nodo
int node::get_channel()
{
    return channel;
}
// Destructor
node::~node() {}
// Constuctor
nodes_class::nodes_class(int channels)
{
    for (int i = 0; i < n_nodes; i++)
    {
        // Creacion de los n nodos
        nodes[i].set_channel(channels);
    }
}
// Regresa el nodo i
node nodes_class::get_node_i(int i)
{
    return nodes[i];
}
nodes_class::~nodes_class() {}