#ifndef nodes_H
#define nodes_H
// Numero total de nodos
#define n_nodes 272
#include <memory>
// Informacion de cada nodo, este contiene el canal aleatorio
class node
{
private:
    // Canal aleatorio
    int channel;

public:
    // Ingresa la informacion del canal
    void set_channel(int);
    // Regresa el canal del nodo
    int get_channel();
    ~node();
};
// Informacion de todos los nodos y sus canales aleatorios
class nodes_class
{
private:
    // Lista de nodos
    node nodes[n_nodes];

public:
    // Constuctor
    nodes_class(int);
    // Regresa el nodo i
    node get_node_i(int);
    // Destructor
    ~nodes_class();
};
#endif