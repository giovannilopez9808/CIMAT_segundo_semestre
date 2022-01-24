#ifndef read_data_H
#define read_data_H
// Numero de lineas en el archivo
#define n_lines 14525
#include "files.h"
#include "nodes.h"
// Informacion de cada linea del archivo de datos
class line
{
private:
    // Nodo i
    int node_i;
    // Nodo j
    int node_j;
    // Distancia
    int d_ij;
    // Interferencia
    int interference;

public:
    // Guardado de la informacion
    void set_information(int,
                         int,
                         int,
                         int);
    //  Regresa el nodo i
    int get_node_i();
    //  Regresa el nodo i
    int get_node_j();
    //  Regresa la distancia minima
    int get_dij();
    // Regresa la interferencia
    int get_interference();
    // Constructor
    line();
    // Destructor
    ~line();
};
// Informacion de todas las lineas del archivo de datos
class lines_class
{
private:
    // Array con cada linea del archivo de datos
    line lines[n_lines];

public:
    // Constructor de la clase
    lines_class(std::string);
    // Calculo del costo de la solucion
    long int cost(int);
    // Obtiene la linea i del archivo de datos
    line get_line_i(int);
    // Destructor
    ~lines_class();
};
#endif