#include "lines.h"
// Guardado de la informacion
void line::set_information(int nodei,
                           int nodej,
                           int dij,
                           int inter)
{
    // Nodo i
    node_i = nodei;
    // Nodo i
    node_j = nodej;
    // Distancia
    d_ij = dij;
    // Interferencia
    interference = inter;
}
//  Regresa el nodo i
int line::get_node_i()
{
    return node_i;
}
//  Regresa el nodo j
int line::get_node_j()
{
    return node_j;
}
//  Regresa la distancia minima
int line::get_dij()
{
    return d_ij;
}
// Regresa la interferencia
int line::get_interference()
{
    return interference;
}
// Constructor
line::line() {}
// Destructor
line::~line() {}
// Constructor de la clase
lines_class::lines_class(std::string filename)
{
    // Inicializacion de las variables para leer la linea del archivo de datos
    int nodei, nodej, dij, inter;
    char R, sign;
    // Apertura del archivo de datos en modo de lectura
    std::fstream file = read_file(filename);
    for (int i = 0; i < n_lines; i++)
    {
        // Lectura de los datos del archivo
        file >> nodei >> nodej >> R >> sign >> dij >> inter;
        // Guardado de los datos en la clase
        lines[i].set_information(nodei,
                                 nodej,
                                 dij,
                                 inter);
    }
    file.close();
}
// Obtiene la linea i del archivo de datos
line lines_class::get_line_i(int i)
{
    return lines[i];
}
// Calculo del costo de la solucion
long int lines_class::cost(int channels)
{
    // Inicalizacion del costo
    long int result = 0;
    // Variables auxiliares
    int c_i, c_j, node_i, node_j, d_ij, interference;
    // Inicalizacion de la linea i
    line line_i;
    // Inicializacion de los nodos con canales aleatorios
    nodes_class nodes(channels);
    for (int i = 0; i < n_lines; i++)
    {
        // Obtiene la informacion de cada linea
        node_i = lines[i].get_node_i();
        node_j = lines[i].get_node_j();
        d_ij = lines[i].get_dij();
        interference = lines[i].get_interference();
        // Obtiene los canales de cada nodo
        c_i = nodes.get_node_i(node_i).get_channel();
        c_j = nodes.get_node_i(node_j).get_channel();
        // Si la diferencia entre nodos es menor a la distancia minima se suma la interferencia
        if (abs(c_i - c_j) <= d_ij)
        {
            result += interference;
        }
    }
    return result;
}
lines_class::~lines_class() {}