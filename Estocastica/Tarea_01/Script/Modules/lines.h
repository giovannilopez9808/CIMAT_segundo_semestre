#ifndef read_data_H
#define read_data_H
#define n_lines 14525
#include "files.h"
#include "nodes.h"
class line
{
private:
    int node_i;
    int node_j;
    int d_ij;
    int interference;

public:
    void set_information(int,
                         int,
                         int,
                         int);
    int get_node_i();
    int get_node_j();
    int get_dij();
    int get_interference();
    line();
    ~line();
};
class lines_class
{
private:
    line lines[n_lines];

public:
    lines_class(std::string);
    long int cost(int);
    line get_line_i(int);
    ~lines_class();
};
#endif