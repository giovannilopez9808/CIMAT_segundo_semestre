#include "data.h"
void line::set_information(int nodei,
                           int nodej,
                           int dij,
                           int inter)
{
    node_i = nodei;
    node_j = nodej;
    d_ij = dij;
    interference = inter;
}
int line::get_node_i()
{
    return node_i;
}
int line::get_node_j()
{
    return node_j;
}
int line::get_dij()
{
    return d_ij;
}
int line::get_interference()
{
    return interference;
}
line::line()
{
}
line::~line()
{
}

lines_class::lines_class(std::string filename)
{
    int nodei, nodej, dij, inter;
    char R, sign;
    std::fstream file = read_file(filename);
    for (int i = 0; i < n_lines; i++)
    {
        file >> nodei >> nodej >> R >> sign >> dij >> inter;
        lines[i].set_information(nodei,
                                 nodej,
                                 dij,
                                 inter);
    }
}
line lines_class::get_line_i(int i)
{
    return lines[i];
}
lines_class::~lines_class()
{
}