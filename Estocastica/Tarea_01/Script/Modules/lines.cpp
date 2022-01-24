#include "lines.h"
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
    file.close();
}
line lines_class::get_line_i(int i)
{
    return lines[i];
}
long int lines_class::cost(int channels)
{
    long int result = 0;
    int c_i, c_j, node_i, node_j, d_ij, interference;
    line line_i;
    nodes_class nodes(channels);
    for (int i = 0; i < n_lines; i++)
    {
        node_i = lines[i].get_node_i();
        node_j = lines[i].get_node_j();
        d_ij = lines[i].get_dij();
        interference = lines[i].get_interference();
        c_i = nodes.get_node_i(node_i).get_channel();
        c_j = nodes.get_node_i(node_j).get_channel();
        if (abs(c_i - c_j) <= d_ij)
        {
            result += interference;
        }
    }
    return result;
}
lines_class::~lines_class()
{
}
