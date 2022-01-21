#include "Modules/data.h"
#include "Modules/nodes.h"
#include "Modules/files.h"
int main()
{
    unsigned seed = 123456;
    int channels = 10;
    // std::string filename_output = "../Output/test.txt";
    std::string filename_data = "../Data/GSM2-272.ctr";
    // std::fstream file_output = open_file(filename_output);
    lines_class lines(filename_data);
    nodes_class nodes(seed,
                      channels);
    // file_output.close();
    return 0;
}