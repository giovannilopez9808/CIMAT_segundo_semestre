#include "Modules/data.h"
#include "Modules/nodes.h"
#include "Modules/files.h"
int main()
{
    unsigned seed = 12345625;
    int channels = 10;
    // std::string filename_output = "../Output/test.txt";
    std::string filename_data = "../Data/GSM2-272.ctr";
    // std::fstream file_output = open_file(filename_output);
    lines_class lines(filename_data);
    for (int i = 0; i < 1000; i++)
    {
        seed = rand();
        long int cost = lines.cost(seed,
                                   channels);
        std::cout << cost << std::endl;
        // file_output.close();
    }
    return 0;
}