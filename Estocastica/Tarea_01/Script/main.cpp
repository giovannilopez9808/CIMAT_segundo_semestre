#include "Modules/data.h"
#include "Modules/nodes.h"
#include "Modules/files.h"
int main(int argc, char* argv[])
{
    if(argc==5)
    {
    std::string filename_data = argv[1];
    std::string filename_output = argv[2];
    unsigned seed =atoi(argv[3]);
    int channels = atoi(argv[4]);
    long int cost, min;
    int iterations = 2;
    std::fstream file_output = open_file(filename_output);
    lines_class lines(filename_data);
    min = 100000000000000000;
    for (int i = 0; i < iterations; i++)
    {
        seed = rand();
        cost = lines.cost(seed,
                          channels);
        if (min > cost)
        {
            min = cost;
        }
    }
    file_output << min << std::endl;
    }else
    {
        std::cout << "Faltan argumentos" << std::endl;
    }
    return 0;
}
