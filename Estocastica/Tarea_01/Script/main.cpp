#include "Modules/data.h"
#include "Modules/nodes.h"
#include "Modules/files.h"
int main()
{
    unsigned seed = 12345625;
    int channels = 34;
    long int cost, min;
    int iterations = 100;
    std::string filename_output = "../Output/test.txt";
    std::string filename_data = "../Data/GSM2-272.ctr";
    std::fstream file_output = open_file(filename_output);
    lines_class lines(filename_data);
    min = 100000000000000000;
    for (int i = 0; i < iterations; i++)
    {
        std::cout << i << " de " << iterations << std::endl;
        seed = rand();
        cost = lines.cost(seed,
                          channels);
        if (min > cost)
        {
            min = cost;
        }
    }
    file_output << min << std::endl;
    return 0;
}