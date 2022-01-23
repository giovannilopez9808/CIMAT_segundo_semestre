#include "Modules/lines.h"
#include "Modules/files.h"
int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        std::cout << "Faltan argumentos" << std::endl;
        return 1;
    }
    std::string filename_data = argv[1];
    std::string filename_output = argv[2];
    unsigned seed = atoi(argv[3]);
    int channels = atoi(argv[4]);
    long int cost, min;
    int iterations = 1000;
    std::fstream file_output = open_file(filename_output);
    lines_class lines(filename_data);
    for (int j = 0; j < 100; j++)
    {
        min = 100000000000000000;
        std::cout << j + 1 << " de " << 100 << std::endl;
        for (int i = 0; i < iterations; i++)
        {
            seed = rand();
            srand(seed);
            cost = lines.cost(seed,
                              channels);
            if (min > cost)
            {
                min = cost;
            }
        }
        file_output << min << std::endl;
    }
    return 0;
}
