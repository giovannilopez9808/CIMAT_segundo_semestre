#include "Modules/lines.h"
#include "Modules/files.h"
int main(int argc, char *argv[])
{
    if (argc != 5)
    {

        std::cout << "------------------------" << std::endl;
        std::cout << "    Faltan argumentos" << std::endl;
        std::cout << "------------------------" << std::endl;
        return 1;
    }
    std::string filename_data = argv[1];
    std::string filename_output = argv[2];
    unsigned seed = atoi(argv[3]);
    srand(seed);
    int channels = atoi(argv[4]);
    long int cost, min;
    int iterations = 100000;
    std::fstream file_output = open_file(filename_output);
    lines_class lines(filename_data);
    for (int j = 0; j < 100; j++)
    {
        min = 100000000000000000;
        std::cout << j + 1 << " de " << 100 << std::endl;
        for (int i = 0; i < iterations; i++)
        {
            cost = lines.cost(channels);
            if (min > cost)
            {
                min = cost;
            }
        }
        std::cout << min << std::endl;
        file_output << min << std::endl;
    }
    return 0;
}
