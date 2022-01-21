#include "files.h"
std::fstream open_file(std::string filename)
{
    std::fstream file;
    file.open(filename, std::ios::out);
    if (!file)
    {
        std::cout << "Documento no creado";
        exit(1);
    }
    return file;
}
std::fstream read_file(std::string filename)
{
    std::fstream file;
    file.open(filename, std::ios::in);
    return file;
}