#include "files.h"
// Apertura del archivo para escritura
std::fstream open_file(std::string filename)
{
    std::fstream file;
    // Apertura del archivo
    file.open(filename, std::ios::out);
    // Verificacion del archivo
    if (!file)
    {
        std::cout << "Documento no creado";
        exit(1);
    }
    return file;
}
// Apertura del archivo para la lectura
std::fstream read_file(std::string filename)
{
    std::fstream file;
    // Apertura del archivo
    file.open(filename, std::ios::in);
    // Verifiacion del archivo
    if (!file)
    {
        std::cout << "Documento no localizado";
        exit(1);
    }
    return file;
}