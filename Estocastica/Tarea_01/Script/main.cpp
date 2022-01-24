/*
Este programa realiza el calculo del costo minimo dado
un archivo de nodos con la informacion de la conexion entre
dos nodos, distancia minima y la interferencia que produce

Argumentos:
    argv[1] Direccion absoluta del fichero de datos
    argv[2] Direccion absoluta del fichero de resultados
    argv[3] Semila
    argv[4] Numero de canales
 */
#include "Modules/lines.h"
#include "Modules/files.h"
#include <climits>
int main(int argc, char *argv[])
{
    if (argc != 5)
    {

        std::cout << "------------------------" << std::endl;
        std::cout << "    Faltan argumentos" << std::endl;
        std::cout << "------------------------" << std::endl;
        return 1;
    }
    // Direccion absoluta del archivo de datos
    std::string filename_data = argv[1];
    // Direccion absoluta del archivo de salida
    std::string filename_output = argv[2];
    // Semilla
    unsigned seed = atoi(argv[3]);
    srand(seed);
    // Numero de canales
    int channels = atoi(argv[4]);
    long int cost, min;
    // Numero de iteraciones para encontrar un minimo
    int iterations = 100000;
    // Apertura del archivo de salida
    std::fstream file_output = open_file(filename_output);
    // Carga de las lineas de conexion
    lines_class lines(filename_data);
    for (int j = 0; j < 100; j++)
    {
        // Inicializacion del minimo
        min = LONG_MAX;
        for (int i = 0; i < iterations; i++)
        {
            // Calculo del costo de la solucion
            cost = lines.cost(channels);
            // Si el costo es menor al minimo se guarda el valor
            if (min > cost)
            {
                min = cost;
            }
        }
        // Guardado del minimo
        file_output << min << std::endl;
    }
    // Cierre del archivo de salida
    file_output.close();
    return 0;
}
