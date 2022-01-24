#ifndef files_H
#define files_H
#include <fstream>
#include <iostream>
// Apertura del archivo para escritura
std::fstream open_file(std::string);
// Apertura del archivo para su lectura
std::fstream read_file(std::string);
#endif