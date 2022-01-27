# Tarea 01 - Optimización estocastica

## Compilación

La compilación del programa se realiza con los siguientes comandos:

```bash

mdkir build
cd build
cmake ..
make

```

Este creará un archivo llamado `Main.out`, el cual es el programa compilado.

## Ejecución

Para realizar la ejecución del programa se necesitan las siguientes entradas:

- Dirección absoluta del archivo de datos
- Dirección absoluta del archivo de salida
- Semilla
- Número de canales disponibles

Se creo un programa en python el cual automatiza la creación del archivo `task`. Este archivo es usado por slurm para realizar la ejecución de forma paralela.
