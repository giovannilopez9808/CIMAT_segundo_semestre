parameters = {
    # Nombre de archivo que leera slurm
    "task file": "task",
    # Numero de canales por ejecucion
    "n task": 100,
    # Dirección absoluta del ejecutable
    "executable": "/home/est_posgrado_giovanni.lopez/src/build/Main.out",
    # Dirección absoluta del archivo de entrada
    "input": "/home/est_posgrado_giovanni.lopez/src/Data/GSM2-272.ctr",
    # Dirección absoluta del archivo de salida
    "path output": "/home/est_posgrado_giovanni.lopez/src/Output/channel_",
    # Canales disponibles en cada ejecución
    "channels": [34, 39, 49]}
# Apertura del archivo que leera slurm
file_output = open(parameters["task file"], "w")
for channel in parameters["channels"]:
    # Direccion absoluta del archivo de salida
    path_output = "{}{}/".format(parameters["path output"],
                                 channel)
    for i in range(100):
        # Nombre del archivo de salida
        filename = "{}{}.dat".format(path_output, i)
        # Escritura en el archivo que leera slurm
        file_output.write("{} {} {} {} {}\n".format(parameters["executable"],
                                                    parameters["input"],
                                                    filename,
                                                    i+1,
                                                    channel))
file_output.close()
