parameters = {"task file": "task",
              "n task": 100,
              "executable": "/home/est_posgrado_giovanni.lopez/src/build/Main.out",
              "input": "/home/est_posgrado_giovanni.lopez/src/Data/GSM2-272.ctr",
              "path output": "/home/est_posgrado_giovanni.lopez/src/Output/channel_",
              "channels": [34, 39, 49]}
file_output = open(parameters["task file"], "w")
for channel in parameters["channels"]:
    path_output = "{}{}/".format(parameters["path output"],
                                 channel)
    for i in range(100):
        filename = "{}{}.dat".format(path_output, i)
        file_output.write("{} {} {} {} {}\n".format(parameters["executable"],
                                                    parameters["input"],
                                                    filename,
                                                    i+1,
                                                    channel))
file_output.close()
