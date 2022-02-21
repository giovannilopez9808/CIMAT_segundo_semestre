import matplotlib.pyplot as plt
from os import listdir as ls
from numpy import log10
from auxiliar import *

parameters = {"path results": "../Results/",
              "path graphics": "../Graphics/",
              "function": "wood"}

files = ls(parameters["path results"])
files = obtain_files_per_function(files, parameters)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
for file in files:
    filename = join_path(parameters["path results"], file)
    data = read_data(filename)
    ax1.plot(data.index, log10(data["fx"]+1))
    ax2.plot(data.index, log10(data["dfx"]+1))
file_graphics = "{}.png".format(parameters["function"])
plt.savefig(join_path(parameters["path graphics"],
                      file_graphics))
