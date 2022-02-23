from auxiliar import join_path, obtain_filename
from datasets import obtain_dataset
import matplotlib.pyplot as plt
from pandas import read_csv
parameters = obtain_dataset("lambda 1")
filename = obtain_filename(parameters)
filename = join_path(parameters["path results"],
                     filename)
data = read_csv(filename)
plt.plot(data["t"], data["y"])
plt.plot(data["t"], data["x"])
plt.show()
