from auxiliar import join_path, read_data
import matplotlib.pyplot as plt
from datasets import *

datasets = datasets_statistics_class()
parameters = datasets.obtain_dataset("rosembrock 100 newton")
filename = join_path(parameters["path results"], parameters["filename"])
data = read_data(filename)
results = data.describe()
results = results.drop(["25%", "50%", "75%"])
results.columns = ["f(x)", "||nabla f(x)||"]
results.index = ["n", "Media", "\sigma", "Mínimo", "Máximo"]
print(results.to_latex())
