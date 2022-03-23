from Modules.datasets import obtain_all_params
from Modules.models import model_class
from Modules.solver import solver
from numpy.random import uniform
from tabulate import tabulate
from numpy import square

model_names = [
    # "SGD",
    # "NAG",
    "ADAM",
    "ADADELTA"
]
results = {}
params, gd_params = obtain_all_params()
models = model_class()
y = uniform(0, 1, params["n"])
for model_name in model_names:
    print("Resolviendo por medio de {}".format(model_name))
    results[model_name] = {}
    models.select_method(model_name)
    phi, alpha, time = solver(models, y, params, gd_params)
    error = round(((phi @ alpha - y)**2).mean(), 8)
    results[model_name]["time"] = time
    results[model_name]["error"] = error
table = []
for model_name in model_names:
    time = results[model_name]["time"]
    error = results[model_name]["error"]
    table += [[model_name, time, error]]
print(tabulate(table,
               headers=["Model",
                        "Time",
                        "Error"]))
