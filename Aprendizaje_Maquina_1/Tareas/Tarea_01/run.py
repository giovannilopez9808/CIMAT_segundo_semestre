from Modules.datasets import obtain_all_params
from Modules.functions import print_results
from Modules.models import model_class
from Modules.solver import solver
from numpy.random import uniform

results = {}
params, gd_params = obtain_all_params()
models = model_class()
x = uniform(1, 2, (params["n"], 1))
y = uniform(0, 1, (params["n"], 1))
params["x"] = x
params["y"] = y
for model_name in params["models"]:
    print("Resolviendo por medio de {}".format(model_name))
    results[model_name] = {}
    models.select_method(model_name)
    phi, alpha, time = solver(models, params, gd_params)
    error = round(((phi @ alpha - y)**2).mean(), 8)
    results[model_name]["time"] = time
    results[model_name]["error"] = error
print_results(params, results)
