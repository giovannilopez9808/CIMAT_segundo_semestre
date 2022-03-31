from Modules.functions import print_results, function_class, write_results
from Modules.datasets import obtain_all_params
from Modules.models import model_class
from Modules.solver import solver
from numpy import array, linspace
from pandas import read_csv
from os.path import join

results = {}
function = function_class()
params, gd_params = obtain_all_params()
models = model_class()
filename = join(params["path data"],
                params["file data"])
y = read_csv(filename)
y = y.dropna(axis=0)
y = array(y[params["data column"]])
params["n"] = len(y)
x = linspace(1, params["n"], params["n"])
params["x"] = x
params["y"] = y
for model_name in params["models"]:
    print("Resolviendo por medio de {}".format(model_name))
    results[model_name] = {}
    models.select_method(model_name)
    alpha, mu, time_solver = solver(models, params, gd_params)
    phi = function.phi(x, mu, params["sigma"])
    error = round(((phi @ alpha - y)**2).mean(), 8)
    results[model_name]["time"] = time_solver
    results[model_name]["error"] = error
    write_results(params,
                  model_name,
                  alpha,
                  mu)
print_results(params, results)
