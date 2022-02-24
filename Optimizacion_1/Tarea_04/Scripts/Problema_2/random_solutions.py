from unittest import result
from datasets import *
from auxiliar import *
from methods import *
datasets = datasets_class()
datasets_list = [
    dataset for dataset in datasets.datasets if "random" in dataset]
for dataset in datasets_list:
    parameters = datasets.obtain_dataset(dataset)
    parameters["path results"] += "Random_results/"
    filename = obtain_filename_for_random_results(dataset)
    filename = join_path(parameters["path results"],
                         filename)
    results = DataFrame(columns=["fx", "dfx"])
    results.index.name = "iterations"
    for i in range(2):
        print("Ejecuntando {} en la interacion {} de {}".format(dataset, i+1, 100))
        problem = problem_class(parameters)
        problem.solve()
        fx = problem.function.f(problem.algorithm.xj)
        dfx = np.linalg.norm(problem.function.gradient(problem.algorithm.xj))
        results.loc[i] = [fx, dfx]
    print(results)
