from datasets import *
from auxiliar import *
from methods import *
datasets = datasets_class()
for dataset in datasets.datasets:
    write_status(dataset)
    parameters = datasets.obtain_dataset(dataset)
    filename = obtain_filename(parameters)
    problem = problem_class(parameters)
    problem.solve()
    problem.save()
