from Modules.datasets import datasets_class
from Modules.methods import problem_class
datasets = datasets_class()
for dataset in datasets.datasets:
    parameters = datasets.obtain_dataset(dataset)
    problem = problem_class(parameters)
    problem.solve()
    problem.save()
