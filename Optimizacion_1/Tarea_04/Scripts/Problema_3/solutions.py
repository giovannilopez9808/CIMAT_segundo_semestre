from datasets import datasets_class
from methods import problem_class
datasets = datasets_class()
for dataset in datasets.datasets:
    parameters = datasets.obtain_dataset("lambda 1000 test")
    problem = problem_class(parameters)
    problem.solve()
    problem.save()
