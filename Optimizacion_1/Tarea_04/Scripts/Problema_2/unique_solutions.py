from datasets import datasets_class
from auxiliar import write_status
from methods import problem_class
datasets = datasets_class()
for dataset in datasets.datasets:
    write_status(dataset)
    parameters = datasets.obtain_dataset(dataset)
    problem = problem_class(parameters)
    problem.solve()
    problem.save()
