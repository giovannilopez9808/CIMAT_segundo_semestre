from datasets import *
from auxiliar import *
from methods import *
parameters = obtain_dataset("lambda 1000")
filename = obtain_filename(parameters)
problem = problem_class(parameters)
problem.solve()
problem.save()
