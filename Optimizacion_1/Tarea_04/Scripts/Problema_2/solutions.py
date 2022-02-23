from datasets import *
from auxiliar import *
from methods import *
parameters = obtain_dataset("lambda 3 predefined gradient")
filename = obtain_filename(parameters)
problem = problem_class(parameters)
problem.solve()
problem.save()
