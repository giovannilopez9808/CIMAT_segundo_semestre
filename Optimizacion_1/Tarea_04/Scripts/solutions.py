from datasets import *
from auxiliar import *
from methods import *
parameters = obtain_dataset("wood 4 predefined newton")
filename = obtain_filename(parameters)
problem = problem_class(parameters)
problem.solve()
problem.save()
