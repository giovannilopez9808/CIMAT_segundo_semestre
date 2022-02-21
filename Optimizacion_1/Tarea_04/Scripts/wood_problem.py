from auxiliar import *
from methods import *
parameters = {"path results": "../Results/",
              "n": 20,
              "tau": 1e-12,
              "alpha": 1e-4,
              "problem name": "wood",
              "initial point": "predefined",
              #   "algorithm name": "newton"
              "algorithm name": "descent gradient"
              }
filename = obtain_filename(parameters["problem name"],
                           parameters["algorithm name"])
problem = problem_class(parameters)
problem.solve()
problem.algorithm.results.to_csv(join_path(parameters["path results"],
                                           filename))
