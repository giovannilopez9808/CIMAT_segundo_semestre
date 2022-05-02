from Modules.methods import problem_class
from Modules.params import get_params

params = get_params()
problem = problem_class()
problem.init(params, "flower")
problem.solve()
