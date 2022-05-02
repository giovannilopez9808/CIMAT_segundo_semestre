from Modules.methods import problem_class
from Modules.params import get_params

params = get_params()
problem = problem_class()
for image in params["Images"]:
    print("Optimizando {}".format(image))
    problem.init(params, image)
    problem.solve()
