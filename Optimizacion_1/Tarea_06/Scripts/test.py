from Modules.datasets import datasets_class
from Modules.methods import problem_class
datasets = datasets_class()
for dataset in datasets.datasets:
    parameters = datasets.obtain_dataset(dataset)
    problem = problem_class(parameters)
    problem.solve()
    problem.save()
# from Modules.functions import functions_class
# from Modules.datasets import datasets_class
# from Modules.mnist import mnist_model
# from numpy import ones
# from numpy.linalg import norm

# datasets = datasets_class()
# parameters = datasets.obtain_dataset("Log likehood gradient")
# mnist = mnist_model(parameters)
# function = functions_class("log likehood")
# beta = ones(len(mnist.train_data[0]))
# print(norm(function.gradient(mnist.train_data,
#                              mnist.train_label,
#                              beta)))
