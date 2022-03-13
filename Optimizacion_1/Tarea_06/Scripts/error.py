from Modules.functions import functions_class, join_path, obtain_filename
from Modules.datasets import datasets_class
from Modules.mnist import mnist_model
from numpy import loadtxt

dataset = datasets_class()
function = functions_class("log likelihood")
parameters = dataset.obtain_dataset("log likelihood bisection")
mnist = mnist_model(parameters)
filename = obtain_filename(parameters)
filename = "beta_{}".format(filename)
beta = loadtxt(join_path(parameters["path results"],
                         filename),
               delimiter=",")
print(function.error(beta,
                     mnist.test_data,
                     mnist.test_label))
