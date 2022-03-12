from Modules.functions import functions_class, join_path
from Modules.datasets import datasets_class
from Modules.mnist import mnist_model
from numpy import loadtxt

dataset = datasets_class()
function = functions_class("log likehood")
parameters = dataset.obtain_dataset("log likehood bisection")
mnist = mnist_model(parameters)
beta = loadtxt(join_path(parameters["path results"],
                         "beta.csv"),
               delimiter=",")
print(function.error(beta,
                     mnist.test_data,
                     mnist.test_label))
