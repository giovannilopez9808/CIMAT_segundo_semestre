from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functions import *
from datasets import *
import numpy as np

parameters = obtain_dataset("lambda 1")
x0 = np.random.randn(128)
function = functions_lambda(parameters)
res = minimize(function.f,
               x0,
               method='BFGS',
               jac=function.gradient)
print(function.f(res.x))
plt.plot(function.t, function.y)
plt.plot(function.t, res.x)
plt.show()
