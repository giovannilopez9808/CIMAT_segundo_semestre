from Modules.functions import function_class
from numpy import ones, array

x = ones(20)
y = x+1
mu = array(range(10))
alpha = array(range(10))
function = function_class()
f_params = {}
f_params['x'] = x
f_params['alpha'] = alpha
f_params['n'] = 20
f_params['y'] = y
f_params["sigma"] = 1
f_params["mu"] = mu
f_params["m"] = 10
print(function.gradient_gaussian_alpha(alpha, f_params))
