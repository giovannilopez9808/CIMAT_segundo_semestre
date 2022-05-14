from Modules.params import get_params, get_params_from_image
from Modules.methods import optimize_method
from Modules.functions import read_image

datasets = {}
params = get_params(datasets)
for lambda_value in params["lambda values"]:
    for method in params["methods"]:
        print("Optimizando imagen con metodo {} y lambda {}".format(method,
                                                                    lambda_value))
        datasets = {
            "lambda": lambda_value,
            "tau": 1e-3,
            "GC method": method
        }
        params = get_params(datasets)
        image = read_image(params)
        params = get_params_from_image(params,
                                       image)
        optimize = optimize_method(params)
        optimize.run()
        optimize.save()
