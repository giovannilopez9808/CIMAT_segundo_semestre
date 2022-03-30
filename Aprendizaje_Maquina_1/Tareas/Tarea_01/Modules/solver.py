from Modules.functions import function_class
from Modules.models import model_class
from numpy.random import rand
from numpy import linspace
import time


def solver(models: model_class, params: dict, gd_params: dict) -> tuple:
    """
    Funcion que ejecuta un algoritmo para realizar la optimización de la función dado un diccionario de parametros

    Parámetros
    -----------------------
    models -> modelo que contiene los métodos de optimización de parámetros
    y -> patrones a aproximar
    params -> diccionario que contiene los parametros de las iteraciones
    gd_params -> diccionario que contiene los parametros del modelo
    """
    max_iteration = params["max iteration"]
    epsilon = params["epsilon"]
    sigma = params["sigma"]
    n = params["n"]
    m = params["m"]
    x = params["x"]
    y = params["y"]
    functions = function_class()
    t_init = time.clock_gettime(0)
    # Valores Iniciales
    mu = linspace(0, x[-1], m)
    alpha = rand(m)
    # Parámetros para el gradiente
    f_params = {
        'mu': mu,
        'x': x,
        'y': y,
        "m": m,
        'alpha': alpha,
        "sigma": sigma,
        'n': n
    }
    iteration = 0
    while iteration < max_iteration:
        print(iteration)
        # descenso para alpha
        alpha = models.method(alpha,
                              grad=functions.gradient_gaussian_alpha,
                              gd_params=gd_params,
                              f_params=f_params)[-1]
        f_params["alpha"] = alpha
        # descenso para mu
        mu = models.method(mu,
                           grad=functions.gradient_gaussian_mu,
                           gd_params=gd_params,
                           f_params=f_params)[-1]
        # Criterio de parada
        f_params["mu"] = mu
        # Número máximo de iteraciones si no hay convergencia
        iteration += 1
    t_end = time.clock_gettime(0)
    total_time = t_end - t_init
    return alpha, mu, total_time
