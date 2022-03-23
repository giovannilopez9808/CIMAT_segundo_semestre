from .functions import function_class
from numpy.random import uniform
from .models import model_class
from numpy.linalg import norm
from numpy import linspace
import time


def solver(models: model_class, y: list, params: dict, gd_params: dict):
    max_iter = params["max_iter"]
    epsilon = params["epsilon"]
    sigma = params["sigma"]
    n = params["n"]
    functions = function_class()
    t_init = time.clock_gettime(0)
    # Valores Iniciales
    mu = linspace(0, 100, n)
    phi = functions.phi(y, mu, sigma)
    alpha = uniform(0, sigma, n)
    # Parámetros para el gradiente
    f_params = {
        'mu': mu,
        'X': phi,
        'y': y,
        'Alpha': alpha,
        'n': n
    }
    num_iter = 0
    while num_iter < max_iter:
        # descenso para alpha
        alpha = models.method(alpha,
                              grad=functions.grad_gaussian_radial_alpha,
                              gd_params=gd_params,
                              f_params=f_params)[-1]
        if norm(phi @ alpha - y) < epsilon:
            break
        # descenso para mu
        mu_old = mu
        mu = models.method(mu,
                           grad=functions.grad_gaussian_radial_mu,
                           gd_params=gd_params,
                           f_params=f_params)[-1]
        # actualizacion
        phi = functions.phi(y, mu, sigma)
        # Criterio de parada
        if norm(mu - mu_old) < epsilon:
            break
        # Número máximo de iteraciones si no hay convergencia
        num_iter += 1
    t_end = time.clock_gettime(0)
    total_time = t_end - t_init
    return phi, alpha, total_time
