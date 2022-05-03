from numpy import exp,  array, sum, zeros_like
from numpy.linalg import norm


def function(f_params: dict) -> float:
    """
    Calcula la función dado unos parametros

    Parametros
    ----------------------
    alpha -> array con los valores de alpha
    mu -> matriz con los valores de mu
    sigma -> parametro de la función de la distribucion normal
    h_list -> array con los valores del histograma
    c_lits -> lista con las posiciones del histograma
    ----------------------
    """
    alpha = f_params["alpha"]
    mu = f_params["mu"]
    sigma = f_params["sigma"]
    h_list = f_params["h"]
    c_list = f_params["c"]
    f = 0
    for h, c in zip(h_list, c_list):
        f_1 = get_exp(c-mu, sigma)
        f_1 = sum(alpha * f_1)
        f_1 = (h-f_1)**2
        f += f_1
    return f


def gradient_gaussian_alpha(alpha: array, f_params: dict) -> array:
    """
    Calcula el gradiente respecto a alpha
   Parametros
    ----------------------
    alpha -> array con los valores de alpha
    mu -> matriz con los valores de mu
    sigma -> parametro de la función de la distribucion normal
    h_list -> array con los valores del histograma
    c_lits -> lista con las posiciones del histograma

    Output
    -----------
        Array gradiente
    """
    # Obtengo Parámetros
    mu = f_params["mu"]
    sigma = f_params["sigma"]
    h_list = f_params["h"]
    c_list = f_params["c"]
    gradient = zeros_like(alpha)
    for i in range(gradient.shape[0]):
        gradient_i = 0
        for h, c in zip(h_list, c_list):
            g = get_exp(c-mu, sigma)
            g = sum(alpha*g)
            e_i = exp(-norm(c - mu[i])**2 / (2*sigma**2))
            gradient_i += (h - g) * e_i
        gradient[i] = -2*gradient_i
    return gradient


def gradient_gaussian_mu(mu: array, f_params: dict) -> array:
    """
    Calcula el gradiente respecto a alpha
    Parametros
    ----------------------
    alpha -> array con los valores de alpha
    mu -> matriz con los valores de mu
    sigma -> parametro de la función de la distribucion normal
    h_list -> array con los valores del histograma
    c_lits -> lista con las posiciones del histograma

    Output
    -----------
        Array gradiente
    """
    # Obtengo Parámetros
    # Obtengo Parámetros
    alpha = f_params["alpha"]
    sigma = f_params["sigma"]
    h_list = f_params["h"]
    c_list = f_params["c"]
    gradient = zeros_like(mu)
    for i in range(gradient.shape[0]):
        gradient_i = 0
        for h, c in zip(h_list, c_list):
            g = get_exp(c-mu, sigma)
            g = sum(alpha*g)
            e_i = exp(-norm(c - mu[i])**2/(2*sigma**2))
            e_i = alpha[i]*e_i
            e_i = e_i*(c-mu[i])
            gradient_i += (h - g) * e_i
        gradient[i] = -2*gradient_i/sigma**2
    return gradient


def get_exp(x, sigma) -> float:
    """
    Generalización de la función exponencial
    """
    f_exp = exp(-(norm(x, axis=1)**2)/(2*sigma**2))
    return f_exp
