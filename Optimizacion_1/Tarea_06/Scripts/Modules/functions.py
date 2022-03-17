from numpy import array, log, sum, abs, mean
from scipy.special import expit as sigmod


def obtain_filename(parameters: dict) -> str:
    """
    Obtiene el nombre del archivo en base a el dataset dado
    """
    problem = parameters["problem name"].replace(" ", "_")
    algorithm = parameters["search name"].replace(" ", "_")
    return "{}_{}.csv".format(problem,
                              algorithm)


def join_path(path: str, filename: str) -> str:
    """
    Creacion de la ruta donde se encuentra un archivo
    """
    return "{}{}".format(path, filename)


class functions_class:
    def __init__(self, name: str) -> None:
        """
        Modelo que contiene las diferentes funciones a utilizar
        """
        if name == "log likelihood":
            self.f = self.function_log_likelihood
            self.gradient = self.grad_log_likelihood

    def function_log_likelihood(self, x: array, y: array, beta: array) -> float:
        """
        Funcion de log-likelihodd    
        """
        epsilon = 1e-6
        pi = self.pi_log_likelihood(x, beta)
        pi2 = 1-pi
        pi[abs(pi) < epsilon] = epsilon
        pi2[abs(pi2) < epsilon] = epsilon
        fx = -sum(y * log(pi) + (1.0 - y) * log(pi2))
        return fx

    def grad_log_likelihood(self, x: array, y: array, beta: array) -> array:
        """
        Función del gradiente de log-likelihood
        """
        pi = self.pi_log_likelihood(x, beta)
        grad = -sum((y - pi)*x.T, axis=1)
        return grad

    def pi_log_likelihood(self, x: array, beta: array) -> float:
        """
        Retorna el valor de pi_i para una x y beta dados
        """
        pi = array(sigmod(x@beta))
        return pi

    def error(self, beta: array, x: array, y: array) -> float:
        """
        Calcula el error definido para una beta obtenida
        """
        pi_ = self.pi_log_likelihood(x, beta)
        return mean(abs((pi_ > 0.5) - y))
