from scipy.special import expit as sigmod
from numpy import array,  log, sum, abs


def obtain_filename(parameters: dict) -> str:
    """
    Obtiene el nombre del archivo en base a el dataset dado
    """
    problem = parameters["problem name"].replace(" ", "_")
    algorithm = parameters["algorithm name"].replace(" ", "_")
    return "{}_{}.csv".format(problem,
                              algorithm)


def join_path(path: str, filename: str) -> str:
    """
    Creacion de la ruta donde se encuentra un archivo
    """
    return "{}{}".format(path, filename)


class functions_class:
    def __init__(self, name: str) -> None:
        if name == "log likehood":
            self.f = self.function_log_likehood
            self.gradient = self.grad_log_likehood

    def function_log_likehood(self, x: array, y: array, beta: array) -> float:
        # fx = 0
        pi = self.pi_log_likehood(x, beta)
        pi2 = 1-pi
        pi[pi == 0] = 1e-6
        pi2[pi2 == 0] = 1e-6
        fx = -sum(y * log(pi) + (1.0 - y) * log(1-pi))
        return fx

    def grad_log_likehood(self, x: array, y: array, beta: array) -> array:
        pi = self.pi_log_likehood(x, beta)
        grad = -sum((y - pi)*x.T, axis=1)
        return grad

    def pi_log_likehood(self, x: array, beta: array) -> float:
        pi = array(sigmod(x@beta))
        return pi
