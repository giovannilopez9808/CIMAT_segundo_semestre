from scipy.optimize import line_search
from .functions import function_class
from .GC_methods import GC_methods
from numpy.linalg import norm
from pandas import DataFrame
from typing import Callable
from os.path import join
from numpy import array
from os import makedirs


class optimize_method:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.functions = function_class()
        self.get_beta = GC_methods(params)
        self.stop = stop_functions(params["tau"])
        self.line_search = get_alpha()

    def get_folder_results(self) -> str:
        folder = self.params["GC method"].replace(" ", "_")
        folder = join(self.params["path results"],
                      folder)
        makedirs(folder,
                 exist_ok=True)
        return folder

    def get_image_filename_results(self) -> str:
        lambda_value = self.params["lambda"]
        filename = self.params["lambda values"][lambda_value]["name"]
        filename = f"{filename}.csv"
        return filename

    def get_iterations_filename_results(self) -> str:
        filename = self.get_image_filename_results()
        filename = f"iteration_{filename}"
        return filename

    def save(self) -> None:
        folder = self.get_folder_results()
        filename = self.get_image_filename_results()
        filename = join(folder,
                        filename)
        results = self.x_j.reshape((self.params["n"],
                                    self.params["n"]))
        results = DataFrame(results)
        results.to_csv(filename,
                       index=0)
        filename = self.get_iterations_filename_results()
        filename = join(folder,
                        filename)
        results = DataFrame()
        results["Function"] = self.function
        results["Gradient"] = self.gradient
        results.index.name = "Iteration"
        results.to_csv(filename)

    def run(self) -> None:
        params = self.params
        x_j = self.params["x"]
        function = self.functions.f
        gradient = self.functions.gradient
        gradient_j = gradient(x_j, params)
        d_j = -gradient_j.copy()
        self.function = []
        self.gradient = []
        while True:
            alpha = self.line_search.run(function,
                                         gradient,
                                         x_j,
                                         d_j,
                                         params)
            alpha = alpha[0]
            x_j = x_j + alpha*d_j
            params["x"] = x_j
            gradient_i = gradient_j.copy()
            gradient_j = gradient(x_j,
                                  params)
            beta = self.get_beta.run(d_j,
                                     [gradient_i,
                                      gradient_j])
            beta = 0
            d_j = beta*d_j-gradient_j
            self.function += [function(x_j,
                                       params)]
            self.gradient += [norm(gradient(x_j,
                                            params))]
            print("GC: {} Lambda: {:>4} f(x): {:>20} g(x): {:>20}".format(self.params["GC method"],
                                                                          self.params["lambda"],
                                                                          self.function[-1],
                                                                          self.gradient[-1]))
            if self.stop.gradient(gradient_j):
                break
        self.x_j = x_j.copy()


class stop_functions:
    """
    Contiene todas las funciones que son usadas para verificar si el metodo llego a un punto estacionario
    """

    def __init__(self, tau: float = 1e-12) -> None:
        self.tau = tau

    def vectors(self, vector_i: array, vector_j: array) -> bool:
        """"
        Comprueba la diferencia entre la posicion actual y la anterior
        """
        up = norm((vector_i-vector_j))
        down = max(norm(vector_j), 1)
        dot = abs(up/down)
        if dot < self.tau:
            return True
        return False

    def gradient(self, gradient: Callable) -> bool:
        """
        Compueba si la norm del gradiente se acerca a cero
        """
        dfx = norm(gradient)
        if dfx < self.tau:
            return True
        return False


class get_alpha():
    """
    Obtiene el alpha siguiendo las condiciones de armijo y Wolfe
    """

    def __init__(self) -> None:
        pass

    def run(self, f: Callable, grad: Callable, x: array, d: array, params: dict) -> float:
        args = ({
                "n": params["n"],
                "lambda": params["lambda"],
                "g": params["g"],
                "n": params["n"]
                },)
        alpha = line_search(f,
                            grad,
                            x,
                            d,
                            args=args)
        return alpha
