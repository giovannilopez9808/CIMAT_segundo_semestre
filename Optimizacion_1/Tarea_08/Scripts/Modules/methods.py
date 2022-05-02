from .functions import function, gradient_gaussian_alpha, gradient_gaussian_mu
from numpy.linalg import norm, solve
from .data_model import data_model
from os import makedirs as mkdir
from .params import get_f_params
from numpy import array, inf
from pandas import DataFrame
from typing import Callable
from os.path import join


class problem_class:
    """
    Orquesta todos los metodos dadas una serie de parametros
    """

    def __init__(self) -> None:
        pass

    def init(self, params: dict, data_name: str) -> None:
        self.params = params
        self.image = data_model(params)
        self.image.read(data_name)
        self.data_name = data_name

    def solve(self):
        """
        Orquesta la solucion del problema
        """
        # Eleccion de la funcion
        self.results = {}
        for class_type in self.image.data:
            self.results[class_type] = {}
            data = self.image.get_data(class_type)
            self.f_params = get_f_params(self.image.n)
            self.f_params["h"] = data[0]
            self.f_params["c"] = data[1]
            self.algorithm = algorithm_class(self.f_params)
            self.algorithm.method(self.f_params)
            self.results[class_type]["alpha"] = self.f_params["alpha"]
            self.results[class_type]["mu"] = self.f_params["mu"]
        self._save_results()

    def _save_results(self):
        for class_type in self.results:
            folder_name = class_type.replace(" ", "_")
            folder_name = join(self.params["path results"],
                               self.data_name,
                               folder_name)
            mkdir(folder_name,
                  exist_ok=True)
            results = self.results[class_type]
            for file in ["mu", "alpha"]:
                filename = "{}.csv".format(file)
                filename = join(folder_name,
                                filename)
                data = DataFrame(results[file])
                data.to_csv(filename,
                            index=False)


class algorithm_class:
    """
    Contenido de los métodos para realizar una optimización
    """

    def __init__(self, params: dict) -> None:
        # Parametros de las funciones y elecciones
        self.params = params
        # Funcion para obtener el alpha que cumpla las condiciones
        self.obtain_alpha = obtain_alpha(params)
        # Funciones para detener el metodo
        self.stop_functions = stop_functios_class(params["tau"])
        # Eleccion del metodo
        self.method = self.descent_gradient

    def descent_gradient(self, f_params: dict):
        """
        Metodo del descenso del gradiente
        """
        # Inicializacion del vector de resultado
        alpha_j = f_params["alpha"].copy()
        mu_j = f_params["mu"].copy()
        iteration = 1
        while(True):
            # Guardado del paso anterior
            alpha_i = alpha_j.copy()
            mu_i = mu_j.copy()
            # Calculo del gradiente en el paso i
            gradient = -gradient_gaussian_alpha(alpha_i, f_params)
            # Siguiente paso
            # alpha = self.obtain_alpha.method(function,
            #                                  gradient_gaussian_alpha,
            #                                  alpha_i,
            #                                  f_params,
            #                                  gradient,
            #                                  "alpha")
            alpha = 0.5
            alpha_j = alpha_i + alpha * gradient
            if self.stop_functions.gradient(gradient_gaussian_alpha,
                                            alpha_j,
                                            f_params):
                break
            gradient = -gradient_gaussian_mu(mu_i, f_params)
            # alpha = self.obtain_alpha.method(function,
            #                                  gradient_gaussian_mu,
            #                                  mu_i,
            #                                  f_params,
            #                                  gradient,
            #                                  "mu")
            mu_j = mu_i + alpha * gradient
            alpha_decision = self.stop_functions.gradient(gradient_gaussian_alpha,
                                                          alpha_j,
                                                          f_params)
            mu_decision = self.stop_functions.gradient(gradient_gaussian_mu,
                                                       mu_j,
                                                       f_params)
            f_params["alpha"] = alpha_j
            f_params["mu"] = mu_j
            function_i = function(f_params)
            grad_alpha = norm(gradient_gaussian_alpha(alpha_j, f_params))
            grad_mu = norm(gradient_gaussian_mu(mu_j, f_params))
            print("{:>3} {:>25} {:>20} {:>20}".format(iteration,
                                                      function_i,
                                                      grad_alpha,
                                                      grad_mu))
            if alpha_decision and mu_decision:
                break
            iteration += 1


class stop_functios_class:
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

    def gradient(self, gradient: Callable, x: array, f_params: dict) -> bool:
        """
        Compueba si la norm del gradiente se acerca a cero
        """
        dfx = gradient(x, f_params)
        dfx = norm(dfx)
        if dfx < self.tau:
            return True
        return False


class obtain_alpha():
    """
    Obtiene el alpha siguiendo las condiciones de armijo y Wolfe
    """

    def __init__(self, params: dict) -> None:
        self.params = params
        if params["search name"] == "bisection":
            self.method = self.bisection
        if params["search name"] == "back tracking":
            self.method = self.back_tracking

    def bisection(self, function_f: Callable, gradient: Callable, x: array, f_params: dict, d: array, name: str) -> float:
        # Inicialización
        alpha = 0.0
        beta_i = inf
        alpha_k = 1
        print(gradient(x, f_params).shape)
        print(d.shape)
        dot_grad = gradient(x, f_params) @ d
        while True:
            armijo_condition = self.obtain_armijo_condition(function_f,
                                                            dot_grad,
                                                            x,
                                                            f_params,
                                                            d,
                                                            alpha_k,
                                                            name)
            wolfe_condition = self.obtain_wolfe_condition(gradient,
                                                          x,
                                                          f_params,
                                                          dot_grad,
                                                          d,
                                                          alpha_k)
            if armijo_condition or wolfe_condition:
                if armijo_condition:
                    beta_i = alpha_k
                    alpha_k = 0.5*(alpha + beta_i)
                else:
                    alpha = alpha_k
                    if beta_i == inf:
                        alpha_k = 2.0 * alpha
                    else:
                        alpha_k = 0.5 * (alpha + beta_i)
            else:
                break
        return alpha_k

    def back_tracking(self, gradient: Callable, x: array, y: array, beta: array, d: array):
        """
        Calcula tamaño de paso alpha

            Parámetros
            -----------
                x_k     : Vector de valores [x_1, x_2, ..., x_n]
                d_k     : Dirección de descenso
                f       : Función f(x)
                f_grad  : Función que calcula gradiente
                alpha   : Tamaño inicial de paso
                ro      : Ponderación de actualización
                c1      : Condición de Armijo
            Regresa
            -----------
                alpha_k : Tamaño actualizado de paso
        """
        # Inicialización
        alpha_k = self.params["alpha bisection"]
        dot_grad = (-gradient(x, y, beta)) @ d
        # Repetir hasta que se cumpla la condición de armijo
        while True:
            armijo_condition = self.obtain_armijo_condition(
                function, dot_grad, x, y, beta, d, alpha_k)
            if armijo_condition:
                alpha_k = self.params["rho"] * alpha_k
            else:
                break
        return alpha_k

    def obtain_armijo_condition(self, function: Callable, dot_grad: float, x: array, f_params: dict, d: array, alpha: float, name: str):
        """
        Condicion de armijo
        """
        f_params_copy = f_params.copy()
        fx_alphagrad = function(f_params)
        f_params_copy[name] = f_params_copy[name]+alpha*d
        fx_alpha = function(f_params_copy)
        fx_alphagrad += self.params["c1"]*alpha*dot_grad
        armijo_condition = fx_alpha > fx_alphagrad
        return armijo_condition

    def obtain_wolfe_condition(self, gradient: Callable,  x: array, f_params: dict, dot_grad: float, d: array, alpha: float):
        """
        Condicion de Wolfe
        """
        dfx_alpha = gradient(x+alpha*d, f_params)
        dfx_alpha = dfx_alpha @ d
        wolfe_condition = dfx_alpha < self.params["c2"]*dot_grad
        return wolfe_condition


def solve_system(matrix: array, vector: array) -> array:
    """
    solucion al sistema de ecuaciones
    """
    solution = solve(matrix, vector)
    return solution
