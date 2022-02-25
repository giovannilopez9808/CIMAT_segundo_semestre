from auxiliar import obtain_filename, join_path
from numpy.linalg import norm as norma
from numpy.linalg import solve
from pandas import DataFrame
from functions import *
import numpy as np


class algorithm_class:
    """
    Contenido de los métodos para realizar una optimización
    """

    def __init__(self, parameters: dict) -> None:
        # Guardado de los parametros
        self.parameters = parameters
        # Algortimo para obtener el alpha
        self.obtain_alpha = obtain_alpha(parameters)
        # Funciones para detener el metodo
        self.stop_functions = stop_functios_class(parameters["tau"])
        # Metodo de solucion
        self.method = self.descent_gradient

    def descent_gradient(self, function: functions_lambda, x0: array):
        """
        Metodo del descenso del gradiente
        """
        xj = x0.copy()
        while (True):
            # Copia del paso anterior
            xi = xj.copy()
            # Calculo del gradiente
            gradient = -function.gradient(xi)
            # Calculo del alpha
            alpha = self.obtain_alpha.bisection(function, xi, gradient)
            xj = xi + alpha * gradient
            # Funciones para detener el algoritmo
            if self.stop_functions.functions(function, xi, xj):
                break
            if self.stop_functions.vectors(xi, xj):
                break
            if self.stop_functions.gradient(function, xj):
                break
        self.x = xj


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
        up = norma((vector_i - vector_j))
        down = max(norma(vector_j), 1)
        dot = abs(up / down)
        if dot < self.tau:
            return True
        return False

    def functions(self, function: functions_lambda, xi: array,
                  xj: array) -> bool:
        """
        Comprueba la diferencia de las funciones evaluadas en la posicion actual y la anterior
        """
        fxi = function.f(xi)
        fxj = function.f(xj)
        if abs(fxi - fxj) < self.tau:
            return True
        return False

    def gradient(self, function: functions_lambda, xi: array) -> bool:
        """
        Compueba si la norma del gradiente se acerca a cero
        """
        dfx = function.gradient(xi)
        dfx = norma(dfx)
        if dfx < self.tau:
            return True
        return False


class problem_class:
    """
    Orquesta todos los metodos dadas una serie de parametros
    """

    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        # Eleccion de la funcion
        self.function = functions_lambda(parameters)
        self.create_x0()

    def create_x0(self) -> None:
        """
        Vector aleatorio normal con media 0 y sigma 0.5
        """
        self.x0 = 0.5 * normal(self.parameters["n"])

    def solve(self):
        """
        Orquesta la solucion del problema
        """
        self.algorithm = algorithm_class(self.parameters)
        self.algorithm.method(self.function, self.x0)

    def save(self):
        """
        Guardado de los resultados en un archivo
        """
        self.results = DataFrame()
        self.results.index.name = "iteration"
        self.results["t"] = self.function.t
        self.results["y"] = self.function.y
        self.results["x"] = self.algorithm.x
        filename = obtain_filename(self.parameters)
        self.results.to_csv(
            join_path(self.parameters["path results"], filename))


class obtain_alpha():
    """
    Obtiene el alpha siguiendo las condiciones de armijo y Wolfe
    """

    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters

    def bisection(self, function: functions_lambda, x: array, d: array) -> float:
        alpha = 0
        alpha_i = 1
        beta = np.inf
        dfx = function.gradient(x)
        dot_grad = np.dot((dfx, d))
        while(True):
            armijo_condition = self.obtain_armijo_conditon(function,
                                                           dot_grad,
                                                           x,
                                                           d,
                                                           alpha_i)
            wolfe_condition = self.obtain_wolfe_condition(function,
                                                          x,
                                                          dot_grad,
                                                          d,
                                                          alpha_i)
            if(armijo_condition or wolfe_condition):
                if armijo_condition:
                    beta = alpha_i
                    alpha_i = 0.5*(alpha+beta)
                elif wolfe_condition:
                    alpha = alpha_i
                    if beta == np.inf:
                        alpha_i = 2*alpha
                    else:
                        alpha_i = 0.5*(alpha+beta)
            else:
                break
        return alpha_i

    def obtain_armijo_conditon(self, function: functions_lambda, dot_grad: float, x: array, d: array, alpha: float):
        """
        Condicion de armijo
        """
        fx_alpha = function.f(x+alpha*d)
        fx_alphagrad = function.f(x)+self.parameters["c1"]*alpha*dot_grad
        armijo_condition = fx_alpha > fx_alphagrad
        return armijo_condition

    def obtain_wolfe_condition(self, function: functions_lambda, x: array, dot_grad: float, d: array, alpha: float):
        """
        Condicion de Wolfe
        """
        dfx_alpha = function.gradient(x+alpha*d)
        dfx_alpha = np.dot((dfx_alpha, d))
        wolfe_condition = dfx_alpha < self.parameters["c2"]*dot_grad
        return wolfe_condition


def solve_system(matrix: array, vector: array) -> array:
    """
    solucion al sistema de ecuaciones
    """
    solution = solve(matrix, vector)
    return solution
