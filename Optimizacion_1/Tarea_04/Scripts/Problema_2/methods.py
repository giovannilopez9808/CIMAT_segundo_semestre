from auxiliar import obtain_filename, join_path
from numpy.linalg import norm as norma
from numpy.linalg import solve
from numpy import array, ones
from pandas import DataFrame
from functions import *
import numpy as np


class algorithm_class:
    """
    Contenido de los métodos para realizar una optimización
    """

    def __init__(self, parameters: dict) -> None:
        # Parametros de las funciones y elecciones
        self.parameters = parameters
        # Resultados de los metodos
        self.results = DataFrame(columns=["fx", "dfx"])
        self.results.index.name = "iteration"
        # Funcion para obtener el alpha que cumpla las condiciones
        self.obtain_alpha = obtain_alpha(parameters)
        # Funciones para detener el metodo
        self.stop_functions = stop_functios_class(parameters["tau"])
        # Eleccion del metodo
        if self.parameters["algorithm name"] == "newton":
            self.method = self.newton
        if self.parameters["algorithm name"] == "descent gradient":
            self.method = self.descent_gradient

    def newton(self, function: functions_class, x0: array) -> array:
        """
        Metodo de newton
        """
        # Inicializacion del vector de resultado
        self.xj = x0.copy()
        # Guardado del punto inicial
        self.results.loc[0] = self.obtain_fx_and_dfx_norm(
            function,
            self.xj)
        i = 1
        while(True):
            # Copia a la posicion anterior
            xi = self.xj.copy()
            # Solucion al sistema de ecuaciones
            di = solve_system(function.hessian(xi),
                              function.gradient(xi))
            #   Obtener un alpha con el metodo de biseccion
            alpha = self.obtain_alpha.bisection(function, xi, -di)
            # Cambio en la posicion
            self.xj = xi-alpha*di
            # Guardado de los resultados
            self.results.loc[i] = self.obtain_fx_and_dfx_norm(
                function,
                self.xj)
            # Comprobacion del metodo
            if self.stop_functions.vectors(xi, self.xj):
                break
            if self.stop_functions.gradient(
                    function,
                    self.xj):
                break
            i += 1

    def descent_gradient(self, function: functions_class, x0: array):
        """
        Metodo del descenso del gradiente
        """
        # Inicializacion del vector de resultado
        self.xj = x0.copy()
        # Guardado de la primer posicion
        self.results.loc[0] = self.obtain_fx_and_dfx_norm(
            function,
            self.xj)
        i = 1
        while(True):
            # Guardado del paso anterior
            xi = self.xj.copy()
            # Calculo del gradiente en el paso i
            gradient = function.gradient(xi)
            # Siguiente paso
            self.xj = xi - self.parameters["alpha"]*gradient
            # Guardado de los resultados
            self.results.loc[i] = self.obtain_fx_and_dfx_norm(
                function,
                self.xj)
            # Compobacion del metodo
            if self.stop_functions.vectors(xi, self.xj):
                break
            if self.stop_functions.gradient(
                    function,
                    self.xj):
                break
            i += 1

    def obtain_fx_and_dfx_norm(self, function: functions_class, x: array) -> tuple:
        """
        Calculo de f(x) y ||gradient(f(x))|
        """
        fx = function.f(x)
        dfx = function.gradient(x)
        dfx = norma(dfx)
        return [fx, dfx]


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
        up = norma((vector_i-vector_j))
        down = max(norma(vector_j), 1)
        dot = abs(up/down)
        if dot < self.tau:
            return True
        return False

    def functions(self, function: functions_class, xi: array, xj: array) -> bool:
        """
        Comprueba la diferencia de las funciones evaluadas en la posicion actual y la anterior
        """
        fxi = function.f(xi)
        fxj = function.f(xj)
        if abs(fxi-fxj) < self.tau:
            return True
        return False

    def gradient(self, function: functions_class, xi: array) -> bool:
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
        self.function = functions_class(parameters["problem name"])
        if parameters["problem name"] == "wood":
            self.use_wood()
        if parameters["problem name"] == "rosembrock":
            self.use_rosembrock(parameters)
        # Eleccion del punto inicial
        if parameters["initial point"] == "random":
            self.random_initial_points(parameters)

    def use_wood(self) -> None:
        """
        Vector inicial predefinido de Wood
        """
        self.x0 = [-1, -3, -1, -3]

    def use_rosembrock(self, parameters: dict) -> None:
        """
        Vector inicial predefinido de la funcion de Rosembrock
        """
        self.x0 = ones(parameters["n"])
        self.x0[0] = -1.2
        self.x0[parameters["n"]-2] = -1.2

    def random_initial_points(self, parameters: dict):
        """
        Vector aleatorio normal con media 0 y sigma 0.5
        """
        if parameters["problem name"] == "wood":
            parameters["n"] = 4
        self.x0 = 0.5*np.random.randn(parameters["n"])

    def solve(self):
        """
        Orquesta la solucion del problema
        """
        self.algorithm = algorithm_class(self.parameters)
        self.algorithm.method(self.function,
                              self.x0)

    def save(self):
        """
        Guardado de los resultados en un archivo
        """
        filename = obtain_filename(self.parameters)
        self.algorithm.results.to_csv(join_path(self.parameters["path results"],
                                                filename))


class obtain_alpha():
    """
    Obtiene el alpha siguiendo las condiciones de armijo y Wolfe
    """

    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters

    def bisection(self, function: functions_class, x: array, d: array) -> float:
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

    def obtain_armijo_conditon(self, function: functions_class, dot_grad: float, x: array, d: array, alpha: float):
        """
        Condicion de armijo
        """
        fx_alpha = function.f(x+alpha*d)
        fx_alphagrad = function.f(x)+self.parameters["c1"]*alpha*dot_grad
        armijo_condition = fx_alpha > fx_alphagrad
        return armijo_condition

    def obtain_wolfe_condition(self, function: functions_class, x: array, dot_grad: float, d: array, alpha: float):
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
