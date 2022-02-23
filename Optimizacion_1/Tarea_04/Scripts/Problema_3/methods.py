from auxiliar import obtain_filename, join_path
from pandas import DataFrame
from functions import *
import numpy as np


class algorithm_class:

    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.obtain_alpha = obtain_alpha(parameters)
        self.stop_functions = stop_functios_class(parameters["tau"])
        self.method = self.descent_gradient

    def descent_gradient(self, function: functions_lambda, x0: np.array):
        xj = x0.copy()
        while (True):
            xi = xj.copy()
            gradient = -function.gradient(xi)
            alpha = self.obtain_alpha.bisection(function, xi, gradient)
            xj = xi + alpha * gradient
            if self.stop_functions.functions(function, xi, xj):
                break
            if self.stop_functions.vectors(xi, xj):
                break
            if self.stop_functions.gradient(function, xj):
                break
            print(function.f(xi), alpha)
        self.x = xj


class stop_functios_class:

    def __init__(self, tau: float = 1e-12) -> None:
        self.tau = tau

    def vectors(self, vector_i: np.array, vector_j: np.array) -> bool:
        up = np.linalg.norm((vector_i - vector_j))
        down = max(np.linalg.norm(vector_j), 1)
        dot = abs(up / down)
        if dot < self.tau:
            return True
        return False

    def functions(self, function: functions_lambda, xi: np.array,
                  xj: np.array) -> bool:
        fxi = function.f(xi)
        fxj = function.f(xj)
        if abs(fxi - fxj) < self.tau:
            return True
        return False

    def gradient(self, function: functions_lambda, xi: np.array) -> bool:
        dfx = function.gradient(xi)
        dfx = np.linalg.norm(dfx)
        if dfx < self.tau:
            return True
        return False


class problem_class:

    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.function = functions_lambda(parameters)
        self.create_x0()

    def create_x0(self) -> None:
        self.x0 = 0.5 * np.random.randn(self.parameters["n"])

    def solve(self):
        self.algorithm = algorithm_class(self.parameters)
        self.algorithm.method(self.function, self.x0)

    def save(self):
        self.results = DataFrame()
        self.results.index.name = "iteration"
        self.results["t"] = self.function.t
        self.results["y"] = self.function.y
        self.results["x"] = self.algorithm.x
        filename = obtain_filename(self.parameters)
        self.results.to_csv(
            join_path(self.parameters["path results"], filename))


class obtain_alpha():

    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters

    def bisection(self, function: functions_lambda, x: np.array,
                  d: np.array) -> float:
        alpha = 0
        alpha_i = 1
        beta = np.inf
        dfx = function.gradient(x)
        dot_grad = np.linalg.multi_dot((dfx, d))
        while (True):
            armijo_condition = obtain_armijo_conditon(function, dot_grad, x, d,
                                                      alpha_i,
                                                      self.parameters["c1"])
            wolfe_condition = obtain_wolfe_condition(function, x, dot_grad, d,
                                                     self.parameters["c2"],
                                                     alpha_i)
            if (armijo_condition or wolfe_condition):
                if armijo_condition:
                    beta = alpha_i
                    alpha_i = 0.5 * (alpha + beta)
                elif wolfe_condition:
                    alpha = alpha_i
                    if beta == np.inf:
                        alpha_i = 2 * alpha
                    else:
                        alpha_i = 0.5 * (alpha + beta)
            else:
                break
        return alpha_i


def solve_system(matrix: np.array, vector: np.array) -> np.array:
    solution = np.linalg.solve(matrix, vector)
    return solution


def obtain_armijo_conditon(function: functions_lambda, dot_grad: float,
                           x: np.array, d: np.array, alpha: float, c1: float):
    fx_alpha = function.f(x + alpha * d)
    fx_alphagrad = function.f(x) + c1 * alpha * dot_grad
    armijo_condition = fx_alpha > fx_alphagrad
    return armijo_condition


def obtain_wolfe_condition(function: functions_lambda, x: np.array,
                           dot_grad: float, d: np.array, c2: float,
                           alpha: float):
    dfx_alpha = function.gradient(x + alpha * d)
    dfx_alpha = np.linalg.multi_dot((dfx_alpha, d))
    wolfe_condition = dfx_alpha < c2 * dot_grad
    return wolfe_condition
