from auxiliar import obtain_filename, join_path
from pandas import DataFrame
from functions import *
import numpy as np


class algorithm_class:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.results = DataFrame(columns=["fx", "dfx"])
        self.results.index.name = "iteration"
        self.obtain_alpha = obtain_alpha(parameters)
        self.stop_functions = stop_functios_class(parameters["tau"])
        if self.parameters["algorithm name"] == "newton":
            self.method = self.newton
        if self.parameters["algorithm name"] == "descent gradient":
            self.method = self.descent_gradient

    def newton(self, function: functions_class, x0: np.array) -> np.array:
        xj = x0.copy()
        self.results.loc[0] = self.obtain_fx_and_dfx_norm(function, xj)
        i = 1
        while(True):
            xi = xj.copy()
            di = solve_system(function.hessian(xi),
                              function.gradient(xi))
            alpha = self.obtain_alpha.bisection(function, xi, -di)
            xj = xi-alpha*di
            self.results.loc[i] = self.obtain_fx_and_dfx_norm(function, xj)
            if self.stop_functions.functions(function, xi, xj):
                break
            if self.stop_functions.vectors(xi, xj):
                break
            if self.stop_functions.gradient(function, xj):
                break
            i += 1

    def descent_gradient(self, function: functions_class, x0: np.array):
        xj = x0.copy()
        self.results.loc[0] = self.obtain_fx_and_dfx_norm(function, xj)
        i = 1
        while(True):
            xi = xj.copy()
            gradient = function.gradient(xi)
            xj = xi - self.parameters["alpha"]*gradient
            self.results.loc[i] = self.obtain_fx_and_dfx_norm(function, xj)
            if self.stop_functions.functions(function, xi, xj):
                break
            if self.stop_functions.vectors(xi, xj):
                break
            if self.stop_functions.gradient(function, xj):
                break
            i += 1

    def obtain_fx_and_dfx_norm(self, function: functions_class, x: np.array) -> tuple:
        fx = function.f(x)
        dfx = function.gradient(x)
        dfx = np.linalg.norm(dfx)
        return [fx, dfx]


class stop_functios_class:
    def __init__(self, tau: float = 1e-12) -> None:
        self.tau = tau

    def vectors(self, vector_i: np.array, vector_j: np.array) -> bool:
        up = np.linalg.norm((vector_i-vector_j))
        down = max(np.linalg.norm(vector_j), 1)
        dot = abs(up/down)
        if dot < self.tau:
            return True
        return False

    def functions(self, function: functions_class, xi: np.array, xj: np.array) -> bool:
        fxi = function.f(xi)
        fxj = function.f(xj)
        if abs(fxi-fxj) < self.tau:
            return True
        return False

    def gradient(self, function: functions_class, xi: np.array) -> bool:
        dfx = function.gradient(xi)
        dfx = np.linalg.norm(dfx)
        if dfx < self.tau:
            return True
        return False


class problem_class:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.function = functions_class(parameters["problem name"])
        if parameters["problem name"] == "wood":
            self.use_wood()
        if parameters["problem name"] == "rosembrock":
            self.use_rosembrock(parameters)
        if parameters["problem name"] == "lambda":
            self.use_lambda_function(parameters)
        if parameters["initial point"] == "random":
            self.random_initial_points(parameters)

    def use_wood(self) -> None:
        self.x0 = [-1, -3, -1, -3]

    def use_rosembrock(self, parameters: dict) -> None:
        self.x0 = np.ones(parameters["n"])
        self.x0[0] = -1.2
        self.x0[parameters["n"]-2] = -1.2

    def random_initial_points(self, parameters: dict):
        if parameters["problem name"] == "wood":
            parameters["n"] = 4
        self.x0 = 0.5*np.random.randn(parameters["n"])

    def solve(self):
        self.algorithm = algorithm_class(self.parameters)
        self.algorithm.method(self.function,
                              self.x0)

    def save(self):
        filename = obtain_filename(self.parameters)
        self.algorithm.results.to_csv(join_path(self.parameters["path results"],
                                                filename))


class obtain_alpha():
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters

    def bisection(self, function: functions_class, x: np.array, d: np.array) -> float:
        alpha = 0
        alpha_i = 1
        beta = np.inf
        dfx = function.gradient(x)
        dot_grad = np.linalg.multi_dot((dfx, d))
        while(True):
            armijo_condition = obtain_armijo_conditon(function,
                                                      dot_grad,
                                                      x,
                                                      d,
                                                      alpha_i,
                                                      self.parameters["c1"])
            wolfe_condition = obtain_wolfe_condition(function,
                                                     x,
                                                     dot_grad,
                                                     d,
                                                     self.parameters["c2"],
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


def solve_system(matrix: np.array, vector: np.array) -> np.array:
    solution = np.linalg.solve(matrix, vector)
    return solution


def obtain_armijo_conditon(function: functions_class, dot_grad: float, x: np.array, d: np.array, alpha: float, c1: float):
    fx_alpha = function.f(x+alpha*d)
    fx_alphagrad = function.f(x)+c1*alpha*dot_grad
    armijo_condition = fx_alpha > fx_alphagrad
    return armijo_condition


def obtain_wolfe_condition(function: functions_class, x: np.array, dot_grad: float, d: np.array, c2: float, alpha: float):
    dfx_alpha = function.gradient(x+alpha*d)
    dfx_alpha = np.linalg.multi_dot((dfx_alpha, d))
    wolfe_condition = dfx_alpha < c2*dot_grad
    return wolfe_condition
