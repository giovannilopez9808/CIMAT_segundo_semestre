from functions import *
import pandas as pd
import numpy as np


class algorithm_class:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.results = pd.DataFrame(columns=["fx", "dfx"])
        self.results.index.name = "iteration"
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
            xj = xi-self.parameters["alpha"]*di
            self.results.loc[i] = self.obtain_fx_and_dfx_norm(function, xj)
            if self.stop_functions.with_functions(function, xi, xj):
                break
            if self.stop_functions.with_vectors(xi, xj):
                break
            i += 1
            print(xj)

    def descent_gradient(self, function: functions_class, x0: np.array):
        xj = x0.copy()
        self.results.loc[0] = self.obtain_fx_and_dfx_norm(function, xj)
        i = 1
        while(True):
            xi = xj.copy()
            gradient = function.gradient(xi)
            xj = xi - self.parameters["alpha"]*gradient
            self.results.loc[i] = self.obtain_fx_and_dfx_norm(function, xj)
            if self.stop_functions.with_functions(function, xi, xj):
                break
            if self.stop_functions.with_vectors(xi, xj):
                break
            i += 1
            print(xj)

    def obtain_fx_and_dfx_norm(self, function: functions_class, x: np.array) -> tuple:
        fx = function.f(x)
        dfx = function.gradient(x)
        dfx = np.linalg.norm(dfx)
        return [fx, dfx]


class stop_functios_class:
    def __init__(self, tau: float = 1e-12) -> None:
        self.tau = tau

    def with_vectors(self, vector_i: np.array, vector_j: np.array) -> bool:
        dot = np.linalg.multi_dot((vector_i, vector_j))
        if dot < self.tau:
            return True
        return False

    def with_functions(self, function: functions_class, xi: np.array, xj: np.array):
        fxi = function.f(xi)
        fxj = function.f(xj)
        if abs(fxi-fxj) < self.tau:
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
        self.x0 = np.random.random(parameters["n"])

    def solve(self):
        self.algorithm = algorithm_class(self.parameters)
        self.algorithm.method(self.function,
                              self.x0)


def solve_system(matrix: np.array, vector: np.array) -> np.array:
    solution = np.linalg.solve(matrix, vector)
    return solution
