import numpy as np


class functions:
    def __init__(self, name: str) -> None:
        if name == "wood":
            self.f = self.f_wood
            self.gradient = self.gradient_wood
            self.hessian = self.hessian_wood
        if name == "rosembrock":
            self.f = self.f_rosembrock
            self.gradient = self.gradient_rosembrock
            self.hessian = self.hessian_rosembrock

    def f_wood(self, x: np.array) -> float:
        fx = 100*(x[0]*x[0]-x[1])*(x[0]*x[0]-x[1])
        fx += (x[0]-1)*(x[0]-1)+(x[2]-1)*(x[2]-1)
        fx += 90*(x[2]*x[2]-x[3])*(x[2]*x[2]-x[3])
        fx += 10.1*((x[1]-1)*(x[1]-1)+(x[3]-1)*(x[3]-1))
        fx += 19.8*(x[1]-1)*(x[3]-1)
        return fx

    def gradient_wood(self, x: np.array) -> np.array:
        n = len(x)
        g = np.zeros(n)
        g[0] = 400*(x[0]*x[0]-x[1])*x[1] + 2*(x[0]-1)
        g[1] = -200*(x[0]*x[0]-x[1])+20.2*(x[1]-1)+19.8*(x[3]-1)
        g[2] = 2*(x[2]-1)+360*(x[2]*x[2]-x[3])*x[2]
        g[3] = -180*(x[2]*x[2]-x[3])+20.2*(x[3]-1)+19.8*(x[1]-1)
        return g

    def hessian_wood(self, x: np.array) -> np.array:
        n = len(x)
        h = np.zeros((n, n))
        h[0, 0] = 400*(x[0]*x[0]-x[1])+800*x[0]*x[0]+2
        h[1, 0] = -400*x[0]
        h[0, 1] = -400*x[0]
        h[1, 1] = 220.2
        h[1, 3] = 19.8
        h[3, 1] = 19.8
        h[2, 2] = 2+720*x[2]*x[2]+360*(x[2]*x[2]-x[3])
        h[2, 3] = -360*x[2]
        h[3, 2] = -360*x[2]
        h[3, 3] = 200.2
        return h

    def f_rosembrock(self, x: np.array) -> float:
        n = len(x)
        fx = 0
        for i in range(n-1):
            fx += 100*(x[i+1]-x[i]*x[i])*(x[i+1]-x[i]*x[i])
            fx += (1-x[i])*(1-x[i])
        return fx

    def gradient_rosembrock(self, x: np.array) -> np.array:
        n = len(x)
        g = np.zeros(n)
        for i in range(n-1):
            g[i] = -400*x[i]*(x[i+1]-x[i]*x[i])-2*(1-x[i])
        i = n-2
        g[i+1] = 200*(x[i+1]-x[i]*x[i])
        return g

    def hessian_rosembrock(self, x: np.array) -> np.array:
        n = len(x)
        h = np.zeros((n, n))
        for i in range(n-1):
            h[i, i] = 1200 * x[i] * x[i] - 400 * x[i+1] + 2
            h[i+1, i] = -400*x[i]
            h[i, i+1] = -400*x[i]
        h[n-1, n-1] = 200*(n-1)
        return h


class algorithm_class:
    def __init__(self, parameters: dict) -> None:
        self.stop_functios = stop_functios_class(parameters["tau"])

    def descent_gradient(self, function: functions, x0: np.array, parameters: dict):
        xj = x0.copy()
        while(True):
            xi = xj.copy()
            gradient = function.gradient(xi)
            xj = xi - parameters["alpha"]*gradient
            if self.stop_functios.with_functions(function, xi, xj):
                break
            if self.stop_functios.with_vectors(xi, xj):
                break
        print(xj)


class stop_functios_class:
    def __init__(self, tau: float = 1e-12) -> None:
        self.tau = tau

    def with_vectors(self, vector_i: np.array, vector_j: np.array) -> bool:
        dot = np.linalg.multi_dot((vector_i, vector_j))
        if dot < self.tau:
            return True
        return False

    def with_functions(self, function: functions, xi: np.array, xj: np.array):
        fxi = function.f(xi)
        fxj = function.f(xj)
        if abs(fxi-fxj) < self.tau:
            return True
        return False


class problem_class:
    def __init__(self, parameters: dict) -> None:
        self.algorithm = algorithm_class(parameters)
        if "wood" == parameters["name"]:
            self.use_wood(parameters)
        if "rosembrock" == parameters["name"]:
            self.use_rosembrock(parameters)

    def use_wood(self, parameters: dict) -> None:
        self.x0 = [-1, -3, -1, -3]
        self.function = functions(parameters["name"])
        self.algorithm.descent_gradient(self.function,
                                        self.x0,
                                        parameters)

    def use_rosembrock(self, parameters: dict) -> None:
        self.x0 = np.ones(parameters["n"])
        self.x0[0] = -1.2
        self.x0[parameters["n"]-2] = -1.2
        self.function = functions(parameters["name"])
        self.algorithm.descent_gradient(self.function,
                                        self.x0,
                                        parameters)


parameters = {"n": 10,
              "tau": 1e-12,
              "alpha": 1e-4,
              "name": "wood"}

problem = problem_class(parameters)
