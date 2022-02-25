from numpy.random import randn as normal
from numpy import array, zeros, loadtxt


class functions_lambda:
    def __init__(self, parameters: dict) -> None:
        self.lambda_i = parameters["lambda"]
        self.create_y_vector(parameters["n"],
                             parameters["sigma"],
                             parameters["test data"])

    def f(self, x: array) -> array:
        n = len(x)
        fx = 0
        for i in range(n-1):
            fx += (x[i]-self.y[i])*(x[i]-self.y[i])
            fx += self.lambda_i*(x[i+1]-x[i])*(x[i+1]-x[i])
        fx += (x[n-1]-self.y[n-1])*(x[n-1]-self.y[n-1])
        return fx

    def gradient(self, x: array) -> array:
        n = len(x)
        g = zeros(n)
        i = 0
        g[i] += 2*(x[i]-self.y[i])
        g[i] += -2*self.lambda_i*(x[i+1]-x[i])
        for i in range(1, n-1):
            g[i] += 2*(x[i]-self.y[i])
            g[i] += -2*self.lambda_i*(x[i+1]-x[i])
            g[i] += 2*self.lambda_i*(x[i]-x[i-1])
        g[n-1] += 2*(x[n-1]-self.y[n-1])
        g[n-1] += 2*self.lambda_i*(x[n-1]-x[n-2])
        return g

    def create_y_vector(self, n: int, sigma: float, test_data: bool = False):
        self.t = array([2*(i)/(n-1)-1 for i in range(n)])
        self.y = self.t*self.t + normal(n)*sigma
        if test_data:
            self.y = loadtxt("Data/y.txt", delimiter=",")
            self.y = self.y[:128]
