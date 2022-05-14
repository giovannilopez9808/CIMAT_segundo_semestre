from numpy import array, vstack, sqrt, sum, divide
from matplotlib.pyplot import imread
from pandas import read_csv
from os.path import join


def read_image(params: dict) -> array:
    filename = join(params["path data"],
                    params["image name"])
    image = imread(filename)
    image = array(image)
    image = image.flatten()
    return image


def read_image_result(filename: str) -> array:
    image = read_csv(filename)
    image = image.to_numpy()
    return image


class function_class:
    def __init__(self) -> None:
        pass

    def f(self, x: array, params: dict) -> array:
        lamb = params['lambda']
        g = params['g']
        n = params['n']
        # Evaluación
        diff = (x - g) ** 2
        neighbours = lamb * self._neighbors_f(x, n)
        f = sum(diff + neighbours)
        return f

    def gradient(self, x: array, params: dict) -> array:
        lamb = params['lambda']
        g = params['g']
        n = params['n']
        grad = 2*(x - g)
        grad += 2*lamb * self._neighbors_grad(x, n)
        return grad

    def _neighbors_f(self, matrix: array, n: int) -> array:
        matrix = matrix.reshape((n, n))
        up = (matrix - vstack((matrix[0],
                               matrix[0:-1])))
        down = (matrix - vstack((matrix[1:],
                                 matrix[n-1])))
        left = (matrix - vstack((matrix.T[0],
                                 matrix.T[0:-1])).T)
        right = (matrix - vstack((matrix.T[1:],
                                  matrix.T[n-1])).T)
        up = self._sqrt_function(up)
        down = self._sqrt_function(down)
        left = self._sqrt_function(left)
        right = self._sqrt_function(right)
        result = up + down + left + right
        return result.flatten()

    def _neighbors_grad(self, matrix, n):
        matrix = matrix.reshape((n, n))
        up = matrix - vstack((matrix[0],
                              matrix[0:-1]))
        down = matrix - vstack((matrix[1:],
                                matrix[n-1]))
        left = matrix - vstack((matrix.T[0],
                                matrix.T[0:-1])).T
        right = matrix - vstack((matrix.T[1:],
                                 matrix.T[n-1])).T
        up = divide(up,
                    self._sqrt_function(up))
        down = divide(down,
                      self._sqrt_function(down))
        left = divide(left,
                      self._sqrt_function(left))
        right = divide(right,
                       self._sqrt_function(right))
        result = up + down + left + right
        return result.flatten()

    def _sqrt_function(self, value: array) -> array:
        mu = 0.01
        f = value**2 + mu
        f = sqrt(f)
        return f
