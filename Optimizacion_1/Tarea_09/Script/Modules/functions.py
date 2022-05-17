from numpy import array, vstack, sqrt, sum, divide
from pandas import read_csv
from os.path import join
from cv2 import imread


def read_image(params: dict) -> array:
    """
    Lectura de la imagen y conversión a vector
    """
    # Localizacion de la imagen
    filename = join(params["path data"],
                    params["image name"])
    # Lectura de la imagen
    image = imread(filename, 0)
    # Conversion a array
    image = array(image)/255
    # Vector
    image = image.flatten()
    return image


def read_image_result(filename: str) -> array:
    """
    Lectura del resultado de la optimización
    """
    image = read_csv(filename)
    image = image.to_numpy()
    return image


class function_class:
    """
    Clase que contiene la función y gradiente para realizar el suavizado de una función
    """

    def __init__(self) -> None:
        pass

    def f(self, x: array, params: dict) -> array:
        """
        Función para el suavizado

        Inputs:
        ----------
        x: vector de la imagen
        params: diccionario con las siguientes caracteristicas
            lambda: parametro de la función
            g: funcion a suavizar
            n: dimensión de la función 
        ----------

        Output:
        f: valor de la función en el punto x
        """
        lamb = params['lambda']
        g = params['g']
        n = params['n']
        diff = (x - g) ** 2
        neighbors = lamb * self._neighbors_f(x, n)
        f = sum(diff + neighbors)
        return f

    def gradient(self, x: array, params: dict) -> array:
        """
        Gradiente de la función para el suavizado

        Inputs:
        ----------
        x: vector de la imagen
        params: diccionario con las siguientes caracteristicas
            lambda: parametro de la función
            g: funcion a suavizar
            n: dimensión de la función 
        ----------

        Output:
        grad: valor del gradiente en el punto x
        """
        lamb = params['lambda']
        g = params['g']
        n = params['n']
        grad = 2*(x - g)
        grad += 2*lamb * self._neighbors_grad(x, n)
        return grad

    def _neighbors_f(self, matrix: array, n: int) -> array:
        """
        Calculo de los vecinos en el punto ij de la imagen para la función

        Inputs:
        ----------
        matrix: vector de la imagen
        n: dimensión de la imagen
        ----------

        Output:
        result: vector con la contribución de los vecinos para todos los puntos ij de la imagen
        """
        # Conversion a matriz
        matrix = matrix.reshape((n, n))
        # Vecino superior
        up = (matrix - vstack((matrix[0],
                               matrix[0:-1])))
        # Vecino inferior
        down = (matrix - vstack((matrix[1:],
                                 matrix[n-1])))
        # Vecino izquierdo
        left = (matrix - vstack((matrix.T[0],
                                 matrix.T[0:-1])).T)
        # Vecino derecho
        right = (matrix - vstack((matrix.T[1:],
                                  matrix.T[n-1])).T)
        up = self._sqrt_function(up)
        down = self._sqrt_function(down)
        left = self._sqrt_function(left)
        right = self._sqrt_function(right)
        result = up + down + left + right
        return result.flatten()

    def _neighbors_grad(self, matrix: array, n: int) -> array:
        """
        Calculo de los vecinos en el punto ij de la imagen para el gradiente

        Inputs:
        ----------
        matrix: vector de la imagen
        n: dimensión de la imagen
        ----------

        Output:
        result: vector con la contribución de los vecinos para todos los puntos ij de la imagen
        """
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
        """
        Estandarizacion del cuadrado de la función
        """
        mu = 0.01
        f = value**2 + mu
        f = sqrt(f)
        return f
