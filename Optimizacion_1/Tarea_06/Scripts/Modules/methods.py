from .functions import functions_class, join_path, obtain_filename
from numpy.linalg import norm, solve
from numpy import array, ones, inf
from .mnist import mnist_model
from numpy.linalg import solve
from pandas import DataFrame


class problem_class:
    """
    Orquesta todos los metodos dadas una serie de parametros
    """

    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        # Eleccion de la funcion
        self.function = functions_class(parameters["problem name"])
        if parameters["problem name"] == "log likehood":
            self.use_log_likehood()

    def use_log_likehood(self) -> None:
        """
        Vector inicial predefinido de log likehood
        """
        mnist = mnist_model(self.parameters)
        self.beta = ones(785)
        self.x = mnist.train_data
        self.y = mnist.train_label

    def solve(self):
        """
        Orquesta la solucion del problema
        """
        self.algorithm = algorithm_class(self.parameters)
        self.algorithm.method(self.function,
                              self.beta,
                              self.x,
                              self.y)

    def save(self):
        """
        Guardado de los resultados en un archivo
        """
        filename = obtain_filename(self.parameters)
        self.algorithm.results.to_csv(join_path(self.parameters["path results"],
                                                filename))
        filename = "beta_{}".format(filename)
        self.algorithm.beta_j.tofile(join_path(self.parameters["path results"],
                                               filename),
                                     ",")


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
        self.method = self.descent_gradient

    def descent_gradient(self, function: functions_class, beta: array, x: array, y: array):
        """
        Metodo del descenso del gradiente
        """
        # Inicializacion del vector de resultado
        self.beta_j = beta.copy()
        # Guardado de la primer posicion
        self.results.loc[0] = self.obtain_fx_and_dfx_norm(function,
                                                          x,
                                                          y,
                                                          self.beta_j)
        i = 0
        while(True):
            # Guardado del paso anterior
            beta_i = self.beta_j.copy()
            # Calculo del gradiente en el paso i
            gradient = -function.gradient(x, y, beta_i)
            # Siguiente paso
            alpha = self.obtain_alpha.method(
                function, x, y, beta_i, gradient)
            self.beta_j = beta_i + alpha * gradient
            # Guardado de los resultados
            self.results.loc[i] = self.obtain_fx_and_dfx_norm(
                function,
                x, y,
                self.beta_j)
            if self.stop_functions.gradient(
                    function,
                    x,
                    y,
                    self.beta_j):
                break
            i += 1

    def obtain_fx_and_dfx_norm(self, function: functions_class, x: array, y: array, beta: array) -> tuple:
        """
        Calculo de f(x) y ||gradient(f(x))|
        """
        fx = function.f(x, y, beta)
        dfx = function.gradient(x, y, beta)
        dfx = norm(dfx)
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
        up = norm((vector_i-vector_j))
        down = max(norm(vector_j), 1)
        dot = abs(up/down)
        if dot < self.tau:
            return True
        return False

    def functions(self, function: functions_class, x: array, y: array, beta_i: array, beta_j: array) -> bool:
        """
        Comprueba la diferencia de las funciones evaluadas en la posicion actual y la anterior
        """
        fbeta_i = function.f(x, y, beta_i)
        fbeta_j = function.f(x, y, beta_j)
        if abs(fbeta_i-fbeta_j) < self.tau:
            return True
        return False

    def gradient(self, function: functions_class, x: array, y: array, beta_i: array) -> bool:
        """
        Compueba si la norm del gradiente se acerca a cero
        """
        dfx = function.gradient(x, y, beta_i)
        dfx = norm(dfx)
        if dfx < self.tau:
            return True
        return False


class obtain_alpha():
    """
    Obtiene el alpha siguiendo las condiciones de armijo y Wolfe
    """

    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        if parameters["search name"] == "bisection":
            self.method = self.bisection

    def bisection(self, function: functions_class, x: array, y: array, beta: array, d: array) -> float:
        # Inicialización
        alpha = 0.0
        beta_i = inf
        alpha_k = 1
        dot_grad = function.gradient(x, y, beta) @ d
        while True:
            armijo_condition = self.obtain_armijo_condition(
                function, dot_grad, x, y, beta, d, alpha_k)
            wolfe_condition = self.obtain_wolfe_condition(
                function, x, y, beta, dot_grad, d, alpha_k)
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

    def obtain_armijo_condition(self, function: functions_class, dot_grad: float, x: array, y: array, beta: array, d: array, alpha: float):
        """
        Condicion de armijo
        """
        fx_alpha = function.f(x, y, beta+alpha*d)
        fx_alphagrad = function.f(x, y, beta) + \
            self.parameters["c1"]*alpha*dot_grad
        armijo_condition = fx_alpha > fx_alphagrad
        return armijo_condition

    def obtain_wolfe_condition(self, function: functions_class, x: array, y: array, beta: array, dot_grad: float, d: array, alpha: float):
        """
        Condicion de Wolfe
        """
        dfx_alpha = function.gradient(x, y, beta+alpha*d)
        dfx_alpha = dfx_alpha @ d
        wolfe_condition = dfx_alpha < self.parameters["c2"]*dot_grad
        return wolfe_condition


def solve_system(matrix: array, vector: array) -> array:
    """
    solucion al sistema de ecuaciones
    """
    solution = solve(matrix, vector)
    return solution
