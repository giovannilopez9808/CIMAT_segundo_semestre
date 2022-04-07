from .params import get_params, get_function_params
from .models import step_model, stop_model
from .functions import functions_class
from numpy.random import uniform
from numpy.linalg import norm
from pandas import DataFrame
from numpy import array
from time import time


class problem_model:
    def __init__(self, dataset: dict) -> None:
        self.function_params = get_function_params(dataset["function"])
        self.function = functions_class(dataset["function"])
        self.params = get_params()
        self._inital_position()
        self.step = step_model()
        self.step.select_model(dataset["step model"])
        self.stop = stop_model(self.function_params)

    def _inital_position(self) -> array:
        """
        Perturbación a la posicion optima de la funcion con una funcion aleatoria U(-2,2)
        """
        x = uniform(-2, 2, self.function_params["n"])
        x = self.function_params["optimal position"] + x
        self.function_params["x"] = x.copy()

    def run(self) -> DataFrame:
        """
        Ejecuccion de la optimizacion de la funcion dada
        """
        # Inicializacion del conteo de pasos
        k = 1
        # Inicializacion del guardado de resultados
        results = DataFrame(columns=["Function",
                                     "Gradient",
                                     "Time"])
        results.index.name = "Iterations"
        # Inicializacion de los parametros para el algoritmo
        function_params = self.function_params
        function = self.function.f
        gradient = self.function.gradient
        hessian = self.function.hessian
        eta = function_params["eta"]
        eta_min = function_params["eta min"]
        eta_max = function_params["eta max"]
        eta1_hat = function_params["eta1 hat"]
        eta2_hat = function_params["eta2 hat"]
        delta_k = function_params["delta k"]
        delta_max = function_params["delta max"]
        # paso inicial
        x_k = function_params["x"]
        # calculo de la funcion en el paso inicial
        f_k = function(x_k, function_params)
        # calculo del gradiente en el paso inicial
        g_k = gradient(x_k, function_params)
        # calculo del hessiano en el paso inicial
        h_k = hessian(x_k, function_params)
        # Guardado del estado inicial
        results.loc[0] = [f_k, norm(g_k), 0]
        time_start = time()
        while True:
            # Copia el paso anterior
            x_i = x_k.copy()
            f_i = f_k
            p_k = self.step.model(g_k, h_k, delta_k)
            # print(f_k)
            # Calculo métrica ro para evaluar modelo
            m_kp = f_k + g_k.dot(p_k) + 0.5 * p_k.dot(h_k).dot(p_k)
            ro_k = f_k - function(x_k + p_k, function_params)
            ro_k = ro_k / (f_k - m_kp)
            # Actualizo radio de la región de confianza
            if ro_k < eta_min:
                delta_k = eta1_hat * delta_k
            elif ro_k > eta_max and abs((norm(p_k)-delta_k)) < 1e-6:
                delta_k = min(eta2_hat * delta_k, delta_max)
            # Actualizo valor del punto
            if ro_k > eta:
                x_k = x_k + p_k
            # Calculo valores de f, gradiente y Hessiano
            f_k = function(x_k, function_params)
            g_k = gradient(x_k, function_params)
            h_k = hessian(x_k, function_params)
            results.loc[k] = [f_k,
                              norm(g_k),
                              time()-time_start]
            # Actualizo contador
            k = k + 1
            if self.stop.gradient(g_k):
                break
            if self.stop.vector(x_i, x_k):
                break
            if self.stop.function(f_i, f_k):
                break
        return results
