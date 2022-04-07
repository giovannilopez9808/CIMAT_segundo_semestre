from numpy.linalg import solve, norm
from numpy import array, sqrt
from typing import Callable


class step_model:
    def __init__(self) -> None:
        pass

    def select_model(self, model_name: str) -> Callable:
        """
        Seleccion del metodo de busqueda de paso

        Inputs
        ----------------
        model_name -> nombre del metodo de busqueda de paso seleccionado
        """
        if model_name == "dogleg":
            self.model = self.dogleg
        if model_name == "cauchy":
            self.model = self.cauchy
        if model_name == "newton":
            self.model = self.newton

    def dogleg(self, g_k, h_k, delta_k) -> array:
        """
        Calcula el tamaño de paso de acuerdo a dogleg

        Inputs
        -----------
        g_k -> Gradiente de la función en la iteración k
        h_k -> Hessiano de la función en la itetación k
        delta_k -> radio de tolerancia para región de confianza

        Output
        -----------
        p_k -> paso calculado por dogleg
        """
        # Modelo cuadrático a lo largo del gradiente
        p_ku = - g_k.dot(g_k) * g_k / (g_k.dot(h_k).dot(g_k))
        # Modelo cuadrático si h_k es positiva definida
        p_kb = self.newton_step(h_k, g_k)
        # p_kb está en el interior de la región de confianza
        if norm(p_kb) <= delta_k:
            p_k = p_kb
        # Intersección con la región de confianza con la trajectoria Dogleg
        elif norm(p_ku) >= delta_k:
            p_k = delta_k * p_ku / norm(p_ku)
        else:
            diff = p_kb - p_ku
            a = diff.dot(diff)
            b = 2.0 * p_kb.dot(diff)
            c = p_ku.dot(p_ku) - delta_k ** 2
            tau_k = 1.0 + (-b + sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
            # 1 <= tau <= 1
            if tau_k <= 1.0:
                p_k = tau_k * p_ku
            # 1 < tau <=2
            else:
                p_k = p_ku + (tau_k - 1.0) * diff
        return p_k

    def cauchy(self, g_k, h_k, delta_k) -> array:
        """
        Inputs
        -----------
        g_k -> Gradiente de la función en la iteración k
        h_k -> Hessiano de la función en la itetación k
        delta_k -> radio de tolerancia para región de confianza

        Output
        -----------
        p_k -> paso calculado por cauchy
        """
        # Resuelve problema lineal
        p_ks = - delta_k * g_k / norm(g_k)
        prod = g_k.dot(h_k).dot(g_k)
        # Se supone el mayor
        tau_k = 1.0
        if prod > 0:
            # Tomo tau en el interior
            tau_k = min(1.0, norm(g_k) ** 3 / (delta_k * prod))
        p_ks = tau_k*p_ks
        return p_ks

    def newton(self, g_k, h_k, delta_k):
        """
        Inputs
        -----------
        g_k -> Gradiente de la función en la iteración k
        h_k -> Hessiano de la función en la itetación k
        delta_k -> radio de tolerancia para región de confianza

        Output
        -----------
        p_k -> paso calculado por newton
        """
        p_k = self.newton_step(h_k, g_k)
        # Tomo paso de Cauchy si paso de Newton está fuera de la region
        if norm(p_k) >= delta_k:
            p_k = self.cauchy(g_k, h_k, delta_k)
        return p_k

    def newton_step(self, h_k, g_k):
        """
        Calculo del tamaño de paso

        Inputs
        ----------
        h_k -> valor del hessiano en la posicion actual
        g_k -> valor del gradiente en la posicon actual

        Output
        ----------
        p_k -> paso calculado usando el metodo de newton
        """
        p_k = solve(h_k, -g_k)
        return p_k


class stop_model:
    def __init__(self, params: dict) -> None:
        self.tau_vector = params["tau vector"]
        self.tau_function = params["tau function"]
        self.tau_gradient = params["tau gradient"]

    def vector(self, xi: array, x: array) -> bool:
        """
        Comparación con el vector

        Inputs
        -----------
        xi -> posicion anterior
        x -> posicion actual

        Outputs
        -----------
        Booleano que indica si se cumple la condicion o no
        """
        return norm(xi - x) / max(norm(x), 1.0) < self.tau_vector

    def function(self, fi: float, f: float) -> bool:
        """
        Comparación con el vector

        Inputs
        -----------
        fi -> valor de la función anterior
        f -> valor de la funcion actual

        Outputs
        -----------
        Booleano que indica si se cumple la condicion o no
        """
        return abs(fi - f) / max(abs(fi), 1.0) < self.tau_function

    def gradient(self, g: array) -> bool:
        """
        Comparación con el vector

        Inputs
        -----------
        g -> gradiente de la función en la posicion actual

        Outputs
        -----------
        Booleano que indica si se cumple la condicion o no
        """
        return norm(g) < self.tau_gradient
