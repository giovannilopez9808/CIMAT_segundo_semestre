from numpy.linalg import solve, norm, eigvals, cholesky, LinAlgError
from numpy import array, eye, sqrt, inf
from typing import Callable


class step_model:
    def __init__(self, function: Callable, params: dict) -> None:
        self.obtain_alpha = obtain_alpha(function, params)

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
        if model_name == "newton-cauchy":
            self.model = self.newton_cauchy
        if model_name == "newton-modification":
            self.model = self.newton_modification

    def dogleg(self, x_k: array, g_k: array, h_k: array, delta_k: float) -> array:
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
        p_kb = self._newton_step(h_k, g_k)
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

    def cauchy(self, x_k: array, g_k: array, h_k: array, delta_k: float) -> array:
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

    def newton_cauchy(self, x_k: array, g_k: array, h_k: array, delta_k: float) -> array:
        """
        Inputs
        -----------
        g_k -> Gradiente de la función en la iteración k
        h_k -> Hessiano de la función en la itetación k
        delta_k -> radio de tolerancia para región de confianza

        Output
        -----------
        p_k -> paso calculado por newton_cauchy
        """
        p_k = self._newton_step(h_k, g_k)
        # Tomo paso de Cauchy si paso de newton_cauchy está fuera de la region
        if norm(p_k) >= delta_k:
            p_k = self.cauchy(x_k, g_k, h_k, delta_k)
        return p_k

    def newton_modification(self, x_k: array, g_k: array, h_k: array, delta_k: float) -> array:
        l_matrix = self._cholesky_modification(h_k)
        # l_matrix = h_k
        d_k = self._newton_step(l_matrix, g_k)
        d_k = solve(l_matrix.T, d_k)
        alpha_k = self.obtain_alpha.method(x_k, d_k)
        p_k = alpha_k*d_k
        return p_k

    def _cholesky_modification(self, h_k: array) -> array:
        beta = 1e-3
        n = h_k.shape[0]
        eigenvalues = eigvals(h_k)
        min_eigenvalue = min(eigenvalues)
        tau = 0
        if min_eigenvalue < 0:
            tau = beta - min_eigenvalue
        while(True):
            b_k = h_k + tau*eye(n)
            try:
                l = cholesky(b_k)
                return l
            except LinAlgError:
                tau = max(2*tau, beta)

    def _newton_step(self, h_k, g_k) -> array:
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


class obtain_alpha():
    """
    Obtiene el alpha siguiendo las condiciones de armijo y Wolfe
    """

    def __init__(self, function: Callable, params: dict) -> None:
        self.params = params
        self.function = function
        if params["search name"] == "bisection":
            self.method = self.bisection
        if params["search name"] == "back tracking":
            self.method = self.back_tracking

    def bisection(self,  x: array, d: array) -> float:
        # Inicialización
        alpha = 0.0
        beta_i = inf
        alpha_k = 1
        dot_grad = self.function.gradient(x, self.params) @ d
        while True:
            armijo_condition = self.obtain_armijo_condition(
                dot_grad, x, d, alpha_k)
            wolfe_condition = self.obtain_wolfe_condition(
                x, dot_grad, d, alpha_k)
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

    def back_tracking(self, x: array,  d: array):
        """
        Calcula tamaño de paso alpha

            Parámetros
            -----------
                x_k     : Vector de valores [x_1, x_2, ..., x_n]
                d_k     : Dirección de descenso
                f       : Función f(x)
                f_grad  : Función que calcula gradiente
                alpha   : Tamaño inicial de paso
                ro      : Ponderación de actualización
                c1      : Condición de Armijo
            Regresa
            -----------
                alpha_k : Tamaño actualizado de paso
        """
        # Inicialización
        alpha_k = self.params["alpha"]
        dot_grad = (-self.function.gradient(x, self.params)) @ d
        # Repetir hasta que se cumpla la condición de armijo
        while True:
            armijo_condition = self.obtain_armijo_condition(
                dot_grad, x,  d, alpha_k)
            if armijo_condition:
                alpha_k = self.params["rho"] * alpha_k
            else:
                break
        return alpha_k

    def obtain_armijo_condition(self,  dot_grad: float, x: array, d: array, alpha: float):
        """
        Condicion de armijo
        """
        fx_alpha = self.function.f(x+alpha*d, self.params)
        fx_alphagrad = self.function.f(x, self.params) + \
            self.params["c1"]*alpha*dot_grad
        armijo_condition = fx_alpha > fx_alphagrad
        return armijo_condition

    def obtain_wolfe_condition(self,  x: array, dot_grad: float, d: array, alpha: float):
        """
        Condicion de Wolfe
        """
        dfx_alpha = self.function.gradient(x+alpha*d, self.params)
        dfx_alpha = dfx_alpha @ d
        wolfe_condition = dfx_alpha < self.params["c2"]*dot_grad
        return wolfe_condition
