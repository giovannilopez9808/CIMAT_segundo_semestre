from numpy import exp, ones, mean, ones_like, array
from tabulate import tabulate


def print_results(params: dict, results: array) -> None:
    table = []
    for model_name in params["models"]:
        time = results[model_name]["time"]
        error = results[model_name]["error"]
        table += [[model_name, time, error]]
    print(tabulate(table,
                   headers=["Model",
                            "Time",
                            "Error"]))


class function_class:
    def __init__(self) -> None:
        pass

    def phi(self, y: array, mu: array, sigma: array) -> array:
        """


        Parámetros
        -----------
        y -> Patrones a Aproximar
        mu -> Array de medias
        sigma -> Vector de Desviaciones
        Output
        -----------
        phi          : matriz de kerneles
        """
        mu_aux = mu.reshape(-1, 1)
        phi = exp(-(y-mu_aux)**2/(2*sigma**2))
        return phi

    def gradient_gaussian_mu(self, theta: None, f_params: dict) -> array:
        """
        Calcula el gradiente respecto a mu
        Parámetros
        -----------
        theta
        f_params -> lista de parametros para la funcion objetivo,
        X = f_params['X'] Variable independiente
        y = f_params['y'] Variable dependiente

        Output
        -----------
            Array gradiente
        """
        # Obtengo Parámetros
        phi = f_params['X']
        alpha = f_params['Alpha']
        n = f_params['n']
        y = f_params['y']
        mu = f_params['mu']
        alpha = alpha.reshape((-1, 1))
        mu = mu.reshape((-1, 1))
        y = y.reshape((-1, 1))
        gradient = (phi @ alpha - y) @ alpha.T * \
            (y @ ones((1, n)) - ones_like(y) @ mu.T)
        return mean(gradient, axis=0)

    def gradient_gaussian_alpha(self, theta: None, f_params: dict) -> array:
        """
        Calcula el gradiente respecto a alpha
        Parámetros
        -----------
            theta
            f_params : lista de parametros para la funcion objetivo,
                        X -> f_params['X'] Variable independiente
                        y -> f_params['y'] Variable dependiente

        Output
        -----------
            Array gradiente
        """
        # Obtengo Parámetros
        phi = f_params['X']
        y = f_params['y']
        alpha = f_params['Alpha']
        gradient = phi.T @ (phi @ alpha - y)
        return mean(gradient, axis=0)
