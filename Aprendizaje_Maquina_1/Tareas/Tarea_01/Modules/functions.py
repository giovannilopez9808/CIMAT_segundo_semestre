from numpy import exp, ones, array, outer
from tabulate import tabulate
from pandas import DataFrame
from os.path import join


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


def write_results(params: dict, model: str, alpha: array, mu: array) -> None:
    filename = "{}.csv".format(model)
    filename = join(params["path results"],
                    filename)
    data = DataFrame()
    data["alpha"] = alpha
    data["mu"] = mu
    data.to_csv(filename,
                index=False)


class function_class:
    def __init__(self) -> None:
        pass

    def phi(self, x: array, mu: array, sigma: array) -> array:
        """


        Parámetros
        -----------
        x -> Patrones a Aproximar
        mu -> Array de medias
        sigma -> Vector de Desviacionesalpha
        -----------
        phi          : matriz de kerneles
        """
        x = x.reshape((-1, 1))
        mu = mu.reshape((-1, 1))
        phi = exp(-(x-mu.T)**2/(2*sigma**2))
        return phi

    def gradient_gaussian_mu(self, mu: None, f_params: dict) -> array:
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
        x = f_params['x']
        alpha = f_params['alpha']
        n = x.shape
        m = f_params["m"]
        m = alpha.shape
        y = f_params['y']
        # print(n, m)
        sigma = f_params["sigma"]
        phi = self.phi(x, mu, sigma)
        Alpha = outer(ones(n), alpha)
        beta = outer(x, ones(m))-outer(ones(n), mu)
        gradient = (phi*Alpha*beta).T @ (phi@alpha-y)
        gradient = gradient/sigma**2
        return gradient

    def gradient_gaussian_alpha(self, alpha: array, f_params: dict) -> array:
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
        x = f_params['x']
        y = f_params['y']
        mu = f_params["mu"]
        sigma = f_params["sigma"]
        phi = self.phi(x, mu, sigma)
        # gradient = phi.T @ (phi @ alpha - y)
        gradient = phi.T @ phi @ alpha - phi.T @ y
        return gradient
