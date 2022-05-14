from numpy import array


class GC_methods:
    """
    Contiene el algortimo para obtener la beta en un método de gradiente conjugado
    """

    def __init__(self, params: dict) -> None:
        self.params = params
        self._select_method()

    def _select_method(self) -> None:
        """
        Seleccion del metodo a utilizar para obtener la beta
        """
        if self.params["GC method"] == "FR":
            self.method = self._FR
        if self.params["GC method"] == "PR":
            self.method = self._PR
        if self.params["GC method"] == "HS":
            self.method = self._HS
        if self.params["GC method"] == "FR PR":
            self.method = self._FR_PR

    def run(self, direction: array, gradient: array) -> float:
        """
        Estandarización del calculo de la beta para una direccion y gradientes dado

        Inputs
        ----------
        direction: vector con la direccion de descenso
        gradiente: lista que contiene los gradientes en el paso actual y el anterior

        Outputs
        ----------
        beta: valor de la beta para la posicion actual
        """
        # Gradiente anterior y actual
        g_i, g_j = gradient
        # Calculo de la beta
        beta = self.method(direction,
                           [g_i, g_j])
        return beta

    def _FR(self, direction: array, gradient: array) -> float:
        """
        Calculo de la beta usando la definición dada por Fletcher-Reeves

        Inputs
        ----------
        direction: vector con la direccion de descenso (estandarizacion con las otras funciones)
        gradiente: lista que contiene los gradientes en el paso actual y el anterior

        Outputs
        ----------
        beta: valor de la beta para la posicion actual
        """
        g_i, g_j = gradient
        beta = (g_j@g_j) / (g_i@g_i)
        return beta

    def _PR(self, direction: array, gradient: array) -> float:
        """
        Calculo de la beta usando la definición dada por Polak-Ribiere

        Inputs
        ----------
        direction: vector con la direccion de descenso (estandarizacion con los demas metodos)
        gradiente: lista que contiene los gradientes en el paso actual y el anterior

        Outputs
        ----------
        beta: valor de la beta para la posicion actual
        """
        g_i, g_j = gradient
        delta = g_i-g_j
        beta = (g_j@delta) / (g_i@g_i)
        # beta = max(0, beta)
        return beta

    def _HS(self, direction: array, gradient: array) -> float:
        """
        Calculo de la beta usando la definición dada por Hestenes-Stiefel

        Inputs
        ----------
        direction: vector con la direccion de descenso (estandarizacion con los demas metodos)
        gradiente: lista que contiene los gradientes en el paso actual y el anterior

        Outputs
        ----------
        beta: valor de la beta para la posicion actual
        """
        g_i, g_j = gradient
        beta = (g_j@(g_j-g_i)) / ((g_j-g_i) @ direction)
        return beta

    def _FR_PR(self, direction: array, gradient: array) -> float:
        """
        Calculo de la beta usando la definición dada por Fletcher-Reeves Polak-Ribiere

        Inputs
        ----------
        direction: vector con la direccion de descenso (estandarizacion con los demas metodos)
        gradiente: lista que contiene los gradientes en el paso actual y el anterior

        Outputs
        ----------
        beta: valor de la beta para la posicion actual
        """
        beta_fr = self._FR(direction,
                           gradient)
        beta_pr = self._PR(direction,
                           gradient)
        if beta_pr < -beta_fr:
            return -beta_fr
        if abs(beta_pr) <= beta_fr:
            return beta_pr
        return beta_fr
