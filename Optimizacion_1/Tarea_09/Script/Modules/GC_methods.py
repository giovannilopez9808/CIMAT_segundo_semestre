from numpy import array


class GC_methods:
    def __init__(self, params: dict) -> None:
        self.params = params
        self._select_method()

    def _select_method(self) -> None:
        if self.params["GC method"] == "FR":
            self.method = self._FR
        if self.params["GC method"] == "PR":
            self.method = self._PR
        if self.params["GC method"] == "HS":
            self.method = self._HS
        if self.params["GC method"] == "FR PR":
            self.method = self._FR_PR

    def run(self, direction: array, gradient: array) -> float:
        g_i, g_j = gradient
        g_i = g_i.flatten()
        g_j = g_j.flatten()
        beta = self.method(direction,
                           [g_i, g_j])
        return beta

    def _FR(self, direction: array, gradient: array) -> float:
        g_i, g_j = gradient
        beta = g_j@g_i / g_i@g_i
        return beta

    def _PR(self, direction: array, gradient: array) -> float:
        g_i, g_j = gradient
        beta = g_j@(g_j-g_i) / g_i@g_i
        return beta

    def _HS(self, direction: array, gradient: array) -> float:
        g_i, g_j = gradient
        beta = g_j@(g_j-g_i) / (g_j-g_i) @ direction
        return beta

    def _FR_PR(self, direction: array, gradient: array) -> float:
        beta_fr = self._FR(direction,
                           gradient)
        beta_pr = self._PR(direction,
                           gradient)
        if beta_pr < -beta_fr:
            return -beta_fr
        if abs(beta_pr) <= beta_fr:
            return beta_pr
        return beta_fr
