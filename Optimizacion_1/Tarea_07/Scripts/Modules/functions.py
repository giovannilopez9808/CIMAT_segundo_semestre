from numpy import array, zeros, sin, cos


class functions_class:
    """
    Funciones que se usaran para ser optimizadas
    """

    def __init__(self, name: str) -> None:
        if name == "wood":
            self.f = self.f_wood
            self.gradient = self.gradient_wood
            self.hessian = self.hessian_wood
        if name == "rosembrock":
            self.f = self.f_rosembrock
            self.gradient = self.gradient_rosembrock
            self.hessian = self.hessian_rosembrock
        if name == "branin":
            self.f = self.f_branin
            self.gradient = self.gradient_branin
            self.hessian = self.hessian_branin

    def f_wood(self, x: array, params: dict) -> float:
        """
        Funcion de Wood

        Inputs
        --------------------------
        x -> array de dimension n con los valores de x
        """
        fx = 100 * (x[0] * x[0] - x[1]) * (x[0] * x[0] - x[1])
        fx += (x[0] - 1) * (x[0] - 1) + (x[2] - 1) * (x[2] - 1)
        fx += 90 * (x[2] * x[2] - x[3]) * (x[2] * x[2] - x[3])
        fx += 10.1 * ((x[1] - 1) * (x[1] - 1) + (x[3] - 1) * (x[3] - 1))
        fx += 19.8 * (x[1] - 1) * (x[3] - 1)
        return fx

    def gradient_wood(self, x: array, params: dict) -> array:
        """
        Gradiente de Wood

        Inputs
        --------------------------
        x -> array de dimension n con los valores de x
        """
        n = len(x)
        g = zeros(n)
        g[0] = 400 * (x[0] * x[0] - x[1]) * x[0] + 2 * (x[0] - 1)
        g[1] = -200 * (x[0] * x[0] - x[1]) + 20.2 * (x[1] - 1)
        g[1] += 19.8 * (x[3] - 1)
        g[2] = 2 * (x[2] - 1) + 360 * (x[2] * x[2] - x[3]) * x[2]
        g[3] = -180 * (x[2] * x[2] - x[3]) + 20.2 * (x[3] - 1)
        g[3] += 19.8 * (x[1] - 1)
        return g

    def hessian_wood(self, x: array, params: dict) -> array:
        """
        Hessiano de Wood

        Inputs
        --------------------------
        x -> array de dimension n con los valores de x
        """
        n = params["n"]
        h = zeros((n, n))
        h[0, 0] = 400 * (x[0] * x[0] - x[1]) + 800 * x[0] * x[0] + 2
        h[1, 0] = -400 * x[0]
        h[0, 1] = -400 * x[0]
        h[1, 1] = 220.2
        h[1, 3] = 19.8
        h[3, 1] = 19.8
        h[2, 2] = 2 + 720 * x[2] * x[2] + 360 * (x[2] * x[2] - x[3])
        h[2, 3] = -360 * x[2]
        h[3, 2] = -360 * x[2]
        h[3, 3] = 200.2
        return h

    def f_rosembrock(self, x: array, params: dict) -> float:
        """
        Funcion de Rosembrock

        Inputs
        --------------------
        x -> array de dimension n con los valores de x
        """
        n = params["n"]
        fx = 0
        for i in range(n - 1):
            fx += 100 * (x[i + 1] - x[i] * x[i]) * (x[i + 1] - x[i] * x[i])
            fx += (1 - x[i]) * (1 - x[i])
        return fx

    def gradient_rosembrock(self, x: array, params: dict) -> array:
        """
        Gradiente de Rosembrock

        Inputs
        --------------------
        x -> array de dimension n con los valores de x
        """
        n = params["n"]
        g = zeros(n)
        i = 0
        g[i] = -400 * x[i] * (x[i + 1] - x[i] * x[i]) - 2 * (1 - x[i])
        for i in range(1, n - 1):
            g[i] = 200 * (x[i] - x[i - 1] * x[i - 1])
            g[i] += -400 * x[i] * (x[i + 1] - x[i] * x[i])
            g[i] += -2 * (1 - x[i])
        i = n - 1
        g[i] = 200 * (x[i] - x[i - 1] * x[i - 1])
        return g

    def hessian_rosembrock(self, x: array, params: dict) -> array:
        """
        Hessiano de Rosembrock

        Inputs
        --------------------
        x -> array de dimension n con los valores de x
        """
        n = params["n"]
        h = zeros((n, n))
        h[0, 0] = -200
        for i in range(n - 1):
            h[i, i] += 1200 * x[i] * x[i] - 400 * x[i + 1] + 202
            h[i + 1, i] += -400 * x[i]
            h[i, i + 1] += -400 * x[i]
        h[n - 1, n - 1] = 200
        return h

    def f_branin(self, x: array, params: dict) -> float:
        """
        Funcion de Branin

        Inputs
        --------------------
        params["x"] -> array de dimension n con los valores de x
        params["a"] -> parametros de la función
        params["b"] -> parametros de la función
        params["c"] -> parametros de la función
        params["r"] -> parametros de la función
        params["s"] -> parametros de la función
        params["t"] -> parametros de la función
        """
        a = params["a"]
        b = params["b"]
        c = params["c"]
        r = params["r"]
        s = params["s"]
        t = params["t"]
        f = a * (x[1] - b * x[0]**2 + c * x[0] - r)**2
        f += s * (1 - t) * cos(x[0]) + s
        return f

    def gradient_branin(self, x: array, params: dict) -> array:
        """
        Gradiente de Branin

        Inputs
        --------------------
        params["x"] -> array de dimension n con los valores de x
        params["a"] -> parametros de la función
        params["b"] -> parametros de la función
        params["c"] -> parametros de la función
        params["r"] -> parametros de la función
        params["s"] -> parametros de la función
        params["t"] -> parametros de la función
        """
        a = params["a"]
        b = params["b"]
        c = params["c"]
        r = params["r"]
        s = params["s"]
        t = params["t"]
        n = params["n"]
        g = zeros(n)
        g[1] = 2 * a * (x[1] - b * x[0]**2 + c * x[0] - r)
        g[0] = g[1]
        g[0] *= (-2 * b * x[0] + c)
        g[0] += -s * (1 - t) * sin(x[0])
        return g

    def hessian_branin(self, x: array, params: dict) -> array:
        """
        Hessiano de Branin

        Inputs
        --------------------
        params["x"] -> array de dimension n con los valores de x
        params["a"] -> parametros de la función
        params["b"] -> parametros de la función
        params["c"] -> parametros de la función
        params["r"] -> parametros de la función
        params["s"] -> parametros de la función
        params["t"] -> parametros de la función
        """
        a = params["a"]
        b = params["b"]
        c = params["c"]
        r = params["r"]
        s = params["s"]
        t = params["t"]
        n = params["n"]
        h = zeros((n, n))
        h[0, 0] = 2 * a*(-2 * b * x[0] + c)**2
        h[0, 0] += -4 * a * b * (x[1] - b * x[0]**2 + c * x[0] - r)
        h[0, 0] += -s * (1 - t) * cos(x[0])
        h[1, 0] = 2 * a*(c - 2 * b * x[0])
        h[0, 1] = 2 * a*(c - 2 * b * x[0])
        h[1, 1] = 2 * a
        return h
