from numpy import array, zeros


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

    def f_wood(self, x: array) -> float:
        """
        Funcion de Wood
        """
        fx = 100*(x[0]*x[0]-x[1])*(x[0]*x[0]-x[1])
        fx += (x[0]-1)*(x[0]-1)+(x[2]-1)*(x[2]-1)
        fx += 90*(x[2]*x[2]-x[3])*(x[2]*x[2]-x[3])
        fx += 10.1*((x[1]-1)*(x[1]-1)+(x[3]-1)*(x[3]-1))
        fx += 19.8*(x[1]-1)*(x[3]-1)
        return fx

    def gradient_wood(self, x: array) -> array:
        """
        Gradiente de Wood
        """
        n = len(x)
        g = zeros(n)
        g[0] = 400*(x[0]*x[0]-x[1])*x[0] + 2*(x[0]-1)
        g[1] = -200*(x[0]*x[0]-x[1])+20.2*(x[1]-1)+19.8*(x[3]-1)
        g[2] = 2*(x[2]-1)+360*(x[2]*x[2]-x[3])*x[2]
        g[3] = -180*(x[2]*x[2]-x[3])+20.2*(x[3]-1)+19.8*(x[1]-1)
        return g

    def hessian_wood(self, x: array) -> array:
        """
        Hessiano de Wood
        """
        n = len(x)
        h = zeros((n, n))
        h[0, 0] = 400*(x[0]*x[0]-x[1])+800*x[0]*x[0]+2
        h[1, 0] = -400*x[0]
        h[0, 1] = -400*x[0]
        h[1, 1] = 220.2
        h[1, 3] = 19.8
        h[3, 1] = 19.8
        h[2, 2] = 2+720*x[2]*x[2]+360*(x[2]*x[2]-x[3])
        h[2, 3] = -360*x[2]
        h[3, 2] = -360*x[2]
        h[3, 3] = 200.2
        return h

    def f_rosembrock(self, x: array) -> float:
        """
        Funcion de Rosembrock
        """
        n = len(x)
        fx = 0
        for i in range(n-1):
            fx += 100*(x[i+1]-x[i]*x[i])*(x[i+1]-x[i]*x[i])
            fx += (1-x[i])*(1-x[i])
        return fx

    def gradient_rosembrock(self, x: array) -> array:
        """
        Gradiente de Rosembrock
        """
        n = len(x)
        g = zeros(n)
        i = 0
        g[i] = -400*x[i]*(x[i+1]-x[i]*x[i])-2*(1-x[i])
        for i in range(1, n-1):
            g[i] = 200*(x[i]-x[i-1]*x[i-1])
            g[i] += -400*x[i] * (x[i+1]-x[i]*x[i])
            g[i] += -2*(1-x[i])
        i = n-1
        g[i] = 200*(x[i]-x[i-1]*x[i-1])
        return g

    def hessian_rosembrock(self, x: array) -> array:
        """
        Hessiano de Rosembrock
        """
        n = len(x)
        h = zeros((n, n))
        h[0, 0] = -200
        for i in range(n-1):
            h[i, i] += 1200 * x[i] * x[i] - 400 * x[i+1] + 202
            h[i+1, i] += -400*x[i]
            h[i, i+1] += -400*x[i]
        h[n-1, n-1] = 200
        return h
