from numpy import array, zeros, sqrt
from numpy.random import randint


class model_class:
    def __init__(self) -> None:
        """
        Modelo que reune los metodos de 
        + Descenso de gradiente estocástico.
        + Descenso de gradiente estoc ́astico accelerado de tipo Nesterov.
        + AdaDelta
        + ADAM
        """
        pass

    def select_method(self, method_name: str):
        if method_name == "SGD":
            self.method = self.SGD
        if method_name == "NAG":
            self.method = self.NAG
        if method_name == "ADADELTA":
            self.method = self.ADADELTA
        if method_name == "ADAM":
            self.method = self.ADAM

    def SGD(self, theta: list, grad, gd_params: dict, f_params: dict,) -> array:
        """
        Descenso de gradiente estocástico

        Parámetros
        -----------
        theta     :   condicion inicial
        grad      :   funcion que calcula el gradiente

        gd_params :   lista de parametros para el algoritmo de descenso,
                        nIter = gd_params['nIter'] número de iteraciones
                        alpha = gd_params['alpha'] tamaño de paso alpha
                        batch_size = gd_params['batch_size'] tamaño de la muestra

        f_params  :   lista de parametros para la funcion objetivo,
                        X     = f_params['X'] Variable independiente
                        y     = f_params['y'] Variable dependiente

        Regresa
        -----------
        Theta     :   trayectoria de los parametros
                        Theta[-1] es el valor alcanzado en la ultima iteracion
        """
        (high, dim) = f_params['X'].shape
        batch_size = gd_params['batch_size']
        nIter = gd_params['nIter']
        alpha = gd_params['alpha']
        Theta = []
        for t in range(nIter):
            # Set of sampled indices
            smpIdx = randint(low=0,
                             high=high,
                             size=batch_size,
                             dtype='int32')
            # sample
            smpX = f_params['X'][smpIdx]
            smpy = f_params['y'][smpIdx]
            # parametros de la funcion objetivo
            smpf_params = {"Alpha": f_params["Alpha"],
                           "mu": f_params["mu"],
                           "n": f_params["n"],
                           'X': smpX,
                           'y': smpy}
            p = grad(theta,
                     f_params=smpf_params)
            theta = theta - alpha*p
            Theta.append(theta)
        return array(Theta)

    def NAG(self, theta: list, grad, gd_params: dict, f_params: dict,):
        """
        Descenso acelerado de Nesterov

        Parámetros
        -----------
        theta     :   condicion inicial
        grad      :   funcion que calcula el gradiente
        gd_params :   lista de parametros para el algoritmo de descenso,
                        nIter = gd_params['nIter'] número de iteraciones
                        alpha = gd_params['alpha'] tamaño de paso alpha
                        eta   = gd_params['eta']  parametro de inercia (0,1]
        f_params  :   lista de parametros para la funcion objetivo,
                        kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                        X     = f_params['X'] Variable independiente
                        y     = f_params['y'] Variable dependiente

        Regresa
        -----------
        Theta     :   trayectoria de los parametros
                        Theta[-1] es el valor alcanzado en la ultima iteracion
        """
        nIter = gd_params['nIter']
        alpha = gd_params['alpha']
        eta = gd_params['eta']
        p = zeros(theta.shape)
        Theta = []
        for t in range(nIter):
            pre_theta = theta - 2.0*alpha*p
            g = grad(pre_theta,
                     f_params=f_params)
            p = g + eta*p
            theta = theta - alpha*p
            Theta.append(theta)
        return array(Theta)

    def ADADELTA(self, theta: list, grad, gd_params: dict, f_params: dict,):
        """
        Descenso de Gradiente Adaptable (ADADELTA)

        Parámetros
        -----------
        theta     :   condicion inicial
        grad      :   funcion que calcula el gradiente
        gd_params :   lista de parametros para el algoritmo de descenso,
                        nIter    = gd_params['nIter'] número de iteraciones
                        alphaADA = gd_params['alphaADADELTA'] tamaño de paso alpha
                        eta      = gd_params['eta']  parametro adaptación del alpha
        f_params  :   lista de parametros para la funcion objetivo,
                        kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                        X     = f_params['X'] Variable independiente
                        y     = f_params['y'] Variable dependiente

        Regresa
        -----------
        Theta     :   trayectoria de los parametros
                        Theta[-1] es el valor alcanzado en la ultima iteracion
        """
        epsilon = 1e-8
        nIter = gd_params['nIter']
        alpha = gd_params['alphaADADELTA']
        eta = gd_params['eta']
        G = zeros(theta.shape)
        g = zeros(theta.shape)
        Theta = []
        for t in range(nIter):
            g = grad(theta,
                     f_params=f_params)
            G = eta*g**2 + (1-eta)*G
            p = 1.0/(sqrt(G)+epsilon)*g
            theta = theta - alpha * p
            Theta.append(theta)
        return array(Theta)

    def ADAM(self, theta: list, grad, gd_params: dict, f_params: dict,):
        """
        Descenso de Gradiente Adaptable con Momentum(A DAM)

        Parámetros
        -----------
        theta     :   condicion inicial
        grad      :   funcion que calcula el gradiente
        gd_params :   lista de parametros para el algoritmo de descenso,
                        nIter    = gd_params['nIter'] número de iteraciones
                        alphaADA = gd_params['alphaADAM'] tamaño de paso alpha
                        eta1     = gd_params['eta1'] factor de momentum para la direccion
                                    de descenso (0,1)
                        eta2     = gd_params['eta2'] factor de momentum para la el
                                    tamaño de paso (0,1)
        f_params  :   lista de parametros para la funcion objetivo,
                        kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                        X     = f_params['X'] Variable independiente
                        y     = f_params['y'] Variable dependiente

        Regresa
        -----------
        Theta     :   trayectoria de los parametros
                        Theta[-1] es el valor alcanzado en la ultima iteracion
        """
        epsilon = 1e-8
        nIter = gd_params['nIter']
        alpha = gd_params['alphaADAM']
        eta1 = gd_params['eta1']
        eta2 = gd_params['eta2']
        p = zeros(theta.shape)
        v = 0.0
        Theta = []
        eta1_t = eta1
        eta2_t = eta2
        for t in range(nIter):
            g = grad(theta,
                     f_params=f_params)
            p = eta1*p + (1.0-eta1)*g
            v = eta2*v + (1.0-eta2)*(g**2)
            theta = theta - alpha * p / (sqrt(v)+epsilon)
            eta1_t *= eta1
            eta2_t *= eta2
            Theta.append(theta)
        return array(Theta)
