from numpy import exp, zeros, ones, mean, ones_like


class function_class:
    def __init__(self) -> None:
        pass

    def update_phi(self, Y, mu, sigma, n):
        '''
        Construye  Matriz de Kerneles Phi

        Parámetros
        -----------
            Y            : Patrones a Aproximar
            mu           : Array de medias
            sigma        : Vector de Desviaciones
            num_rad_func : Número de funciones radiales usadas
        Regresa
        -----------
            phi          : matriz de kerneles
        '''
        phi = zeros((Y.shape[0], n))
        for i in range(n):
            phi[:, i] = exp(- 1.0 / (2*sigma**2) * (Y - mu[i])**2)
        return phi

    def grad_gaussian_radial_mu(self, theta, f_params):
        '''
        Calcula el gradiente respecto a mu
        Parámetros
        -----------
            theta
            f_params : lista de parametros para la funcion objetivo,
                        kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                        X     = f_params['X'] Variable independiente
                        y     = f_params['y'] Variable dependiente

        Regresa
        -----------
            Array gradiente
        '''
        # Obtengo Parámetros
        phi = f_params['X']
        alpha = f_params['Alpha']
        n = f_params['n']
        Y = f_params['y']
        mu = f_params['mu']
        alpha = alpha.reshape((-1, 1))
        mu = mu.reshape((-1, 1))
        Y = Y.reshape((-1, 1))
        gradient = (phi @ alpha - Y) @ alpha.T * \
            (Y @ ones((1, n)) - ones_like(Y) @ mu.T)
        return mean(gradient, axis=0)

    def grad_gaussian_radial_alpha(self, theta, f_params):
        '''
        Calcula el gradiente respecto a alpha
        Parámetros
        -----------
            theta
            f_params : lista de parametros para la funcion objetivo,
                        kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                        X     = f_params['X'] Variable independiente
                        y     = f_params['y'] Variable dependiente

        Regresa
        -----------
            Array gradiente
        '''
        # Obtengo Parámetros
        phi = f_params['X']
        Y = f_params['y']
        alpha = f_params['Alpha']
        gradient = phi.T @ (phi @ alpha - Y)
        return mean(gradient, axis=0)
