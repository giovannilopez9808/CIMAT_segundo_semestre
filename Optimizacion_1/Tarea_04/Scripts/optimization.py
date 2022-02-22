import numpy as np

def back_tracking (x_k, d_k, f, f_grad, alpha = 0.1, ro = 0.9, c1 = 1e-4, params = {}) :
    '''
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
    '''
    # Inicialización
    alpha_k = alpha
    
    # Repetir hasta que se cumpla la condición de armijo
    while f(x_k + alpha_k * d_k, params) > f(x_k, params) + c1 * alpha_k * f_grad(x_k, params).dot(d_k) :
        alpha_k = ro * alpha_k
        
    return alpha_k

def bisection (x_k, d_k, f, f_grad,  alpha_0 = 1.0, c1 = 1e-4, c2 = 0.9, max_iter = 100, params = {}) :
    '''
    Calcula tamaño de paso con método de Bisección
    
    Parámetros
    -----------
        x_k      : Vector de valores [x_1, x_2, ..., x_n]
        d_k      : Dirección de descenso
        f        : Función f(x)
        f_grad   : Función que calcula gradiente
        alpha_0  : Tamaño inicial de paso
        c1       : Condición de Wolfe
        c2       : Condición de Wolfe
        max_iter : Iteraciones Máximas permitadas
    Regresa
    -----------
        alpha_k  : Tamaño actualizado de paso
    '''
    
    grad_k = f_grad(x_k, params)
    
        # Inicialización
    alpha    = 0.0
    beta     = np.inf
    counter   = 0
    
    alpha_k = 1
    
    while (f(x_k + alpha_k * d_k, params) > f(x_k, params) + c1 * alpha_k * f_grad(x_k, params).dot(d_k)) or f_grad(x_k + alpha_k * d_k, params).dot(d_k) < c2 * f_grad(x_k, params).dot(d_k):
        if f(x_k + alpha_k * d_k, params) > f(x_k, params) + c1 * alpha_k *  f_grad(x_k, params).dot(d_k) :
            beta = alpha_k
            alpha_k    = 0.5*(alpha + beta)
        else :
            alpha = alpha_k
            if beta == np.inf :
                alpha_k = 2.0 * alpha
            else :
                alpha_k = 0.5* (alpha + beta)
        
        counter = counter + 1
        
        if counter > max_iter :
            break
        
    return alpha_k

# Descenso Gradiente

def des_grad (params = []) :
    '''
    '''
    # Cargo parámetros
    x_k        = params['x_0']
    x_k_next   = None
    
    f          = params['f']
    f_grad     = params['f_grad']
    max_iter   = params['max_iter']
    tau_x      = params['tau_x']
    tau_f      = params['tau_f']
    tau_f_grad = params['tau_grad']
    
    # Identifico parámetros especiales
    if f.__name__ == 'f3' :
        sub_params = {
                        'lambda' : params['lambda'],
                        'eta'    : params['eta'],
                        'sigma'  : params['sigma']
        }
        # Guardo Parámetros
        x_hist = []
        x_hist.append(x_k)
        
    else :
        sub_params = {}
        
        # Guardo Parámetros
        f_hist = []
        f_hist.append(f(x_k, params = sub_params))

        g_hist = []
        g_hist.append(np.linalg.norm(f_grad(x_k, params = sub_params)))
    
    # Identifico función para paso
    if (params['method'] == 'BackTracking') :
        alpha = params['BackTracking']['alpha']
        ro    = params['BackTracking']['ro']
        c1    = params['BackTracking']['c1']
    
    elif (params['method'] == 'Bisection') :
        alpha        = params['Bisection']['alpha']
        c1           = params['Bisection']['c1']
        c2           = params['Bisection']['c2']
        max_iter_bic = params['Bisection']['max_iter'] 
    else :
        print('None')
        return
          
    # Comienza descenso
    k = 0 
    
    while True:
        # Calculo gradiente
        d_k = - f_grad(x_k, params = sub_params)  
        
        # Cálculo tamaño de paso
        if (params['method'] == 'BackTracking') :
            alpha_k = back_tracking(x_k, d_k, f, f_grad, alpha, ro, c1, params = sub_params)
        
        elif (params['method'] == 'Bisection') :
            alpha_k = bisection(x_k, d_k, f, f_grad, alpha, c1, c2, max_iter_bic, params = sub_params)
                        
        # Calculo siguiente valor x_k+1
        x_k_next = x_k + alpha_k * d_k   
        
        ''' Guardo Resultados '''
        
        if f.__name__ == 'f3' :
            # Guardo Parámetros
            x_hist.append(x_k_next)
        else :
            # Guardo Parámetros
            f_hist.append(f(x_k_next, sub_params))
            g_hist.append(np.linalg.norm(f_grad(x_k_next, sub_params)))
        
                  
        # Criterios de paro
        if (k > max_iter) :
            print('Iteraciones: ', k, ' , valor: ', x_k)
            break
            
        if np.linalg.norm(x_k_next - x_k)/max(np.linalg.norm(x_k), 1.0) < tau_x :
            print('Iteraciones: ', k, ' , valor: ', x_k)
            break
                  
        if np.abs(f(x_k_next, sub_params) - f(x_k, sub_params)) / max(np.linalg.norm(f(x_k, sub_params)), 1.0) < tau_f :
            print('Iteraciones: ', k, ' , valor: ', x_k)
            break
        
                  
        if np.linalg.norm(f_grad(x_k_next, sub_params)) < tau_f_grad :
            print('Iteraciones: ', k, ' , valor: ', x_k)
            break
            
        # Guardo valor anterior   
        x_k = x_k_next       
        k   = k + 1
        
    if f.__name__ == 'f3' :
        return np.array(x_k_next)
    else :  
        return np.array(f_hist), np.array(g_hist)