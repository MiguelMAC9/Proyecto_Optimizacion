# Proyecto_Optimizacion
Aquí encontraran 9 algoritmos de optimización, los cuales son básicos en el área de optimización.
También encontraran algunas funciones muy usadas para el uso de estos algoritmos.

def cauchy(funcion, x0, epsilon1, epsilon2, M, optimizador_univariable):
    terminar = False
    xk = x0
    k = 0
    while not terminar:
        grad = np.array(gradiente(funcion, xk))

        if np.linalg.norm(grad) < epsilon1 or k >= M:
            terminar = True
        else:
            def alpha_funcion(alpha):
                return funcion(xk - alpha * grad)

            alpha = optimizador_univariable(alpha_funcion, epsilon2, a=0.0, b=1.0)
            x_k1 = xk - alpha * grad
            print(xk, alpha, grad, x_k1)

            if np.linalg.norm(x_k1 - xk) / (np.linalg.norm(xk) + 0.00001) <= epsilon2:
                terminar = True
            else:
                k = k + 1
                xk = x_k1
    return xk
