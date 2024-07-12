import numpy as np
import matplotlib.pyplot as plt

# Random Walk
def random_walk_2d(n_steps, initial_position, mu=0, sigma=1):
    x, y = [initial_position[0]], [initial_position[1]]  # Posición inicial
    
    for _ in range(n_steps):
        step_x = np.random.normal(mu, sigma)  # Incremento aleatorio en x
        step_y = np.random.normal(mu, sigma)  # Incremento aleatorio en y
        x.append(x[-1] + step_x)
        y.append(y[-1] + step_y)
    return x, y

# Nelder-Mean Simplex
def NelderMead_Simplex(funcion, x0, tol=1e-6, iteraciones=100, alpha=1, beta=0.5, gamma=2):
    n = len(x0)
    # Aqui vamos a hacer un conjunto de vectores que van a ser nuestro vectores xl, xg, xh
    vectores = np.zeros((n + 1, n))
    vectores[0] = x0 
    vectores[1:] = x0 + np.eye(n) * 0.05 # En este punto tenemos nuestros xl = vectores[0], xg = vectores[1], xh = vectores[2] pero no sabemos cual es el mejor y cual es el peor
    valores_vect = np.array([funcion(i) for i in vectores]) # Entonces los evaluamos en nuestra funcion

    for _ in range(iteraciones): # Una vez hecha la evaluacion en nuestro funcion, vamos a ordenar nuestros vectores
        ordenar_vect = np.argsort(valores_vect)  # Aqui ordenamos las posiciones, osea en vez de que sea [0,1,2] va a ser [1,0,2] gracias a la evalucion de los vectores
        vectores = vectores[ordenar_vect] # Ordenamos nuestros vectores, xl = vectores[1], xg = vectores[0], xh = vectores[2]
        valores_vect = valores_vect[ordenar_vect] # Tambien ordenamos nuestras evaluaciones de la misma manera 

        # Aqui calculamos el punto central xc de nuestro mejor punto xl y el punto que esta junto al peor punto xg
        xc = np.mean(vectores[:-1], axis=0) # Esto nos ayuda a saber donde se va a mover nuestros puntos 

        # Hacemos la Reflexión con esta funcion xr = xc + alpha * (xc - vector[-1])
        xr = xc + alpha * (xc - vectores[-1])
        if funcion(xr) < valores_vect[-2] and funcion(xr) >= valores_vect[0]:
            vectores[-1] = xr
            valores_vect[-1] = funcion(xr)
        # Hacemos la Expansión
        elif funcion(xr) < valores_vect[0]:
            expansion = xc + gamma * (xr - xc)
            vectores[-1] = expansion if funcion(expansion) < funcion(xr) else xr
            valores_vect[-1] = funcion(vectores[-1])
        # Hacemos la Contracción
        else:
            contraccion = xc + beta * (vectores[-1] - xc)
            vectores[-1] = contraccion if funcion(contraccion) < valores_vect[-1] else 0.5 * (vectores[0] + vectores[1])
            valores_vect[-1] = funcion(vectores[-1])

    return vectores[0]

# Hooke-Jeeves
def Busqueda(x, d, funcion, limite=1e10):
    x_i = np.copy(x)
    for i in range(len(x)):
        for direction in [-1, 1]:
            x_t = np.copy(x_i)
            x_t[i] += direction * d
            if np.abs(x_t[i]) > limite:
                continue
            try:
                if funcion(x_t) < funcion(x_i):
                    x_i = x_t
            except OverflowError:
                continue
    return x_i

def hooke_jeeves(x_i, delta, alpha, e, n_iter, funcion, limite=1e10):
    x_b = np.array(x_i, dtype=np.float64)
    x_m = np.copy(x_b)
    iter_c = 0
    resul = [x_b.copy()]

    while delta > e and iter_c < n_iter:
        x_n = Busqueda(x_b, delta, funcion, limite)
        try:
            if funcion(x_n) < funcion(x_m):
                x_b = 2 * x_n - x_m
                x_m = np.copy(x_n)
            else:
                delta *= alpha
                x_b = np.copy(x_m)
        except OverflowError:
            break
        resul.append(x_b.copy())
        iter_c += 1

    return x_m, resul

#Cauchy
def gradiente(f, x, deltaX=0.001):
    grad = []
    for i in range(0, len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] = xp[i] + deltaX
        xn[i] = xn[i] - deltaX
        grad.append((f(xp) - f(xn)) / (2 * deltaX))
    return grad

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

# Método de Fletcher-Reeves
def gradient(f, x, deltaX=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += deltaX
        x2[i] -= deltaX
        grad[i] = (f(x1) - f(x2)) / (2 * deltaX)
    return grad

def fletcher_reeves(f, x0, tol=1e-6, max_iter=1000):
    x = x0
    grad_f = gradient(f, x)  # Calcula el gradiente de la función objetivo usando tu función gradient
    r = -grad_f
    d = r
    rsold = np.dot(r, r)
    
    for i in range(max_iter):
        alpha = line_search(f, x, d)
        x = x + alpha * d
        grad_f = gradient(f, x)  # Recalcula el gradiente en el nuevo punto
        r = -grad_f
        rsnew = np.dot(r, r)
        
        if np.sqrt(rsnew) < tol:
            print(f'Convergencia alcanzada en {i+1} iteraciones')
            break
        
        beta = rsnew / rsold
        d = r + beta * d
        rsold = rsnew
        
    return x

def line_search(f, x, d, alpha0=1, c=1e-4, tau=0.9):
    alpha = alpha0
    while f(x + alpha * d) > f(x) + c * alpha * np.dot(gradient(f, x + alpha * d), d):
        alpha *= tau
    return alpha

# Metodo de newton
def hessian_matrix(f, x, deltaX):
    fx = f(x)
    N = len(x)
    H = []
    for i in range(N):
        hi = []
        for j in range(N):
            if i == j:
                xp = x.copy()
                xn = x.copy()
                xp[i] = xp[i] + deltaX
                xn[i] = xn[i] - deltaX
                hi.append((f(xp) - 2 * fx + f(xn)) / (deltaX ** 2))
            else:
                xpp = x.copy()
                xpn = x.copy()
                xnp = x.copy()
                xnn = x.copy()
                xpp[i] = xpp[i] + deltaX
                xpp[j] = xpp[j] + deltaX
                xpn[i] = xpn[i] + deltaX
                xpn[j] = xpn[j] - deltaX
                xnp[i] = xnp[i] - deltaX
                xnp[j] = xnp[j] + deltaX
                xnn[i] = xnn[i] - deltaX
                xnn[j] = xnn[j] - deltaX
                hi.append((f(xpp) - f(xpn) - f(xnp) + f(xnn)) / (4 * deltaX ** 2))
        H.append(hi)
    return np.array(H)

def gradient(f, x, deltaX=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += deltaX
        x2[i] -= deltaX
        grad[i] = (f(x1) - f(x2)) / (2 * deltaX)
    return grad

def newton_method(f, x0, epsilon=1e-6, max_iter=100):
    x = x0.copy()
    for k in range(max_iter):
        grad = gradient(f, x)
        H = hessian_matrix(f, x, 1e-5)
        H_inv = np.linalg.inv(H)
        d = -np.dot(H_inv, grad)
        alpha = 1.0  
        while f(x + alpha * d) > f(x): 
            alpha *= 0.5
        x = x + alpha * d
        if np.linalg.norm(grad) < epsilon:
            break
    return x

