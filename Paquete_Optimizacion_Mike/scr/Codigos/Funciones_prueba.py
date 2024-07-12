import numpy as np 
import matplotlib.pyplot as plt

def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def testfunction(x):
    return x[0]**2 + x[1]**2

def sphere(x):
        return np.sum(np.square(x))

def rastrigin(x, A=10):
    mat = 0
    n = len(x)      
    for i in x:
        mat += (i**2 - A * np.cos(2 * np.pi * i))
    resul = A * n + mat
    return resul

def rosenbrock(x):
    mat = 0
    for i in range(0,len(x)-1):
        mat += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    return mat

def beale(x):
    return ((1.5 - x[0] + x[0] * x[1])**2 +
            (2.25 - x[0] + x[0] * x[1]**2)**2 +
            (2.625 - x[0] + x[0] * x[1]**3)**2)
    
def goldstein(x):
    part1 = (1 + (x[0] + x[1] + 1)**2 * 
            (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2))
    part2 = (30 + (2 * x[0] - 3 * x[1])**2 * 
            (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2))
    return part1 * part2

def boothfunction(x):
    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

def bunkinn6(x):
    return 100 * np.sqrt(np.abs(x[1] - 0.001 * x[0]**2)) + 0.01 * np.abs(x[0] + 10)

def matyas(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def levi(x):
    part1 = np.sin(3 * np.pi * x[0])**2
    part2 = (x[0] - 1)**2 * (1 + np.sin(3 * np.pi * x[1])**2)
    part3 = (x[1] - 1)**2 * (1 + np.sin(2 * np.pi * x[1])**2)
    return part1 + part2 + part3
    
def threehumpcamel(x):
    return 2 * x[0]**2 - 1.05 * x[0]**4 + (x[0]**6) / 6 + x[0] * x[1] + x[1]**2

def easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)

def crossintray(x):
    op = np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))
    return -0.0001 * (op + 1)**0.1

def eggholder(x):
    op1 = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + (x[1] + 47))))
    op2 = -x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
    return op1 + op2

def holdertable(x):
    op = np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))
    return -op

def mccormick(x):
    return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1

def schaffern2(x):
    numerator = np.sin(x[0]**2 - x[1]**2)**2 - 0.5
    denominator = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
    return 0.5 + numerator / denominator

def schaffern4(x):
    num = np.cos(np.sin(np.abs(x[0]**2 - x[1]**2))) - 0.5
    den = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
    return 0.5 + num / den

def styblinskitang(x):
    resul = 0
    for i in x:
        resul += (i**4 - 16 * i**2 + 5 * i) / 2
    return resul
    
def shekel(x, a=None, c=None):
    if a is None:# Esto lo hice para que el usuario pueda meter los pesos que guste, si no se ponen estos
        a = np.array([
            [4.0, 4.0, 4.0, 4.0],
            [1.0, 1.0, 1.0, 1.0],
            [8.0, 8.0, 8.0, 8.0],
            [6.0, 6.0, 6.0, 6.0],
            [3.0, 7.0, 3.0, 7.0],
            [2.0, 9.0, 2.0, 9.0],
            [5.0, 5.0, 3.0, 3.0],
            [8.0, 1.0, 8.0, 1.0],
            [6.0, 2.0, 6.0, 2.0],
            [7.0, 3.6, 7.0, 3.6]
        ])
    if c is None:
        c = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])#Lo mismo que con a
        
    m = len(c)
    s = 0
    for i in range(m):
        s -= 1 / (np.dot(x - a[i, :2], x - a[i, :2]) + c[i])#Esta es la sumatoria dado m, que seria el numero de terminos en la suma
    return s