import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import math

class PowerFunction:
    def __init__(self, p,q):
        self.name="Power Function $f(x)=1/2 ||x||^2$"
        self.p = p
        self.q = q 

    def f(self,x):
        return (1/2)* np.pow(np.linalg.norm(x, ord=self.q), self.q)
    
    def grad(self, x):
        return x

    def find_s(self, x_t, xbounds=(-10,10)):
        res = linprog(x_t, bounds=xbounds)
        return res.x

def FrankeWolfeLoop(T, xbounds, f, x_t):
    x_ts = []

    for t in range(0,T): 
        s_t = f.find_s(x_t, xbounds)
        x_t = x_t + (2/(t+2))*(s_t-x_t)
        x_ts.append(np.linalg.norm(x_t))
        print(x_t)

    print(x_t)
    return x_ts

def NAG(T, f, x_t, beta=0.5):
    x_ts = []
    lambda_t = 0.5
    y_t = 0

    for t in range(0,T):
        lambda_t1 = (1-math.sqrt(1 + 4* lambda_t**2))/2
        gamma_t = (1-lambda_t)/lambda_t1
        y_t1 = x_t - (1/beta) * f.grad(x_t)
        x_t = (1-gamma_t)*y_t1 + gamma_t*y_t
        print(x_t)
        x_ts.append(x_t)

        # save for use in next iteration
        lambda_t = lambda_t1
        y_t = y_t1

    return x_ts

if __name__ == "__main__":
    # Franke-Wolfe Training loop
    T = 1000
    xbounds = [-10,10]
    f = PowerFunction(2,2)

    x2_t = np.array([10,-10])

    x2_ts = NAG(T, f, x2_t)
    plt.plot(x2_ts)
    plt.title("NAG Algorithm")
    plt.show()

    x_t = np.array([10,-10])

    x_ts = FrankeWolfeLoop(T, xbounds, f, x_t)
    plt.plot(x_ts)
    plt.title("FW Algorithm")
    plt.show()
