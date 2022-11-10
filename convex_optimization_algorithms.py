import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import linprog


class PowerFunction:
    def __init__(self, p,q):
        self.name="Power Function $f(x)=1/2 ||x||^2$"
        self.p = p
        self.q = q 

    def f(self,x):
        return (1/2)* np.pow(np.linalg.norm(x, ord=self.q), self.q)
    
    def grad_f(self, x):
        return x

    def find_s(self, x_k, xbounds=(-10,10)):
        res = linprog(x_k, bounds=xbounds)
        return res.x


class FrankeWolfe:
    def __init__(self):
        self.name = "FrankeWolfe Algorithm"

# Franke-Wolfe Training loop
T = 1000
xbounds = [-10,10]
f = PowerFunction(2,2)
x_k = np.array([10,-10])
x_ks = []

for k in range(0,T): 
    s_k = f.find_s(x_k)
    x_k = x_k + (2/(k+2))*(s_k-x_k)
    x_ks.append(np.linalg.norm(x_k))
    print(x_k)

print(x_k)

plt.plot(x_ks)
plt.show()