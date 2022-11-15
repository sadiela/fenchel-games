import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import linprog

### POWER FUNCTION: p,q =2 ###

#############
# FUNCTIONS # 
#############
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

######################
# FENCHEL CONJUGATES # 
######################
class PowerFenchel:
    def __init__(self, p,q):
        self.name = "Power function (fenchel)" 
        self.p = p
        self.q = q

    def fenchel(self, theta): 
        return (1/2) * np.pow(np.linalg.norm(theta, ord = self.q), self.q)

    def payoff(self, x,y):
        return np.dot(x, y) - self.fenchel(y, self.p, self.q)

    def grad_x(self, x, y):
        return y

    def grad_y(self, x, y):
        return x - y

class ExpFenchel:
    def __init__(self):
        self.name = "Exponential function"

    def fenchel(self,theta): 
        return theta* np.log(theta) - theta

    def payoff(self, x,y):
        return np.dot(x,y) - self.fenchel(y)

    def grad_x(self, x,y):
        return y

    def grad_y(self, x,y):
        return x- np.log(y)