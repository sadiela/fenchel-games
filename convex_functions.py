import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import linprog

### POWER FUNCTION: p,q =2 ###

def find_s(x_t, xbounds=(-10,10)):
    res = linprog(x_t, bounds=xbounds)
    return res.x

def bregmanDivergence(phi, x, y):
    return phi.f(x)-phi.f(y) - np.dot(phi.grad(y), x-y)

class L2Reg:
    def __init__(self):
        self.name="Power Function $f(x)=1/2 ||x||^2$"

    def f(self,x, t=1):
        return (1/2)* np.pow(np.linalg.norm(x, ord=2), 2) * np.sqrt(t)
    
    def grad(self, x):
        return x

    def fenchel(self, theta): 
        return (1/2) * np.power(np.linalg.norm(theta, ord = 2), 2)

    def fenchel_grad(self, x, t):
        return x * np.sqrt(t)


#############
# FUNCTIONS # 
#############
class SqrtOneXSquared:
    def __init__(self):
        self.name = "Function: $f(x) = sqrt(1+x^2) $"

    def f(self, x):
        return np.sqrt(1 + np.power(x, 2))

    def grad(self, x):
        gradient = x/np.sqrt(1 + np.power(x,2))
        return gradient

class AbsoluteValueFunction:
    def __init__(self):
        self.name = "Absolute value function $f(x) = |x| $"

    def f(self, x):
        return np.abs(x)

    def grad(self, x):
        gradient = np.copy(x)
        gradient[gradient>0] = 1
        gradient[gradient <0] = -1
        return gradient

class PowerFunction:
    def __init__(self, p,q):
        self.name="Power Function $f(x)=1/2 ||x||^2$"
        self.p = p
        self.q = q 

    def f(self,x):
        return (1/2)* np.pow(np.linalg.norm(x, ord=self.q), self.q)
    
    def grad(self, x):
        return x


class ExpFunction:
    def __init__(self):
        self.name="Exponential Function $f(x)=e^x$"

    def f(self,x):
        return np.exp(x)
    
    def grad(self, x):
        return np.exp(x)

######################
# FENCHEL CONJUGATES # 
######################
class SqrtOneXSquaredFenchel:
    def __init__(self):
        self.name = "Function: $f(x) = sqrt(1+x^2) (fenchel)" 

    def fenchel(self, theta): 
        return -np.sqrt(1 - np.power(theta, 2))

    def grad(self, x):
        return x/np.sqrt(1 + np.power(x, 2))

    def payoff(self, x, y):
        return np.dot(x,y) - self.fenchel(y)

    def grad_x(self, x, y):
        return y

    def grad_y(self, x, y):
        return x - y/np.sqrt(1-np.power(y,2))

class PowerFenchel:
    def __init__(self, p, q):
        self.name = "Power function (fenchel)" 
        self.p = p
        self.q = q

    def fenchel(self, theta): 
        return (1/2) * np.power(np.linalg.norm(theta, ord = self.q), self.q)

    def grad(self, x):
        return x

    def payoff(self, x, y):
        return np.dot(x, y) - self.fenchel(y)

    def grad_x(self, x, y):
        return y

    def grad_y(self, x, y):
        return x - y

class ExpFenchel:
    def __init__(self):
        self.name = "Exponential function"

    def fenchel(self,theta): 
        if theta == 0: 
            return 0
        if theta > 0:
            return theta* np.log(theta) - theta

    def payoff(self, x,y):
        return np.dot(x,y) - self.fenchel(y)

    def grad_x(self, x, y):
        return y

    def grad_y(self, x,y):
        return x- np.log(y) - 2