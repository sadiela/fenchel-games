import numpy as np 

#### NO REGRET ALGORITHMS ####

class BestResponse: # implemented for power function only 

    # Might need some projections here...This is also very messy, need to be careful about who is actually running BestResp
    def __init__(self, d, f, weights, X, Y):
        self.name = "BestResp"
        self.d = d
        self.f = f              # This is the Function Wrapper
        self.alpha_t = weights
        self.X = X
        self.Y = Y

    # For player X, f(x, yt) = <x|yt> - f*(yt); this quantity is minimized for bounded domain X, we just need to match signs
    # 1) Assume X is bounded by hypercube - then each axis is separable, and we can just compare axes
    # 2) Assume X is bounded by L2 sphere or some other object
    def get_update_x(self, x, y, xbounds, t):
        x_ret = np.ones(shape = (self.d))
        for i in range(0, self.d):
            x_ret[i] = self.alpha_t[t] * max(abs(xbounds[i][0]), abs(xbounds[i][1])) * -1 * np.sign(y[i])

        return x_ret

    def get_update_y(self, x, y, t, ybounds):
        return x

class OMD():
    def __init__(self, f, eta_t):
        self.name = "OMD"
        self.eta_t = eta_t
        self.f = f
    
    def get_update(self, x, g, t):
        return x- self.eta_t[t]*g

class OOMD():
    def __init__(self, f, eta_t):
        self.name = "OOMD"

class FTL:

    def __init__(self, f, d , weights):
        self.name = "FTL"
        self.d = d
        self.alpha_t = weights
        self.f = f

    # This doesn't do anything
    def get_update_y(self, x, t):

        weighted_sum = np.zeros((self.d))
        for i in range(0, t):
            print(x[i])
            weighted_sum += self.alpha_t[i] * x[i]
        return weighted_sum / sum(self.alpha_t[0:t])

class FTRL:

    def __init__(self):
        self.name = "FTRL"
        self.regularizer = r"$\frac{1}{2} \sqrt{t} ||x||^2$"

    def get_update(self, g, t):
        return -np.sum(g, axis = 1) / np.sqrt(t) # R(x,t) = 1/2 sqrt(t) ||x||^2