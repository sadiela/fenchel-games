import numpy as np 

#### NO REGRET ALGORITHMS ####

#FUNCTION_DICT = {"FTRL" : FTRL, "FTL": FTL, "BestResp": BestResponse, "OMD": OMD, "OOMD": OOMD}

class BestResponse: # implemented for power function only 

    # Might need some projections here...This is also very messy, need to be careful about who is actually running BestResp
    def __init__(self, f, d, weights, xbounds, ybounds):
        self.name = "BestResp"
        self.f = f   
        self.d = d   
        self.alpha_t = weights
        self.xbounds = xbounds
        self.ybounds = ybounds

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

    def __init__(self, f, d, eta_t):
        self.name = "OMD"
        self.f = f
        self.d = d
        self.eta_t = eta_t
        
    def get_update(self, x, g, t):
        return x- self.eta_t[t] * g

class OOMD():

    def __init__(self, f, d, eta_t):
        self.name = "OOMD"

class FTL:

    def __init__(self, f, d, weights, z0):
        self.name = "FTL"
        self.f = f
        self.d = d
        self.z0 = z0
        self.alpha_t = weights

    def get_update_y(self, x, t):

        if t == 0:
            return self.z0
        else:
            weighted_sum = np.zeros(shape = (self.d))
            for i in range(0, t):
                #print(x[i])
                weighted_sum += (self.alpha_t[i] * x[i])
            return weighted_sum / sum(self.alpha_t[0:t])

class FTRL:

    def __init__(self):
        self.name = "FTRL"
        self.regularizer = r"$\frac{1}{2} \sqrt{t} ||x||^2$"

    def get_update(self, g, t):
        return -np.sum(g, axis = 1) / np.sqrt(t) # R(x,t) = 1/2 sqrt(t) ||x||^2