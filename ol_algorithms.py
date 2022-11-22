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
        self.alpha_t = weights
        self.z0 = z0
        
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

    def __init__(self, f, d, weights, eta, z0):
        self.name = "FTRL"
        self.f = f
        self.d = d
        self.alpha_t = weights
        self.z0 = z0
        self.eta = eta  
        self.regularizer = r"$\frac{1}{2} \sqrt{t} ||x||^2$"

    # Assumes R(x,t) = 1/2 sqrt(t) ||x||^2
    def get_update_x(self, g, t):
        return -np.average(g, axis = 1, weights = self.alpha_t[0:t]) * np.sum(self.alpha_t[0:t]) / np.sqrt(t)


class OFTRL:

    def __init__(self, f, d, weights, hints, z0):
        self.name = "Optimistic FTRL"
        self.f = f
        self.d = d
        self.alpha_t = weights
        self.hints = None       # Uses hint m(t) = \ell(t-1), the previous loss
        self.z0 = z0

    # This is the closed form for the update of x - the x update is really easy,
    # since there is no dependence on the Fenchel conjugate.

    # DOUBLE CHECK THERE ARE NO OFF-BY-ONE ERRORS!
    def get_update_x(self, y, t):

        weighted_sum = np.zeros(shape = (self.d))
        for i in range(0, t-1):
            weighted_sum += self.alpha_t[i] * y[i]
        weighted_sum += self.alpha_t[t] * y[t-1]

        return -weighted_sum / np.sqrt(t)

    # This update rule is tightly coupled to the Fenchel conjugate... not sure
    # how to decouple it to get "general" update rules that are independent of 
    # the specific function we are trying to implement...maybe a numerical
    # solution?  We can get decent closed form expressions, maybe we can compute
    # an equality...

    # DOUBLE CHECK THERE ARE NO OFF-BY-ONE-ERRORS!
    def get_update_y(self, x, t):
        
        weighted_sum = np.zeros(shape = (self.d))
        for i in range(0, t-1):
            weighted_sum += self.alpha_t[i] * x[i]
        weighted_sum + self.alpha_t[t] * x[t-1]

        return weighted_sum / (np.sqrt(t) + np.sum(self.alpha_t[0:t]))

