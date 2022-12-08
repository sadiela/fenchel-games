import numpy as np 

#### NO REGRET ALGORITHMS ####

#FUNCTION_DICT = {"FTRL" : FTRL, "FTL": FTL, "BestResp": BestResponse, "OMD": OMD, "OOMD": OOMD}

class BestResponse: # implemented for power function only 

    # Might need some projections here...This is also very messy, need to be careful about who is actually running BestResp
    def __init__(self, f, d, z0, weights, xbounds, ybounds):
        self.name = "BestResp"
        self.f = f   
        self.d = d   
        self.alpha_t = weights
        self.z0 = z0
        self.xbounds = xbounds
        self.ybounds = ybounds

    # For player X, f(x, yt) = <x|yt> - f*(yt); this quantity is minimized for bounded domain X, we just need to match signs
    # 1) Assume X is bounded by hypercube - then each axis is separable, and we can just compare axes
    # 2) Assume X is bounded by L2 sphere or some other object
    def get_update_x(self, y, t):
        x_ret = np.ones(shape = (self.d))
        for i in range(0, self.d):
            x_ret[i] = self.alpha_t[t] * max(abs(self.xbounds[i][0]), abs(self.xbounds[i][1])) * -1 * np.sign(y[-1][i])

        return x_ret

    def get_update_y(self, x, t):
        return self.alpha_t[t] * self.f.grad(x[-1])

class OMD():

    def __init__(self, f, d, weights, z0, y0, eta_t, bounds, prescient = False):
        self.name = "OMD"
        self.prescient = prescient
        if self.prescient:
            self.name += "+"
        self.f = f
        self.d = d
        self.alpha_t = weights
        self.eta_t = eta_t
        self.z0 = z0
        self.z = z0
        self.y0 = y0
        self.bounds = bounds
        

    def get_update_x(self, y, t):

        #if t == 1 and not self.prescient:
        #    self.z = self.z = self.eta_t[t] * self.alpha_t[t] * self.y0
        #else:
            #print("Start t = %d, z = %lf:" % (t, self.z))
    
        if t == 1 and not self.prescient:
            self.z = self.z - self.eta_t[t] * self.alpha_t[t] * self.y0
        elif not self.prescient:
            print("OMD update: %lf, %lf, %lf" % (self.z, self.eta_t[t], y[-1]))
            self.z = self.z - self.eta_t[t] * self.alpha_t[t-1] * y[-1]

        if self.prescient:
            self.z = self.z - (self.eta_t[t] * self.alpha_t[t] * y[-1])

            #print("%f %lf %f" % (self.eta_t[t], self.alpha_t[t], y[-1]))
            #print(self.eta_t[t] * self.alpha_t[t] * y[-1])

        #if len(y) > 0:
        #    self.z = self.z - self.eta_t[t] * self.alpha_t[t] * y[-1]
        #else:
        #    self.z = self.z - self.eta_t[t] * self.alpha_t[t] * self.y0
        #print("Before projection: ", self.z)
        return self.z
        
    #def get_update(self, x, g, t):
    #    return x- self.eta_t[t] * g

class OOMD():

    def __init__(self, f, d, weights, x0, y0, xminushalf, yminushalf, eta_t, bounds, yfirst):
        self.name = "Opt-OMD"
        self.f = f
        self.d = d
        self.alpha_t = weights
        self.eta_t = eta_t
        self.z0 = x0
        self.z = x0
        self.z_half = xminushalf
        self.y0 = y0
        self.yfirst = yfirst
        self.yminushalf = yminushalf
        self.bounds = bounds

    def get_update_x(self, y, t):
        
        self.z = self.z_half - self.eta_t[t] * self.alpha_t[t] * y[-1]
        return self.z

        '''if not self.yfirst:
            if t == 0:
                self.z = self.z_half - self.eta_t[t] * self.alpha_t[t] * self.y0
            else:
                self.z = self.z_half - self.eta_t[t] * self.alpha_t[t] * y[-1]
            return self.z
        else:
            if t <= 1:
                self.z = self.z_half - self.eta_t[t] * self.alpha_t[t] * self.y0
            else:
                self.z = self.z_half - self.eta_t[t] * self.alpha_t[t] * y[-2]
            return self.z'''
        #if len(y) >= 2:
        #    if t == 0:
        #        self.z = self.z_half - self.eta_t[t] * self.alpha_t[t] * self.y0
        #        self.z_half = self.z_half + self.eta_t[t] * self.alpha_t[t] * y[-1]
        #    else:
        #        self.z = self.z_half - self.eta_t[t] * self.alpha_t[t] * y[-2]
        #        self.z_half = self.z_half + self.eta_t[t] * self.alpha_t[t] * y[-1]
        #else:
        #if t == 0:
        #    self.z = self.z_half - self.eta_t[t] * self.alpha_t[t] * self.y0
        #    self.z_half = self.z_half - self.eta_t[t] * self.alpha_t[t] * self.y0
        #elif t == 1:
        #    self.z = self.z_half - self.eta_t[t] * self.alpha_t[t] * self.y0
        #    print(self.z)
        #    self.z_half = self.z_half - self.eta_t[t] * self.alpha_t[t] * y[-1]
        #    print(self.z_half)
        #else:
        #    self.z = self.z_half - self.eta_t[t] * self.alpha_t[t] * y[-2]
        #    self.z_half = self.z_half - self.eta_t[t] * self.alpha_t[t] * y[-1]
        #print("xhalf[%d] = %lf" % (t, self.z_half))
    
    def update_half(self, y, t):
        self.z_half = self.z_half - self.eta_t[t] * self.alpha_t[t] * y[-1]



        

    

class FTL:

    def __init__(self, f, d, weights, z0, bounds, prescient = False):
        self.name = "FTL"
        self.prescient = prescient
        if self.prescient:
            self.name += "+"
        self.f = f
        self.d = d
        self.alpha_t = weights
        self.z0 = z0
        self.weighted_sum = np.zeros(shape = (self.d))
        self.bounds = bounds
        

    def get_update_y(self, x, t):

        if t == 1 and not self.prescient:
            #self.weighted_sum += self.alpha_t[t] * self.z0
            return self.z0
        elif not self.prescient:
            self.weighted_sum += self.alpha_t[t-1] * x[-1]
            update = self.weighted_sum / np.sum(self.alpha_t[1 : t])
            return self.f.grad(update)
        else:
            self.weighted_sum += self.alpha_t[t] * x[-1]
            print("FTL weighted sum %lf" % self.weighted_sum)
            update = self.weighted_sum / np.sum(self.alpha_t[1 : t+1])
            return self.f.grad(update)
            #self.weighted_sum += self.alpha_t[t-1] * x[-1]
            #update = self.weighted_sum / np.sum(self.alpha_t[0 : t])

            #return self.f.grad(update)

            #weighted_sum = np.zeros(shape = (self.d))
            #return self.f.grad(update)
            #for i in range(0, t):
            #    print(i)
            #    weighted_sum += (self.alpha_t[i] * x[i])
            #print(weighted_sum)
            #print(self.f.grad(weighted_sum / sum(self.alpha_t[0:t])))
            #update = weighted_sum / np.sum(self.alpha_t[0:t])
            
            #return weighted_sum / sum(self.alpha_t[0:t])
            
            # I think we can just do this...this way we defer computation of the gradient to the actual function.
            # Maybe this is a little silly, because I need to compute the gradient to perform the update, but if I could
            # compute the gradient then I wouldn't need to do this...anyway, as a proof of concept yes this works

            
            #return update * np.sqrt(1.0 / (1 + np.power(update, 2)))
            #return np.sqrt(weighted_sum / (sum(self.alpha_t[0:t]) * (1 + weighted_sum / (sum(self.alpha_t[0:t])))))

class OFTL:

    # Not sure where there is an eta here?
    def __init__(self, f, d, weights, z0, bounds):
        self.name = "Opt-FTL"
        self.f = f
        self.d = d
        self.alpha_t = weights
        self.z0 = z0
        #self.eta = eta
        self.weighted_sum = np.zeros(shape = (self.d))
        self.bound = bounds

    def get_update_y(self, x, t):

        if t == 1:
            return self.f.grad(x[-1])
        else:
            self.weighted_sum += self.alpha_t[t-1] * x[-1]
            update = (self.weighted_sum + self.alpha_t[t] * x[-1]) / np.sum(self.alpha_t[1:t+1])
            return self.f.grad(update)

class FTRL:

    def __init__(self, f, d, weights, z0, eta, reg, bounds, prescient = False):
        self.name = "FTRL"
        self.prescient = prescient
        if self.prescient:
            self.name += "+"
        self.f = f
        self.d = d
        self.alpha_t = weights
        self.z0 = z0
        self.eta = eta
        self.regularizer = reg #r"$\frac{1}{2} \sqrt{t} ||x||^2$"
        self.bounds = bounds
        self.weighted_sum = np.zeros(shape = (self.d))
        #self.weighted_sum += self.z0
        

    def get_update_x(self, y, t):
        
        if t == 1 and not self.prescient:
            return self.regularizer.fenchel_grad(0, 1)
        elif not self.prescient:
            self.weighted_sum += self.alpha_t[t-1] * y[-1]
            return self.regularizer.fenchel_grad(-self.eta * self.weighted_sum, 1)
        elif self.prescient:
            self.weighted_sum += self.alpha_t[t] * y[-1]
            return self.regularizer.fenchel_grad(-self.eta * self.weighted_sum, 1)
        #return -self.eta * self.weighted_sum 
        #return -self.weighted_sum / np.sqrt(t + 1)

    # Assumes R(x,t) = 1/2 sqrt(t) ||x||^2
    #def get_update_x(self, g, t):
    #    return -np.average(g, axis = 1, weights = self.alpha_t[0:t]) * np.sum(self.alpha_t[0:t]) / np.sqrt(t)

class OFTRL:

    def __init__(self, f, d, weights, z0, bounds):
        self.name = "Opt-FTRL"
        self.f = f
        self.d = d
        self.alpha_t = weights
        self.hints = None       # Uses hint m(t) = \ell(t-1), the previous loss
        self.z0 = z0
        self.regularizer = r"$\frac{1}{2} \sqrt{t} ||x||^2$"
        self.weighted_sum = np.zeros(shape = (self.d))

    # This is the closed form for the update of x - the x update is really easy,
    # since there is no dependence on the Fenchel conjugate.

    # DOUBLE CHECK THERE ARE NO OFF-BY-ONE ERRORS!
    def get_update_x(self, y, t):

        if t == 1:
            return self.regularizer.fenchel_grad(-self.alpha_[t] * y[-1])
        else:
            self.weighted_sum += self.alpha_t[t-1] * y[-1]
            update = -(self.weighted_sum + self.alpha_t[t] * y[-1])
            return self.regularizer.fenchel_grad(update)

        #self.weighted_sum += self.alpha_t[t-1] * y[-1]
        #update = (self.weighted_sum + self.alpha_t[t] * y[-1])
        #return -update / np.sqrt(t + 1)
        #weighted_sum = np.zeros(shape = (self.d))
        #for i in range(0, t-1):
        #    weighted_sum += self.alpha_t[i] * y[i]
        #weighted_sum += self.alpha_t[t] * y[t-1]

        #return -weighted_sum / np.sqrt(t + 1)

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

