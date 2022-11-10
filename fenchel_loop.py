import numpy as np 
import matplotlib.pyplot as plt

#### NO REGRET ALGORITHMS ####

def OMD():
    return 0

def OOMD():
    return 0

def FTL():
    return 0

def FTRL(g, t):
    return -np.sum(g, axis = 1) / np.sqrt(t)

#def get_subgradient():

FUNCTION_DICT = {"FTRL" : FTRL}

### POWER FUNCTION: p,q =2 ###

class PowerFenchel:
    def __init__(self, p,q):
        self.name = "Power function (fenchel)" 
        self.p = p
        self.q = q

    def fenchel(self,theta, p=2, q=2): 
        return (1/2)* np.pow(np.linalg.norm(theta, ord=q), q)

    def payoff(self, x,y, p=2, q=2):
        np.dot(x,y) - self.fenchel(y, p, q)

    def grad_x(self, x,y):
        return y

    def grad_y(self, x,y):
        return x-y

class ExpFenchel:
    def __init__(self):
        self.name = "Exponential function"

    def fenchel(self,theta): 
        return theta* np.log(theta) - theta

    def payoff(self, x,y):
        np.dot(x,y) - self.fenchel(y)

    def grad_x(self, x,y):
        return y

    def grad_y(self, x,y):
        return x- np.log(y)

class Fenchel_Game:

    def __init__(self, f, xbounds, ybounds, iterations, weights, d = 1):
        
        self.f = f
        self.d = d
        self.x_star = 0
        self.y_star = 0
        self.Player_X = None
        self.Player_Y = None
        self.xbounds = xbounds
        self.ybounds = ybounds

        self.T = iterations
        self.alpha = weights    # Doesn't do anything yet

        self.x = np.random.rand(self.d, self.T+1)#, dtype = float)
        self.x[:,0] = 0.5
        self.y = np.random.rand(self.d, self.T+1) #, dtype = float)
        self.y[:,0] = 0.5

        self.gx = np.random.rand(self.d, self.T+1) #zeros(shape = (self.d, self.T), dtype = float)
        self.gy = np.random.rand(self.d, self.T+1) #zeros(shape = (self.d, self.T), dtype = float)

    def set_player(self, player_name, alg_name, params = None):

        if player_name == "X":
            self.algo_X = FUNCTION_DICT[alg_name]
        elif player_name == "Y":
            self.algo_Y = FUNCTION_DICT[alg_name]

    # This just runs FTRL vs. FTRL like the homework
    def run(self):

        for t in range(1, self.T):

            # update y
            self.y[:,t] = np.minimum(np.maximum(self.algo_Y(self.gy[:, 0:t], t), self.ybounds[0]), self.ybounds[1])

            # Compute the subgradients
            self.gx[:, t] = self.f.grad_x(self.x[:,t-1], self.y[:,t])

            # update x based on new y value
            self.x[:,t] = np.minimum(np.maximum(self.algo_X(self.gx[:, 0:t+1], t), self.xbounds[0]), self.xbounds[1]) # WILL HAVE TO CHANGE FOR LARGER DIMS

            # new gy subgradient
            self.gy[:, t+1] = -self.f.grad_y(self.x[:,t], self.y[:,t]) # 1 - self.x[:, t]

            #print("xys", self.x[:, t],self.y[:, t], "gs:", self.gx[:, t],self.gy[:, t])

            # Get the losses - doesn't do anything yet
            #self.lt_x[t] = self.f(x(t), y(t))
            #self.lt_y[t] = -self.f(x(t), y(t))

        self.x_star = np.average(self.x[:,:-1])
        self.y_star = np.average(self.y[:,:-1])

    def plot_trajectory_2D(self):

        plt.plot(self.x[0, :-1], self.y[0, :-1], '--b', linewidth = 1.0)
        plt.show()

    def get_subgradient(f, val):
        return 0


#alpha = np.ones((1, T))
#x = np.zeros((d, T))
#y = np.zeros((d, T))

def OL_Y(t, lxt, yt):
    print("Player Y taking turn t = %d" % t)

def OL_X(t, lyt, xt):
    print("Player X taking turn t = %d" % t)

def f_homework(x, y):
    return (x - 1) * (y - 1)

def reg(x, t):
    return 0.5 * np.sqrt(t) * x**2


if __name__ == "__main__":
    
    '''T = 1000
    alpha_t = np.ones(shape = (1, T), dtype = int)
    m_game = Fenchel_Game(f = f_homework, d = 1)

    m_game.set_player("X", "FTRL", params = None)
    m_game.set_player("Y", "FTRL", params = None)

    m_game.run(iterations = T, weights = alpha_t)
    
    print("Saddle Point (x*, y*) = (%0.3f, %0.3f)" % (m_game.x_star, m_game.y_star))
    
    m_game.plot_trajectory_2D()'''

    function = PowerFenchel(2,2) #ExpFenchel()
    T = 100
    alpha_t = np.ones(shape = (1, T), dtype = int)
    xbounds = [-10,10]
    ybounds = [-10,10]
    m_game = Fenchel_Game(f = function, xbounds=xbounds, ybounds=ybounds, iterations=T, weights=alpha_t, d = 1)

    m_game.set_player("X", "FTRL", params = None)
    m_game.set_player("Y", "FTRL", params = None)

    m_game.run()
    
    print("Saddle Point (x*, y*) = (%0.3f, %0.3f)" % (m_game.x_star, m_game.y_star))
    print("Final iterate:",m_game.x[:,-2], m_game.y[:,-2])
    
    m_game.plot_trajectory_2D()

    # schedules
    #T = 10000 # number of rounds
    #alpha_ts = 1 # or a  vector of length T 

    # Function Initialization - f(x), f*, etc.

    # Algorithm Initialization

    # Set schedule

    # Pick to algorithms

    # Run Fenchel game loop
    #for t in range(1, T + 1):

    #    x[t] = OL_X(t, lxt, y[t-1])
    #    y[t] = OL_Y(t, lyt, x[t-1])

    # Return saddle point guess - use to find min f(x)

        


    # Pair up "fair players" FTL & OMD

    # what functions are we interested in optimizing? How hard is it to calc the fenchel conjugate?!?
    # power function
    # search OCO literature
    # convex loss functions? square loss, hinge loss, smoothed hinge loss, modified square loss, exponential loss function
    # log loss function 
    # which loss function/OL algorithm combos are "allowed"
    # https://core.ac.uk/download/pdf/213011306.pdf