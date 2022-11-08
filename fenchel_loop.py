import numpy as np 
import matplotlib.pyplot as plt

def FTRL(g, t):
    return -np.sum(g, axis = 1) / np.sqrt(t)

#def get_subgradient():

FUNCTION_DICT = {"FTRL" : FTRL}

class Fenchel_Game:

    def __init__(self, f, iterations, weights, d = 1):
        
        self.f = f
        self.d = d
        self.x_star = 0
        self.y_star = 0
        self.Player_X = None
        self.Player_Y = None

        self.T = iterations
        self.alpha = weights    # Doesn't do anything yet

        self.x = np.random.rand(self.d, self.T)#, dtype = float)
        self.x[:,0] = 0.5
        self.y = np.random.rand(self.d, self.T) #, dtype = float)
        self.y[:,0] = 0.5

        self.gx = np.random.rand(self.d, self.T) #zeros(shape = (self.d, self.T), dtype = float)
        self.gy = np.random.rand(self.d, self.T) #zeros(shape = (self.d, self.T), dtype = float)

    def set_player(self, player_name, alg_name, params = None):

        if player_name == "X":
            self.algo_X = FUNCTION_DICT[alg_name]
        elif player_name == "Y":
            self.algo_Y = FUNCTION_DICT[alg_name]

    # This just runs FTRL vs. FTRL like the homework
    def run(self):

        for t in range(0, self.T):

            # Compute the subgradients
            self.gx[:, t] = grad_power_x(self.x[:,t], self.y[:,t]) # self.y[:, t] - 1
            self.gy[:, t] = -grad_power_y(self.x[:,t], self.y[:,t]) # 1 - self.x[:, t]


            # Run iteration t
            self.x[:, t] = self.algo_X(self.gx[:, 0:t], t+1)
            self.y[:, t] = self.algo_Y(self.gy[:, 0:t], t+1)

            #print("xys", self.x[:, t],self.y[:, t], "gs:", self.gx[:, t],self.gy[:, t])

            #print(self.x[:, t],self.y[:, t])

            # Get the losses - doesn't do anything yet
            #self.lt_x[t] = self.f(x(t), y(t))
            #self.lt_y[t] = -self.f(x(t), y(t))

            # Doesn't do anything yet
            #self.gx[t] = getSubgradient(self.lt_x, x[t])
            #self.gy[t] = getSubgradient(self.lt_y, y[t])

        self.x_star = np.average(self.x)
        self.y_star = np.average(self.y)

    def plot_trajectory_2D(self):

        plt.plot(self.x[0, :], self.y[0, :], '--b', linewidth = 1.0)
        plt.show()

    def get_subgradient(f, val):
        return 0


#alpha = np.ones((1, T))
#x = np.zeros((d, T))
#y = np.zeros((d, T))

### POWER FUNCTION: p,q =2 ###
def power_fenchel(theta, p=2, q=2): 
    return (1/2)* np.pow(np.linalg.norm(theta, ord=q), q)

def power_payoff(x,y):
    np.dot(x,y) - power_fenchel(y)

def grad_power_x(x,y):
    return y

def grad_power_y(x,y):
    return x-y


def g_fenchel_hinge(): 
    return 0

def f():
    return 0

def g(x,y):
    return 0 

def OMD():
    return 0

def OOMD():
    return 0

def FTL():
    return 0

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

    T = 1000
    alpha_t = np.ones(shape = (1, T), dtype = int)
    m_game = Fenchel_Game(f = power_payoff, iterations=T, weights=alpha_t, d = 1)

    m_game.set_player("X", "FTRL", params = None)
    m_game.set_player("Y", "FTRL", params = None)

    m_game.run()
    
    print("Saddle Point (x*, y*) = (%0.3f, %0.3f)" % (m_game.x_star, m_game.y_star))
    
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