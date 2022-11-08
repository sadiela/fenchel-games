import numpy as np 
import matplotlib.pyplot as plt

def FTRL(g, t):
    return -np.sum(g, axis = 1) / np.sqrt(t)

#def get_subgradient():

FUNCTION_DICT = {"FTRL" : FTRL}

class Fenchel_Game:

    def __init__(self, f, d = 1):
        
        self.f = f
        self.d = d
        self.x_star = 0
        self.y_star = 0
        self.Player_X = None
        self.Player_Y = None

    def set_player(self, player_name, alg_name, params = None):

        if player_name == "X":
            self.algo_X = FUNCTION_DICT[alg_name]
        elif player_name == "Y":
            self.algo_Y = FUNCTION_DICT[alg_name]

    # This just runs FTRL vs. FTRL like the homework
    def run(self, iterations, weights):
        self.T = iterations
        self.alpha = weights    # Doesn't do anything yet

        self.x = np.zeros(shape = (self.d, self.T), dtype = float)
        self.y = np.zeros(shape = (self.d, self.T), dtype = float)

        self.gx = np.zeros(shape = (self.d, self.T), dtype = float)
        self.gy = np.zeros(shape = (self.d, self.T), dtype = float)

        for t in range(1, self.T):

            # Run iteration t
            self.x[:, t] = self.algo_X(self.gx[:, 1:t], t+1)
            self.y[:, t] = self.algo_Y(self.gy[:, 1:t], t+1)

            # Get the losses - doesn't do anything yet
            #self.lt_x[t] = self.f(x(t), y(t))
            #self.lt_y[t] = -self.f(x(t), y(t))

            # Compute the subgradients
            self.gx[:, t] = self.y[:, t] - 1
            self.gy[:, t] = 1 - self.x[:, t]

            # Doesn't do anything yet
            #self.gx[t] = getSubgradient(self.lt_x, x[t])
            #self.gy[t] = getSubgradient(self.lt_y, y[t])

        self.x_star = np.average(self.x)
        self.y_star = np.average(self.y)

    def plot_trajectory_2D(self):

        plt.plot(self.x[0, :], self.y[0, :], '--b', linewidth = 1.0)
        plt.show()


#alpha = np.ones((1, T))
#x = np.zeros((d, T))
#y = np.zeros((d, T))

def OMD():
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
    
    T = 1000
    alpha_t = np.ones(shape = (1, T), dtype = int)
    m_game = Fenchel_Game(f = f_homework, d = 1)

    m_game.set_player("X", "FTRL", params = None)
    m_game.set_player("Y", "FTRL", params = None)

    m_game.run(iterations = T, weights = alpha_t)
    
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