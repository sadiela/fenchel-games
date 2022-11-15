import numpy as np 
import matplotlib.pyplot as plt
import csv as csv
from ol_algorithms import *
from convex_functions import *

FUNCTION_DICT = {"FTRL" : FTRL, "FTL": FTL, "BestResp": BestResponse, "OMD": OMD, "OOMD": OOMD}

def projection(x, bounds):
    # projection into hypercube
    # for n dimensional x, you will have a list of n sets of bounds
    for i in range(len(bounds)): 
        x[i] = np.minimum(np.maximum(x, bounds[i][0]), bounds[i][1])
    return x

class Fenchel_Game:

    def __init__(self, f, x_init, y_init, xbounds, ybounds, iterations, weights, d = 1):
        
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

        self.x = [x_init] #np.zeros(shape = (self.d, self.T+1), dtype = float)
        self.y = [y_init] # np.zeros(shape = (self.d, self.T+1), dtype = float)

        self.gx = [self.f.grad_x(self.x[0], self.y[0])] #np.zeros(shape = (self.d, self.T+1), dtype = float)
        self.gy = [-self.f.grad_y(self.x[0], self.y[0])] #np.zeros(shape = (self.d, self.T+1), dtype = float)

    def set_players(self, x_alg, y_alg):
        self.algo_X = x_alg
        self.algo_Y = y_alg

    # This just runs FTRL vs. FTRL like the homework
    def run(self, yfirst=True):
        print(self.y[-1], self.x[-1])
        # FTL + BEST RESP
        for t in range(1, self.T):

            print("Updating round t = %d" % t)

            # update y
            if yfirst: 
                print("--> Return y[%d], computing with N = %d gradients from t = 0 to t = %d" % (t, len(self.gy), t-1))
                #self.y[:, t] = np.minimum(np.maximum(self.algo_Y.get_update(self.gy[:, 0:t], t+1), self.ybounds[0][0]), self.ybounds[0][1])
                self.y.append(projection(self.algo_Y.get_update_y(self.x[:], t), self.ybounds))
                print("y[%d] = %lf" % (t, self.y[-1]))

                # Compute the subgradients
                print("--> Update gx[%d], using f(x, y[%d])" % (t, t))
                self.gx.append(self.alpha[t] * self.f.grad_x(self.x[-1], self.y[-1]))
                print("gx[%d] = %lf" % (t, self.gx[-1]))

            # update x based on new y value
            print("--> Return x[%d], computing with N = %d gradients from t = 0 to t = %d" % (t, len(self.gx), t))
            #self.x[:, t] = np.minimum(np.maximum(self.algo_X.get_update(self.gx[:, 0:t+1], t+1), self.xbounds[0][0]), self.xbounds[0][1]) # WILL HAVE TO CHANGE FOR LARGER DIMS
            self.x.append(projection(self.algo_X.get_update_x(self.x[-1], self.y[-1], self.xbounds, t), self.xbounds))
            print("x[%d] = %lf" % (t, self.x[-1]))

            # new gy subgradient
            print("--> Update gy[%d], using f(x[%d], y[%d])" % (t, t, t))
            self.gy.append(-self.alpha[t] * self.f.grad_y(self.x[-1], self.y[-1])) # 1 - self.x[:, t]
            print("gy[%d] = %lf" % (t+1, self.gy[-1]))

            if not yfirst and t==T-1:
                print("Finish something")

            #print("xys", self.x[:, t],self.y[:, t], "gs:", self.gx[:, t],self.gy[:, t])

            # Get the losses - doesn't do anything yet
            #self.lt_x[t] = self.f(x(t), y(t))
            #self.lt_y[t] = -self.f(x(t), y(t))
        print(len(self.x), len(self.y))
        print(self.x)

        self.x_star = np.average(np.concatenate(self.x, axis=0), weights=self.alpha, axis=0)
        self.y_star = np.average(np.concatenate(self.y, axis=0), weights=self.alpha, axis=0)

    def plot_trajectory_2D(self):

        plt.plot(self.x[:-1], self.y[:-1], '--b', linewidth = 1.0)
        plt.plot(self.x[:-1], self.y[:-1], '*r', linewidth = 1.0)
        plt.show()

    def plot_xbar(self):

        weighted_sum = np.zeros(shape = (self.d, 1))
        self.xbar = np.zeros(shape = (self.d, self.T))

        for t in range(1, self.T):
            print(sum(self.alpha[0:t]))
            weighted_sum += (self.alpha[t-1] * self.x[t])
            self.xbar[:, t] = weighted_sum / np.sum(self.alpha[0:t])
            print( self.xbar[:, t])

        t_plot = np.linspace(1, self.T, self.T)

        plt.plot(t_plot, self.xbar[0, :], '--b', linewidth = 1.0)
        #plt.plot(self.x[0, :-1], self.y[0, :-1], '*r', linewidth = 1.0)
        plt.show()

    def plot_x(self):

        t_plot = np.linspace(1, self.T, self.T)
        plt.plot(t_plot, self.x[0:self.T], '--b', linewidth = 1.0)
        #plt.plot(self.x[0, :-1], self.y[0, :-1], '*r', linewidth = 1.0)
        plt.show()

    def save_trajectories(self):

        log_file = "x_" + self.algo_X.name + "_y_" + self.algo_Y.name + "_data.csv"

        with open(log_file, 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow(self.x[:])
            writer.writerow(self.y[:])
        csvfile.close()


if __name__ == "__main__":
    
    '''T = 1000
    alpha_t = np.ones(shape = (1, T), dtype = int)
    m_game = Fenchel_Game(f = f_homework, d = 1)

    m_game.set_player("X", "FTRL", params = None)
    m_game.set_player("Y", "FTRL", params = None)

    m_game.run(iterations = T, weights = alpha_t)
    
    print("Saddle Point (x*, y*) = (%0.3f, %0.3f)" % (m_game.x_star, m_game.y_star))
    
    m_game.plot_trajectory_2D()'''

    function = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    T = 10
    #alpha_t = np.ones(shape = (1, T), dtype = int)
    alpha_t = np.linspace(1, T, T)
    print(alpha_t)
    xbounds = [[-10, 10]]
    ybounds = [[-10, 10]]

    print(xbounds[0][0])
    print(xbounds[0][1])

    x_init = np.array([0.4])
    y_init = np.array([0.3])

    m_game = Fenchel_Game(f = function, x_init=x_init, y_init=y_init, xbounds=xbounds, ybounds=ybounds, iterations=T, weights=alpha_t, d = 1)

    bestresp = BestResponse(m_game.d, m_game.f, m_game.alpha, m_game.xbounds, m_game.ybounds)
    ftl = FTL(m_game.f, m_game.d, m_game.alpha)

    m_game.set_players(bestresp, ftl)

    m_game.run()
    
    print("Saddle Point (x*, y*) = (%0.3f, %0.3f)" % (m_game.x_star, m_game.y_star))
    print("Final iterate:",m_game.x[-1], m_game.y[-1])
    
    m_game.plot_trajectory_2D()

    #m_game.save_trajectories()

    m_game.plot_xbar()

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