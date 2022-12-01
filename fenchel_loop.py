import numpy as np 
import matplotlib.pyplot as plt
import csv as csv
from ol_algorithms import *
from convex_functions import *

def projection(x, bounds):
    # projection into hypercube
    # for n dimensional x, you will have a list of n sets of bounds
    for i in range(len(bounds)): 
        x[i] = np.minimum(np.maximum(x[i], bounds[i][0]), bounds[i][1])
    return x

def debug_print(print_str, T):
    if T <= 100:
        print(print_str)

class Fenchel_Game:

    def __init__(self, f, xbounds, ybounds, iterations, weights, d = 1):
        
        self.f = f
        self.d = d
        self.x_star = 0
        self.y_star = 0
        self.xbounds = xbounds
        self.ybounds = ybounds

        self.T = iterations
        self.alpha = weights    

        #self.acl_x = np.zeros(shape = (1, self.T))
        #self.acl_y = np.zeros(shape = (1, self.T))

        # Initialize each algorithm
        self.x = [] #[x_init] #np.zeros(shape = (self.d, self.T+1), dtype = float)
        self.y = [] #[y_init] # np.zeros(shape = (self.d, self.T+1), dtype = float)

        # Pay loss
        #self.loss_x = [(self.alpha[0] * self.f.payoff(self.x[0], self.y[0]))]
        #self.loss_y = [(self.alpha[0] * -self.f.payoff(self.x[0], self.y[0]))]

        self.loss_x = []
        self.loss_y = []

        # Update gradients
        #self.gx = [self.f.grad_x(self.x[0], self.y[0])] #np.zeros(shape = (self.d, self.T+1), dtype = float)
        #self.gy = [-self.f.grad_y(self.x[0], self.y[0])] #np.zeros(shape = (self.d, self.T+1), dtype = float)

        self.gx = []
        self.gy = []

        self.acl_x = []
        self.acl_y = []

        #self.acl_x = [self.loss_x[0] / self.alpha[0]]
        #self.acl_y = [self.loss_y[0] / self.alpha[0]]

    def set_players(self, x_alg, y_alg):
        self.algo_X = x_alg
        self.algo_Y = y_alg

    def set_teams(self, x_team, y_team, lr = 0.5, wtx1 = (0.5, 0.75), wty1 = (0.5, 0.75)):
        self.team_X = x_team
        self.team_Y = y_team
        self.lr = lr
        self.x_dist = []
        self.x_dist.append(wtx1[0])
        self.y_dist = []
        self.y_dist.append(wty1[0])

    def run_teams(self, yfirst = True):

        for t in range(0, self.T):

            if t % (self.T/10) == 0:
                print("Updating round t = %d" % t)

            # Update both Y players

            # Sample from Y team according to current probability distribution

            # Update both X players

            # Sample from X team according to current probability distribution

            # Update probability distributions and regret accordingly

    def run(self, yfirst = True):
        #print(self.y[-1], self.x[-1])

        for t in range(0, self.T):

            if t % (self.T/10) == 0:
                print("Updating round t = %d" % t)

            # update y
            if yfirst: 
                debug_print("--> Return y[%d], computing with N = %d gradients from t = 0 to t = %d" % (t, len(self.gy), t), self.T)
                #self.y[:, t] = np.minimum(np.maximum(self.algo_Y.get_update(self.gy[:, 0:t], t+1), self.ybounds[0][0]), self.ybounds[0][1])
                
                self.y.append(projection(self.algo_Y.get_update_y(self.x, t), self.ybounds))
                debug_print("y[%d] = %lf" % (t, self.y[-1]), self.T)
                
                # Compute the subgradients
                #debug_print("--> Update gx[%d], using f(x, y[%d])" % (t, t), self.T)
                #self.gx.append(self.alpha[t] * self.f.grad_x(0, self.y))
                #debug_print("gx[%d] = %lf" % (t, self.gx[-1]), self.T)

            # update x based on new y value
            debug_print("--> Return x[%d], computing with N = %d gradients from t = 0 to t = %d" % (t, len(self.gx), t), self.T)
            #self.x[:, t] = np.minimum(np.maximum(self.algo_X.get_update(self.gx[:, 0:t+1], t+1), self.xbounds[0][0]), self.xbounds[0][1]) # WILL HAVE TO CHANGE FOR LARGER DIMS
            self.x.append(projection(self.algo_X.get_update_x(self.y, t), self.xbounds))
            #self.loss_x.append((self.alpha[t] * self.f.payoff(self.x[-1], self.y[-1])))
            #self.acl_x.append(np.sum(self.loss_x) / np.sum(self.alpha[0:t+1]))

            debug_print("x[%d] = %lf" % (t, self.x[-1]), self.T)
            #debug_print("lx[%d] = \u03B1[%d] * g(x[%d], y[%d]) = %lf" % (t, t, t, t, self.loss_x[-1]), self.T)

            # new gy subgradient
            #debug_print("--> Update gy[%d], using f(x[%d], y[%d])" % (t, t, t), self.T)
            #self.gy.append(-self.alpha[t] * self.f.grad_y(self.x[-1], self.y[-1])) # 1 - self.x[:, t]
            #debug_print("gy[%d] = %lf" % (t+1, self.gy[-1]), self.T)

            #self.loss_y.append((self.alpha[t] * -self.f.payoff(self.x[-1], self.y[-1])))
            #self.acl_y.append(np.sum(self.loss_y) / np.sum(self.alpha[0:t+1]))
            #debug_print("ly[%d] = -\u03B1[%d] * g(x[%d], y[%d]) = %lf" % (t, t, t-1, t, self.loss_y[-1]), self.T)
            

            if not yfirst and t==T-1:
                print("Finish something")

        print("Fenchel game complete, T = [%d, %d, %d] rounds" % (self.T, len(self.x), len(self.y)))

        weighted_sum = self.x[0] * self.alpha[0]
        self.xbar = [weighted_sum / self.alpha[0]] 

        for t in range(1, self.T):
            print(sum(self.alpha[0:t]))
            weighted_sum += (self.alpha[t] * self.x[t])
            self.xbar.append(weighted_sum / np.sum(self.alpha[0:t+1]))

        self.x_star = np.average(np.concatenate(self.x, axis=0), weights = self.alpha, axis=0)
        self.y_star = np.average(np.concatenate(self.y, axis=0), weights = self.alpha, axis=0)

    # Only plot if the data is 2D...difficult to visualize otherwise.
    def plot_trajectory_2D(self):

        if self.d > 1:
            return

        plt.figure()
        plt.plot(self.x[:-1], self.y[:-1], '--b', linewidth = 1.0)
        plt.plot(self.x[:-1], self.y[:-1], '*r', linewidth = 1.0)
        plt.title("Trajectory plot")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def plot_xbar(self):

        if self.d > 1:
            return

        plt.figure()
        plt.plot(self.xbar, '-b', linewidth = 1.0)
        plt.title("xbar plot")
        plt.xlabel("Iteration t")
        plt.ylabel("xbar")
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

    def plot_acl(self):

        print("R(T = %d, X) = %lf" % (self.T, self.acl_x[-1]))
        print("R(T = %d, Y) = %lf" % (self.T, self.acl_y[-1]))

        #t_plot = np.linspace(1, self.T, self.T)
        plt.figure()
        plt.plot(self.acl_x, '-b', label = self.algo_X.name)
        plt.plot(self.acl_y, '-r', label = self.algo_Y.name)

        plt.title("Average Cumulative Loss: X: " + self.algo_X.name + ", Y: " + self.algo_Y.name)
        plt.xlabel("Iteration t")
        plt.ylabel("ACL")
        plt.legend()


        plt.show()
        


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