import numpy as np 
import matplotlib.pyplot as plt
import csv as csv

#### NO REGRET ALGORITHMS ####

def OMD():
    return 0

def OOMD():
    return 0

#def FTL():
#    return 0

# This uses the regularizer \psi(t) = 0.5 * sqrt(t) * |x|^2
#def FTRL(g, t):
#    return -np.sum(g, axis = 1) / np.sqrt(t)

class FTRL:

    def __init__(self):
        self.name = "FTRL"

    def get_update(self, g, t):
        return -np.sum(g, axis = 1) / np.sqrt(t)

class BestResponse:

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
    def get_update_x(self, x, y, xbounds):

        x_ret = np.ones(shape = (self.d, 1))
        for i in range(0, self.d):
            x_ret[i] = self.alpha_t[i] * max(abs(xbounds[i][0]), abs(xbounds[i][1])) * -1 * np.sign(y[i])

        return x_ret

    # This doesn't do anything yet either
    def get_update_y(self, x, y, t, ybounds):
        return x


class FTL:

    def __init__(self, d, z0, weights):
        self.name = "FTL"
        self.d = d
        self.z0 = z0
        self.alpha_t = weights

    # This doesn't do anything
    def get_update_y(self, x, t):

        weighted_sum = np.zeros(shape = (self.d, 1))
        for i in range(0, t):
            weighted_sum += self.alpha_t[i] * x[:, i]
        return weighted_sum / sum(self.alpha_t[0:t])

#def get_subgradient():

FUNCTION_DICT = {"FTRL" : FTRL}

### POWER FUNCTION: p,q =2 ###

class PowerFenchel:
    def __init__(self, p,q):
        self.name = "Power function (fenchel)" 
        self.p = p
        self.q = q

    def fenchel(self, theta, p = 2, q = 2): 
        return (1/2) * np.pow(np.linalg.norm(theta, ord = q), q)

    def payoff(self, x,y, p = 2, q = 2):
        return np.dot(x, y) - self.fenchel(y, p, q)

    def grad_x(self, x, y):
        return y

    def grad_y(self, x, y):
        return x - y

class ExpFenchel:
    def __init__(self):
        self.name = "Exponential function"

    def fenchel(self,theta): 
        return theta* np.log(theta) - theta

    def payoff(self, x,y):
        return np.dot(x,y) - self.fenchel(y)

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

        self.x = np.zeros(shape = (self.d, self.T+1), dtype = float)
        
        self.y = np.zeros(shape = (self.d, self.T+1), dtype = float)
       

        self.gx = np.zeros(shape = (self.d, self.T+1), dtype = float)
        self.gy = np.zeros(shape = (self.d, self.T+1), dtype = float)

        

        # This is the old init style, will remove once we verify style above works
        #self.x = np.random.rand(self.d, self.T+1)#, dtype = float)
        #self.x[:,0] = 0.5
        #self.y = np.random.rand(self.d, self.T+1) #, dtype = float)
        #self.y[:,0] = 0.5

        #self.gx = np.random.rand(self.d, self.T+1) #zeros(shape = (self.d, self.T), dtype = float)
        #self.gy = np.random.rand(self.d, self.T+1) #zeros(shape = (self.d, self.T), dtype = float)

    def set_player(self, player_name, alg_name, params = None):

        if player_name == "X":
            self.algo_X = BestResponse(self.d, self.f, self.alpha, self.xbounds, self.ybounds)
            self.x[:, 0] = 0.4

            #self.algo_X = FTRL()
        elif player_name == "Y":
            #self.algo_Y = FTRL()
            self.algo_Y = FTL(z0 = 0.3, d = self.d, weights = self.alpha)

            self.y[:, 0] = self.algo_Y.z0
            
        self.gx[:, 0] = self.f.grad_x(self.x[:, 0], self.y[:, 0])
        self.gy[:, 0] = -self.f.grad_y(self.x[:, 0], self.y[:, 0])

        #if player_name == "X":
        #    self.algo_X = FUNCTION_DICT[alg_name]
        #elif player_name == "Y":
        #    self.algo_Y = FUNCTION_DICT[alg_name]

    # This just runs FTRL vs. FTRL like the homework
    def run(self):

        for t in range(1, self.T):

            print("Updating round t = %d" % t)

            # update y
            print("--> Return y[%d], computing with N = %d gradients from t = 0 to t = %d" % (t, len(self.gy[0, 0:t]), t-1))
            #self.y[:, t] = np.minimum(np.maximum(self.algo_Y.get_update(self.gy[:, 0:t], t+1), self.ybounds[0][0]), self.ybounds[0][1])
            self.y[:, t] = np.minimum(np.maximum(self.algo_Y.get_update_y(self.x[:, 0:t], t), self.ybounds[0][0]), self.ybounds[0][1])
            print("y[%d] = %lf" % (t, self.y[:, t]))

            # Compute the subgradients
            print("--> Update gx[%d], using f(x, y[%d])" % (t, t))
            self.gx[:, t] = self.alpha[t] * self.f.grad_x(self.x[:, t-1], self.y[:, t])
            print("gx[%d] = %lf" % (t, self.gx[:, t]))

            # update x based on new y value
            print("--> Return x[%d], computing with N = %d gradients from t = 0 to t = %d" % (t, len(self.gx[0, 0:t+1]), t))
            #self.x[:, t] = np.minimum(np.maximum(self.algo_X.get_update(self.gx[:, 0:t+1], t+1), self.xbounds[0][0]), self.xbounds[0][1]) # WILL HAVE TO CHANGE FOR LARGER DIMS
            self.x[:, t] = np.minimum(np.maximum(self.algo_X.get_update_x(self.x[:, t], self.y[:, t], self.xbounds), self.xbounds[0][0]), self.xbounds[0][1]) # WILL HAVE TO CHANGE FOR LARGER DIMS
            print("x[%d] = %lf" % (t, self.x[:, t]))

            # new gy subgradient
            print("--> Update gy[%d], using f(x[%d], y[%d])" % (t, t, t))
            self.gy[:, t+1] = -self.alpha[t] * self.f.grad_y(self.x[:, t], self.y[:, t]) # 1 - self.x[:, t]
            print("gy[%d] = %lf" % (t+1, self.gy[:, t+1]))

            #print("xys", self.x[:, t],self.y[:, t], "gs:", self.gx[:, t],self.gy[:, t])

            # Get the losses - doesn't do anything yet
            #self.lt_x[t] = self.f(x(t), y(t))
            #self.lt_y[t] = -self.f(x(t), y(t))

        self.x_star = np.average(self.x[:, :-1])
        self.y_star = np.average(self.y[:, :-1])

    def plot_trajectory_2D(self):

        plt.plot(self.x[0, :-1], self.y[0, :-1], '--b', linewidth = 1.0)
        plt.plot(self.x[0, :-1], self.y[0, :-1], '*r', linewidth = 1.0)
        plt.show()

    def plot_xbar(self):

        weighted_sum = np.zeros(shape = (self.d, 1))
        xbar = np.zeros(shape = (self.d, self.T))

        for t in range(0, self.T):
            print(sum(self.alpha[0:t]))
            weighted_sum += (self.alpha[t] * self.x[:, t])
            xbar[:, t] = weighted_sum / np.sum(self.alpha[0:t+1])
            print(xbar[:, t])

        t_plot = np.linspace(1, self.T, self.T)

        plt.plot(t_plot, xbar[0, :], '--b', linewidth = 1.0)
        #plt.plot(self.x[0, :-1], self.y[0, :-1], '*r', linewidth = 1.0)
        plt.show()

    def plot_x(self):

        t_plot = np.linspace(1, self.T, self.T)
        plt.plot(t_plot, self.x[0, 0:self.T], '--b', linewidth = 1.0)
        #plt.plot(self.x[0, :-1], self.y[0, :-1], '*r', linewidth = 1.0)
        plt.show()

    def save_trajectories(self):

        log_file = "x_" + self.algo_X.name + "_y_" + self.algo_Y.name + "_data.csv"

        with open(log_file, 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow(self.x[0, :])
            writer.writerow(self.y[0, :])
        csvfile.close()

        '''
        log_file_x = "x_data.csv"
        log_file_y = "y_data.csv"

        with open(log_file_x, 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow(self.x[0, :])
        csvfile.close()

        with open(log_file_y, 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow(self.y[0, :])
        csvfile.close()
        '''

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

    function = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    T = 10
    #alpha_t = np.ones(shape = (1, T), dtype = int)
    alpha_t = np.linspace(1, T, T)
    print(alpha_t)
    xbounds = [[-10, 10]]
    ybounds = [[-10, 10]]

    print(xbounds[0][0])
    print(xbounds[0][1])

    m_game = Fenchel_Game(f = function, xbounds=xbounds, ybounds=ybounds, iterations=T, weights=alpha_t, d = 1)

    m_game.set_player("X", "FTRL", params = None)
    m_game.set_player("Y", "FTRL", params = None)

    m_game.run()
    
    print("Saddle Point (x*, y*) = (%0.3f, %0.3f)" % (m_game.x_star, m_game.y_star))
    print("Final iterate:",m_game.x[:,-2], m_game.y[:,-2])
    
    m_game.plot_trajectory_2D()

    m_game.save_trajectories()

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