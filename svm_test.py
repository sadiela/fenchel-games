import matplotlib.pyplot as plt
import numpy as np
from fenchel_loop import *
from convex_optimization_algorithms import *
from ol_algorithms import *
from algorithm_comparison import *
from weights import *

class HingeLoss:

    def __init__(self, xi, yi):
        self.name = "Hinge Loss"
        self.xi = xi
        self.yi = yi

    def f(self, z):
        return np.max(0, 1 - self.yi * np.dot(self.xi, z))

    def grad(self, z):

        if 1 - self.yi * np.dot(self.xi, z) <= 0:
            return 0
        else:
            return -self.yi * self.xi

class SVM_Objective:

    def __init__(self, n, d, x, y, fi, eta):
        
        self.name = "SVM Objective Function"
        self.n = n
        self.d = d
        self.x = x
        self.y = y
        self.m_f = [HingeLoss(self.x[i], self.y[i]) for i in range(0, self.n)]
        self.eta = eta

    def f(self, w):
        
        ret_value = np.zeros(shape = (self.d))
        for i in range(0, self.n):
            ret_value += self.m_f[i].f(w)
        ret_value += (0.5 * self.eta * np.power(np.linalg.norm(w, ord = 2), 2))
        return ret_value

    def grad(self, w):
        ret_value = np.zeros(shape = (self.d))
        for i in range(0, self.n):
            ret_value += self.m_f[i].grad(w)
        ret_value += (self.eta * w)
        return ret_value

    def payoff(self, x, y):
        return 


#n = np.asarray([np.sqrt(3), 0.5])
#w = np.zeros(2)
#w[0] = -1
#w[1] = -n[0]/n[1] * w[0]

#w /= la.norm(w)

#w = np.asarray([np.sqrt(2)/2, np.sqrt(2)/2])

def generate_data_and_labels(w, b, N, d):

    X = np.random.uniform(low = -1.0, high = 1.0, size = (N, d))
    Y = np.zeros(shape = N)

    for i in range(0, N):
        #Y[i] = 1 if np.inner(self.w, X[i]) >= self.b else -1
        Y[i] = np.sign(np.inner(w, X[i]) + b)

    while np.abs(np.sum(Y)) == N:
        X = np.random.uniform(low = -1.0, high = 1.0, size = (N, d))
    
        Y = np.zeros(shape = N)
        for i in range(0, N):
            #Y[i] = 1 if np.inner(self.w, X[i]) >= self.b else -1
            Y[i] = np.sign(np.inner(w, X[i]) + b)

    return X, Y

if __name__ == '__main__':

    print("Salve Munde")
    d = 2
    w_star = np.asarray([np.sqrt(2)/2, np.sqrt(2)/2])
    b = 0

    NUM_TRAINING_POINTS = 200
    NUM_TEST_POINTS = 100

    X, Y = generate_data_and_labels(w = w_star, b = b, N = NUM_TRAINING_POINTS, d = d)

    Z = np.column_stack((X, Y))
   
    X_pos = [z[0:d] for z in Z if z[-1] == 1]    
    X_neg = [z[0:d] for z in Z if z[-1] == -1]

    t = np.linspace(-1, 1, 100)
    plt.figure()
    plt.scatter([x[0] for x in X_pos], [x[1] for x in X_pos], color = 'b', label = "+")
    plt.scatter([x[0] for x in X_neg], [x[1] for x in X_neg], color = 'r', label = "-")

    X_TEST, Y_TEST = generate_data_and_labels(w = w_star, b = b, N = NUM_TRAINING_POINTS, d = d)
    Z_TEST = np.column_stack((X_TEST, Y_TEST))
    X_TEST_pos = [z[0:d] for z in Z_TEST if z[-1] == 1]    
    X_TEST_neg = [z[0:d] for z in Z_TEST if z[-1] == -1]

    #plt.show()

    T = 500
    d = 2

    alpha_t = np.linspace(1, T, T)
    eta_t = 0.25*np.ones(T)

    XBOUNDS = [[-10, 10], [-10, 10]]
    YBOUNDS = [[-10, 10], [-10, 10]]

    X_INIT = np.array([1.0, 1.0])
    Y_INIT = np.array([1.0, 1.0])

    phi = L2Reg()

    f_game = SVM_Objective(n = NUM_TRAINING_POINTS, d = 2, x = X, y = Y, fi = None, eta = 0.5)
    f_opt = SVM_Objective(n = NUM_TRAINING_POINTS, d = 2, x = X, y = Y, fi = None, eta = 0.5)


    x_0 = np.array([1, 1], dtype='float64')

    alpha_t = Weights("linear", T = T)

    ftrl = FTRL(f = f_game, d = d, weights = alpha_t, z0 = x_0, eta = 0.25, reg = phi, bounds = XBOUNDS, prescient = False)
    omd = OMD(f = f_game, d = d, weights = alpha_t, z0 = x_0, y0 = x_0, eta_t = eta_t, bounds = XBOUNDS, prescient = False)
    optimistic_omd = OOMD(f = f_game, d = d, weights = alpha_t, x0 = x_0, xminushalf = x_0, y0 = x_0, yminushalf = x_0, eta_t = eta_t, bounds = XBOUNDS, yfirst = False)
    optimistic_ftrl = OFTRL(f = f_game, d = d, weights = alpha_t, z0 = x_0, reg = phi, bounds = XBOUNDS)

    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t, z0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)
    optimistic_ftl = OFTL(f = f_game, d = d, weights = alpha_t, z0 = f_opt.grad(x_0), bounds = YBOUNDS)
    ftl = FTL(f = f_game, d = d, weights = alpha_t, z0 = x_0, bounds = YBOUNDS, prescient = True)
        
    game_xbar, _ = run_helper(f_game = f_game, x_alg = omd, y_alg = optimistic_ftl, T = T, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = True)

    print(game_xbar[-1]/np.linalg.norm(game_xbar[-1]))

    #plt.figure()
    #plt.plot(x_ts_n_onemem[1:], color = 'blue', label = "NesterovsOne-Memory")
    #plt.plot(x_ts_n_infmem[1:], color = 'blue', label = "NesterovsInf-Memory")
    #plt.scatter([x[0]/np.linalg.norm(x) for x in game_xbar], [x[1]/np.linalg.norm(x) for x in game_xbar], color = 'red', linestyle = '--', label = 'FGNRD Recovery')
    #plt.scatter([game_xbar[-1][0]/np.linalg.norm(game_xbar[-1])], [game_xbar[-1][1]/np.linalg.norm(game_xbar[-1])], color = 'blue')
    #plt.plot(v_ts[1:], color = 'green', linestyle = '--', label = 'vts')
    #plt.plot(xt, color = 'purple', linestyle = '--', label = 'xt')
    #plt.plot(x_ts_n_infmem, color='gray', label="NesterovsInf-Memory")
    #plt.title("Algorithm Recovery")
    #plt.legend()
    #plt.show()

    w_time = []
    t_time = []

    NUM_PLOTS = 6
    NUM_ROWS = 2
    NUM_COLS = 3

    errors = []
    for i in range(0, T):

        # Get current hyperplane
        wi = game_xbar[i] / np.linalg.norm(game_xbar[i])
        errors.append(np.linalg.norm(wi - w_star))

    plt.figure()
    plt.plot(errors, color = 'blue', linestyle = '-', linewidth = 1.5)
    plt.title("Norm Error: " + r'$\|\| w_{t} - w_{*} \|\|$' + " vs. Iteration t")
    plt.xlabel("Iteration t")
    plt.ylabel("Error")
    plt.show()


    i = 1
    j = 1
    while i <= NUM_PLOTS:

        w = game_xbar[j] / np.linalg.norm(game_xbar[j])
        w_time.append(w)
        t_time.append(j)

        i += 1
        j += int(T/NUM_PLOTS)
        print("Using w[%d]..." % j)

    wstar = game_xbar[-1] / np.linalg.norm(game_xbar[-1])



    t = np.linspace(-1, 1, 100)

    fig, axs = plt.subplots(NUM_ROWS, NUM_COLS, sharex = True)
    fontsize = 9

    row = 0
    col = 0
    for x, w in enumerate(w_time):

        print("row = %d, col = %d, w = (%lf, %lf)" % (row, col, w[0], w[1]))

        axs[row, col].set_title("Linearly Separable Hyperplane in " + r'$\mathbb{R}^{2}$' + ", iteration t = " + str(t_time[x]) + ", " + r'$\hat{w} = $' + " [%0.2lf, %0.2lf]^T" % (w[0], w[1]), fontsize = fontsize)
        axs[row, col].scatter([x[0] for x in X_pos], [x[1] for x in X_pos], color = 'b', label = "+")
        axs[row, col].scatter([x[0] for x in X_neg], [x[1] for x in X_neg], color = 'r', label = "-")
        axs[row, col].plot(t, -w[0]/w[1] * t - b/w[1], linewidth = 2.5, linestyle = '--', color = 'k', label = "w")
        axs[row, col].set_xlim([-1, 1])
        axs[row, col].set_ylim([-1, 1])
        axs[row, col].legend()

        col = (col + 1) % NUM_COLS
        if col == 0:
            row += 1

    plt.show()

    w_time = []
    t_time = []

    i = 1
    j = 1
    while i <= NUM_PLOTS:

        w = game_xbar[i] / np.linalg.norm(game_xbar[i])
        w_time.append(w)
        t_time.append(i)

        i += 1
   
    t = np.linspace(-1, 1, 100)

    fig, axs = plt.subplots(NUM_ROWS, NUM_COLS, sharex = True)
    fontsize = 9

    row = 0
    col = 0
    for x, w in enumerate(w_time):

        print("row = %d, col = %d, w = (%lf, %lf)" % (row, col, w[0], w[1]))

        axs[row, col].set_title("Linearly Separable Hyperplane in " + r'$\mathbb{R}^{2}$' + ", iteration t = " + str(t_time[x]) + ", " + r'$\hat{w} = $' + " [%0.2lf, %0.2lf]^T" % (w[0], w[1]), fontsize = fontsize)
        axs[row, col].scatter([x[0] for x in X_pos], [x[1] for x in X_pos], color = 'b', label = "+")
        axs[row, col].scatter([x[0] for x in X_neg], [x[1] for x in X_neg], color = 'r', label = "-")
        axs[row, col].plot(t, -w[0]/w[1] * t - b/w[1], linewidth = 2.5, linestyle = '--', color = 'k', label = "w")
        axs[row, col].set_xlim([-1, 1])
        axs[row, col].set_ylim([-1, 1])
        axs[row, col].legend()

        col = (col + 1) % NUM_COLS
        if col == 0:
            row += 1

    plt.show()
        
        #plt.plot(t, -w[0]/w[1] * t - b/w[1], linewidth = 2.5, linestyle = '--', color = 'k', label = "w")
        #plt.scatter([x[0] for x in X_pos], [x[1] for x in X_pos], color = 'b', label = "+")
        #plt.scatter([x[0] for x in X_neg], [x[1] for x in X_neg], color = 'r', label = "-")
        #plt.title("t = %d" % t_time[x])
        #plt.xlim(-1, 1)
        #plt.ylim(-1, 1)
        #plt.show()


        #axs[0,0].set_title("HI", fontsize=fontsize)
        #axs[0,0].plt.scatter([x[0] for x in X_pos], [x[1] for x in X_pos], color = 'b', label = "+")
        #axs[0,0].plt.scatter([x[0] for x in X_neg], [x[1] for x in X_neg], color = 'r', label = "-")
        #axs[0,0].plt.plot(t, -w[0]/w[1] * t - b/w[1], linewidth = 2.5, linestyle = '--', color = 'k', label = "w")
        #axs[0,0].legend()
'''
    axs[0,1].set_title("GradDescent w/Averaging <-> X:OMD,Y:FTL+", fontsize=fontsize)
    axs[0,1].plot(GDavgco, color = 'blue', linewidth = 1.5, label = "GDw/AVG")
    axs[0,1].plot(GDavgsaddle, color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    axs[0,1].legend()

    axs[0,2].set_title("CumulativeGradDescent <-> X:OMD,Y:FTL+", fontsize=fontsize)
    axs[0,2].plot(CGDco, color = 'blue', linewidth = 1.5, label = "CGD")
    axs[0,2].plot(CGDsaddle, color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    axs[0,2].legend()

    axs[0,3].set_title("SCExtraGradw/Averaging <-> X:OOMD, Y:BR+", fontsize=fontsize)
    axs[0,3].plot(SCEGwAVGco, color = 'blue', linewidth = 1.5, label = "SCEGw/AVG")
    axs[0,3].plot(SCEGwAVGsaddle, color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    axs[0,3].legend()

    axs[1,0].set_title("Nesterov's1-Mem <-> X:OMD+,Y:OFTL", fontsize=fontsize)
    axs[1,0].plot(N1co, color = 'blue', linewidth = 1.5, label = "Nesterov's 1Mem")
    axs[1,0].plot(N1saddle, color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    axs[1,0].legend()

    axs[1,1].set_title("Nesterov'sInf-Mem <-> X:FTRL+,Y:OFTL", fontsize=fontsize)
    axs[1,1].plot(Ninfco, color = 'blue', linewidth = 1.5, label = "Nesterov's 1Mem")
    axs[1,1].plot(Ninfsaddle, color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    axs[1,1].legend()

    axs[1,2].set_title("HeavyBall <-> X:OMD+,Y:FTL", fontsize=fontsize)
    axs[1,2].plot(hbco, color = 'blue', linewidth = 1.5, label = "Heavy Ball")
    axs[1,2].plot(hbsaddle, color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    axs[1,2].legend()

    axs[0,0].tick_params(axis='both', which='major', labelsize=8)
    axs[0,1].tick_params(axis='both', which='major', labelsize=8)
    axs[0,2].tick_params(axis='both', which='major', labelsize=8)
    axs[0,3].tick_params(axis='both', which='major', labelsize=8)
    axs[1,0].tick_params(axis='both', which='major', labelsize=8)
    axs[1,1].tick_params(axis='both', which='major', labelsize=8)
    axs[1,2].tick_params(axis='both', which='major', labelsize=8)
    axs[1,3].tick_params(axis='both', which='major', labelsize=8)
    #axs.tick_params(axis='both', which='minor', labelsize=8)
    for ax in axs.flat:
        ax.set(xlabel='t', ylabel='Iterates')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.show()
'''   
        
    
