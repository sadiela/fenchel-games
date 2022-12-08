
import matplotlib.pyplot as plt
import numpy as np
from fenchel_loop import *
from convex_optimization_algorithms import *
from ol_algorithms import *

class Weights:

    def __init__(self, name, T):
        self.name = name
        self.T = T

        if self.name == "ones":
            self.weights = np.ones(self.T + 1)
        elif self.name == "linear":
            self.weights = np.linspace(0, self.T, self.T + 1)
        elif self.name == "sqrt":
            self.weights = [np.sqrt(t) for t in range(0, self.T + 1)]
            self.weights[0] = 0
        elif self.name == "log":
            self.weights = [np.log(t) for t in range(0, self.T + 1)]
            self.weights[0] = 0

    def print_weights(self):

        for t in range(0, self.T):
            print("\u03B1[%d] = %lf" % (t, self.weights[t]))

    def plot_weights(self):

        plt.plot(self.weights, '-b', linewidth = 1.5)
        plt.title("Weight schedule: " + self.name)
        plt.show()



def run_helper(f_game, x_alg, y_alg, T, d, weights, xbounds, ybounds, yfirst = True):

    print("Running algorithms X = " + x_alg.name + ", Y = " + y_alg.name)

    m_game = Fenchel_Game(f = f_game, xbounds = xbounds, ybounds = ybounds, iterations = T, weights = weights, d = d)
    m_game.set_players(x_alg = x_alg, y_alg = y_alg)
    
    m_game.run(yfirst = yfirst)
    
    print("Saddle Point (x*, y*) = (%0.3f, %0.3f)" % (m_game.x_star, m_game.y_star))
    print("Final iterate:", m_game.x[-1], m_game.y[-1])

    #game_xbar = m_game.xbar

    #print(game_xbar[-1]/np.linalg.norm(game_xbar[-1]))

    #plt.figure()
    #plt.plot(x_ts_n_onemem[1:], color = 'blue', label = "NesterovsOne-Memory")
    #plt.plot(x_ts_n_infmem[1:], color = 'blue', label = "NesterovsInf-Memory")
    #plt.plot([x[0]/np.linalg.norm(x) for x in game_xbar], [x[1]/np.linalg.norm(x) for x in game_xbar], color = 'red', linestyle = '--', label = 'FGNRD Recovery')
    #plt.show()


    #m_game.plot_xbar()



    #m_game.plot_trajectory_2D()

    #m_game.save_trajectories()

    #m_game.plot_xbar()

    #m_game.plot_acl()

    #print(m_game.acl_x)

    return m_game.xbar, m_game.x

def FW_Recovery():

    T = 10
    d = 1

    alpha_t = Weights("linear", T = T)
   
    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    f_game = PowerFenchel(p = 2, q = 2)
    f_opt = PowerFunction(p = 2, q = 2)

    x_0 = np.array([5], dtype='float64')

    x_ts_FW = frankWolfe(f = f_opt, T = T, w_0 = x_0, xbounds = XBOUNDS)

    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t.weights, z0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)
    ftl = FTL(f = f_game, d = d, weights = alpha_t.weights, z0 = f_opt.grad(x_0), bounds = YBOUNDS)
  
    game_xbar, _ = run_helper(f_game = f_game, x_alg = bestresp, y_alg = ftl, T = T + 1, d = d, weights = alpha_t.weights, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = True)
           
    plt.figure()
    plt.title("Algorithm Recovery: Frank-Wolfe <--> X: BestResp+, Y: FTL")
    plt.plot(x_ts_FW[1:], color = 'blue', linewidth = 1.5, label = "Frank-Wolfe")
    plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()

def GDwAVG_Recovery():

    T = 100
    d = 1

    alpha_t = Weights("ones", T = T)
    eta_t = 0.5 * np.ones(T+1)
   
    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    f_game = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)

    x_0 = np.array([5], dtype='float64')

    
    x_ts_GDAvg = gradDescentAveraging(f = f_opt, T = T, w_0 = x_0, L = 1) 

    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t.weights, z0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)
    omd = OMD(f = f_game, d = d, weights = alpha_t.weights, z0 = x_0, y0 = x_0, eta_t = eta_t, bounds = XBOUNDS)
  
    game_xbar, _ = run_helper(f_game = f_game, x_alg = omd, y_alg = bestresp, T = T + 1, d = d, weights = alpha_t.weights, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = False)
           
    plt.figure()
    plt.title("Algorithm Recovery: Gradient Descent w/Averaging <--> X: OMD, Y: FTL+")
    plt.plot(x_ts_GDAvg, color = 'blue', linewidth = 1.5, label = "GDw/AVG")
    plt.plot(game_xbar[1:T], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()

def CGD_Recovery():

    T = 10
    d = 1

    R = 5
    G = 100

    alpha_t = Weights("ones", T = T)
    eta_t = (R / (G * np.sqrt(T))) * np.ones(T+1)
   
    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    f_game = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)

    x_0 = np.array([5], dtype='float64')

    x_ts_cumulativeGD = cumulativeGradientDescent(f = f_opt, T = T, w_0 = x_0, R = R, G = G)

    omd = OMD(f = f_game, d = d, weights = alpha_t.weights, z0 = x_0, y0 = x_0, eta_t = eta_t, bounds = XBOUNDS)
    ftl = FTL(f = f_game, d = d, weights = alpha_t.weights, z0 = f_opt.grad(x_0), bounds = YBOUNDS, prescient = True)

    game_xbar, _ = run_helper(f_game = f_game, x_alg = omd, y_alg = ftl, T = T + 1, d = d, weights = alpha_t.weights, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = False)
           
    print(x_ts_cumulativeGD)
    print(game_xbar)

        
    plt.figure()
    plt.title("Algorithm Recovery: Cumulative Gradient Descent <--> X: OMD, Y: FTL+")
    plt.plot(x_ts_cumulativeGD[1:], color = 'blue', linewidth = 1.5, label = "CGD")
    plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()

def SCEGwAVG_Recovery():

    T = 10
    d = 1

    L = 1

    alpha_t = Weights("ones", T = T)
    eta_t = 0.5 * np.ones(T+1)
   
    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    f_game = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)

    x_0 = np.array([5], dtype='float64')
    phi = L2Reg()

    x_ts_sgc_eg = singleGradientCallExtraGradientWithAveraging(f = f_opt, T = T, w_0 = x_0, phi = phi)

    optimistic_omd = OOMD(f = f_game, d = d, weights = alpha_t.weights, x0 = x_0, xminushalf = x_0, y0 = x_0, yminushalf = x_0, eta_t = eta_t, bounds = XBOUNDS, yfirst = False)
    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t.weights, z0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)

    game_xbar, _ = run_helper(f_game = f_game, x_alg = optimistic_omd, y_alg = bestresp, T = T + 1, d = d, weights = alpha_t.weights, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = False)
           
    plt.figure()
    plt.title("Algorithm Recovery: Single-Call Extra-Gradient with Averaging <--> X: Opt OMD, Y: BR+")
    plt.plot(x_ts_sgc_eg[1:], color = 'blue', linewidth = 1.5, label = "SCEGw/AVG")
    plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()

def Nesterov1Mem_Recovery():

    T = 100
    d = 1

    L = 1

    alpha_t = Weights("linear", T = T)
    alpha_t.print_weights()
    eta_t = 0.25 * np.ones(T+1)
   
    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    f_game = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)

    x_0 = np.array([5], dtype='float64')
    phi = L2Reg()

    x_ts_n_onemem, _ = nesterovOneMemory(f = f_opt, T = T, w_0 = x_0, phi = phi, L = 1)  

    omd = OMD(f = f_game, d = d, weights = alpha_t.weights, z0 = x_0, y0 = x_0, eta_t = eta_t, bounds = XBOUNDS, prescient = True)
    optimistic_ftl = OFTL(f = f_game, d = d, weights = alpha_t.weights, z0 = f_opt.grad(x_0), bounds = YBOUNDS)

    game_xbar, _ = run_helper(f_game = f_game, x_alg = omd, y_alg = optimistic_ftl, T = T + 1, d = d, weights = alpha_t.weights, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = True)
           
    plt.figure()
    plt.title("Algorithm Recovery: Nesterov's 1-Memory Method <--> X: OMD+, Y: Opt-FTL")
    plt.plot(x_ts_n_onemem[1:], color = 'blue', linewidth = 1.5, label = "Nesterov's 1Mem")
    plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()

def NesterovInfMem_Recovery():

    T = 100
    d = 1

    L = 1

    alpha_t = Weights("linear", T = T)
    alpha_t.print_weights()
    eta_t = 0.25 * np.ones(T+1)
    eta = 0.25
   
    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    f_game = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)

    x_0 = np.array([5], dtype='float64')
    phi = L2Reg()

    x_ts_n_infmem, _ = nesterovInfMemory(f = f_opt, T = T, w_0 = x_0, R = phi, L = 1) 

    ftrl = FTRL(f = f_game, d = d, weights = alpha_t.weights, z0 = np.array([5.0]), eta = eta, reg = phi, bounds = XBOUNDS, prescient = True)
    optimistic_ftl = OFTL(f = f_game, d = d, weights = alpha_t.weights, z0 = f_opt.grad(x_0), bounds = YBOUNDS)

    game_xbar, _ = run_helper(f_game = f_game, x_alg = ftrl, y_alg = optimistic_ftl, T = T + 1, d = d, weights = alpha_t.weights, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = True)
           
    plt.figure()
    plt.title("Algorithm Recovery: Nesterov's Inf-Memory Method <--> X: FTRL+, Y: Opt-FTL")
    plt.plot(x_ts_n_infmem[1:], color = 'blue', linewidth = 1.5, label = "Nesterov's 1Mem")
    plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()

def HeavyBall_Recovery():

    T = 10
    d = 1
    L = 1

    alpha_t = Weights("linear", T = T)
    alpha_t.print_weights()
    eta_t = 0.125 * np.ones(T+1)
   
    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    f_game = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)

    x_0 = np.array([5], dtype='float64')
    phi = L2Reg()

    x_ts_heavyball = heavyBall(f = f_opt, T = T, w_0 = x_0, L = 1) 

    omd = OMD(f = f_game, d = d, weights = alpha_t.weights, z0 = x_0, y0 = x_0, eta_t = eta_t, bounds = XBOUNDS, prescient = True)
    ftl = FTL(f = f_game, d = d, weights = alpha_t.weights, z0 = f_opt.grad(x_0), bounds = YBOUNDS, prescient = False)
  
    game_xbar, _ = run_helper(f_game = f_game, x_alg = omd, y_alg = ftl, T = T + 1, d = d, weights = alpha_t.weights, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = True)
           
    print(x_ts_heavyball)
    print(game_xbar)
        
    plt.figure()
    plt.title("Algorithm Recovery: Heavy Ball <--> X: OMD+, Y: FTL")
    plt.plot(x_ts_heavyball[2:], color = 'blue', linewidth = 1.5, label = "Heavy Ball")
    plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()

if __name__ == '__main__':

    #print("Salve Munde")
    
    # STATUS: OPERATIONAL
    # alpha_t = t
    FW_Recovery()

    # STATUS: OPERATIONAL
    # alpha_t = 1, eta_t = (1/2)*L
    GDwAVG_Recovery()

    # STATUS: CLOSE BUT NOT EXACT - NEED TO FIX PRESCIENT FLAG
    # alpha_t = 1, eta_t = R/Gsqrt(T)
    CGD_Recovery()

    # STATUS: OPERATIONAL
    # alpha_t = 1, eta_t = (1/2)*L
    SCEGwAVG_Recovery()

    # STATUS: OPERATIONAL
    # alpha_t = t, eta_t = (1/2)*L
    Nesterov1Mem_Recovery()

    # STATUS: OPERATIONAL
    # alpha_t = t, eta_t = (1/4)*L
    NesterovInfMem_Recovery()

    # STATUS: OPERATIONAL
    # alpha_t = t, eta_t = (1/8)*L
    HeavyBall_Recovery()

    T = 100

    #weight_schedules = ["ones", "linear", "sqrt", "log"]

    #for wt in weight_schedules:

    #    alpha_t = Weights(name = wt, T = T)
    #    alpha_t.plot_weights()


'''
    T = 100
    d = 1

    alpha_t = Weights("linear", T = T)
    alpha_t.print_weights()
    alpha_t.plot_weights()


    #alpha_t = np.linspace(1, T, T)
    #alpha_t = np.ones(T)
    eta_t = 0.5*np.ones(T)

    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    X_INIT = np.array([1.0])
    Y_INIT = np.array([1])

    phi = L2Reg()

    f_game = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)

    x_0 = np.array([5], dtype='float64')


    x_ts_FW = frankWolfe(f = f_opt, T = T, w_0 = x_0, xbounds = XBOUNDS)
    # alpha_t = t, eta_t = (1/2)*L
    #x_ts_n_onemem, v_ts = nesterovOneMemory(f = f_opt, T = T, w_0 = x_0, phi = phi, L = 1)  

    # alpha_t = t, eta_t = (1/4)*L
    #x_ts_n_infmem, v_ts = nesterovInfMemory(f = f_opt, T = T, w_0 = x_0, R = phi, L = 1)

    # alpha_t = 1, eta_t = (1/2)*L
    #x_ts_GDAvg = gradDescentAveraging(f = f_opt, T = T, w_0 = x_0, L = 1) 

    # alpha_t = 1, eta_t = (1/2)*L
    #x_ts_sgc_eg = singleGradientCallExtraGradientWithAveraging(f = f_opt, T = T, w_0 = x_0, phi = phi)


    #f_game = SqrtOneXSquaredFenchel()
    #f_opt = SqrtOneXSquared()

    # OL ALGORITHMS
    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t.weights, z0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)
    ftl = FTL(f = f_game, d = d, weights = alpha_t.weights, z0 = f_opt.grad(x_0), bounds = YBOUNDS)
    #optimistic_ftl = OFTL(f = f_game, d = d, weights = alpha_t, z0 = f_opt.grad(np.array([5.0])), bounds = YBOUNDS)

    #ftrl = FTRL(f = f_game, d = d, weights = alpha_t, z0 = np.array([5.0]), eta = 0.25, reg = phi, bounds = XBOUNDS)

    #optimistic_ftrl = OFTRL(f = f_game, d = d, weights = alpha_t, z0 = np.array([1.0]), bounds = XBOUNDS)
    #omd = OMD(f = f_game, d = d, weights = alpha_t, z0 = np.array([5.0]), y0 = np.array([5.0]), eta_t = eta_t, bounds = XBOUNDS)

    #optimistic_omd = OOMD(f = f_game, d = d, weights = alpha_t, x0 = np.array([5.0]), xminushalf = np.array([5.0]), y0 = np.array([5.0]), yminushalf = np.array([0]), eta_t = eta_t, bounds = XBOUNDS, yfirst = False)

    #X_LIST = [ftrl, optimistic_ftrl, omd, optimistic_omd]
    #Y_LIST = [ftl, optimistic_ftl]

    #run_helper(f_game = f_game, x_alg = optimistic_omd, y_alg = bestresp, T = T, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = False)
        
    #for x in X_LIST:
    #    for y in Y_LIST:
    #        print("RUNNING ALGORITHM PAIR (" + x.name + ", " + y.name + ")")
    #        game_xbar, xt = run_helper(f_game = f_game, x_alg = x, y_alg = y, T = T, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = True)
    #        print()


    game_xbar, xt = run_helper(f_game = f_game, x_alg = bestresp, y_alg = ftl, T = T + 1, d = d, weights = alpha_t.weights, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = True)
           


    #game_xbar, xt = run_helper(f_game = f_game, f_opt = f_opt, x_alg = optimistic_omd, y_alg = bestresp, T = T, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = False)




    #print(x_ts_sgc_eg)
    #plt.figure()

    #t_plot = np.linspace(0, T, T+1)
    plt.plot(x_ts_FW[1:], color = 'blue', label = "Frank-Wolfe")
    #plt.plot(x_ts_n_onemem[1:], color = 'blue', label = "NesterovsOne-Memory")
    #plt.plot(x_ts_n_infmem[1:], color = 'blue', label = "NesterovsInf-Memory")
    #plt.plot(t_plot[0:len(x_ts_GDAvg)], x_ts_GDAvg[0:], color = 'blue', label = "Gradient Descent w/ Averaging")
    #plt.plot(x_ts_sgc_eg[1:], color = 'blue', label = "Single-Call Extra Gradient with Averaging")
    plt.plot(game_xbar[1:], color = 'red', linestyle = '--', label = 'FGNRD Recovery')
    #plt.plot(v_ts[1:], color = 'green', linestyle = '--', label = 'vts')
    #plt.plot(xt, color = 'purple', linestyle = '--', label = 'xt')
    #plt.plot(x_ts_n_infmem, color='gray', label="NesterovsInf-Memory")

    plt.title("Algorithm Recovery")
    plt.legend()
    plt.show()

    #print(x_ts_n_onemem)
    #print(game_xbar)
'''

    
