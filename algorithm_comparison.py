
import matplotlib.pyplot as plt
import numpy as np
from fenchel_loop import *
from convex_optimization_algorithms import *
from ol_algorithms import *

def run_helper(f_game, f_opt, x_alg, y_alg, T, d, weights, xbounds, ybounds):

    print("Running algorithms X = " + x_alg.name + ", Y = " + y_alg.name)

    m_game = Fenchel_Game(f = f_game, xbounds = xbounds, ybounds = ybounds, iterations = T, weights = weights, d = d)
    m_game.set_players(x_alg = x_alg, y_alg = y_alg)
    
    m_game.run()
    
    print("Saddle Point (x*, y*) = (%0.3f, %0.3f)" % (m_game.x_star, m_game.y_star))
    print("Final iterate:", m_game.x[-1], m_game.y[-1])

    #m_game.plot_trajectory_2D()

    m_game.save_trajectories()

    m_game.plot_xbar()

    #m_game.plot_acl()

    #print(m_game.acl_x)


def FW_Recovery():

    # ---------------------------- RUN FENCHEL GAME ----------------------------

    ftrl = FTRL(f = f_game, d=d, weights=alpha_t, eta=0, z0=0)
    
    # --------------------------------------------------------------------------

    # ---------------------- RUN FRANK-WOLFE OPTIMIZATION ----------------------

    xbounds = [-1,1]
    
    #x_t = np.array([10,-10])
    #print(x_t)

    #x_ts = frankWolfe(f_opt, T, X_INIT, L=1) #(f_opt, T, X_INIT, xbounds)
    x_ts = frankWolfe(f_opt, T, X_INIT, xbounds)

    plt.figure()
    plt.title("Frank-Wolfe <--> X: BestResp+, Y: FTL")
    plt.plot(x_ts[1:], '-r', label = "FW")
    plt.plot(m_game.xbar, '--b', label = "BR + FTL")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    #print("Salve Munde")
    #FW_Recovery()

    T = 100
    d = 1

    alpha_t = np.linspace(1, T, T)
    eta_t = np.ones(T)

    XBOUNDS = [[-1, 1]]
    YBOUNDS = [[-1, 1]]

    X_INIT = np.array([1.0])
    Y_INIT = np.array([1])

    f_game = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)
    #f_game = SqrtOneXSquaredFenchel()
    #f_opt = SqrtOneXSquared()

    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS)
    
    ftl = FTL(f = f_game, d = d, weights = alpha_t, z0 = np.array([1.0]), bounds = YBOUNDS)
    optimistic_ftl = OFTL(f = f_game, d = d, weights = alpha_t, z0 = np.array([1.0]), bounds = YBOUNDS)

    ftrl = FTRL(f = f_game, d = d, weights = alpha_t, z0 = np.array([1.0]), bounds = XBOUNDS)

    optimistic_ftrl = OFTRL(f = f_game, d = d, weights = alpha_t, z0 = np.array([1.0]), bounds = XBOUNDS)

    omd = OMD(f = f_game, d = 1, weights = alpha_t, eta_t = eta_t, bounds = XBOUNDS)

    run_helper(f_game = f_game, f_opt = f_opt, x_alg = omd, y_alg = ftl, T = T, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS)

    
