
import matplotlib.pyplot as plt
import numpy as np
from fenchel_loop import *
from convex_optimization_algorithms import *
from ol_algorithms import *

def FW_Recovery():

    f_game = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)
    
    T = 10
    d = 1

    alpha_t = np.linspace(1, T, T)

    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    X_INIT = np.array([0.4])
    Y_INIT = np.array([0.4])

    # ---------------------------- RUN FENCHEL GAME ----------------------------

    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS)
    ftl = FTL(f = f_game, d = d, weights = alpha_t, z0 = np.array([0.4]))

    m_game = Fenchel_Game(f = f_game, xbounds = XBOUNDS, ybounds = YBOUNDS, iterations = T, weights = alpha_t, d = d)
    m_game.set_players(x_alg = bestresp, y_alg = ftl)
    
    m_game.run()
    
    print("Saddle Point (x*, y*) = (%0.3f, %0.3f)" % (m_game.x_star, m_game.y_star))
    print("Final iterate:", m_game.x[-1], m_game.y[-1])

    #m_game.plot_trajectory_2D()

    m_game.save_trajectories()

    m_game.plot_xbar()

    m_game.plot_acl()

    #print(m_game.acl_x)
    
    # --------------------------------------------------------------------------

    # ---------------------- RUN FRANK-WOLFE OPTIMIZATION ----------------------

    xbounds = [-10,10]
    
    #x_t = np.array([10,-10])
    x_t = np.array([0.4])
    #print(x_t)

    x_ts = FrankeWolfeLoop(T, xbounds, f_opt, X_INIT)

    plt.figure()
    plt.title("Frank-Wolfe <--> X: BestResp+, Y: FTL")
    plt.plot(x_ts, '-r', label = "FW")
    plt.plot(m_game.xbar, '--b', label = "BR + FTL")
    plt.legend()
    plt.show()





if __name__ == '__main__':

    #print("Salve Munde")

    FW_Recovery()

    