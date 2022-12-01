
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

    return m_game.xbar, m_game.x

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

    T = 10
    d = 1

    alpha_t = np.linspace(1, T, T)
    eta_t = 0.25*np.ones(T)

    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    X_INIT = np.array([1.0])
    Y_INIT = np.array([1])

    phi = L2Reg()

    f_game = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)

    x_0 = np.array([5], dtype='float64')

    x_ts_n_onemem, v_ts = nesterovOneMemory(f = f_opt, T = T, w_0 = x_0, phi = phi, L = 1)
    x_ts_n_infmem = nesterovInfMemory(f = f_opt, T = T, w_0 = x_0, L = 1)

    #f_game = SqrtOneXSquaredFenchel()
    #f_opt = SqrtOneXSquared()

    # OL ALGORITHMS
    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS)
    ftl = FTL(f = f_game, d = d, weights = alpha_t, z0 = np.array([5.0]), bounds = YBOUNDS)
    optimistic_ftl = OFTL(f = f_game, d = d, weights = alpha_t, z0 = f_opt.grad(np.array([5.0])), bounds = YBOUNDS)
    ftrl = FTRL(f = f_game, d = d, weights = alpha_t, z0 = np.array([5.0]), bounds = XBOUNDS)
    optimistic_ftrl = OFTRL(f = f_game, d = d, weights = alpha_t, z0 = np.array([1.0]), bounds = XBOUNDS)
    omd = OMD(f = f_game, d = 1, weights = alpha_t, z0 = np.array([5.0]), eta_t = eta_t, bounds = XBOUNDS)


    game_xbar, xt = run_helper(f_game = f_game, f_opt = f_opt, x_alg = omd, y_alg = optimistic_ftl, T = T, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS)

    plt.figure()
    plt.plot(x_ts_n_onemem[1:], color = 'blue', label = "NesterovsOne-Memory")
    #plt.plot(x_ts_n_infmem[1:], color = 'blue', label = "NesterovsInf-Memory")
    plt.plot(game_xbar, color = 'red', linestyle = '--', label = 'FGNRD Recovery')
    #plt.plot(v_ts[1:], color = 'green', linestyle = '--', label = 'vts')
    #plt.plot(xt, color = 'purple', linestyle = '--', label = 'xt')
    #plt.plot(x_ts_n_infmem, color='gray', label="NesterovsInf-Memory")

    plt.title("Algorithm Recovery")
    plt.legend()
    plt.show()


    
