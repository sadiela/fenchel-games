
import matplotlib.pyplot as plt
import numpy as np
from fenchel_loop import *
from convex_optimization_algorithms import *
from ol_algorithms import *

def run_helper(f_game, x_alg, y_alg, T, d, weights, xbounds, ybounds, yfirst = True):

    print("Running algorithms X = " + x_alg.name + ", Y = " + y_alg.name)

    m_game = Fenchel_Game(f = f_game, xbounds = xbounds, ybounds = ybounds, iterations = T, weights = weights, d = d)
    m_game.set_players(x_alg = x_alg, y_alg = y_alg)
    
    m_game.run(yfirst = yfirst)
    
    print("Saddle Point (x*, y*) = (%0.3f, %0.3f)" % (m_game.x_star, m_game.y_star))
    print("Final iterate:", m_game.x[-1], m_game.y[-1])

    game_xbar = m_game.xbar

    print(game_xbar[-1]/np.linalg.norm(game_xbar[-1]))

    plt.figure()
    #plt.plot(x_ts_n_onemem[1:], color = 'blue', label = "NesterovsOne-Memory")
    #plt.plot(x_ts_n_infmem[1:], color = 'blue', label = "NesterovsInf-Memory")
    plt.plot([x[0]/np.linalg.norm(x) for x in game_xbar], [x[1]/np.linalg.norm(x) for x in game_xbar], color = 'red', linestyle = '--', label = 'FGNRD Recovery')
    plt.show()


    #m_game.plot_xbar()



    #m_game.plot_trajectory_2D()

    #m_game.save_trajectories()

    #m_game.plot_xbar()

    #m_game.plot_acl()

    #print(m_game.acl_x)

    #return m_game.xbar, m_game.x

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

    #alpha_t = np.linspace(1, T, T)
    alpha_t = np.ones(T)
    eta_t = 0.5*np.ones(T)

    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    X_INIT = np.array([1.0])
    Y_INIT = np.array([1])

    phi = L2Reg()

    f_game = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)

    x_0 = np.array([5], dtype='float64')

    # alpha_t = t, eta_t = (1/4)*L
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
    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS)
    ftl = FTL(f = f_game, d = d, weights = alpha_t, z0 = np.array([5.0]), bounds = YBOUNDS)
    optimistic_ftl = OFTL(f = f_game, d = d, weights = alpha_t, z0 = f_opt.grad(np.array([5.0])), bounds = YBOUNDS)

    ftrl = FTRL(f = f_game, d = d, weights = alpha_t, z0 = np.array([5.0]), eta = 0.25, reg = phi, bounds = XBOUNDS)

    optimistic_ftrl = OFTRL(f = f_game, d = d, weights = alpha_t, z0 = np.array([1.0]), bounds = XBOUNDS)
    omd = OMD(f = f_game, d = d, weights = alpha_t, z0 = np.array([5.0]), y0 = np.array([5.0]), eta_t = eta_t, bounds = XBOUNDS)

    optimistic_omd = OOMD(f = f_game, d = d, weights = alpha_t, x0 = np.array([5.0]), xminushalf = np.array([5.0]), y0 = np.array([5.0]), yminushalf = np.array([0]), eta_t = eta_t, bounds = XBOUNDS, yfirst = False)

    X_LIST = [ftrl, optimistic_ftrl, omd, optimistic_omd]
    Y_LIST = [ftl, optimistic_ftl]

    #run_helper(f_game = f_game, x_alg = optimistic_omd, y_alg = bestresp, T = T, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = False)
        
    for x in X_LIST:
        for y in Y_LIST:
            print("RUNNING ALGORITHM PAIR (" + x.name + ", " + y.name + ")")
            run_helper(f_game = f_game, x_alg = x, y_alg = y, T = T, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = True)
            print()

    #game_xbar, xt = run_helper(f_game = f_game, f_opt = f_opt, x_alg = optimistic_omd, y_alg = bestresp, T = T, d = d, weights = alpha_t, xbounds = XBOUNDS, ybounds = YBOUNDS, yfirst = False)




    #print(x_ts_sgc_eg)
    #plt.figure()

    #t_plot = np.linspace(0, T, T+1)
    #plt.plot(x_ts_n_onemem[1:], color = 'blue', label = "NesterovsOne-Memory")
    #plt.plot(x_ts_n_infmem[1:], color = 'blue', label = "NesterovsInf-Memory")
    #plt.plot(t_plot[0:len(x_ts_GDAvg)], x_ts_GDAvg[0:], color = 'blue', label = "Gradient Descent w/ Averaging")
    #plt.plot(x_ts_sgc_eg[1:], color = 'blue', label = "Single-Call Extra Gradient with Averaging")
    #plt.plot(game_xbar, color = 'red', linestyle = '--', label = 'FGNRD Recovery')
    #plt.plot(v_ts[1:], color = 'green', linestyle = '--', label = 'vts')
    #plt.plot(xt, color = 'purple', linestyle = '--', label = 'xt')
    #plt.plot(x_ts_n_infmem, color='gray', label="NesterovsInf-Memory")

    #plt.title("Algorithm Recovery")
    #plt.legend()
    #plt.show()

    #print(x_ts_n_onemem)
    #print(game_xbar)


    
