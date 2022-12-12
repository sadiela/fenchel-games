
import matplotlib.pyplot as plt
import numpy as np
from fenchel_loop import *
from convex_optimization_algorithms import *
from ol_algorithms import *
from weights import *

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

    m_game.plot_acl()

    #print(m_game.acl_x)

    return m_game.xbar, m_game.x

def FW_Recovery(f_game, f_opt, T, d, x_0, xbounds, ybounds):

    alpha_t = Weights("linear", T = T)

    x_0 = np.array([5], dtype='float64')
   
    x_ts_FW = frankWolfe(f = f_opt, T = T, w_0 = x_0, xbounds = xbounds)

    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t, z0 = x_0, xbounds = xbounds, ybounds = ybounds)
    ftl = FTL(f = f_game, d = d, weights = alpha_t, z0 = f_opt.grad(x_0), bounds = ybounds)
  
    game_xbar, _ = run_helper(f_game = f_game, x_alg = bestresp, y_alg = ftl, T = T + 1, d = d, weights = alpha_t, xbounds = xbounds, ybounds = ybounds, yfirst = True)

    return x_ts_FW[1:], game_xbar[1:]  
    '''
    plt.figure()
    plt.suptitle("Algorithm Recovery: Frank-Wolfe <--> X: BestResp+, Y: FTL")
    plt.title("T = " + str(T))
    plt.plot(x_ts_FW[1:], color = 'blue', linewidth = 1.5, label = "Frank-Wolfe")
    plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()
    plt.clf()
    #plt.savefig("FW_power_T_" + str(T) + ".png")'''

def GDwAVG_Recovery(f_game, f_opt, T, d, x_0, xbounds, ybounds):

    alpha_t = Weights("ones", T = T)
    eta_t = 0.5 * np.ones(T+1)

    x_0 = np.array([5], dtype='float64')
   
    x_ts_GDAvg = gradDescentAveraging(f = f_opt, T = T, w_0 = x_0, L = 1, xbounds = xbounds) 

    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t, z0 = x_0, xbounds = xbounds, ybounds = ybounds)
    omd = OMD(f = f_game, d = d, weights = alpha_t, z0 = x_0, y0 = f_opt.grad(x_0), eta_t = eta_t, bounds = xbounds, prescient = False)
  
    game_xbar, _ = run_helper(f_game = f_game, x_alg = omd, y_alg = bestresp, T = T + 1, d = d, weights = alpha_t, xbounds = xbounds, ybounds = ybounds, yfirst = False)

    return x_ts_GDAvg, game_xbar[1:T]      
    #print(game_xbar)
    #print(x_ts_GDAvg)
    '''
    plt.figure()
    plt.suptitle("Algorithm Recovery: Gradient Descent w/Averaging <--> X: OMD, Y: FTL+")
    plt.title("T = " + str(T))
    plt.plot(x_ts_GDAvg, color = 'blue', linewidth = 1.5, label = "GDw/AVG")
    plt.plot(game_xbar[1:T], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()
    plt.savefig("GDAVG_power_T_" + str(T) + ".png")'''

def CGD_Recovery(f_game, f_opt, T, d, x_0, xbounds, ybounds):

    R = 10
    G = 1

    alpha_t = Weights("ones", T = T)
    eta_t = (R / (G * np.sqrt(T))) * np.ones(T+1)
    print(eta_t)

    x_0 = np.array([5], dtype='float64')
    #x_0 = np.array([1], dtype='float64')
    #z0ftl = np.array([1], dtype='float64') # f_opt.grad(x_0)

    x_ts_cumulativeGD = cumulativeGradientDescent(f = f_opt, T = T, w_0 = x_0, R = R, G = G, xbounds = xbounds) 

    omd = OMD(f = f_game, d = d, weights = alpha_t, z0 = np.array([5], dtype='float64'), y0 = np.array([5], dtype='float64'), eta_t = eta_t, bounds = xbounds, prescient = False)
    
    x_0 = np.array([5], dtype='float64')
    ftl = FTL(f = f_game, d = d, weights = alpha_t, z0 = np.array([5], dtype='float64'), bounds = ybounds, prescient = True)

    game_xbar, _ = run_helper(f_game = f_game, x_alg = omd, y_alg = ftl, T = T + 1, d = d, weights = alpha_t, xbounds = xbounds, ybounds = ybounds, yfirst = False)
           
    print(x_ts_cumulativeGD)
    print(game_xbar)

    return x_ts_cumulativeGD[1:], game_xbar[1:]

    '''plt.figure()
    plt.suptitle("Algorithm Recovery: Cumulative Gradient Descent <--> X: OMD, Y: FTL+")
    plt.title("T = " + str(T))
    plt.plot(x_ts_cumulativeGD[1:], color = 'blue', linewidth = 1.5, label = "CGD")
    plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()
    plt.savefig("CGD_power_T_" + str(T) + ".png")'''

def SCEGwAVG_Recovery(f_game, f_opt, T, d, x_0, xbounds, ybounds):

    L = 2

    alpha_t = Weights("ones", T = T)
    alpha_t.print_weights()
    eta_t = 0.5 * np.ones(T+1)

    phi = L2Reg()

    x_ts_sgc_eg = singleGradientCallExtraGradientWithAveraging(f = f_opt, T = T, w_0 = x_0, phi = phi, L = L, xbounds = xbounds)

    #optimistic_omd = OOMD(f = f_game, d = d, weights = alpha_t, x0 = x_0, xminushalf = x_0, y0 = f_opt.grad(x_0), yminushalf = f_opt.grad(x_0), eta_t = eta_t, bounds = xbounds, yfirst = False)
    #bestresp = BestResponse(f = f_game, d = d, weights = alpha_t, z0 = x_0, xbounds = xbounds, ybounds = ybounds)

    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    optimistic_omd = OOMD(f = f_game, d = d, weights = alpha_t, x0 = x_0, xminushalf = x_0, y0 = x_0, yminushalf = x_0, eta_t = eta_t, bounds = XBOUNDS, yfirst = False)
    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t, z0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)

    game_xbar, _ = run_helper(f_game = f_game, x_alg = optimistic_omd, y_alg = bestresp, T = T + 1, d = d, weights = alpha_t, xbounds = xbounds, ybounds = ybounds, yfirst = False)
           
    return x_ts_sgc_eg[1:], game_xbar[1:]
    '''plt.figure()
    plt.suptitle("Algorithm Recovery: Single-Call Extra-Gradient with Averaging <--> X: Opt OMD, Y: BR+")
    plt.title("T = " + str(T))
    plt.plot(x_ts_sgc_eg[1:], color = 'blue', linewidth = 1.5, label = "SCEGw/AVG")
    plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()
    plt.savefig("SCEGwAVG_power_T_" + str(T) + ".png")'''

def Nesterov1Mem_Recovery(f_game, f_opt, T, d, x_0, xbounds, ybounds):

    L = 1

    alpha_t = Weights("linear", T = T)
    eta_t = 0.25 * np.ones(T+1)
   
    x_0 = np.array([5], dtype='float64')
    phi = L2Reg()

    x_ts_n_onemem, _ = nesterovOneMemory(f = f_opt, T = T, w_0 = x_0, phi = phi, L = 1, xbounds = xbounds)  

    omd = OMD(f = f_game, d = d, weights = alpha_t, z0 = x_0, y0 = x_0, eta_t = eta_t, bounds = xbounds, prescient = True)
    optimistic_ftl = OFTL(f = f_game, d = d, weights = alpha_t, z0 = f_opt.grad(x_0), bounds = ybounds)

    game_xbar, _ = run_helper(f_game = f_game, x_alg = omd, y_alg = optimistic_ftl, T = T + 1, d = d, weights = alpha_t, xbounds = xbounds, ybounds = ybounds, yfirst = True)
           
    return x_ts_n_onemem[1:], game_xbar[1:]
    '''plt.figure()
    plt.suptitle("Algorithm Recovery: Nesterov's 1-Memory Method <--> X: OMD+, Y: Opt-FTL")
    plt.title("T = " + str(T))
    plt.plot(x_ts_n_onemem[1:], color = 'blue', linewidth = 1.5, label = "Nesterov's 1Mem")
    plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()
    plt.savefig("Nesterov1_power_T_" + str(T) + ".png")'''

def NesterovInfMem_Recovery(f_game, f_opt, T, d, x_0, xbounds, ybounds):

    L = 1

    alpha_t = Weights("linear", T = T)
    alpha_t.print_weights()
    eta_t = 0.25 * np.ones(T+1)
    eta = 0.25
   
    x_0 = np.array([5], dtype='float64')
    phi = L2Reg()

    x_ts_n_infmem, _ = nesterovInfMemory(f = f_opt, T = T, w_0 = x_0, R = phi, L = 1, xbounds = xbounds) 

    ftrl = FTRL(f = f_game, d = d, weights = alpha_t, z0 = np.array([5.0]), eta = eta, reg = phi, bounds = xbounds, prescient = True)
    optimistic_ftl = OFTL(f = f_game, d = d, weights = alpha_t, z0 = f_opt.grad(x_0), bounds = ybounds)

    game_xbar, _ = run_helper(f_game = f_game, x_alg = ftrl, y_alg = optimistic_ftl, T = T + 1, d = d, weights = alpha_t, xbounds = xbounds, ybounds = ybounds, yfirst = True)
           
    return x_ts_n_infmem[1:], game_xbar[1:]
    ''' 
    plt.figure()
    plt.suptitle("Algorithm Recovery: Nesterov's Inf-Memory Method <--> X: FTRL+, Y: Opt-FTL")
    plt.title("T = " + str(T))
    plt.plot(x_ts_n_infmem[1:], color = 'blue', linewidth = 1.5, label = "Nesterov's 1Mem")
    plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()
    plt.savefig("NesterovInf_power_T_" + str(T) + ".png")'''

def HeavyBall_Recovery(f_game, f_opt, T, d, x_0, xbounds, ybounds):

    L = 1

    alpha_t = Weights("linear", T = T)
    alpha_t.print_weights()
    eta_t = 0.125 * np.ones(T+1)

    phi = L2Reg()

    x_ts_heavyball = heavyBall(f = f_opt, T = T, w_0 = x_0, L = 1, xbounds = xbounds) 

    omd = OMD(f = f_game, d = d, weights = alpha_t, z0 = x_0, y0 = x_0, eta_t = eta_t, bounds = xbounds, prescient = True)
    ftl = FTL(f = f_game, d = d, weights = alpha_t, z0 = f_opt.grad(x_0), bounds = ybounds, prescient = False)
  
    game_xbar, _ = run_helper(f_game = f_game, x_alg = omd, y_alg = ftl, T = T + 1, d = d, weights = alpha_t, xbounds = xbounds, ybounds = ybounds, yfirst = True)
           
    #print(x_ts_heavyball)
    #print(game_xbar)
    return x_ts_heavyball[2:], game_xbar[1:]
        
    '''plt.figure()
    plt.suptitle("Algorithm Recovery: Heavy Ball <--> X: OMD+, Y: FTL")
    plt.title("T = " + str(T))
    plt.plot(x_ts_heavyball[2:], color = 'blue', linewidth = 1.5, label = "Heavy Ball")
    plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()
    plt.savefig("Heavyball_power_T_" + str(T) + ".png")'''

def run_teams():

    T = 100
    d = 1

    alpha_t = Weights("ones", T = T)
    eta_t = 0.5 * np.ones(T+1)
    eta = 0.25
    phi = L2Reg()
   
    XBOUNDS = [[-1, 1]]
    YBOUNDS = [[-1, 1]]

    f_game = PowerFenchel(p = 2, q = 2) #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)

    x_0 = np.array([5], dtype='float64')
    z0ftl = np.array([-5], dtype='float64') # f_opt.grad(x_0)

    bestresp = BestResponse(f = f_game, d = d, weights = alpha_t, z0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)
    ftl = FTL(f = f_game, d = d, weights = alpha_t, z0 = z0ftl, bounds = YBOUNDS, prescient = True)

    omd = OMD(f = f_game, d = d, weights = alpha_t, z0 = x_0, y0 = x_0, eta_t = eta_t, bounds = XBOUNDS)
    ftrl = FTRL(f = f_game, d = d, weights = alpha_t, z0 = np.array([5.0]), eta = eta, reg = phi, bounds = XBOUNDS, prescient = False)
  
    m_game = Fenchel_Game(f = f_game, xbounds = XBOUNDS, ybounds = YBOUNDS, iterations = T, weights = alpha_t, d = d)
    m_game.set_teams(x_team = (omd, ftrl), y_team = (ftl, bestresp), lr = 0.5, w1xB = 0.75, w1yB = 0.75)
    
    m_game.run_teams(yfirst = False)

    game_xbar = m_game.xbar
    
    #print("Saddle Point (x*, y*) = (%0.3f, %0.3f)" % (m_game.x_star, m_game.y_star))
    print("Final iterate:", m_game.x[-1], m_game.y[-1])
   
    plt.figure()
    plt.title("Playing with teams: X Team: (OMD, FTRL), Y Team: (FTL+, BestResp+)")
    plt.plot(game_xbar[1:T], color = 'blue', linewidth = 1.5, linestyle = '-', label = "FGNRD Recovery")
    plt.legend()
    plt.show()

    stx = [0]
    sty = [0]
    for t in range(1, T):
        stx.append(m_game.x_dist[t] / (m_game.x_dist[t] + m_game.w1xB))
        sty.append(m_game.y_dist[t] / (m_game.y_dist[t] + m_game.w1yB))

    plt.figure()
    plt.title("Weights vs. Iteration")
    plt.plot(stx, color = 'blue', linewidth = 1.5, linestyle = '--', label = "Prob. of Selecting X_A (OMD)")
    plt.plot(sty, color = 'red', linewidth = 1.5, linestyle = '--', label = "Prob. of Selecting Y_A (FTL)")
    plt.xlabel("Iteration t")
    plt.ylabel("Probability")
    #plt.plot(m_game.x_dist[1:T], color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    plt.legend()
    plt.show()

def test_suite_X_prescient(f_game, f_opt, T, d, xbounds, ybounds, weight_schedule):

    print("Running test suite for prescient X!")

    L = 1
    eta = 0.25
    eta_t = 0.25 * np.ones(T+1)
    x_0 = np.array([5], dtype = 'float64')
    phi = L2Reg()

    for wt in weight_schedule:
        alpha_t = Weights(name = wt, T = T)
        print("Running test suite for weight schedule:" + alpha_t.name)

        bestresp = BestResponse(f = f_game, d = d, weights = alpha_t, z0 = x_0, xbounds = xbounds, ybounds = ybounds)
        ftrl = FTRL(f = f_game, d = d, weights = alpha_t, z0 = x_0, eta = eta, reg = phi, bounds = xbounds, prescient = True)
        omd = OMD(f = f_game, d = d, weights = alpha_t, z0 = x_0, y0 = x_0, eta_t = eta_t, bounds = xbounds, prescient = True)

        ftl = FTL(f = f_game, d = d, weights = alpha_t, z0 = x_0, bounds = ybounds, prescient = False)
        optimistic_ftl = OFTL(f = f_game, d = d, weights = alpha_t, z0 = f_opt.grad(x_0), bounds = ybounds)

        # Prescient X Algorithms: FTRL+, OMD+, BR+
        X_ALG = [bestresp, ftrl, omd]

        # Non-Prescient Y Algorithms: FTL, Opt-FTL
        Y_ALG = [ftl, optimistic_ftl]

        for x in X_ALG:
            for y in Y_ALG:
                game_xbar, _ = run_helper(f_game = f_game, x_alg = x, y_alg = y, T = T + 1, d = d, weights = alpha_t, xbounds = xbounds, ybounds = ybounds, yfirst = True)
                #print(game_xbar[1:T])

                plt.figure()
                plt.suptitle("Algorithm Testing: X = " + x.name + ", Y = " + y.name)
                plt.title("f(x) = " + f_game.latex_print + ", " + alpha_t.latex_print + ", T = %d" % T)
                plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '-', label = "FGNRD Recovery")
                plt.legend()
                save_str = x.name + "_" + y.name + "_" + alpha_t.name + "_" + str(T)
                plt.savefig("./test_results/" + save_str)
                plt.show()

                x.reset()
                y.reset()

def test_suite_Y_prescient(f_game, f_opt, T, d, xbounds, ybounds, weight_schedule):

    print("Running test suite for prescient Y!")

    L = 1
    eta = 0.25
    eta_t = 0.25 * np.ones(T+1)
    x_0 = np.array([5], dtype = 'float64')
    phi = L2Reg()

    for wt in weight_schedule:
        alpha_t = Weights(name = wt, T = T)
        print("Running test suite for weight schedule:" + alpha_t.name)

        ftrl = FTRL(f = f_game, d = d, weights = alpha_t, z0 = x_0, eta = eta, reg = phi, bounds = xbounds, prescient = False)
        omd = OMD(f = f_game, d = d, weights = alpha_t, z0 = x_0, y0 = x_0, eta_t = eta_t, bounds = xbounds, prescient = False)
        optimistic_omd = OOMD(f = f_game, d = d, weights = alpha_t, x0 = x_0, xminushalf = x_0, y0 = x_0, yminushalf = x_0, eta_t = eta_t, bounds = XBOUNDS, yfirst = False)
        optimistic_ftrl = OFTRL(f = f_game, d = d, weights = alpha_t, z0 = x_0, reg = phi, bounds = xbounds)

        bestresp = BestResponse(f = f_game, d = d, weights = alpha_t, z0 = x_0, xbounds = xbounds, ybounds = ybounds)
        ftl = FTL(f = f_game, d = d, weights = alpha_t, z0 = x_0, bounds = ybounds, prescient = True)
        
        # Non-Prescient X Algorithms: FTRL, OMD, Opt-FTRL, Opt-OMD
        X_ALG = [ftrl, omd, optimistic_omd, optimistic_ftrl]

        # Prescient Y Algorithms: FTL, Opt-FTL
        Y_ALG = [bestresp, ftl]

        for x in X_ALG:
            for y in Y_ALG:
                game_xbar, _ = run_helper(f_game = f_game, x_alg = x, y_alg = y, T = T + 1, d = d, weights = alpha_t, xbounds = xbounds, ybounds = ybounds, yfirst = False)
                #print(game_xbar[1:T])

                plt.figure()
                plt.suptitle("Algorithm Testing: X = " + x.name + ", Y = " + y.name)
                plt.title("f(x) = " + f_game.latex_print + ", " + alpha_t.latex_print + ", T = %d" % T)
                plt.plot(game_xbar[1:], color = 'red', linewidth = 1.5, linestyle = '-', label = "FGNRD Recovery")
                plt.legend()
                save_str = x.name + "_" + y.name + "_" + alpha_t.name + "_" + str(T)
                #plt.savefig("./test_results/" + save_str)
                plt.show()

                x.reset()
                y.reset()


if __name__ == '__main__':

    #print("Salve Munde")
    #run_teams()

    f_game = PowerFenchel(p = 2, q = 2)#SqrtOneXSquaredFenchel()  #ExpFenchel()
    f_opt = PowerFunction(p = 2, q = 2)#SqrtOneXSquared() 
    
    T = 90
    d = 1
    x_0 = np.array([5.0], dtype = 'float64')

    XBOUNDS = [[-10, 10]]
    YBOUNDS = [[-10, 10]]

    # STATUS: OPERATIONAL
    # alpha_t = t
    FWco, FWsaddle = FW_Recovery(f_game = f_game, f_opt = f_opt, T = T, d = d, x_0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)

    # STATUS: OPERATIONAL
    # alpha_t = 1, eta_t = (1/2)*L
    GDavgco, GDavgsaddle = GDwAVG_Recovery(f_game, f_opt, T = T, d = d, x_0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)

    # STATUS: OPERATIONAL -YOU CANNOT PICK T = 100
    # alpha_t = 1, eta_t = R/Gsqrt(T)
    CGDco, CGDsaddle = CGD_Recovery(f_game = f_game, f_opt = f_opt, T = T, d = d, x_0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)

    # STATUS: OPERATIONAL
    # alpha_t = 1, eta_t = (1/2)*L
    SCEGwAVGco, SCEGwAVGsaddle = SCEGwAVG_Recovery(f_game, f_opt, T = T, d = d, x_0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)

    # STATUS: OPERATIONAL
    # alpha_t = t, eta_t = (1/2)*L
    N1co, N1saddle = Nesterov1Mem_Recovery(f_game, f_opt, T = T, d = d, x_0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)

    # STATUS: OPERATIONAL
    # alpha_t = t, eta_t = (1/4)*L
    Ninfco, Ninfsaddle = NesterovInfMem_Recovery(f_game, f_opt, T = T, d = d, x_0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)

    # STATUS: OPERATIONAL
    # alpha_t = t, eta_t = (1/8)*L
    hbco, hbsaddle = HeavyBall_Recovery(f_game, f_opt, T = T, d = d, x_0 = x_0, xbounds = XBOUNDS, ybounds = YBOUNDS)

    fig, axs = plt.subplots(2,4, sharex=True)
    fontsize=9
    axs[0,0].set_title("FrankWolfe <-> X:BR+,Y:FTL", fontsize=fontsize)
    #plt.title("T = " + str(T))
    axs[0,0].plot(FWco, color = 'blue', linewidth = 1.5, label = "Frank-Wolfe")
    axs[0,0].plot(FWsaddle, color = 'red', linewidth = 1.5, linestyle = '--', label = "FGNRD Recovery")
    axs[0,0].legend()

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


    #T = 100
    #d = 1
    

    #XBOUNDS = [[-10, 10]]
    #YBOUNDS = [[-10, 10]]

    #wt_sch = ["ones", "linear", "sqrt", "log"]

    

    #test_suite_X_prescient(f_game = f_game, f_opt = f_opt, T = T, d = 1, xbounds = XBOUNDS, ybounds = YBOUNDS, weight_schedule = wt_sch)

    #test_suite_Y_prescient(f_game = f_game, f_opt = f_opt, T = T, d = 1, xbounds = XBOUNDS, ybounds = YBOUNDS, weight_schedule = wt_sch)

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

    