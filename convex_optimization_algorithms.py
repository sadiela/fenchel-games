import numpy as np 
import matplotlib.pyplot as plt
import math
from convex_functions import *

def FrankeWolfeLoop(T, xbounds, f, w_t):
    w_ts = []

    for t in range(0,T): 
        s_t = f.find_s(w_t, xbounds)
        print("st", s_t)
        w_t = w_t + (2/(t+2))*(s_t-w_t)
        #x_ts.append(np.linalg.norm(x_t))
        w_ts.append(w_t)
        print(w_t)

    print(w_t)
    return w_ts


# Heavy Ball Method
def heavy_ball(T, f, w_0, L=2):
    w_ts = [w_0, w_0]

    for t in range(1,T): 
        eta_t = t/(4*(t+1)*L)
        beta_t = (t-2)/(t+1)
        v_t = w_ts[-1] - w_ts[-2]
        w_t = w_ts[-1] - eta_t * f.grad(w_ts[-1]) + beta_t*v_t

        w_ts.append(w_t)

    return w_t[-1]



# Unconstrained Nesterov Accelerated Gradient Descent (Algorithm 12)
def alg_12(T, f, x_t, beta=0.5, L=2):
    # NEED TO FIX
    x_ts = []
    lambda_t = 0.5
    y_t = 0

    for t in range(1,T):
        theta_t = t/(2*(t+1)*L)
        beta_t = (t-2)/(t+1)
        w_t1 = z_t - theta_t * f.grad(z_t)
        z_t1 = w_t + beta_t(w_t1 - w_t)


        z_t = z_t1
        w_t = w_t1

    return w_t

if __name__ == "__main__":
    # Franke-Wolfe Training loop
    T = 10
    xbounds = [-10,10]
    f = PowerFunction(2,2)

    x2_t = np.array([10,-10])


    #x2_ts = NAG(T, f, x2_t)
    #plt.plot(x2_ts)
    #plt.title("NAG Algorithm")
    #plt.show()

    x_t = np.array([10,-10])
    x_t = np.array([0.4])
    print(x_t)

    x_ts = FrankeWolfeLoop(T, xbounds, f, x_t)

    print(x_ts)
    plt.plot(x_ts)
    plt.title("FW Algorithm")
    plt.show()
