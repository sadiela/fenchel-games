import numpy as np 
import matplotlib.pyplot as plt
import math
from convex_functions import *

def projection(x, bounds):
    # projection into hypercube
    # for n dimensional x, you will have a list of n sets of bounds
    for i in range(len(bounds)): 
        x[i] = np.minimum(np.maximum(x[i], bounds[i][0]), bounds[i][1])
    return x

# Gradient descent with averaging (different rate depending on smooth or not)
# X: OMD, Y: BestResp+, alpha_t = 1
def gradDescentAveraging(f, T, w_0, L=2, xbounds=[[-10,10]]): # ASSUMING SMOOTH
    w_ts = []
    w_ts.append(w_0)
    avg_ws = []
    eta = 1/(2*L)
    for t in range(1,T): 
        #w_ts.append(w_ts[-1] - eta*f.grad(w_ts[-1]))
        w_ts.append(projection(w_ts[-1] - eta*f.grad(w_ts[-1]), xbounds))
        avg_ws.append((1/t)*np.sum(w_ts))
    return avg_ws

# Cumulative gradient descent
# X: OMD, Y: FTL+, alpha_t = 1
def cumulativeGradientDescent(f, T, w_0, R, G, xbounds=[[-10,10]]):
    eta = (R/G)/math.sqrt(T)
    w_ts = []
    w_ts.append(w_0)
    grad_sum = f.grad(w_0)
    for t in range(1,T):
        w_t = (1-(1/t))*w_ts[-1] - (1/t)*eta*grad_sum
        #w_ts.append(w_t)
        w_ts.append(projection(w_t, xbounds))
        grad_sum += f.grad(w_t)
    return w_ts

# Frank-Wolfe
# X: BestResp+, Y: FTL, alpha_t = t
def frankWolfe(f, T, w_0, xbounds):
    w_ts = []
    w_ts.append(w_0)
    for t in range(0,T): 
        s_t = find_s(f.grad(w_ts[-1]), xbounds)
        w_t = w_ts[-1] + (2/(t+2))*(s_t-w_ts[-1])
        w_ts.append(w_t)
    return w_ts

# Linear rate FW 
# X: BestResp+, Y: AFTL, alpha_t = 1/|| \ell(xt) ||^2

# Single-call extra-gradient with averaging
# SPECIFIC TO EXPONENTIAL FUNCTION WITH L2 NORM AS 
# Y: BestResp+, X: Optimistic OMD, alpha_t = 1
def singleGradientCallExtraGradientWithAveraging(f, T, w_0, phi, L=2, xbounds=[[-10,10]], alphas=None):
    gamma = 1/L
    w_halfs = []
    w_halfs.append(w_0)
    w_ts = []
    w_ts.append(w_0)
    avg_ws = []
    avg_ws.append(w_0)
    if not alphas:
        alphas = [1 for i in range(0,T)]
    for t in range(0,T):
        q_1 = -gamma*alphas[t]*f.grad(w_ts[-1]) + phi.grad(w_halfs[-1])
        w_t = phi.fenchel_grad(q_1) #-gamma*f.grad(w_ts[-1]) + w_halfs[-1] #argmin(alpha_ts[t]*np.dot(w, f.grad(w_ts[-1])) + bregmanDivergence(phi, w ,w_halfs[-1]))
        w_ts.append(projection(w_t, xbounds))
        q_2 = -gamma*alphas[t]*f.grad(w_ts[-1]) + phi.grad(w_halfs[-1])
        w_t_12 = phi.fenchel_grad(q_2) #-gamma*f.grad(w_ts[-1]) + w_halfs[-1] #argmin(alpha_ts[t]*np.dot(w, f.grad(w_ts[-1])) + bregmanDivergence(phi, w ,w_halfs[-1]))
        w_halfs.append(projection(w_t_12, xbounds))
        avg_ws.append((1/(t+1))*np.sum(w_ts))
    return avg_ws

# Nesterov's 1-memory method PSEUDOCODE!!!
# X: OMD+, Y: Optimistic FTL, alpha_t = t
def nesterovOneMemory(f, T, w_0, phi, L=2, xbounds=[[-10,10]]):
    w_ts = []
    w_ts.append(w_0)
    v_ts = []
    v_ts.append(w_0)
    z_ts = []
    for t in range(1,T):
        beta_t = 2/(t+1)
        gamma_t = t/(4*L)
        z_t = (1-beta_t)*w_ts[-1] + beta_t*v_ts[-1]
        z_ts.append(z_t)
        q = phi.grad(v_ts[-1]) - gamma_t*f.grad(z_ts[-1])
        v_t = phi.fenchel_grad(q, 1)
        #v_t = v_ts[-1] - gamma_t*f.grad(z_t[-1])#argmin(gamma_t*np.dot(f.grad(z_t), x) + bregmanDivergence(phi, x, v_ts[-1]))
        v_ts.append(projection(v_t, xbounds))
        w_t = (1-beta_t)*w_ts[-1]+ beta_t*v_ts[-1]
        w_ts.append(w_t)
    return w_ts, v_ts

# Nesterov's infinity-memory method PSEUDOCODE
# X: FTRL+, Y: Optimistic FTL, alpha_t = t
def nesterovInfMemory(f, T, w_0, R, L=8, xbounds=[[-10,10]]):
    w_ts = []
    w_ts.append(w_0)
    v_ts = []
    v_ts.append(w_0)
    z_ts = []
    sums = 0
    for t in range(1,T):
        beta_t = 2/(t+1)
        gamma_t = t/(4*L)
        z_t = (1-beta_t)*w_ts[-1] + beta_t*v_ts[-1]
        z_ts.append(z_t)
        sums -= gamma_t*f.grad(z_ts[-1])
        v_t = R.fenchel_grad(sums)
        v_t = projection(v_t, xbounds) #argmin(sum_{s=1}^t gamma_s*np.dot(f.grad(z_s), x) + R.f(x))
        v_ts.append(v_t)
        w_t = (1-beta_t)*w_ts[-1]+ beta_t*v_ts[-1]
        w_ts.append(w_t)
    return w_ts

# Nesterov's first acceleration method
# Unconstrained Nesterov Accelerated Gradient Descent (Algorithm 12)
# X: OMD+, Y: Optimistic FTL, alpha_t = t
# (QUESTION) IS THIS THE SAME AS NESTEROV 1-MEMORY IF REGULARIZER IS THE SAME?
def NAG(f, T, w_0, L=2, xbounds=[[-10,10]]):
    # NEED TO FIX
    w_ts = []
    w_ts.append(w_0)
    z_s = []
    z_s.append(w_0)
    for t in range(1,T):
        theta_t = t/(2*(t+1)*L)
        beta_t = (t-2)/(t+1)
        w_t = z_s[-1] - theta_t * f.grad(z_s[-1])
        #w_ts.append(w_t)
        w_ts.append(projection(w_t, xbounds))
        z_t = w_ts[-1] + beta_t*(w_ts[-1] - w_ts[-2])
        z_s.append(z_t)
    return w_ts

# Heavy Ball Method
# X: FTRL+, Y: FTL, alpha_t = t
def heavyBall(f, T, w_0, L=2, xbounds=[[-10,10]]):
    w_ts = [w_0, w_0]
    for t in range(1,T): 
        eta_t = t/(4*(t+1)*L)
        beta_t = (t-2)/(t+1)
        v_t = w_ts[-1] - w_ts[-2]
        w_t = w_ts[-1] - eta_t * f.grad(w_ts[-1]) + beta_t*v_t
        #w_ts.append(w_t)
        w_ts.append(projection(w_t, xbounds))
    return w_ts

# Nesterov's method (REQUIRES STRONG CONVEXITY AND SMOOTHNESS)
#     

if __name__ == "__main__":
    # Franke-Wolfe Training loop
    T = 1500
    xbounds = [-10,10]
    f = ExpFunction() #2,2)#PowerFunction(2,2)
    phi = L2Reg()

    x_0 = np.array([5], dtype='float64')
    print(x_0)

    x_ts_FW = frankWolfe(f, T, x_0, xbounds)
    x_ts_GDAvg = gradDescentAveraging(f, T, x_0, L=1) 
    x_ts_cumulativeGD = cumulativeGradientDescent(f, T, x_0, R=10, G=1)
    x_ts_NAG = NAG(f, T, x_0, L=2)
    x_ts_heavyball = heavyBall(f, T, x_0, L=2)
    x_ts_sgc_eg = singleGradientCallExtraGradientWithAveraging(f, T, x_0, phi= phi)
    x_ts_n_onemem, _ = nesterovOneMemory(f,T,x_0, phi=phi)
    x_ts_n_infmem = nesterovInfMemory(f,T,x_0, R = phi)

    plt.plot(x_ts_FW, color='red', label="FrankWolfe")
    plt.plot(x_ts_GDAvg, color='blue', label="GradientDescentwithAveraging")
    plt.plot(x_ts_cumulativeGD, color='green', label="CumulativeGradientDescent")
    plt.plot(x_ts_NAG, color='purple', label="NAG")
    plt.plot(x_ts_heavyball, color='orange', label="HeavyBall")
    # SPECIFIC FOR EXP!
    plt.plot(x_ts_sgc_eg, color='black', label="Single-gradient-callExtra-gradientWithAveraging")
    plt.plot(x_ts_n_onemem, color='yellow', label="NesterovsOne-Memory")
    plt.plot(x_ts_n_infmem, color='gray', label="NesterovsInf-Memory")

    plt.title("Convex Optimization Algorithms on $f(x) = |x|")
    plt.legend()
    plt.show()
