import numpy as np 
import matplotlib.pyplot as plt
import math
from convex_functions import *

# Gradient descent with averaging (different rate depending on smooth or not)
# (equiv to OMD + bestresp)
def gradDescentAveraging(f, T, w_0, L=2): # ASSUMING SMOOTH
    w_ts = []
    w_ts.append(w_0)
    avg_ws = []
    eta = 1/(2*L)
    for t in range(1,T): 
        w_ts.append(w_ts[-1] - eta*f.grad(w_ts[-1]))
        avg_ws.append((1/t)*np.sum(w_ts))
    return avg_ws

# Cumulative gradient descent
def cumulativeGradientDescent(f, T, w_0, R, G):
    eta = (R/G)/math.sqrt(T)
    w_ts = []
    w_ts.append(w_0)
    grad_sum = f.grad(w_0)
    for t in range(1,T):
        w_t = (1-(1/t))*w_ts[-1] - (1/t)*eta*grad_sum
        w_ts.append(w_t)
        grad_sum += f.grad(w_t)
    return w_ts

# Frank-Wolfe
def frankWolfe(f, T, w_0, xbounds):
    w_ts = []
    w_ts.append(w_0)
    for t in range(0,T): 
        s_t = find_s(f.grad(w_ts[-1]), xbounds)
        w_t = w_ts[-1] + (2/(t+2))*(s_t-w_ts[-1])
        w_ts.append(w_t)
    return w_ts

# Linear rate FW 


# Single-call extra-gradient with averaging
'''
def singleGradientCallExtraGradientWithAveraging(f, T, w_0, L=2, phi, alpha_ts):
    gamma = 1/L
    w_halfs = []
    w_halfs.append(w_0)
    w_ts = []
    w_ts.append(w_0)
    avg_ws = []
    avg_ws.append(w_0)
    # PSEUDOCODE
    for t in range(0,T):
        w_t = argmin(alpha_ts[t]*np.dot(w, f.grad(w_ts[-1])) + bregmanDivergence(phi, w ,w_halfs[-1]))
        w_ts.append(w_t)
        w_t_12 = argmin(alpha_ts[t]*np.dot(w, f.grad(w_ts[-1])) + bregmanDivergence(phi, w ,w_halfs[-1]))
        w_halfs.append(w_t_12)
        avg_ws.append((1/t+1)*np.sum(w_ts))
    return avg_w_ts
'''

# Nesterov's 1-memory method PSEUDOCODE!!!
'''
def nesterovOneMemory(f, T, w_0, phi, L=2):
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
        v_t = argmin(gamma_t*np.dot(f.grad(z_t), x) + bregmanDivergence(phi, x, v_ts[-1]))
        v_ts.append(v_t)
        w_t = (1-beta_t)*w[t-1]+ beta_t*v_ts[-1]
        w_ts.append(w_t)
'''

# Nesterov's infinity-memory method PSEUDOCODE
'''
def nesterovInfMemory(f, T, w_0, R, L=2):
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
        v_t = argmin(sum_{s=1}^t gamma_s*np.dot(f.grad(z_s), x) + R.f(x))
        v_ts.append(v_t)
        w_t = (1-beta_t)*w[t-1]+ beta_t*v_ts[-1]
        w_ts.append(w_t)
'''

# Nesterov's first acceleration method
# Unconstrained Nesterov Accelerated Gradient Descent (Algorithm 12)
def NAG(f, T, w_0, L=2):
    # NEED TO FIX
    w_ts = []
    w_ts.append(w_0)
    z_s = []
    z_s.append(w_0)
    for t in range(1,T):
        theta_t = t/(2*(t+1)*L)
        beta_t = (t-2)/(t+1)
        w_t = z_s[-1] - theta_t * f.grad(z_s[-1])
        w_ts.append(w_t)
        z_t = w_ts[-1] + beta_t*(w_ts[-1] - w_ts[-2])
        z_s.append(z_t)
    return w_ts

# Heavy Ball Method
def heavyBall(f, T, w_0, L=2):
    w_ts = [w_0, w_0]
    for t in range(1,T): 
        eta_t = t/(4*(t+1)*L)
        beta_t = (t-2)/(t+1)
        v_t = w_ts[-1] - w_ts[-2]
        w_t = w_ts[-1] - eta_t * f.grad(w_ts[-1]) + beta_t*v_t
        w_ts.append(w_t)
    return w_ts

# Nesterov's method (REQUIRES STRONG CONVEXITY AND SMOOTHNESS)
#     

if __name__ == "__main__":
    # Franke-Wolfe Training loop
    T = 100
    xbounds = [-10,10]
    f = SqrtOneXSquared()

    x_0 = np.array([5], dtype='float64')
    print(x_0)

    x_ts_FW = frankWolfe(f, T, x_0, xbounds)
    x_ts_GDAvg = gradDescentAveraging(f, T, x_0, L=1) 
    x_ts_cumulativeGD = cumulativeGradientDescent(f, T, x_0, R=10, G=1)
    x_ts_NAG = NAG(f, T, x_0, L=2)
    x_ts_heavyball = heavyBall(f, T, x_0, L=2)

    plt.plot(x_ts_FW, color='red', label="FrankWolfe")
    plt.plot(x_ts_GDAvg, color='blue', label="GradientDescentwithAveraging")
    plt.plot(x_ts_cumulativeGD, color='green', label="CumulativeGradientDescent")
    plt.plot(x_ts_NAG, color='purple', label="NAG")
    plt.plot(x_ts_heavyball, color='orange', label="HeavyBall")

    plt.title("Convex Optimization Algorithms")
    plt.legend()
    plt.show()
