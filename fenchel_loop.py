import numpy as np 


def f():
    return 0

def g(x,y):
    return 0 

def OMD():
    return 0

def OOMD():
    return 0

def FTL():
    return 0

if __name__ == "__main__":
    # schedules
    x_t = np.random.rand(10)
    y_t = np.random.rand(10)
    T = 10000 # number of rounds
    alpha_ts = 1 # or a  vector of length T 
    # Pick to algorithms
    for i in range(T): 
        print(i)


    # Pair up "fair players" FTL & OMD

    # what functions are we interested in optimizing? How hard is it to calc the fenchel conjugate?!?
    # power function
    # search OCO literature
    # convex loss functions? square loss, hinge loss, smoothed hinge loss, modified square loss, exponential loss function
    # log loss function 
    # which loss function/OL algorithm combos are "allowed"
    # https://core.ac.uk/download/pdf/213011306.pdf