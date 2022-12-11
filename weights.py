
import matplotlib.pyplot as plt
import numpy as np

class Weights:

    def __init__(self, name, T):
        self.name = name
        self.T = T

        if self.name == "ones":
            self.weights = np.ones(self.T + 1)
            self.latex_print = r'$\alpha_{t} = 1$'
        elif self.name == "linear":
            self.weights = np.linspace(0, self.T, self.T + 1)
            self.latex_print = r'$\alpha_{t} = t$'
        elif self.name == "sqrt":
            self.weights = [np.sqrt(t) for t in range(0, self.T + 1)]
            self.weights[0] = 0
            self.latex_print = r'$\alpha_{t} = \sqrt{t}$'
        elif self.name == "log":
            self.weights = [0]
            self.latex_print = r'$\alpha_{t} = \log{(t + 1)}$'
            for t in range(1, self.T + 1):
                self.weights.append(np.log(t + 1))
                
    def print_weights(self):

        print("Weight Schedule: %s, Number of Weights = %d" % (self.name, len(self.weights)))
        for t in range(0, self.T):
            print("\u03B1[%d] = %lf" % (t, self.weights[t]))

    def plot_weights(self):

        plt.plot(self.weights[1:self.T+1], '-b', linewidth = 1.5)
        plt.suptitle("Weight schedule: " + self.name)
        plt.title(self.latex_print)
        plt.show()

if __name__ == '__main__':

    T = 10
    weight_schedules = ["ones", "linear", "sqrt", "log"]

    for wt in weight_schedules:
        alpha_t = Weights(name = wt, T = T)
        alpha_t.print_weights()
        alpha_t.plot_weights()