
import numpy as np
import numpy.linalg as la
from sklearn import svm
import matplotlib.pyplot as plt

class SVM_Learner():

    def __init__(self, d, w, b):

        self.d = d
        self.w = w
        self.b = b

    def run_learning_algorithm(self, display, num_train, num_test):

        self.initialize_dataset(dataset_type = "TRAIN", N = num_train)
        self.initialize_dataset(dataset_type = "TEST", N = num_test)

        self.train_classifier()
        
        self.compute_error(error_type = "TRAIN")
        self.compute_error(error_type = "TEST")

        self.generalization_error = np.abs(self.test_error - self.training_error)

        if display:

            self.print_results()
            self.plot_results() 

    def initialize_dataset(self, dataset_type, N):

        if dataset_type == "TRAIN":
            self.N_TRAIN = N
            self.X_train, self.Y_train = self.generate_data_and_labels(N = self.N_TRAIN, d = self.d)

            Z_train = np.column_stack((self.X_train, self.Y_train))
   
            self.X_train_pos = [z[0:self.d] for z in Z_train if z[-1] == 1]    
            self.X_train_neg = [z[0:self.d] for z in Z_train if z[-1] == -1]
        
        elif dataset_type == "TEST":
            self.N_TEST = N
            self.X_test, self.Y_test = self.generate_data_and_labels(N = self.N_TEST, d = self.d)
            
            Z_test = np.column_stack((self.X_test, self.Y_test))
            
            self.X_test_pos = [z[0:self.d] for z in Z_test if z[-1] == 1]    
            self.X_test_neg = [z[0:self.d] for z in Z_test if z[-1] == -1]

    def generate_data_and_labels(self, N, d):

        X = np.random.uniform(low = -1.0, high = 1.0, size = (N, d))
        
        Y = np.zeros(shape = N)
        for i in range(0, N):
            #Y[i] = 1 if np.inner(self.w, X[i]) >= self.b else -1
            Y[i] = np.sign(np.inner(self.w, X[i]) + self.b)

        while np.abs(np.sum(Y)) == N:
            X = np.random.uniform(low = -1.0, high = 1.0, size = (N, d))
        
            Y = np.zeros(shape = N)
            for i in range(0, N):
                #Y[i] = 1 if np.inner(self.w, X[i]) >= self.b else -1
                Y[i] = np.sign(np.inner(self.w, X[i]) + self.b)

        return X, Y

    def train_classifier(self):

        self.classifier = svm.SVC(kernel = "linear")
        self.classifier.fit(self.X_train, self.Y_train)

        self.w_guess = (self.classifier.coef_[0])
        self.w_guess_NORMALIZED = (self.classifier.coef_[0])/la.norm(self.w_guess)
        
        self.b_guess = self.classifier.intercept_
        self.b_guess_NORMALIZED = self.classifier.intercept_/self.w_guess[1]

        

    def compute_error(self, error_type):

        if error_type == "TRAIN":
            self.num_training_errors, self.mis_training_points = self.svm_error(X = self.X_train, Y = self.Y_train)
            self.training_error = self.num_training_errors/self.N_TRAIN
        elif error_type == "TEST":
            self.num_test_errors, self.mis_test_points = self.svm_error(X = self.X_test, Y = self.Y_test)
            self.test_error = self.num_test_errors/self.N_TEST

    def svm_error(self, X, Y):

        num_errors = 0
        misclassified_points = []
        Y_guess = self.classifier.predict(X)
        
        for i in range(0, len(Y)):

            if Y[i] != Y_guess[i]:
                num_errors += 1
                misclassified_points.append(list(X[i]))

        return num_errors, misclassified_points

    def print_results(self):

        print("w = ")
        with np.printoptions(precision = 5):
            print(self.w)
        print("b = %0.5lf" % self.b)

        print("w_guess = ")
        with np.printoptions(precision = 5):
            print(self.w_guess_NORMALIZED)
        print("b_guess = %0.5lf" % self.b_guess_NORMALIZED)

        print("Number of training samples N_train = %d" % self.N_TRAIN)
        print("Number of test samples N_test = %d" % self.N_TEST)
        print("Training error = %lf" % self.training_error)
        print("Test Error = %lf" % self.test_error)
        print("Generalization Error = %lf" % self.generalization_error)

    def plot_results(self):

        t = np.linspace(-1, 1, 100)
        plt.figure()
        plt.scatter([x[0] for x in self.X_train_pos], [x[1] for x in self.X_train_pos], color = 'b', label = "+")
        plt.scatter([x[0] for x in self.X_train_neg], [x[1] for x in self.X_train_neg], color = 'r', label = "-")
        #plt.plot(t, -self.w[0]/self.w[1] * t - self.b/self.w[1], linewidth = 1.5, linestyle = '--', color = 'k', label = "w")
        
        #plt.plot(t, -self.w_guess[0]/self.w_guess[1] * t - self.b_guess/self.w_guess[1], linewidth = 1.5, linestyle = '--', color = 'g', label = "${\hat{w}}$")
        #plt.plot(t, t + self.b, linewidth = 1.5, linestyle = '--', color = 'orange')
        
        #plt.scatter([x[0] for x in self.mis_training_points], [x[1] for x in self.mis_training_points], marker = '.', color = 'c')
        
        #plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100, linewidth = 1.0, facecolors = "none", edgecolors = "k")

        # create grid to evaluate model
        xx = np.linspace(-1, 1, 100)
        yy = np.linspace(-1, 1, 100)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.classifier.decision_function(xy).reshape(XX.shape)

        #print(XX)
        #print(Z)
        # plot decision boundary and margins
        #plt.contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])

        #plt.title("Learning Hyperplane in ${{\mathbb{{R}}^{0}}}$: Training Data".format(self.d))
        plt.title("Training")
        #plt.grid(which = "major", color = 'gray', alpha = 0.5)
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(t, -self.w[0]/self.w[1] * t - self.b/self.w[1], linewidth = 1.5, linestyle = '--', color = 'k', label = "w")
        plt.plot(t, -self.w_guess[0]/self.w_guess[1] * t - self.b_guess/self.w_guess[1], linewidth = 1.5, linestyle = '--', color = 'g', label = "${\hat{w}}$")
        
        plt.scatter([x[0] for x in self.X_test_pos], [x[1] for x in self.X_test_pos], color = 'b', label = "+")
        plt.scatter([x[0] for x in self.X_test_neg], [x[1] for x in self.X_test_neg], color = 'r', label = "-")
        plt.scatter([x[0] for x in self.mis_test_points], [x[1] for x in self.mis_test_points], marker = '.', color = 'c', label = "Misclassified")
        #plt.contour(XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
        #plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100, linewidth = 1.0, facecolors = "none", edgecolors = "k")
        plt.title("Learning Hyperplane in ${{\mathbb{{R}}^{0}}}$: Test Data".format(self.d))
        plt.grid(which = "major", color = 'gray', alpha = 0.5)
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        plt.show()

d = 2
w = np.asarray([np.sqrt(2)/2, np.sqrt(2)/2])
#n = np.asarray([np.sqrt(3), 0.5])
#w = np.zeros(2)
#w[0] = -1
#w[1] = -n[0]/n[1] * w[0]

#w /= la.norm(w)

#w = np.asarray([np.sqrt(2)/2, np.sqrt(2)/2])
b = 0

NUM_TRAINING_POINTS = 25
NUM_TEST_POINTS = 100

def main():
    print("SALVE MUNDI")
    t = np.linspace(-1, 1, 100)

    np.random.seed(0x0B1)
    #np.random.seed(0x66023C)

    #print(w)
    robo = SVM_Learner(d = d, w = w, b = b)
    robo.run_learning_algorithm(display = True, num_train = NUM_TRAINING_POINTS, num_test = NUM_TEST_POINTS)
    
    X_train_pos1 = robo.X_train_pos
    X_train_neg1 = robo.X_train_neg
    w_guess1 = robo.w_guess
    b_guess1 = robo.b_guess

    '''
    plt.scatter([x[0] for x in X_train_pos1], [x[1] for x in X_train_pos1], color = 'b')
    plt.scatter([x[0] for x in X_train_neg1], [x[1] for x in X_train_neg1], color = 'r')
    plt.plot(t, -w_guess1[0]/w_guess1[1] * t - b_guess1/w_guess1[1], linewidth = 1.5, linestyle = '--', color = 'g', label = "${\hat{w}}$")
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.show()
    '''
    

    robo.run_learning_algorithm(display = True, num_train = NUM_TRAINING_POINTS, num_test = NUM_TEST_POINTS)
    
    X_train_pos2 = robo.X_train_pos
    X_train_neg2 = robo.X_train_neg
    w_guess2 = robo.w_guess
    b_guess2 = robo.b_guess

    '''
    plt.scatter([x[0] for x in X_train_pos2], [x[1] for x in X_train_pos2], color = 'b')
    plt.scatter([x[0] for x in X_train_neg2], [x[1] for x in X_train_neg2], color = 'r')
    plt.plot(t, -w_guess2[0]/w_guess2[1] * t - b_guess2/w_guess2[1], linewidth = 1.5, linestyle = '--', color = 'g', label = "${\hat{w}}$")
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.show()
    '''

    plt.scatter([x[0] for x in X_train_pos1], [x[1] for x in X_train_pos1], color = 'b', label = "+")
    plt.scatter([x[0] for x in X_train_neg1], [x[1] for x in X_train_neg1], color = 'r', label = "-")
    plt.scatter([x[0] for x in X_train_pos2], [x[1] for x in X_train_pos2], color = 'b')
    plt.scatter([x[0] for x in X_train_neg2], [x[1] for x in X_train_neg2], color = 'r')
    #plt.plot(t, -w_guess1[0]/w_guess1[1] * t - b_guess1/w_guess1[1], linewidth = 1.5, linestyle = '--', color = 'g', label = "${\hat{w}}$")
    plt.plot(t, -w_guess2[0]/w_guess2[1] * t - b_guess2/w_guess2[1], linewidth = 1.5, linestyle = '--', color = 'g', label = "${\hat{w}}$")
    plt.title("Training")
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.legend()
    plt.show()

    plt.scatter([x[0] for x in X_train_pos1], [x[1] for x in X_train_pos1], color = 'b', label = "+")
    plt.scatter([x[0] for x in X_train_neg1], [x[1] for x in X_train_neg1], color = 'r', label = "-")

    #plt.scatter([x[0] for x in X_train_pos1], [x[1] for x in X_train_pos1], s = 75, facecolors = 'c', alpha = 0.5, edgecolor = 'k', label = "${\\tilde{Z}_{S}}$")
    #plt.scatter([x[0] for x in X_train_neg1], [x[1] for x in X_train_neg1], s = 75, facecolors = 'c', alpha = 0.5, edgecolor = 'k')

    plt.scatter([x[0] for x in X_train_pos2], [x[1] for x in X_train_pos2], color = 'b')
    plt.scatter([x[0] for x in X_train_neg2], [x[1] for x in X_train_neg2], color = 'r')

    plt.scatter([x[0] for x in X_train_pos2], [x[1] for x in X_train_pos2], s = 75, facecolors = 'c', alpha = 0.5, edgecolor = 'k', label = "${\\tilde{Z}_{S}}$")
    plt.scatter([x[0] for x in X_train_neg2], [x[1] for x in X_train_neg2], s = 75, facecolors = 'c', alpha = 0.5, edgecolor = 'k')
    
    #plt.plot(t, -w_guess1[0]/w_guess1[1] * t - b_guess1/w_guess1[1], linewidth = 1.5, linestyle = '--', color = 'g', label = "${\hat{w}}$")
    plt.plot(t, -w_guess2[0]/w_guess2[1] * t - b_guess2/w_guess2[1], linewidth = 1.5, linestyle = '--', color = 'g', label = "${\hat{w}}$")
    plt.title("Training")
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.legend()
    plt.show()

def bound_plots():

    D = np.linspace(1, 9, 3, dtype = int)
    n = np.linspace(2, 101, 100, dtype = int)
    plt.figure()

    color_array = ['red', 'blue', 'green']

    for count, d in enumerate(D):
        plt.plot(n, np.sqrt(d/n), linestyle = '-', linewidth = 1.5, color = color_array[count], label = "VC, d = {0}".format(d))
        plt.plot(n, np.sqrt(d*np.log(n)/n), linestyle = '--', linewidth = 1.5, color = color_array[count], label = "CMI, d = {0}".format(d))
    plt.axhline(y = 1.0, linestyle = '--', linewidth = 1.0, color = 'k')
    plt.title("Learning Hyperplanes in ${\mathbb{R}^d}$: EGE vs. Number of training samples ${n}$ \n VC bound: ${EGE \leq O\left( \sqrt{d/n} \\right)}$, CMI bound: ${EGE \leq O\left( \sqrt{d \ln{(n)}/n} \\right)}$")
    plt.xlabel("Number of training examples ${n}$")
    plt.ylabel("EGE")
    plt.legend()
    plt.show()

    D = np.linspace(1, 10, 5, dtype = int)
    n = np.linspace(2, 101, 100, dtype = int)

    for d in D:
        plt.plot(n, np.sqrt(d/n), linewidth = 1.5, label = "d = {0}".format(d))
    plt.axhline(y = 1.0, linestyle = '--', linewidth = 1.0, color = 'k')
    plt.title("Learning Hyperplanes in ${\mathbb{R}^d}$, VC bound:\n ${EGE \leq O\left( \sqrt{d/n} \\right)}$ vs. Number of training samples ${n}$")
    plt.xlabel("Number of training examples ${n}$")
    plt.ylabel("EGE")
    plt.legend()
    plt.show()

    plt.figure()
    
    for d in D:
        plt.plot(n, np.sqrt(d*np.log(n)/n), linewidth = 1.5, label = "d = {0}".format(d))
    plt.axhline(y = 1.0, linestyle = '--', linewidth = 1.0, color = 'k')
    plt.title("Learning Hyperplanes in ${\mathbb{R}^d}$, CMI bound:\n ${EGE \leq O\left( \sqrt{d \ln{(n)}/n} \\right)}$ vs. Number of training samples ${n}$")
    plt.xlabel("Number of training examples ${n}$")
    plt.ylabel("EGE")
    plt.legend()
    plt.show()

    d = 2
    #VCdim = d + 1

    n = np.linspace(2, 1001, 1000, dtype = int)
    plt.figure()
    plt.plot(n, np.sqrt(d/n), linewidth = 1.5, color = 'b', label = "VC bound")
    plt.plot(n, np.sqrt(d*np.log(n)/n), linewidth = 1.5, color = 'r', label = "CMI bound")
    plt.title("Learning Hyperplane in ${\mathbb{R}^2}$: EGE vs. Number of training samples ${n}$ \n VC bound: ${EGE \leq O\left( \sqrt{2/n} \\right)}$, CMI bound: ${EGE \leq O\left( \sqrt{2 \ln{(n)}/n} \\right)}$")
    plt.xlabel("Number of training examples ${n}$")
    plt.ylabel("EGE")
    plt.legend()
    plt.show()

MAX_NUMBER_TRAINING_POINTS = 100
MAX_NUMBER_TEST_POINTS = 300
NUM_ITERATIONS = 1000

def run(print_flag):

    display_flag = False

    starting_sample_count = 10
    N = np.linspace(starting_sample_count, MAX_NUMBER_TRAINING_POINTS, MAX_NUMBER_TRAINING_POINTS - starting_sample_count + 1, dtype = int)
    robo = SVM_Learner(d = 2, w = w, b = b)
    
    expected_generalization_error = np.zeros(len(N))
    expected_generalization_error_var = np.zeros(len(N))
    for count, n in enumerate(N):

        print("Current number of training samples n = %d" % n)

        current_generalization_error = np.zeros(NUM_ITERATIONS)

        for i in range(0, NUM_ITERATIONS):

            if print_flag and i % (NUM_ITERATIONS/10) == 0:
                print("i = %d" % i)

            if (n == 25 and i == 0) or (n == 50 and i == 0) or (n == 100 and i == 0):
                display_flag = True
            else:
                display_flag = False

            robo.run_learning_algorithm(display = display_flag, num_train = n, num_test = MAX_NUMBER_TEST_POINTS)
            current_generalization_error[i] = robo.generalization_error
    
        expected_generalization_error[count] = np.average(current_generalization_error)
        expected_generalization_error_var[count] = np.std(current_generalization_error)

    d = 2
    n = np.linspace(10, 100, 90, dtype = int)
    plt.errorbar(N, expected_generalization_error, yerr = expected_generalization_error_var, linewidth = 1.5, color = 'k', ecolor = 'gray', elinewidth = 1.0, capsize = 1.5, label = "Empirical generalization error")
    plt.plot(n, np.sqrt(d/n), linewidth = 1.5, color = 'b', label = "VC bound")
    plt.plot(n, np.sqrt(d*np.log(n)/n), linewidth = 1.5, color = 'r', label = "CMI bound")
    plt.title("Learning Hyperplane in ${\mathbb{R}^2}$: EGE vs. Number of training samples ${n}$ \n VC bound: ${EGE \leq O\left( \sqrt{2/n} \\right)}$, CMI bound: ${EGE \leq O\left( \sqrt{2 \ln{(n)}/n} \\right)}$")
    plt.xlabel("Number of training examples ${n}$")
    plt.ylabel("EGE")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
    #run(print_flag = True)