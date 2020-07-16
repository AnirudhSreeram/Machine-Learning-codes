import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        # Initialize all the parameters required
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None

    def fit(self, X, y):
        # convert all the labels into -1 and +1
        y_ = np.where(y <= 0, -1, 1)
        # Get the samples and features
        n_samples, n_features = X.shape
        # initialize weights and biases
        self.w = np.zeros((n_features))
        self.b = 0
        # number of iterations to be performed
        for _i in range(self.n_iterations):
            # For all the data in X taking 1 at a time
            for index, x_i in enumerate(X):
                # Check for the condition yi * f(x) >=1
                condition = y_[index] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dJW = 2 * self.lambda_param * self.w  # dJw = 2 * lambda * w
                    dJb = 0   # dJb = 0
                    self.w -= self.lr * dJW  # w = w - lr *dJW
                else:
                    dJW = 2 * self.lambda_param * self.w - \
                        np.dot(x_i, y_[index])  # dJW = 2 * lambda * w -yi * xi
                    dJb = y_[index]  # dJb = yi
                    self.w -= self.lr * dJW  # w = w - lr *dJW
                    self.b -= self.lr * dJb  # b = b - lr *dJb

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)


X, y = datasets.make_blobs(n_samples=50, n_features=2,
                           centers=2, cluster_std=1.05, random_state=40)
y = np.where(y == 0, -1, 1)
clf = SVM()
clf.fit(X, y)
print(clf.w, clf.b)


def visualize_svm():
    def get_hyperplane_value(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
    x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

    x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
    x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

    x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
    x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min-3, x1_max+3])
    location = "/home/anirudh/Documents/Machine_learning_algorithms/scatter_SVM_plot"
    plt.savefig(location)


visualize_svm()
