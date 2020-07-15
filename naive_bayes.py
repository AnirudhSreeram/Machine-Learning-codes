import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


class NaiveBayes:

    def fit(self, X, y):
        # Initialize the variables that are necessary
        n_samples, n_features = X.shape  # X.shape == (800,10)
        self._classes = np.unique(y)     # (2,)
        n_classes = len(self._classes)  # 2 classes 0 and 1

        self._mean = np.zeros((n_classes, n_features),
                              dtype=np.float64)  # (2,10)
        self._vars = np.zeros((n_classes, n_features),
                              dtype=np.float64)  # (2,10)
        self._priors = np.zeros((n_classes), dtype=np.float64)  # (2,)

        for i in self._classes:
            # taking samples that have class "i" as label
            X_i = X[i == y]
            # Calculate mean, varience and prior for each class "i"
            self._mean[i, :] = X_i.mean(axis=0)
            self._vars[i, :] = X_i.var(axis=0)
            self._priors[i] = X_i.shape[0]/float(n_samples)

    def predict(self, X):
        # Do predictions for all the test cases
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        # To compute posterior probabilities
        posteriors = []
        # run for 2 time as classes are 2 in number
        for index, classes in enumerate(self._classes):
            # calculate prior probabilities and take log
            prior = np.log(self._priors[index])  # log(P(y))
            # Compute class conditional probability
            # sum( log ( P ( Xi | y ) ) )
            class_cond = np.sum(np.log(self._pdf(index, x)))
            # Add the probabilities and take argmax
            posterior = prior + class_cond
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_indx, x):
        out = (1/np.sqrt(2*np.pi*self._vars[class_indx]**2)) * np.exp(-(
            (x-self._mean[class_indx])**2/(2*self._vars[class_indx])))
        return out


# code to test the algorithm
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


X, y = datasets.make_classification(
    n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
print("Naive Bayes classifiction accuracy : ",
      str(accuracy(y_test, predictions)))
