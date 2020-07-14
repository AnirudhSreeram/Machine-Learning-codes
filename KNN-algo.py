import os
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from collections import Counter
cmap = ListedColormap(['r','g','b'])

def euclidian_distances(x1,x2):
    return (np.sqrt(np.sum((x1-x2)**2)))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        predicted_lables = [self._predict(x) for x in X]
        return np.array(predicted_lables)

    def _predict(self, x):
        # compute distances
        distances = [euclidian_distances(x,all_samples) for all_samples in self.X_train]

        # get K nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_lable = [self.y_train[index] for index in k_indices]

        # so a majority vote
        most_common = Counter(k_nearest_lable).most_common(1)
        return most_common[0][0]


class testCase:
    def __init__(self):
        self.iris = datasets.load_iris()
        self.X, self.y = self.iris.data, self.iris.target

    def makeSplit(self, X, y):
        self.X_train,self.X_test, self.y_train,self.y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
        print("Xtrain shape : " + str(self.X_train.shape))
        print("Xtrain sample : " + str(self.X_train[0]))

        print("ytrain shape : " + str(self.y_train.shape))
        print("ytrain sample : " + str(self.y_train))
        
        if os.path.isfile('/home/anirudh/Documents/Machine_learning_algorithms/scatter_plot.png') is False:
            location = "/home/anirudh/Documents/Machine_learning_algorithms/scatter_plot"
            plt.figure()
            plt.scatter(X[:,2],X[:,3],c=y,cmap=cmap,edgecolor='k',s=20)
            plt.savefig(location)

    def test_Knn(self):
        clf = KNN(k=3)
        clf.fit(self.X_train, self.y_train)
        predictions = clf.predict(self.X_test)
        acc = np.sum(predictions == self.y_test)/len(self.y_test)
        print(acc)

test = testCase()
test.makeSplit(test.X,test.y)
test.test_Knn()
