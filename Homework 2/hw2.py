from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

def scatterPlot():
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=20, cmap='plasma')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    plt.clf()

def elbow():
    visualizer = KElbowVisualizer(KMeans(), k=(1,12))
    visualizer.fit(X)
    visualizer.show()

def confusionMatrix():
    X_train, X_test, y_train, y_test = train_test_split(X, y_true, random_state=0)
    clf = SVC(random_state=0)
    clf.fit(X_train, y_train)
    SVC(random_state=0)
    plot_confusion_matrix(clf, X_test, y_test)  
    plt.show()
    plt.clf()

#uncomment to run functions
scatterPlot()
elbow()
confusionMatrix()