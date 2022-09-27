import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer

#from sklearn.datasets.samples_generator import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s = 30, color ='b')
plt.xlabel('X')
plt.ylabel('Y')
#uncomment to show scatter plot
#plt.show()
plt.clf()

visualizer = KElbowVisualizer(KMeans(), k=(1,10))
visualizer.fit(X)
visualizer.show()

# TODO determine the best k for k-means
# TODO calculate accuracy for best K
# TODO draw a confusion matrix