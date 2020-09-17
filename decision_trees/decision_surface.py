import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris


iris = load_iris()

cla_cnt = {}
for cla in list(iris.target):
    cla_cnt[cla] = cla_cnt.get(cla, 0) + 1

n_classes = len(cla_cnt)
plot_colors = 'ryb'
plot_step= 0.02

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    X = iris.data[:, pair]
    y = iris.target

    clf = DecisionTreeClassifier().fit(X, y)

    plt.subplot(2, 3, pairidx + 1)
    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = clf.predict(xy).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    edgecolor='black', s=15)

plt.suptitle('Decision surface of DTs using paired features')
plt.legend(loc='lower right', borderpad=0, handletextpad=0)


plt.figure(figsize=(20, 30))
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf, filled=True)
plt.show()