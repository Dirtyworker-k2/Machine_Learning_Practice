'''
Descripttion: 
version: 
Author: sch
Date: 2020-09-14 10:47:54
LastEditors: sch
LastEditTime: 2020-09-15 11:51:30
'''
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier, plot_tree


iris = load_iris()
cla_cnt = {}
for cla in list(iris.target):
    cla_cnt[cla] = cla_cnt.get(cla, 0) + 1

n_classes = len(cla_cnt)
plot_colors = 'ryb'
plot_step = 0.02


for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    X = iris.data[:, pair]
    y = iris.target

    clf = DecisionTreeClassifier().fit(X, y)

    plt.subplot(2, 3, pairidx+1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T

    Z = clf.predict(xy).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    plt.tight_layout(pad=2.5, h_pad=0.5, w_pad=0.5)
    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target[i], s=15,
                    edgecolors='black')

plt.suptitle('Decision surface of DTs for paired features')
plt.show()