'''
Descripttion: 
version: 
Author: sch
Date: 2020-09-13 21:41:58
LastEditors: sch
LastEditTime: 2020-09-13 22:06:28
'''
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# EG.1
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = DecisionTreeClassifier()
clf.fit(X, y)
print(clf.predict([[2, 2]]))


X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier()
clf.fit(X, y)
fig, ax = plt.subplots(figsize=(20, 20))
plot_tree(clf, max_depth=4, fontsize=10, ax=ax)
plt.show()