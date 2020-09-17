from sklearn.datasets import load_iris 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np


def custom_cv_2folds(X):
     n = X.shape[0]
     i = 1
     while i <= 2:
         idx = np.arange(n * (i - 1) / 2, n * i / 2, dtype=int)
         yield idx, idx
         i += 1

X, y = load_iris(return_X_y=True)
clf = SVC(kernel='linear', C=1)
custom_cv = custom_cv_2folds(X)
print(custom_cv)
scores = cross_val_score(clf, X, y, cv=custom_cv)
print(scores)