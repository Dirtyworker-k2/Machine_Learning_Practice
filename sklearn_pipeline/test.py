'''
Descripttion: 
version: 
Author: sch
Date: 2020-09-12 21:45:23
LastEditors: sch
LastEditTime: 2020-09-13 09:55:55
'''
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_val_score 
from sklearn.feature_selection import SelectPercentile, chi2 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline 
from sklearn.svm import SVC
from sklearn.datasets import load_iris


X, y = load_iris(return_X_y=True)
np.random.seed(0)
X = np.hstack([X, np.random.random((X.shape[0], 36))])


clf = Pipeline([('anova', SelectPercentile(chi2)),
                ('scaler', StandardScaler()),
                ('svc', SVC(gamma='auto'))])


score_means = list()
score_stds = list()

percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)


for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    this_scores = cross_val_score(clf, X, y)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())


plt.errorbar(percentiles, score_means, score_stds)
plt.xlabel('Percentile')
plt.ylabel('Accuracy score')
plt.show()