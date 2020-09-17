'''
Descripttion: 
version: 
Author: sch
Date: 2020-08-31 14:13:27
LastEditors: sch
LastEditTime: 2020-08-31 15:14:02
'''
from sklearn import svm 
# from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import numpy 
import pandas as pd 
import scipy.io as sio 
import numpy as np 


# 1 读取数据
path = '/Users/mac/Desktop/ML/SVM/data/ex6data3.mat'
mat = sio.loadmat(path)
training = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
training['y'] = mat.get('y')

cv = pd.DataFrame(mat.get('Xval'), columns=['X1', 'X2'])
cv['y'] = mat.get('yval')


# 2 manual grid search for C and theta
candidate = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
combination = [(C, gamma) for C in candidate for gamma in candidate]

search = []

for C, gamma in combination:
    svc = svm.SVC(C=C, gamma=gamma, kernel='rbf')
    svc.fit(training[['X1', 'X2']], training['y'])
    search.append(svc.score(cv[['X1', 'X2']], cv['y']))

best_score = search[np.argmax(search)]
best_params = combination[np.argmax(search)]
# print(best_score, best_params)


best_svc = svm.SVC(C=100, gamma=0.3, kernel='rbf')
best_svc.fit(training[['X1', 'X2']], training['y'])
ypred = best_svc.predict(cv[['X1', 'X2']])

print(metrics.classification_report(cv['y'], ypred))



# sklearn GridSearchCV
parameters = {'C':candidate, 'gamma':candidate}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, n_jobs=-1)
clf.fit(training[['X1', 'X2']], training['y'])