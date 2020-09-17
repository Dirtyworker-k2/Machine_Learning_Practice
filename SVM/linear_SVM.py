'''
Descripttion: 
version: 
Author: sch
Date: 2020-08-30 20:47:27
LastEditors: sch
LastEditTime: 2020-08-31 10:41:31
'''
import scipy.io as sio 
import numpy as np 
import pandas as pd 
import sklearn.svm 
import seaborn as sns 
import matplotlib.pyplot as plt 


# 1 读取数据
path = '/Users/mac/Desktop/ML/SVM/data/ex6data1.mat'
mat = sio.loadmat(path)
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')

# sns.lmplot(x='X1', y='X2', hue='y', data=data, fit_reg=False)
# ax = plt.gca()
# ax.set_xlabel('liu')
# plt.show()


# 2 C=1
svc1 = sklearn.svm.LinearSVC(C=1, loss='hinge')
svc1.fit(data[['X1', 'X2']], data['y'])
score1 = svc1.score(data[['X1', 'X2']], data['y'])
print(score1)
data['SVM1 Confidence'] = svc1.decision_function(data[['X1', 'X2']])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='RdBu')
ax.set_title('SVM(C=1) Decision Confidence')



# try C=100
svc100 = sklearn.svm.LinearSVC(C=100, loss='hinge')
svc100.fit(data[['X1', 'X2']], data['y'])
score100 = svc100.score(data[['X1', 'X2']], data['y'])
print(score100)
data['SVM100 Confidence'] = svc100.decision_function(data[['X1', 'X2']])


fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='RdBu')
ax.set_title('SVM(C=100) Decision Confidence')
plt.show()

print(data.head())