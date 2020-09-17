'''
Descripttion: 
version: 
Author: sch
Date: 2020-08-31 10:05:25
LastEditors: sch
LastEditTime: 2020-08-31 10:31:50
'''
import scipy.io as sio 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn.svm


# 1 读取数据
path = '/Users/mac/Desktop/ML/SVM/data/ex6data1.mat'
mat = sio.loadmat(path)
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')

sns.lmplot('X1', 'X2', hue='y', data=data, fit_reg=False)
ax = plt.gca()
ax.set_title('DataSet')


# 2 try C=1
svc1 = sklearn.svm.LinearSVC(C=100, loss='hinge')
svc1.fit(data[['X1', 'X2']], data['y'])
score100 = svc1.score(data[['X1', 'X2']], data['y'])
data['SVM1 Confidence'] = svc1.decision_function(data[['X1', 'X2']])
print(score100)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='RdBu')
# plt.show()