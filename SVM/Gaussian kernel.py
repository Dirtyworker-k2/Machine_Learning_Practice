'''
Descripttion: 
version: 
Author: sch
Date: 2020-08-31 10:42:11
LastEditors: sch
LastEditTime: 2020-08-31 14:04:17
'''
import scipy.io as sio 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import sklearn.svm 
import matplotlib.pyplot as plt 


# 1 高斯核函数
def gaussian_krenel(x1, x2, sigma):
    return np.exp(- np.power(x1 - x2, 2).sum() / (2 * np.power(sigma, 2)))

x1 = np.array([1, 2, 1]) 
x2 = np.array([0, 4, -1])
sigma = 2

# print(gaussian_krenel(x1, x2, sigma))



# 2 读取数据
path = '/Users/mac/Desktop/ML/SVM/data/ex6data2.mat'
mat = sio.loadmat(path)
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')


# 3 数据可视化
sns.set(context='notebook', style='white', palette=sns.diverging_palette(240, 10, n=2))
sns.lmplot(x='X1', y='X2', hue='y', data=data, fit_reg=False, scatter_kws={'s':10})
# plt.show()



# 4 SVC
svc = sklearn.svm.SVC(C=100, kernel='rbf', gamma=10, probability=True)

svc.fit(data[['X1', 'X2']], data['y'])
score = svc.score(data[['X1', 'X2']], data['y'])

predict_prob = svc.predict_proba(data[['X1', 'X2']])[:, 0]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=30, c=predict_prob, cmap='Reds')
plt.show()