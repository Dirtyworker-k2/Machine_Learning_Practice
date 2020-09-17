'''
@Descripttion: 
@version: 
@Author: sch
@Date: 2020-06-17 21:18:20
@LastEditors: sch
@LastEditTime: 2020-07-18 14:24:00
'''
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


path = '/Users/mac/Desktop/ML/Logistic Regression/ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Test1', 'Test2', 'Accepted'])
data.insert(data.shape[1], "Ones", 1)

positive = data[data.Accepted.isin([1])]
negative = data[data.Accepted.isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive.Test1, positive.Test2, marker='o', c='r', label='Accepted')
ax.scatter(negative.Test1, negative.Test2, marker='x', c='b', label='Not Accepted')
ax.set_xlabel('Test1')
ax.set_ylabel('Test2')
ax.legend()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

degree = 5
x1 = data.Test1.values
x2 = data.Test2.values

for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data.drop('Test1', axis=1, inplace=True)
data.drop('Test2', axis=1, inplace=True)

print(data.head())

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
    second = np.multiply(1-y, np.log(1 - sigmoid(X*theta.T)))
    reg = learningRate / (2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))

    return np.sum(first - second) / len(X) + reg 


def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X*theta.T) - y 


    for i in range(parameters):
        term = np.multiply(error, X[:,i])

        if (i==0):
            grad[i] = np.sum(term) / len(X)
        
        else:
            grad[i] = (np.sum(term) / len(X)) + learningRate / len(X) * theta[:, i]
        
    return grad 


cols = data.shape[1]
X = data.iloc[:, 1:cols].values
y = data.iloc[:, 0:1].values
theta = np.zeros(11)
learningRate = 1

print(gradientReg(theta, X, y, learningRate))
print(costReg(theta, X, y, learningRate))

import scipy.optimize as opt 

result = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(X, y, learningRate))
print(result)