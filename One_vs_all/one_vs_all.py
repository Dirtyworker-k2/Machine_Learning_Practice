'''
@Descripttion: 
@version: 
@Author: sch
@Date: 2020-07-16 11:35:03
@LastEditors: sch
@LastEditTime: 2020-07-24 19:41:09
'''
from scipy.io import loadmat 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize


num_labels=10
path = '/Users/mac/Desktop/ML/Neural Networks/ex3data1.mat'
data = loadmat(path)
X = data.get('X')
X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
y = data.get('y')
rows = X.shape[0]
params = X.shape[1]
theta = np.zeros(params)


def sigmoid(z):
    return 1 / (1 + np.exp(-z)) 


def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1-y, np.log(1 - sigmoid(X * theta.T)))
    reg = learningRate / (2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    
    return np.sum(first - second) / len(X) + reg 



def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    rows = X.shape[0]
    params = X.shape[1]

    error = sigmoid(X * theta.T) - y

    grad = (X.T * error / len(X)).T + learningRate / len(X) * theta
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()

print(gradient(theta, X, y, 1).shape)

def one_vs_all(X, y, num_labels,learningRate):
    all_theta = np.zeros((num_labels, params))

    for i in range(1, num_labels + 1):
        theta = np.zeros(params)
        y_i = np.array([1 if label==i else 0 for label in y])
        y_i = np.reshape(y_i, (rows,1))
        
        print(theta.shape)
        print(y_i.shape)
        print(X.shape)
    
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learningRate), method='TNC',
                        jac=gradient)
        all_theta[i-1, :] = fmin.x
    
    return all_theta


print(one_vs_all(X, y, 10, 1))