'''
@Descripttion: 
@version: 
@Author: sch
@Date: 2020-07-24 22:55:40
@LastEditors: sch
@LastEditTime: 2020-07-26 18:19:47
'''
import scipy.io as sio 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize


path = '/Users/mac/Desktop/ML/One_vs_all/ex3data1.mat'
data = sio.loadmat(path)
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

    error = sigmoid(X * theta.T) - y
    
    grad = (X.T * error / len(X)).T + learningRate / len(X) * theta
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
    
    return np.array(grad).ravel()



def one_vs_all(X, y, num_labels, learningRate):
    X = np.matrix(X)
    y = np.matrix(y)
    all_theta = np.zeros((num_labels, params))

    for i in range(1, num_labels + 1):
        theta = np.zeros(params)
        y_i = np.array([1 if label==i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learningRate),
                        method='TNC', jac=gradient)
    
        all_theta[i-1, :] = fmin.x
    
    return all_theta



def predict_all(theta, X):
    X = np.matrix(X)

    h = sigmoid(X * theta.T)
    h_argmax = np.argmax(h, axis=1)
    h_argmax = h_argmax + 1

    return h_argmax



if __name__ == '__main__':
    all_theta = one_vs_all(X, y, 10, 1)
    y_pred = predict_all(all_theta, X)
    correct = [1 if a==b else 0 for (a,b) in zip(y, y_pred)]
    accurancy = sum(map(int, correct)) / len(X)
    print('The accurancy of the model is {0}%'.format(accurancy * 100))