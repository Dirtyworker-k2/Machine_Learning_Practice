import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.io as sio 
import scipy.optimize as opt 
import seaborn as sns 
learningRate = 1


path = '/Users/mac/Desktop/ML/Variance_vs_Bias/ex5data1.mat'
theta = np.ones(2)
def load_data(path):
    d = sio.loadmat(path)
    return d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']

X, y, Xval, yval, Xtest, ytest = load_data(path)

df = pd.DataFrame({'water_level':np.array(X).ravel(), 'flow':np.array(y).ravel()})
sns.lmplot('water_level', 'flow', data=df, fit_reg=False, size=7)


X, Xval, Xtest = [np.insert(x, 0, values=np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]


def regularized_cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    rows = X.shape[0]

    inner = np.power(X @ theta.T - y, 2)
    reg = learningRate * np.sum(theta[1:])

    return np.sum(inner) / (2 * rows) + reg / (2 * rows)


def regularized_gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    rows = X.shape[0]

    inner = np.array(X.T @ (X @ theta.T - y)).ravel()
    reg = learningRate / rows * theta

    grad = inner / rows + reg
    grad[0, 0] = np.sum(np.multiply((X @ theta.T - y), X[:, 0])) / rows

    return np.array(grad).ravel()


def linear_regression_np(theta, X, y, learningRate):
    res = opt.minimize(fun=regularized_cost, x0=theta,
                        args=(X, y, learningRate), method='TNC',
                        jac=regularized_gradient,
                        options={'disp':True})
    return res 


final_theta = linear_regression_np(theta, X, y, 1).x
fig, ax = plt.subplots(figsize=(6, 4))
b = final_theta[0]
m = final_theta[1]
ax.scatter(X[:, 1], y, label='Training data', c='g')
ax.plot(X[:, 1], X[:, 1] * m + b, label='Prediction', c='r')
ax.legend(loc=2)



training_cost, cv_cost = [], []
rows = X.shape[0]
for i in range(1, rows + 1):
    res = linear_regression_np(theta, X[:i, :], y[:i, :], 0)

    tc = regularized_cost(res.x, X[:i, :], y[:i, :], 0)
    cv = regularized_cost(res.x, Xval, yval, 0)

    training_cost.append(tc)
    cv_cost.append(cv)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.arange(1, rows+1), training_cost, label='Training cost')
ax.plot(np.arange(1, rows+1), cv_cost, label='CV cost')
ax.legend(loc=1)



