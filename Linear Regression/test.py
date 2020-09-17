import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


path = '/Users/mac/Desktop/ML/Linear Regression/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.plot(kind='scatter', x="Population", y='Profit', figsize=(12,8))

data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))


def computeCost(X, y, theta):
    inner = np.power((X * theta.T) - y, 2)
    return np.sum(inner) / (2 * len(X))

print(computeCost(X, y, theta))


def gradientDescent(X, y, theta, alpha, iters):
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    temp = np.matrix(np.zeros(theta.shape))

    for i in range(iters):
        error = X * theta.T - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - (alpha / len(X)) * np.sum(term)
        
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost


alpha = 0.01
iters = 1000
g, cost = gradientDescent(X, y, theta, alpha, iters)


x = np.linspace(data.Population.min(), data.Profit.max(), 100)
f = g[0,0] + g[0,1] * x


fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(x, f, c='r', label='Prediction')
ax1.scatter(data.Population, data.Profit, c='g', label='Profit')
ax1.set_xlabel('Prediction')
ax1.set_ylabel('Profit')
ax1.legend()



ax2.plot(np.arange(iters), cost, c='r')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Cost')
ax2.set_title('Error vs. Training set')
ax2.legend()
plt.show()


