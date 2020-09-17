import scipy.io as sio 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder


path = '/Users/mac/Desktop/ML/NN_backpropagation/ex4data1.mat'
data = sio.loadmat(path)
X = data.get('X')
y = data.get('y')
encoder = OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore')
y_onehot = encoder.fit_transform(y)



# initialize NN architecture
input_size = 400 
hidden_size = 25
num_labels = 10 

params = (np.random.random(size=(input_size + 1) * hidden_size + (hidden_size + 1) * num_labels) - 0.5) * 0.25
theta1 = np.matrix(np.reshape(params[:(input_size + 1) * hidden_size],
                                (hidden_size, input_size + 1)))
theta2 = np.matrix(np.reshape(params[(input_size + 1) * hidden_size:],
                                (num_labels, hidden_size + 1)))

learningRate = 1



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivatives(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))


def forwardPropagation(X, theta1, theta2):
    X = np.matrix(X)
    rows = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(rows), axis=1)
    z2 = a1 * theta1.T 
    a2 = np.insert(sigmoid(z2), 0 ,values=np.ones(rows), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3) 

    return a1, z2, a2, z3, h


def cost(params, input_size, hidden_size, num_labels, X, y, learningRate):
    X = np.matrix(X)
    y = np.matrix(y)
    rows = X.shape[0]

    a1, z2, a2, z3, h = forwardPropagation(X, theta1, theta2)

    J = 0
    for i in range(rows):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply(1 - y[i, :], np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    
    J = J / rows
    J += learningRate/ (2 * rows) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    return J


def backPropagation(params, input_size, hidden_size, num_labels, X, y, learningRate):
    X = np.matrix(X)
    y = np.matrix(y)
    rows = X.shape[0]

    a1, z2, a2, z3, h = forwardPropagation(X, theta1, theta2)

    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    for t in range(rows):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        z3t = z3[t, :]  # (1, 10)
        ht = h[t, :]    # (1, 10)
        yt = y[t, :]    # (1, 10)

        d3t = ht - yt   # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_derivatives(z2t))

        delta1 = delta1 + d2t[:, 1:].T * a1t
        delta2 = delta2 + d3t.T * a2t

        delta1 = delta1 / rows
        delta2 = delta2 / rows

        delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learningRate) / rows
        delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learningRate) / rows

        grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

        return grad
        
print(backPropagation(params, input_size, hidden_size, num_labels, X, y, learningRate).shape)
