import numpy as np 
import scipy.io as sio 
import scipy.optimize as opt 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
learningRate = 1


def load_data(path):
    d = sio.loadmat(path)
    return d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']

path = '/Users/mac/Desktop/ML/Variance_vs_Bias/ex5data1.mat'
X, y, Xval, yval, Xtest, ytest = load_data(path)
# print(X.shape, y.shape, Xval.shape, yval.shape, Xtest.shape, ytest.get_shape)

df = pd.DataFrame({'water_level':np.array(X).ravel(), 'flow':np.array(y).ravel()})
# sns.lmplot('water_level', 'flow', data=df, fit_reg=False, size=7)
# plt.show()

X, Xval, Xtest = [np.insert(x, 0, values=np.ones(x.shape[0]), axis=1) for x in (X, Xval, Xtest)]


def regularized_cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    rows = X.shape[0]

    inner = X @ theta.T - y
    reg = learningRate / (2 * rows) * np.sum(np.power(theta[1:], 2))

    return np.sum(np.power(inner, 2)) / (2 * rows) + reg

theta = np.ones(2)
# print(cost(theta, X, y, 1))



def regularized_gradient(theta, X, y,learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    rows = X.shape[0]

    inner = np.array(X.T @ (X @ theta.T - y)).ravel() / rows
    reg = learningRate / rows * theta

    grad = inner + reg
    grad[0, 0] = np.sum(np.multiply((X @ theta.T - y), X[:, 0])) / rows
    
    return grad 

# print(regularized_gradient(theta, X, y, learningRate))


def linear_regression_np(theta, X, y, learningRate):
    res = opt.minimize(fun=regularized_cost, x0=theta, args=(X, y, learningRate), 
                            method='TNC', jac=regularized_gradient, options={'disp':True})

    return res


final_theta = linear_regression_np(theta, X, y, learningRate).x
# (2, )

b = final_theta[0]
m = final_theta[1]

fig, ax = plt.subplots(figsize=(8, 6))
# ax.scatter(X[:, 1], y, label='Training data') # X.shape=(12,1)
# ax.plot(X[:, 1], X[:, 1] * m + b, label='Prediction')
plt.legend(loc=2)
# plt.show()


training_cost, cv_cost = [], []

rows = X.shape[0]
for i in range(1, rows+1):
    res = linear_regression_np(theta, X[0:i, :], y[:i, :], 0)

    tc = regularized_cost(res.x, X[0:i, :], y[:i, :], 0)
    cv = regularized_cost(res.x, Xval, yval, 0)
    training_cost.append(tc)
    cv_cost.append(cv)


fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(np.arange(1, rows+1), training_cost, label='training cost')
# ax.plot(np.arange(1, rows+1), cv_cost, label='cv cost')
ax.legend(loc=1)



X, y, Xval, yval, Xtest, ytest = load_data(path)

def poly_features(x, power, as_adarray=False):
    data = {'f{0}'.format(i): np.array(np.power(x, i)).ravel() for i in range(1, power + 1)}
    df = pd.DataFrame(data)

    return df.as_matrix() if as_adarray else df


def normalize_features(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


def prepare_poly_data(*args, power):
    def prepare(x):
        df = poly_features(x, power=power)

        ndarr = np.matrix(normalize_features(df))

        return np.insert(ndarr, 0, values=np.ones(ndarr.shape[0]), axis=1)
    
    return [prepare(x) for x in args]


X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)

theta = np.ones(9)
def plot_learning_curves(theta, X, y, Xval, yval, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    training_cost, cv_cost = [], []
    rows = X.shape[0]

    for i in range(1, rows+1):
        res = linear_regression_np(theta, X[:i, :], y[:i, :], learningRate)

        tc = regularized_cost(res.x, X[:i, :], y[:i, :], learningRate)
        cv = regularized_cost(res.x, Xval, yval, learningRate)

        training_cost.append(tc)
        cv_cost.append(cv)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, rows+1), training_cost, label='training cost')
    ax.plot(np.arange(1, rows+1), cv_cost, label='cv cost')
    ax.legend(loc=1)

# plot_learning_curves(theta, X_poly, y, Xval_poly, yval, learningRate=100)


learningRate_candidate = [0 ,0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []

for learningRate in learningRate_candidate:
    res = linear_regression_np(theta, X_poly[:i, :], y[:i, :], learningRate)

    tc = regularized_cost(res.x, X_poly[:i, :], y[:i, :], learningRate)
    cv = regularized_cost(res.x, Xval_poly, yval, learningRate)

    training_cost.append(tc)
    cv_cost.append(cv)


fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(learningRate_candidate, training_cost, label='training cost')
ax.plot(learningRate_candidate, cv_cost, label='cv cost')
ax.legend(loc=1)
plt.xlabel('learningRate')
plt.ylabel('cost')
# plt.show()

for learningRate in learningRate_candidate:
    res = linear_regression_np(theta, X_poly, y, learningRate)
    print('test cost(learningRate={0}) = {1}'.format(learningRate, regularized_cost(res.x, Xtest_poly, ytest, learningRate)))