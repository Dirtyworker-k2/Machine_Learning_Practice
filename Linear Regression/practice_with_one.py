import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# 1 读取数据
path = "/Users/mac/Desktop/ML/Linear Regression/ex1data1.txt"
data = pd.read_csv(path,header=None,names=['Population','Profit'])
data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8))
plt.show()


# 2 构造代价函数
def computeCost(X,y,theta):
    inner = np.power((X * theta.T) - y,2)
    return np.sum(inner) / (2 * len(X))


# 实例：计算代价函数
data.insert(0,'Ones',1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
print(computeCost(X,y,theta))

# 3 批处理梯度下降算法
def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = X * theta.T - y
        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - (alpha / len(X) * np.sum(term))
        
        theta = temp
        cost[i] = computeCost(X,y,theta)
    
    return theta, cost 

alpha = 0.01
iters = 1000
g, cost = gradientDescent(X,y,theta,alpha,iters)
print(computeCost(X,y,g))


# 绘制拟合图像
x = np.linspace(data.Population.min(),data.Population.max(),100)
f = g[0,0] + g[0,1] * x

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x,f,'r',label="Prediction")
ax.scatter(data.Population,data.Profit,label="Training Data")
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title("Preddiction  vs.  Training Data")
plt.show()


# 绘制cost-iterations图像
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title("Error vs. Training Data")
plt.show()