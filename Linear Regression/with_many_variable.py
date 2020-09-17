import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


# 1 数据读取
path = "/Users/mac/Desktop/ML/Linear Regression/ex1data2.txt"
data2 = pd.read_csv(path,header=None,names=['Size','Bearoomes','Price'])

# 2 特征归一化处理
print(data2)
print(data2.mean())
print(data2.std())
data2 = (data2 - data2.mean()) / data2.std()


# 首先建立一个以theta为特征值的代价函数
def computeCost(X,y,theta):
    inner = np.power((X*theta.T)-y,2)
    return np.sum(inner) / (2*len(X))

# 批处理梯度下降
def gradientDescent(X,y,theta,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost=np.zeros(iters)

    for i in range(iters):
        error = (X*theta.T)-y

        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
        
        theta = temp
        cost[i]=computeCost(X,y,theta)
    
    return theta,cost

# 实例计算代价函数
data2.insert(0,'Ones',1)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

alpha = 0.01
iters = 1000
g2, cost2 = gradientDescent(X2,y2,theta2,alpha,iters)
print(computeCost(X2,y2,g2))


# 3 查看训练进程
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost2,'r')
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
ax.set_title("Error vs. Training Data")
plt.show()