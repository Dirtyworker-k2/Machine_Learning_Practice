import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 



# 单变量线性回归

path = "/Users/mac/Desktop/ML/Linear Regression/ex1data1.txt"
data = pd.read_csv(path,header=None,names=['Population','Profit'])  # 读取数据
print(data.head())      # 查看数据的前5行
print(data.describe())      # 查看数据的统计数值
data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8))      # 绘图
plt.show()



# 使用梯度下降算法实现线性回归，以最小化成本函数

# 首先建立一个以theta为特征值的代价函数
def computeCost(X,y,theta):
    inner = np.power((X*theta.T)-y,2)
    return np.sum(inner) / (2*len(X))

# 在Features前加一列x0=1
data.insert(0,'Ones',1)

#set X(training example) and y(target variable)
cols=data.shape[1]      # shape[0]=rows， shape[1]=columns
X=data.iloc[:,0:cols-1]     # X是所有行，去掉最后一列 
y=data.iloc[:,cols-1:cols]      # 所有行，y是最后一列
print(X.head())
print(y.head())
# 代价函数应为numpy矩阵，所以我们需要转换X和y，然后才能继续使用它们。我们还需要初始化theta
X=np.matrix(X.values)
y=np.matrix(y.values)
theta=np.matrix(np.array([0,0]))    # 初始化theta，theta是一个（1，2）矩阵
# X.shape(97,2)  theta.shape(1,2)  y.shape(97,1)

# 计算代价函数初始值（theta初始值为0）
print("\nOrigin cost:%d",computeCost(X,y,theta))


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

alpha=0.01
iters=1000
g,cost = gradientDescent(X,y,theta,alpha,iters)
print("\nparameters:%d\n",g)



# 绘制线性模型以及数据，直观地看出它的拟合
x = np.linspace(data.Population.min(),data.Population.max(),100)
f = g[0,0]+(g[0,1]*x)

fig,ax = plt.subplots(figsize=(12,8))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

# 梯度方程式函数也在每个训练迭代中输出一个代价的向量，所以我们也可以绘制。
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()