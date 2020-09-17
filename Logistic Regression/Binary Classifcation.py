import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns       # 基于matpoltlib，并修饰matplotlib
import scipy.optimize as opt    # 用于优化
from sklearn.metrics import classification_report # 这个包是评价报告

# 1 读取数据
path = '/Users/mac/Desktop/ML/Logistic Regression/ex2data1.txt'
data = pd.read_csv(path,header=None,names=['exam1','exam2','admitted'])
sns.set(context='notebook',style='darkgrid',palette=sns.color_palette('RdBu',2))
sns.lmplot(x='exam1',y='exam2',hue='admitted',data=data,
            size=6,
            fit_reg=False,
            scatter_kws={'s':50})
# plt.show()

# 得到的是ndarray，但形式与matrix相同 eg.[[1 2 3] 
#                                       4 5 6]]
def get_X(df):
    ones = pd.DataFrame({'Ones':np.ones(len(df))})
    data = pd.concat([ones,df],axis=1)
    return data.iloc[:,:-1].values

# 得到的是ndarray，发生了坍缩eg.[0 1 2 3]
def get_y(df):
    return data.iloc[:,-1].values
# 特征归一化

def normalize_feature(df):
    return (df - df.mean()) /  df.std()

# X.shape=(100,3)  y.shape=(100,)
X = get_X(data)
y = get_y(data)



# 2 sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(np.arange(-10,10,step=0.01),
        sigmoid(np.arange(-10,10,step=0.01)))
ax.set_ylim(-0.1,1.1)
ax.set_xlabel('z',fontsize=18)
ax.set_ylabel('g(z)',fontsize=18)
ax.set_title('sigmoid function',fontsize=18)
# plt.show()

# theta.shape = (3,) theta=[0. 0. 0.]
theta = np.zeros(3)


# 3 cost function
def cost(theta,X,y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1-y) * np.log(1 - sigmoid(X @ theta)))
# print(cost(theta,X,y))


# 4 gradient descent
def gradient(theta,X,y):
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)
# print(gradient(theta,X,y))


# 5 拟合参数
res = opt.minimize(fun=cost,x0=theta,args=(X,y),method='Newton-CG',jac=gradient)
# print(res)


# 6 用训练集预测和验证
def predict(X,theta):
    prob = sigmoid(X @ theta)
    return (prob >= 0.5).astype(int)

final_theta = res.x
y_pred = predict(X, final_theta)

# print(classification_report(y,y_pred))


# 7 寻找决策边界
coef = -(res.x / res.x[2])

# 找到决策边界的x和y坐标
x = np.arange(130,step=0.1)
y = coef[0] + coef[1] * x

sns.set(context='notebook',style='ticks',font_scale=1.5)
sns.lmplot(x='exam1',y='exam2',hue='admitted',data=data,
            size=6,
            fit_reg=False,
            scatter_kws={'s':25})
plt.plot(x,y,'grey')
plt.xlim(0,130)
plt.ylim(0,130)
plt.title('Decision Boundary')
#plt.show()

