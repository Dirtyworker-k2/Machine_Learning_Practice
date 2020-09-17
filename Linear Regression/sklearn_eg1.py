import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model


path = "/Users/mac/Desktop/ML/Linear Regression/ex1data1.txt"
data = pd.read_csv(path,header=None,names=['Population','Profit'])  # 读取数据

data.insert(0,'Ones',1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y= data.iloc[:,cols-1:cols]
X = np.matrix(X.values)
y = np.matrix(y.values)


model = linear_model.LinearRegression()     # 模型实例化
model.fit(X,y)      # 训练模型

x = np.array(X[:,1].A1)
f = model.predict(X).flatten() 

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()