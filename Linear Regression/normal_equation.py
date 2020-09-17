import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

path = "/Users/mac/Desktop/ML/Linear Regression/ex1data1.txt"
data = pd.read_csv(path,header=None,names=['Population','Profit'])

data.insert(0,'Ones',1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y= data.iloc[:,cols-1:cols]
X = np.matrix(X.values)
y = np.matrix(y.values)

# 正规方程法
def normalEqn(X,y):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta 

final_theta2 = normalEqn(X,y)
print(final_theta2)