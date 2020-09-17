import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# 1 读取数据
path = '/Users/mac/Desktop/ML/Logistic Regression/ex2data2.txt'
data = pd.read_csv(path,header=None,names=['test1','test2','accepted']) 

sns.set(context='notebook',style='ticks',font_scale=1.5)
sns.lmplot(x='test1',y='test2',hue='accepted',data=data,
            size=6,
            fit_reg=False,
            scatter_kws={'s':50})
plt.title('Regularized Logistic Regression')
plt.show()


# 2 feature mapping(特征映射)